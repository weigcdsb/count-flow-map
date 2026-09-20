from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr, wasserstein_distance

from .simulation_metrics import block_rbf_mmd2, sliced_w2


@dataclass
class PCAProjection:
    mean: torch.Tensor
    components: torch.Tensor

    def transform(self, counts: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(counts, dtype=torch.float32)
        library = x.sum(dim=1, keepdim=True).clamp_min(1.0)
        normalized = torch.log1p(1e4 * x / library)
        return (normalized - self.mean) @ self.components.T


def fit_count_pca(
    counts: torch.Tensor,
    *,
    n_components: int = 50,
    max_cells: int = 20000,
    seed: int = 42,
) -> PCAProjection:
    x = torch.as_tensor(counts, dtype=torch.float32).cpu()
    if x.shape[0] > int(max_cells):
        generator = torch.Generator().manual_seed(seed)
        x = x[torch.randperm(x.shape[0], generator=generator)[: int(max_cells)]]
    library = x.sum(dim=1, keepdim=True).clamp_min(1.0)
    normalized = torch.log1p(1e4 * x / library)
    mean = normalized.mean(dim=0, keepdim=True)
    centered = normalized - mean
    q = min(int(n_components), centered.shape[0] - 1, centered.shape[1])
    if q < 1:
        raise ValueError("Not enough observations to fit PCA.")
    # Isolate and seed the randomized SVD as well as row subsampling.
    # Inputs are explicitly on CPU, so only the CPU generator is changed.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        _, _, v = torch.pca_lowrank(centered, q=q, center=False)
    return PCAProjection(mean.squeeze(0), v[:, :q].T.contiguous())


def _safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    value = pearsonr(x, y).statistic if kind == "pearson" else spearmanr(x, y).statistic
    return float(value)


def _correlation_matrix(x: torch.Tensor) -> torch.Tensor:
    x = x.double()
    centered = x - x.mean(dim=0)
    cov = centered.T @ centered / max(x.shape[0] - 1, 1)
    sd = torch.sqrt(torch.diag(cov).clamp_min(1e-12))
    return cov / (sd[:, None] * sd[None, :]).clamp_min(1e-12)


def _r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
    denom = float(np.sum((target - target.mean()) ** 2))
    return float(1.0 - np.sum((target - prediction) ** 2) / denom) if denom > 1e-12 else float("nan")


def _explained_variance(target: np.ndarray, prediction: np.ndarray) -> float:
    denom = float(np.var(target))
    return float(1.0 - np.var(target - prediction) / denom) if denom > 1e-12 else float("nan")


def raw_effect_cross_moments(generated, target, control) -> Dict[str, float]:
    """Cross-row raw-count moments; do not include same-row sampling noise.

    For independent row triplets (C_i,T_i,G_i), O_i=T_i-C_i and P_i=G_i-C_i,
    cross-products with i != j estimate ||E O||^2, <E O,E P>, and ||E P||^2.
    Within-row dependence (including G_i depending on C_i) is allowed. Genes
    may be dependent. Values are averaged over genes and can be negative at
    finite sample size. These are cell-sampling diagnostics, not uncertainty
    over biological replicates, and the identities apply to RAW mean effects.
    """
    keys = ("raw_effect_cross_signal", "raw_effect_cross_alignment",
            "raw_effect_cross_prediction", "raw_effect_cross_mse")
    g, t, c = [torch.as_tensor(x, dtype=torch.float64).cpu() for x in (generated, target, control)]
    if g.shape != t.shape or g.shape != c.shape or g.ndim != 2 or len(g) < 2:
        return dict.fromkeys(keys, float("nan"))
    o, p = t - c, g - c
    n, d = o.shape
    def cross(a, b):
        return float((a.sum(0).dot(b.sum(0)) - (a * b).sum()) / (n * (n - 1) * d))
    signal, alignment, prediction = cross(o, o), cross(o, p), cross(p, p)
    return dict(zip(keys, (signal, alignment, prediction, signal + prediction - 2 * alignment)))


def scrna_condition_metrics(
    generated: torch.Tensor,
    target: torch.Tensor,
    control: torch.Tensor,
    projection: PCAProjection,
    *,
    seed: int = 42,
    mmd_max_points: int = 3000,
    sw2_max_points: int = 5000,
    n_projections: int = 128,
    top_response_genes: int = 200,
    logfc_pseudocount: float = 0.1,
) -> Dict[str, float]:
    """Distributional and perturbation-effect metrics for one held-out condition.

    Delta and logFC are kept distinct.  Earlier application code accidentally
    reported the log1p-effect correlation again under the delta metric names.
    """
    generated = torch.as_tensor(generated).cpu().long()
    target = torch.as_tensor(target).cpu().long()
    control = torch.as_tensor(control).cpu().long()
    gen_pca = projection.transform(generated)
    target_pca = projection.transform(target)
    result = {
        "sliced_w2": sliced_w2(
            gen_pca, target_pca, n_projections=n_projections,
            max_points=sw2_max_points, seed=seed,
        ),
        "mmd2_rbf": block_rbf_mmd2(
            gen_pca, target_pca, max_points=mmd_max_points, seed=seed,
        ),
    }

    gen = generated.double(); tgt = target.double(); ctl = control.double()
    result.update(raw_effect_cross_moments(gen, tgt, ctl))
    gen_mean, tgt_mean = gen.mean(0), tgt.mean(0)
    gen_var = gen.var(0, unbiased=True); tgt_var = tgt.var(0, unbiased=True)
    gen_zero = (gen == 0).double().mean(0); tgt_zero = (tgt == 0).double().mean(0)
    result.update({
        "gene_mean_relative_l1": float(torch.mean(torch.abs(gen_mean - tgt_mean) / (tgt_mean.abs() + 1e-3)).item()),
        "gene_variance_relative_l1": float(torch.mean(torch.abs(gen_var - tgt_var) / (tgt_var.abs() + 1e-3)).item()),
        "zero_fraction_l1": float(torch.mean(torch.abs(gen_zero - tgt_zero)).item()),
        "library_size_w1": float(wasserstein_distance(gen.sum(1).numpy(), tgt.sum(1).numpy())),
    })

    control_mean = ctl.mean(0)
    real_delta = (tgt_mean - control_mean).numpy()
    generated_delta = (gen_mean - control_mean).numpy()

    pseudocount = max(float(logfc_pseudocount), 1e-12)
    real_logfc = torch.log2((tgt_mean + pseudocount) / (control_mean + pseudocount)).numpy()
    generated_logfc = torch.log2((gen_mean + pseudocount) / (control_mean + pseudocount)).numpy()

    real_log1p_effect = (torch.log1p(tgt_mean) - torch.log1p(control_mean)).numpy()
    generated_log1p_effect = (torch.log1p(gen_mean) - torch.log1p(control_mean)).numpy()

    k = min(int(top_response_genes), real_logfc.size)
    real_top = np.argpartition(np.abs(real_logfc), -k)[-k:] if k > 0 else np.asarray([], dtype=int)
    generated_top = np.argpartition(np.abs(generated_logfc), -k)[-k:] if k > 0 else np.asarray([], dtype=int)
    overlap = len(set(real_top.tolist()) & set(generated_top.tolist())) / max(k, 1)
    direction = (
        float(np.mean(np.sign(real_logfc[real_top]) == np.sign(generated_logfc[real_top])))
        if k > 0 else float("nan")
    )

    result.update({
        "delta_pearson": _safe_corr(real_delta, generated_delta, "pearson"),
        "delta_spearman": _safe_corr(real_delta, generated_delta, "spearman"),
        "delta_mae": float(np.mean(np.abs(real_delta - generated_delta))),
        "delta_r2": _r2_score(real_delta, generated_delta),
        "delta_explained_variance": _explained_variance(real_delta, generated_delta),
        "logfc_pearson": _safe_corr(real_logfc, generated_logfc, "pearson"),
        "logfc_spearman": _safe_corr(real_logfc, generated_logfc, "spearman"),
        "logfc_mae": float(np.mean(np.abs(real_logfc - generated_logfc))),
        "logfc_r2": _r2_score(real_logfc, generated_logfc),
        "logfc_explained_variance": _explained_variance(real_logfc, generated_logfc),
        "log1p_effect_slope": float(np.dot(real_log1p_effect, generated_log1p_effect) / max(float(np.dot(real_log1p_effect, real_log1p_effect)), 1e-12)),
        "log1p_effect_r2": _r2_score(real_log1p_effect, generated_log1p_effect),
        "log1p_effect_rmse": float(np.sqrt(np.mean((real_log1p_effect-generated_log1p_effect)**2))),
        "log1p_effect_pearson": _safe_corr(real_log1p_effect, generated_log1p_effect, "pearson"),
        "log1p_effect_spearman": _safe_corr(real_log1p_effect, generated_log1p_effect, "spearman"),
        "top_response_gene_overlap": float(overlap),
        "top_response_direction_accuracy": direction,
        "top_de_overlap": float(overlap),
    })
    if k > 1:
        result.update({
            "deg_delta_pearson": _safe_corr(real_delta[real_top], generated_delta[real_top], "pearson"),
            "deg_delta_spearman": _safe_corr(real_delta[real_top], generated_delta[real_top], "spearman"),
            "deg_logfc_pearson": _safe_corr(real_logfc[real_top], generated_logfc[real_top], "pearson"),
            "deg_logfc_spearman": _safe_corr(real_logfc[real_top], generated_logfc[real_top], "spearman"),
        })
    else:
        result.update({
            "deg_delta_pearson": float("nan"),
            "deg_delta_spearman": float("nan"),
            "deg_logfc_pearson": float("nan"),
            "deg_logfc_spearman": float("nan"),
        })
    return result


def aggregate_condition_metrics(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row if key != "condition_id"})
    result: Dict[str, float] = {}
    for key in keys:
        values = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
        finite = np.isfinite(values)
        result[key] = float(values[finite].mean()) if finite.any() else float("nan")
    return result


def evaluate_scrna_by_condition(
    generated: torch.Tensor,
    split_x0: torch.Tensor,
    split_x1: torch.Tensor,
    condition_id: torch.Tensor,
    projection: PCAProjection,
    *,
    seed: int = 42,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    rows = []
    for condition in sorted(condition_id.unique().tolist()):
        mask = condition_id == int(condition)
        metric = scrna_condition_metrics(
            generated[mask], split_x1[mask], split_x0[mask], projection, seed=seed + int(condition)
        )
        metric["condition_id"] = float(condition)
        rows.append(metric)
    return aggregate_condition_metrics(rows), rows


def multivariate_energy_score(
    samples: torch.Tensor,
    observations: torch.Tensor,
    *,
    max_cases: int = 2000,
    max_draws: int = 64,
    seed: int = 42,
) -> float:
    """Energy score for samples [cases,draws,dim] and observations [cases,dim]."""
    samples = torch.as_tensor(samples, dtype=torch.float32).cpu()
    observations = torch.as_tensor(observations, dtype=torch.float32).cpu()
    if samples.ndim != 3 or observations.shape != (samples.shape[0], samples.shape[2]):
        raise ValueError("samples must be [cases,draws,dim] and observations [cases,dim].")
    generator = torch.Generator().manual_seed(seed)
    if samples.shape[0] > int(max_cases):
        index = torch.randperm(samples.shape[0], generator=generator)[: int(max_cases)]
        samples, observations = samples[index], observations[index]
    if samples.shape[1] > int(max_draws):
        draw = torch.randperm(samples.shape[1], generator=generator)[: int(max_draws)]
        samples = samples[:, draw]
    first = torch.linalg.norm(samples - observations[:, None, :], dim=-1).mean()
    draw_count = samples.shape[1]
    if draw_count < 2:
        return float(first.item())
    perm = torch.randperm(draw_count, generator=generator)
    second = torch.linalg.norm(samples - samples[:, perm, :], dim=-1).mean()
    return float((first - 0.5 * second).item())


def _pearson_torch(x: torch.Tensor, y: torch.Tensor, dim: int) -> torch.Tensor:
    x = x.float()
    y = y.float()
    x = x - x.mean(dim=dim, keepdim=True)
    y = y - y.mean(dim=dim, keepdim=True)
    numerator = (x * y).sum(dim=dim)
    denominator = torch.sqrt((x.square().sum(dim=dim) * y.square().sum(dim=dim)).clamp_min(1e-12))
    return numerator / denominator


def spikeprophecy_population_metrics(
    predictive_mean: torch.Tensor,
    observations: torch.Tensor,
) -> Dict[str, float]:
    prediction = torch.as_tensor(predictive_mean, dtype=torch.float32).cpu()
    target = torch.as_tensor(observations, dtype=torch.float32).cpu()
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must have shape [time,neurons].")
    pop_rate_r = _pearson_torch(target.sum(1), prediction.sum(1), dim=0)
    spatial = _pearson_torch(target, prediction, dim=1)
    cosine = torch.nn.functional.cosine_similarity(target, prediction, dim=1, eps=1e-8)
    neuron_r = _pearson_torch(target.T, prediction.T, dim=1)
    weights = target.var(0, unbiased=True)
    weighted_r = (weights * neuron_r).sum() / weights.sum().clamp_min(1e-12)
    return {
        "weighted_neuron_r": float(weighted_r.item()),
        "population_rate_r": float(pop_rate_r.item()),
        "spatial_r": float(torch.nanmean(spatial).item()),
        "cosine_similarity": float(torch.nanmean(cosine).item()),
        "median_neuron_r": float(torch.nanmedian(neuron_r).item()),
    }


def neural_calibration_metrics(samples: torch.Tensor, observations: torch.Tensor) -> Dict[str, float]:
    samples = torch.as_tensor(samples, dtype=torch.float64).cpu()
    observations = torch.as_tensor(observations, dtype=torch.float64).cpu()
    if samples.ndim != 3:
        raise ValueError("samples must have shape [cases,draws,neurons].")
    generated = samples.reshape(-1, samples.shape[-1])
    repeated_target = observations[:, None, :].expand_as(samples).reshape(-1, samples.shape[-1])
    gen_mean = generated.mean(0)
    target_mean = repeated_target.mean(0)
    gen_var = generated.var(0, unbiased=True)
    target_var = repeated_target.var(0, unbiased=True)
    gen_zero = (generated == 0).double().mean(0)
    target_zero = (repeated_target == 0).double().mean(0)
    gen_population = samples.sum(-1).reshape(-1).numpy()
    target_population = observations.sum(-1).numpy()
    predictive_cov = _correlation_matrix(generated)
    target_cov = _correlation_matrix(observations)
    return {
        "mean_relative_l1": float(torch.mean(torch.abs(gen_mean - target_mean) / (target_mean.abs() + 1e-3)).item()),
        "variance_relative_l1": float(torch.mean(torch.abs(gen_var - target_var) / (target_var.abs() + 1e-3)).item()),
        "zero_fraction_l1": float(torch.mean(torch.abs(gen_zero - target_zero)).item()),
        "correlation_fro_per_neuron": float((torch.linalg.norm(predictive_cov - target_cov) / observations.shape[1]).item()),
        "population_count_w1": float(wasserstein_distance(gen_population, target_population)),
    }


def evaluate_neural_forecast_samples(
    samples: torch.Tensor,
    observations: torch.Tensor,
    *,
    seed: int = 42,
) -> Dict[str, float]:
    samples = torch.as_tensor(samples).cpu()
    observations = torch.as_tensor(observations).cpu()
    predictive_mean = samples.float().mean(dim=1)
    result = {
        "energy_score": multivariate_energy_score(samples, observations, seed=seed),
    }
    result.update(spikeprophecy_population_metrics(predictive_mean, observations))
    result.update(neural_calibration_metrics(samples, observations))
    return result


def select_best_nfe(
    nfe_values: Sequence[int],
    validation_errors: Sequence[float],
) -> int:
    """Select the validation-minimizing NFE without making a saturation claim."""
    if len(nfe_values) != len(validation_errors) or not nfe_values:
        raise ValueError("nfe_values and validation_errors must be aligned and nonempty.")
    errors = np.asarray(validation_errors, dtype=np.float64)
    if not np.isfinite(errors).any():
        raise ValueError("validation_errors contain no finite values.")
    return int(nfe_values[int(np.nanargmin(errors))])


def select_saturated_nfe(
    nfe_values: Sequence[int],
    validation_errors: Sequence[float],
    *,
    relative_tolerance: float = 0.01,
) -> int:
    if len(nfe_values) != len(validation_errors) or not nfe_values:
        raise ValueError("nfe_values and validation_errors must be aligned and nonempty.")
    best = float(np.nanmin(np.asarray(validation_errors, dtype=np.float64)))
    threshold = (1.0 + float(relative_tolerance)) * best
    candidates = [int(nfe) for nfe, error in zip(nfe_values, validation_errors) if float(error) <= threshold]
    return min(candidates) if candidates else int(nfe_values[int(np.nanargmin(validation_errors))])
