from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .exact_2d import Grid2D, PMFSolution, total_variation


def empirical_pmf_2d(samples: torch.Tensor, grid: Grid2D) -> PMFSolution:
    samples = torch.as_tensor(samples).long().cpu()
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("samples must have shape [N,2].")
    h, w = grid.shape
    inside = (
        (samples[:, 0] >= 0)
        & (samples[:, 0] < h)
        & (samples[:, 1] >= 0)
        & (samples[:, 1] < w)
    )
    pmf = np.zeros((h, w), dtype=np.float64)
    if inside.any():
        flat = samples[inside, 0].numpy() * w + samples[inside, 1].numpy()
        counts = np.bincount(flat, minlength=h * w).reshape(h, w)
        pmf = counts.astype(np.float64) / float(samples.shape[0])
    overflow = float((~inside).float().mean().item())
    return PMFSolution(pmf, overflow)


def empirical_w2(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    max_points: int = 1000,
    repeats: int = 3,
    seed: int = 123,
) -> float:
    """Empirical 2-Wasserstein via optimal bipartite matching."""
    x = torch.as_tensor(x, dtype=torch.float64).cpu()
    y = torch.as_tensor(y, dtype=torch.float64).cpu()
    n = min(x.shape[0], y.shape[0], int(max_points))
    if n < 2:
        return float("nan")
    generator = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(int(repeats)):
        xi = x[torch.randperm(x.shape[0], generator=generator)[:n]]
        yi = y[torch.randperm(y.shape[0], generator=generator)[:n]]
        cost = torch.cdist(xi, yi).pow(2).numpy()
        row, col = linear_sum_assignment(cost)
        values.append(math.sqrt(float(cost[row, col].mean())))
    return float(np.mean(values))


def sliced_w2(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    n_projections: int = 128,
    max_points: int = 10000,
    seed: int = 123,
) -> float:
    x = torch.as_tensor(x, dtype=torch.float32).cpu()
    y = torch.as_tensor(y, dtype=torch.float32).cpu()
    n = min(x.shape[0], y.shape[0], int(max_points))
    generator = torch.Generator().manual_seed(seed)
    x = x[torch.randperm(x.shape[0], generator=generator)[:n]]
    y = y[torch.randperm(y.shape[0], generator=generator)[:n]]
    direction = torch.randn(x.shape[1], int(n_projections), generator=generator)
    direction = direction / direction.norm(dim=0, keepdim=True).clamp_min(1e-12)
    x_proj = torch.sort(x @ direction, dim=0).values
    y_proj = torch.sort(y @ direction, dim=0).values
    return float(torch.sqrt(torch.mean((x_proj - y_proj).pow(2))).item())


def _median_sigma(x: torch.Tensor, y: torch.Tensor, max_points: int, seed: int) -> float:
    generator = torch.Generator().manual_seed(seed)
    z = torch.cat([x, y], dim=0)
    if z.shape[0] > max_points:
        z = z[torch.randperm(z.shape[0], generator=generator)[:max_points]]
    distance = torch.pdist(z)
    positive = distance[distance > 0]
    return 1.0 if positive.numel() == 0 else float(positive.median().item())


def block_rbf_mmd2(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    sigma: float = 0.0,
    max_points: int = 5000,
    block_size: int = 512,
    seed: int = 123,
) -> float:
    """Unbiased MMD^2 computed in blocks without an O(N^2) allocation."""
    x = torch.as_tensor(x, dtype=torch.float32).cpu()
    y = torch.as_tensor(y, dtype=torch.float32).cpu()
    generator = torch.Generator().manual_seed(seed)
    if x.shape[0] > max_points:
        x = x[torch.randperm(x.shape[0], generator=generator)[:max_points]]
    if y.shape[0] > max_points:
        y = y[torch.randperm(y.shape[0], generator=generator)[:max_points]]
    if sigma <= 0:
        sigma = _median_sigma(x, y, min(1000, max_points), seed)
    gamma = 1.0 / max(2.0 * sigma * sigma, 1e-12)

    def kernel_sum(a: torch.Tensor, b: torch.Tensor, remove_diagonal: bool) -> Tuple[float, int]:
        total = 0.0
        count = 0
        for i in range(0, a.shape[0], block_size):
            ai = a[i : i + block_size]
            for j in range(0, b.shape[0], block_size):
                bj = b[j : j + block_size]
                kernel = torch.exp(-gamma * torch.cdist(ai, bj).pow(2))
                block_count = kernel.numel()
                if remove_diagonal and a.data_ptr() == b.data_ptr() and i == j:
                    diagonal = kernel.diag().sum()
                    total += float((kernel.sum() - diagonal).item())
                    block_count -= min(ai.shape[0], bj.shape[0])
                else:
                    total += float(kernel.sum().item())
                count += block_count
        return total, count

    xx_sum, xx_count = kernel_sum(x, x, True)
    yy_sum, yy_count = kernel_sum(y, y, True)
    xy_sum, xy_count = kernel_sum(x, y, False)
    value = xx_sum / max(xx_count, 1) + yy_sum / max(yy_count, 1) - 2.0 * xy_sum / max(xy_count, 1)
    return float(max(value, 0.0))


def count_summary_errors(generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    generated = torch.as_tensor(generated, dtype=torch.float64).cpu()
    target = torch.as_tensor(target, dtype=torch.float64).cpu()
    gen_mean = generated.mean(0)
    tgt_mean = target.mean(0)
    gen_var = generated.var(0, unbiased=True)
    tgt_var = target.var(0, unbiased=True)
    gen_zero = (generated == 0).double().mean(0)
    tgt_zero = (target == 0).double().mean(0)

    def corr(x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(0)
        cov = centered.T @ centered / max(x.shape[0] - 1, 1)
        sd = torch.sqrt(torch.diag(cov).clamp_min(1e-12))
        return cov / (sd[:, None] * sd[None, :]).clamp_min(1e-12)

    gen_corr = corr(generated)
    tgt_corr = corr(target)
    return {
        "mean_relative_l1": float(
            torch.mean(torch.abs(gen_mean - tgt_mean) / (torch.abs(tgt_mean) + 1e-6)).item()
        ),
        "variance_relative_l1": float(
            torch.mean(torch.abs(gen_var - tgt_var) / (torch.abs(tgt_var) + 1e-6)).item()
        ),
        "zero_fraction_l1": float(torch.mean(torch.abs(gen_zero - tgt_zero)).item()),
        "correlation_fro_per_dim": float(
            (torch.linalg.norm(gen_corr - tgt_corr) / generated.shape[1]).item()
        ),
    }


def evaluate_samples(
    generated: torch.Tensor,
    target: torch.Tensor,
    *,
    exact_grid: Optional[Grid2D] = None,
    exact_target_pmf: Optional[PMFSolution] = None,
    seed: int = 123,
    mmd_max_points: int = 5000,
    w2_max_points: int = 1000,
    w2_repeats: int = 3,
    sliced_w2_projections: int = 128,
    sliced_w2_max_points: int = 10000,
) -> Dict[str, float]:
    result = count_summary_errors(generated, target)
    result["mmd2_rbf"] = block_rbf_mmd2(
        generated, target, seed=seed, max_points=mmd_max_points
    )
    if generated.shape[1] == 2:
        result["w2"] = empirical_w2(
            generated, target, seed=seed, max_points=w2_max_points, repeats=w2_repeats
        )
    else:
        result["sliced_w2"] = sliced_w2(
            generated,
            target,
            seed=seed,
            n_projections=sliced_w2_projections,
            max_points=sliced_w2_max_points,
        )
    if exact_grid is not None and exact_target_pmf is not None:
        generated_pmf = empirical_pmf_2d(generated, exact_grid)
        result["tv"] = total_variation(generated_pmf, exact_target_pmf)
        result["overflow_mass"] = generated_pmf.overflow_mass
    return result


def interpolated_equal_quality_nfe(
    reference_nfe: Sequence[float],
    reference_error: Sequence[float],
    target_error: float,
) -> float:
    """Log-log interpolation of the NFE needed to reach target_error."""
    nfe = np.asarray(reference_nfe, dtype=np.float64)
    error = np.asarray(reference_error, dtype=np.float64)
    order = np.argsort(nfe)
    nfe, error = nfe[order], error[order]
    for left in range(len(nfe) - 1):
        e0, e1 = error[left], error[left + 1]
        if (e0 - target_error) * (e1 - target_error) <= 0 and e0 != e1:
            x0, x1 = np.log(nfe[left]), np.log(nfe[left + 1])
            y0, y1 = np.log(max(e0, 1e-16)), np.log(max(e1, 1e-16))
            y = np.log(max(target_error, 1e-16))
            fraction = (y - y0) / (y1 - y0)
            return float(np.exp(x0 + fraction * (x1 - x0)))
    return float("nan")
