from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .application_data import ConditionalPairBundle, ConditionalPairSplit
from .conditional_model import VectorContextEncoder
from .model import MLP
from .utils import resolve_device, set_seed


SCRNA_RESPONSE_BASELINE_VERSION = "scrna-response-baselines-v3"
SCGEN_LABEL = "scGen (nearest-dose)"
SCVIDR_LABEL = "scVIDR"
CPA_LABEL = "CPA"
NBVAE_LABEL = "Conditional NB-VAE"
LINEAR_LABEL = "Linear dose-response"
SINKHORN_LABEL = "Sinkhorn OT (dose interpolation)"


@dataclass(frozen=True)
class ConditionInfo:
    condition_id: int
    cell_line_id: str
    drug: str
    dose: float


@dataclass
class PublishedBaselineTrainConfig:
    steps: int = 30_000
    epochs: Optional[int] = None
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    adam_eps: float = 1e-8
    grad_clip: float = 5.0
    log_every: int = 500
    seed: int = 42


def published_train_config(cfg: Mapping[str, object], seed: int) -> PublishedBaselineTrainConfig:
    epochs_value = cfg.get("epochs")
    return PublishedBaselineTrainConfig(
        steps=int(cfg.get("steps", 30_000)),
        epochs=(None if epochs_value is None else int(epochs_value)),
        batch_size=int(cfg.get("batch_size", 64)),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-6)),
        adam_eps=float(cfg.get("adam_eps", 1e-8)),
        grad_clip=float(cfg.get("grad_clip", 5.0)),
        log_every=max(1, int(cfg.get("log_every", 500))),
        seed=int(seed),
    )


def _condition_lookup(bundle: ConditionalPairBundle) -> Dict[int, ConditionInfo]:
    raw = bundle.metadata.get("conditions", [])
    lookup: Dict[int, ConditionInfo] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, Mapping) or "condition_id" not in item:
                continue
            condition_id = int(item["condition_id"])
            lookup[condition_id] = ConditionInfo(
                condition_id=condition_id,
                cell_line_id=str(item.get("cell_line_id", "")),
                drug=str(item.get("drug", "")),
                dose=float(item.get("dose", 0.0)),
            )
    if lookup:
        return lookup

    # Synthetic smoke bundles predate condition metadata. Their context layout is
    # [cell one-hot, drug one-hot, log1p(dose)], so reconstruct metadata only for tests.
    n_cells = int(bundle.metadata.get("n_cell_lines", 0))
    n_drugs = int(bundle.metadata.get("n_drugs", 0))
    if n_cells < 1 or n_drugs < 1:
        raise ValueError("Published scRNA baselines require condition metadata.")
    all_splits = (bundle.train, bundle.val, bundle.test)
    for split in all_splits:
        for condition_id in map(int, split.condition_id.unique().tolist()):
            if condition_id in lookup:
                continue
            row = int(torch.nonzero(split.condition_id == condition_id, as_tuple=False)[0].item())
            context = split.context[row]
            cell = int(torch.argmax(context[:n_cells]).item())
            drug = int(torch.argmax(context[n_cells : n_cells + n_drugs]).item())
            dose = float(torch.expm1(context[n_cells + n_drugs]).item())
            lookup[condition_id] = ConditionInfo(condition_id, f"cell{cell}", f"drug{drug}", dose)
    return lookup


def _train_pair_conditions(
    bundle: ConditionalPairBundle,
    lookup: Mapping[int, ConditionInfo],
    cell_line_id: str,
    drug: str,
) -> List[ConditionInfo]:
    train_ids = set(map(int, bundle.train.condition_id.unique().tolist()))
    rows = [
        info for condition_id, info in lookup.items()
        if condition_id in train_ids and info.cell_line_id == cell_line_id and info.drug == drug
    ]
    return sorted(rows, key=lambda value: value.dose)


def _log_normalize(counts: torch.Tensor, target_sum: float = 1e4) -> torch.Tensor:
    value = torch.as_tensor(counts, dtype=torch.float32)
    library = value.sum(-1, keepdim=True).clamp_min(1.0)
    return torch.log1p(float(target_sum) * value / library)


def _training_library_log_slope(
    bundle: ConditionalPairBundle,
    lookup: Mapping[int, ConditionInfo],
    cell: str,
    drug: str,
) -> float:
    conditions = _train_pair_conditions(bundle, lookup, cell, drug)
    numerator = 0.0
    denominator = 0.0
    for info in conditions:
        mask = bundle.train.condition_id == info.condition_id
        if not bool(mask.any()):
            continue
        source_lib = bundle.train.x0[mask].double().sum(-1).mean().clamp_min(1.0)
        target_lib = bundle.train.x1[mask].double().sum(-1).mean().clamp_min(1.0)
        x = float(np.log1p(max(info.dose, 0.0)))
        y = float(torch.log(target_lib / source_lib).item())
        numerator += x * y
        denominator += x * x
    return numerator / max(denominator, 1e-12)


def _normalized_to_counts(
    normalized_log: torch.Tensor,
    source_counts: torch.Tensor,
    *,
    library_log_multiplier: float = 0.0,
) -> torch.Tensor:
    normalized_log = torch.as_tensor(normalized_log, dtype=torch.float32)
    source = torch.as_tensor(source_counts, dtype=torch.float32, device=normalized_log.device)
    abundance = torch.expm1(normalized_log.clamp(min=0.0, max=12.0)).clamp_min(0.0)
    total = abundance.sum(-1, keepdim=True)
    fallback = torch.full_like(abundance, 1.0 / max(abundance.shape[1], 1))
    proportions = torch.where(total > 1e-12, abundance / total.clamp_min(1e-12), fallback)
    source_library = source.sum(-1, keepdim=True).clamp_min(1.0)
    target_library = source_library * float(np.exp(np.clip(library_log_multiplier, -3.0, 3.0)))
    return torch.round(proportions * target_library).clamp_min(0.0).long()


def _mlp_block(input_dim: int, hidden_dim: int, depth: int, dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    current = int(input_dim)
    for _ in range(int(depth)):
        layers.extend([
            nn.Linear(current, int(hidden_dim)),
            nn.BatchNorm1d(int(hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.Dropout(float(dropout)),
        ])
        current = int(hidden_dim)
    return nn.Sequential(*layers)


class ScGenVAE(nn.Module):
    """PyTorch reproduction of the scGen VAE architecture used for latent arithmetic.

    scGen's public model defaults are 800 hidden units, 100 latent dimensions,
    two hidden layers and 0.2 dropout. Training uses normalized/log1p expression;
    the perturbation prediction itself is latent-vector arithmetic.
    """

    def __init__(
        self,
        dim: int,
        *,
        hidden_dim: int = 800,
        latent_dim: int = 100,
        depth: int = 2,
        dropout: float = 0.2,
        kl_weight: float = 5e-5,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.latent_dim = int(latent_dim)
        self.kl_weight = float(kl_weight)
        self.encoder = _mlp_block(self.dim, hidden_dim, depth, dropout)
        self.mean = nn.Linear(int(hidden_dim), self.latent_dim)
        self.logvar = nn.Linear(int(hidden_dim), self.latent_dim)
        self.decoder_hidden = _mlp_block(self.latent_dim, hidden_dim, depth, dropout)
        self.decoder = nn.Linear(int(hidden_dim), self.dim)

    def encode(self, normalized_log: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(normalized_log)
        return self.mean(hidden), self.logvar(hidden).clamp(-10.0, 10.0)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.decoder_hidden(z))

    def forward(self, normalized_log: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(normalized_log)
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return self.decode(z), mean, logvar


@torch.no_grad()
def _encode_counts(
    model: ScGenVAE,
    counts: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    model = model.to(device).eval()
    outputs: List[torch.Tensor] = []
    for first in range(0, counts.shape[0], int(batch_size)):
        x = _log_normalize(counts[first : first + int(batch_size)].to(device))
        mean, _ = model.encode(x)
        outputs.append(mean.cpu())
    return torch.cat(outputs, dim=0)


def build_scgen(bundle: ConditionalPairBundle, cfg: Mapping[str, object]) -> ScGenVAE:
    return ScGenVAE(
        bundle.dim,
        hidden_dim=int(cfg.get("hidden_dim", 800)),
        latent_dim=int(cfg.get("latent_dim", 100)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.2)),
        kl_weight=float(cfg.get("kl_weight", 5e-5)),
    )


def train_scgen(
    model: ScGenVAE,
    train: ConditionalPairSplit,
    config: PublishedBaselineTrainConfig,
    *,
    device: Optional[str] = None,
    verbose: bool = True,
    run_state=None,
) -> Tuple[ScGenVAE, Dict[str, list]]:
    set_seed(config.seed)
    torch_device = resolve_device(device)
    model = model.to(torch_device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, eps=config.adam_eps
    )
    generator = torch.Generator(device=train.x0.device).manual_seed(config.seed + 1701)
    history: Dict[str, list] = {"step": [], "loss": [], "reconstruction": [], "kl": []}
    model.train()
    n = train.n
    steps = int(config.steps)
    if config.epochs is not None:
        # scGen/scVIDR report training in epochs.  The VAE sees both DMSO and
        # treated training cells, so one epoch contains 2*n observations.
        steps = int(config.epochs) * max(1, int(np.ceil((2 * n) / float(config.batch_size))))
    optimizers = {"main": optimizer}
    start_step = run_state.begin(model, optimizers, generator, history, steps) if run_state else 1
    for step in range(start_step, steps + 1):
        index = torch.randint(2 * n, (int(config.batch_size),), generator=generator, device=train.x0.device)
        is_target = index >= n
        row = index.remainder(n)
        counts = torch.where(is_target[:, None], train.x1[row], train.x0[row]).to(torch_device)
        normalized = _log_normalize(counts)
        reconstruction, mean, logvar = model(normalized)
        recon = (reconstruction - normalized).square().sum(-1)
        kl = 0.5 * (mean.square() + logvar.exp() - logvar - 1.0).sum(-1)
        # This matches the weighting used in the public scGen VAE implementation.
        loss = (0.5 * recon + 0.5 * model.kl_weight * kl).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        if step == 1 or step % config.log_every == 0 or step == steps:
            history["step"].append(step)
            history["loss"].append(float(loss.detach().cpu()))
            history["reconstruction"].append(float(recon.mean().detach().cpu()))
            history["kl"].append(float(kl.mean().detach().cpu()))
            if verbose:
                print(
                    f"step={step:6d} scgen_loss={history['loss'][-1]:9.5f} "
                    f"recon={history['reconstruction'][-1]:9.3f} kl={history['kl'][-1]:9.3f}",
                    flush=True,
                )
        if run_state is not None:
            run_state.after_step(model, optimizers, generator, history, step)
    if run_state is not None:
        run_state.finish(model, history)
    model.eval()
    return model, history


def _latent_condition_effects(
    model: ScGenVAE,
    bundle: ConditionalPairBundle,
    lookup: Mapping[int, ConditionInfo],
    *,
    device: torch.device,
    batch_size: int,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    z0 = _encode_counts(model, bundle.train.x0, device=device, batch_size=batch_size)
    z1 = _encode_counts(model, bundle.train.x1, device=device, batch_size=batch_size)
    source: Dict[int, torch.Tensor] = {}
    delta: Dict[int, torch.Tensor] = {}
    for condition_id in map(int, bundle.train.condition_id.unique().tolist()):
        if condition_id not in lookup:
            continue
        mask = bundle.train.condition_id == condition_id
        source[condition_id] = z0[mask].mean(0)
        delta[condition_id] = z1[mask].mean(0) - z0[mask].mean(0)
    return source, delta


def _fit_effect_slope(
    conditions: Sequence[ConditionInfo],
    effects: Mapping[int, torch.Tensor],
) -> Optional[torch.Tensor]:
    usable = [(float(np.log1p(max(info.dose, 0.0))), effects.get(info.condition_id)) for info in conditions]
    usable = [(dose, effect) for dose, effect in usable if effect is not None and dose > 0]
    if not usable:
        return None
    denominator = sum(dose * dose for dose, _ in usable)
    result = sum(dose * effect for dose, effect in usable) / max(denominator, 1e-12)
    return result


@torch.no_grad()
def generate_scgen(
    model: ScGenVAE,
    bundle: ConditionalPairBundle,
    split: ConditionalPairSplit,
    *,
    device: Optional[str] = None,
    batch_size: int = 256,
) -> torch.Tensor:
    torch_device = resolve_device(device)
    model = model.to(torch_device).eval()
    lookup = _condition_lookup(bundle)
    _, effects = _latent_condition_effects(
        model, bundle, lookup, device=torch_device, batch_size=batch_size
    )
    generated = torch.empty_like(split.x1)
    for condition_id in sorted(map(int, split.condition_id.unique().tolist())):
        info = lookup[condition_id]
        candidates = _train_pair_conditions(bundle, lookup, info.cell_line_id, info.drug)
        slope = _fit_effect_slope(candidates, effects)
        if slope is None:
            raise ValueError(f"scGen found no training response for {info.cell_line_id}/{info.drug}.")
        target_delta = slope * float(np.log1p(max(info.dose, 0.0)))
        mask = split.condition_id == condition_id
        x0 = split.x0[mask]
        z0 = _encode_counts(model, x0, device=torch_device, batch_size=batch_size).to(torch_device)
        decoded = model.decode(z0 + target_delta.to(torch_device))
        lib_slope = _training_library_log_slope(bundle, lookup, info.cell_line_id, info.drug)
        generated[mask] = _normalized_to_counts(
            decoded, x0.to(torch_device),
            library_log_multiplier=lib_slope * float(np.log1p(max(info.dose, 0.0))),
        ).cpu()
    return generated


@torch.no_grad()
def generate_scvidr(
    model: ScGenVAE,
    bundle: ConditionalPairBundle,
    split: ConditionalPairSplit,
    *,
    ridge: float = 0.0,
    device: Optional[str] = None,
    batch_size: int = 256,
) -> torch.Tensor:
    """scVIDR-style latent dose regression using the shared scGen VAE.

    scVIDR builds on the scGen VAE and predicts dose-response latent vectors from
    control-state latent representations. Here the held-out object is dose, not
    cell type, so all training doses remain eligible and only held-out target
    expression is excluded.
    """
    torch_device = resolve_device(device)
    model = model.to(torch_device).eval()
    lookup = _condition_lookup(bundle)
    source_centroids, effects = _latent_condition_effects(
        model, bundle, lookup, device=torch_device, batch_size=batch_size
    )
    train_infos = [lookup[int(i)] for i in map(int, bundle.train.condition_id.unique().tolist())]
    cells = sorted({info.cell_line_id for info in train_infos})
    drugs = sorted({info.drug for info in train_infos})

    # Cell-level control states are drug-independent DMSO summaries. Average all
    # available plate-matched control centroids for stability.
    cell_control: Dict[str, torch.Tensor] = {}
    for cell in cells:
        values = [source_centroids[info.condition_id] for info in train_infos if info.cell_line_id == cell and info.condition_id in source_centroids]
        if values:
            cell_control[cell] = torch.stack(values).mean(0)

    drug_regression: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    direct_slopes: Dict[Tuple[str, str], torch.Tensor] = {}
    for drug in drugs:
        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        for cell in cells:
            conditions = sorted(
                [info for info in train_infos if info.cell_line_id == cell and info.drug == drug],
                key=lambda value: value.dose,
            )
            slope = _fit_effect_slope(conditions, effects)
            if slope is None or cell not in cell_control:
                continue
            direct_slopes[(cell, drug)] = slope
            xs.append(cell_control[cell])
            ys.append(slope)
        if len(xs) >= 2:
            x = torch.stack(xs).double()
            y = torch.stack(ys).double()
            design = torch.cat([torch.ones(x.shape[0], 1, dtype=x.dtype), x], dim=1)
            if float(ridge) > 0.0:
                eye = torch.eye(design.shape[1], dtype=x.dtype)
                eye[0, 0] = 0.0
                beta = torch.linalg.solve(design.T @ design + float(ridge) * eye, design.T @ y)
            else:
                # sklearn LinearRegression (used by scVIDR) is ordinary least squares;
                # lstsq gives the corresponding minimum-norm solution when the
                # latent dimension exceeds the number of observed cell lines.
                beta = torch.linalg.lstsq(design, y).solution
            drug_regression[drug] = (beta.float(), torch.stack(xs).mean(0))

    generated = torch.empty_like(split.x1)
    for condition_id in sorted(map(int, split.condition_id.unique().tolist())):
        info = lookup[condition_id]
        slope = direct_slopes.get((info.cell_line_id, info.drug))
        if info.drug in drug_regression and info.cell_line_id in cell_control:
            beta, _ = drug_regression[info.drug]
            feature = torch.cat([torch.ones(1), cell_control[info.cell_line_id]])
            # scVIDR uses ordinary least-squares regression from the control-state
            # latent centroid to the perturbation delta.  Use that prediction
            # directly rather than introducing an unpublished blending heuristic.
            slope = feature @ beta
        if slope is None:
            raise ValueError(f"scVIDR found no training response for {info.cell_line_id}/{info.drug}.")
        target_delta = slope * float(np.log1p(max(info.dose, 0.0)))
        mask = split.condition_id == condition_id
        x0 = split.x0[mask]
        z0 = _encode_counts(model, x0, device=torch_device, batch_size=batch_size).to(torch_device)
        decoded = model.decode(z0 + target_delta.to(torch_device))
        lib_slope = _training_library_log_slope(bundle, lookup, info.cell_line_id, info.drug)
        generated[mask] = _normalized_to_counts(
            decoded, x0.to(torch_device),
            library_log_multiplier=lib_slope * float(np.log1p(max(info.dose, 0.0))),
        ).cpu()
    return generated


class _CPAMLP(nn.Module):
    def __init__(self, widths: Sequence[int], *, batch_norm: bool = True, last_activation: bool = False) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for index in range(len(widths) - 1):
            layers.append(nn.Linear(int(widths[index]), int(widths[index + 1])))
            if index < len(widths) - 2 or last_activation:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(int(widths[index + 1])))
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CPAModel(nn.Module):
    """Self-contained reproduction of CPA's public architecture and update rule."""

    def __init__(
        self,
        dim: int,
        n_drugs: int,
        n_cells: int,
        *,
        latent_dim: int = 128,
        autoencoder_width: int = 128,
        autoencoder_depth: int = 3,
        adversary_width: int = 64,
        adversary_depth: int = 2,
        doser_width: int = 128,
        doser_depth: int = 2,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_drugs = int(n_drugs)
        self.n_cells = int(n_cells)
        self.latent_dim = int(latent_dim)
        self.encoder = _CPAMLP([self.dim] + [autoencoder_width] * autoencoder_depth + [latent_dim])
        self.decoder = _CPAMLP([latent_dim] + [autoencoder_width] * autoencoder_depth + [2 * self.dim], batch_norm=True)
        self.adversary_drugs = _CPAMLP([latent_dim] + [adversary_width] * adversary_depth + [n_drugs], batch_norm=True)
        self.adversary_cells = _CPAMLP([latent_dim] + [adversary_width] * adversary_depth + [n_cells], batch_norm=True)
        self.drug_embeddings = nn.Embedding(self.n_drugs, self.latent_dim)
        self.cell_embeddings = nn.Embedding(self.n_cells, self.latent_dim)
        self.dosers = nn.ModuleList([
            _CPAMLP([1] + [doser_width] * doser_depth + [1], batch_norm=False)
            for _ in range(self.n_drugs)
        ])

    def dose_weights(self, doses: torch.Tensor) -> torch.Tensor:
        weights = []
        for drug in range(self.n_drugs):
            value = doses[:, drug : drug + 1]
            weights.append(torch.sigmoid(self.dosers[drug](value)) * (value > 0).float())
        return torch.cat(weights, dim=1)

    def perturbation_embedding(self, doses: torch.Tensor) -> torch.Tensor:
        return self.dose_weights(doses) @ self.drug_embeddings.weight

    def forward(
        self,
        normalized_log: torch.Tensor,
        doses: torch.Tensor,
        cell_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_basal = self.encoder(normalized_log)
        latent_treated = latent_basal + self.perturbation_embedding(doses) + self.cell_embeddings(cell_index)
        decoded = self.decoder(latent_treated)
        mean = decoded[:, : self.dim]
        variance = F.softplus(decoded[:, self.dim :]) + 1e-3
        return mean, variance, latent_basal, latent_treated


def _categorical_layout(bundle: ConditionalPairBundle) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, ConditionInfo]]:
    lookup = _condition_lookup(bundle)
    cells = sorted({info.cell_line_id for info in lookup.values()})
    drugs = sorted({info.drug for info in lookup.values()})
    return {name: i for i, name in enumerate(cells)}, {name: i for i, name in enumerate(drugs)}, lookup


def build_cpa(bundle: ConditionalPairBundle, cfg: Mapping[str, object]) -> CPAModel:
    cell_map, drug_map, _ = _categorical_layout(bundle)
    return CPAModel(
        bundle.dim,
        len(drug_map),
        len(cell_map),
        latent_dim=int(cfg.get("latent_dim", 128)),
        autoencoder_width=int(cfg.get("autoencoder_width", 128)),
        autoencoder_depth=int(cfg.get("autoencoder_depth", 3)),
        adversary_width=int(cfg.get("adversary_width", 64)),
        adversary_depth=int(cfg.get("adversary_depth", 2)),
        doser_width=int(cfg.get("doser_width", 128)),
        doser_depth=int(cfg.get("doser_depth", 2)),
    )


def _cpa_batch_metadata(
    condition_ids: torch.Tensor,
    is_treated: torch.Tensor,
    lookup: Mapping[int, ConditionInfo],
    cell_map: Mapping[str, int],
    drug_map: Mapping[str, int],
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = condition_ids.shape[0]
    doses = torch.zeros(batch, len(drug_map), dtype=torch.float32, device=device)
    cells = torch.empty(batch, dtype=torch.long, device=device)
    for row, condition_id in enumerate(map(int, condition_ids.cpu().tolist())):
        info = lookup[condition_id]
        cells[row] = int(cell_map[info.cell_line_id])
        if bool(is_treated[row].item()):
            doses[row, int(drug_map[info.drug])] = float(info.dose)
    return doses, cells


def _set_requires_grad(module: nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def train_cpa(
    model: CPAModel,
    bundle: ConditionalPairBundle,
    config: PublishedBaselineTrainConfig,
    cfg: Mapping[str, object],
    *,
    device: Optional[str] = None,
    verbose: bool = True,
    run_state=None,
) -> Tuple[CPAModel, Dict[str, list]]:
    set_seed(config.seed)
    torch_device = resolve_device(device)
    model = model.to(torch_device)
    cell_map, drug_map, lookup = _categorical_layout(bundle)
    autoencoder_lr = float(cfg.get("autoencoder_lr", 3e-4))
    adversary_lr = float(cfg.get("adversary_lr", 3e-4))
    doser_lr = float(cfg.get("doser_lr", 4e-3))
    autoencoder_wd = float(cfg.get("autoencoder_wd", 4e-7))
    adversary_wd = float(cfg.get("adversary_wd", 4e-7))
    doser_wd = float(cfg.get("doser_wd", 1e-7))
    reg_adversary = float(cfg.get("reg_adversary", 60.0))
    penalty_adversary = float(cfg.get("penalty_adversary", 60.0))
    adversary_steps = max(2, int(cfg.get("adversary_steps", 3)))

    ae_parameters = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.drug_embeddings.parameters()) + list(model.cell_embeddings.parameters())
    optimizer_ae = torch.optim.Adam(ae_parameters, lr=autoencoder_lr, weight_decay=autoencoder_wd)
    optimizer_adv = torch.optim.Adam(
        list(model.adversary_drugs.parameters()) + list(model.adversary_cells.parameters()),
        lr=adversary_lr, weight_decay=adversary_wd,
    )
    optimizer_doser = torch.optim.Adam(model.dosers.parameters(), lr=doser_lr, weight_decay=doser_wd)
    generator = torch.Generator(device=bundle.train.x0.device).manual_seed(config.seed + 1901)
    history: Dict[str, list] = {"step": [], "reconstruction": [], "adv_drug": [], "adv_cell": []}
    model.train()
    n = bundle.train.n
    optimizers = {"autoencoder": optimizer_ae, "adversary": optimizer_adv, "doser": optimizer_doser}
    start_step = run_state.begin(model, optimizers, generator, history, int(config.steps)) if run_state else 1
    for step in range(start_step, int(config.steps) + 1):
        index = torch.randint(2 * n, (int(config.batch_size),), generator=generator, device=bundle.train.x0.device)
        is_treated = index >= n
        row = index.remainder(n)
        counts = torch.where(is_treated[:, None], bundle.train.x1[row], bundle.train.x0[row]).to(torch_device)
        condition_ids = bundle.train.condition_id[row]
        doses, cells = _cpa_batch_metadata(condition_ids, is_treated, lookup, cell_map, drug_map, device=torch_device)
        genes = _log_normalize(counts)

        if step % adversary_steps != 0:
            with torch.no_grad():
                latent = model.encoder(genes)
            latent = latent.detach().requires_grad_(True)
            pred_drug = model.adversary_drugs(latent)
            pred_cell = model.adversary_cells(latent)
            drug_target = (doses > 0).float()
            loss_drug = F.binary_cross_entropy_with_logits(pred_drug, drug_target)
            loss_cell = F.cross_entropy(pred_cell, cells)
            grad_drug = torch.autograd.grad(pred_drug.sum(), latent, create_graph=True, retain_graph=True)[0].square().mean()
            grad_cell = torch.autograd.grad(pred_cell.sum(), latent, create_graph=True)[0].square().mean()
            loss = loss_drug + loss_cell + penalty_adversary * (grad_drug + grad_cell)
            optimizer_adv.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.adversary_drugs.parameters()) + list(model.adversary_cells.parameters()),
                config.grad_clip,
            )
            optimizer_adv.step()
            reconstruction_value = float("nan")
        else:
            _set_requires_grad(model.adversary_drugs, False)
            _set_requires_grad(model.adversary_cells, False)
            mean, variance, latent, _ = model(genes, doses, cells)
            reconstruction = F.gaussian_nll_loss(mean, genes, variance, reduction="mean")
            loss_drug = F.binary_cross_entropy_with_logits(model.adversary_drugs(latent), (doses > 0).float())
            loss_cell = F.cross_entropy(model.adversary_cells(latent), cells)
            loss = reconstruction - reg_adversary * (loss_drug + loss_cell)
            optimizer_ae.zero_grad(set_to_none=True)
            optimizer_doser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(ae_parameters + list(model.dosers.parameters()), config.grad_clip)
            optimizer_ae.step()
            optimizer_doser.step()
            _set_requires_grad(model.adversary_drugs, True)
            _set_requires_grad(model.adversary_cells, True)
            reconstruction_value = float(reconstruction.detach().cpu())

        if step == 1 or step % config.log_every == 0 or step == config.steps:
            history["step"].append(step)
            history["reconstruction"].append(reconstruction_value)
            history["adv_drug"].append(float(loss_drug.detach().cpu()))
            history["adv_cell"].append(float(loss_cell.detach().cpu()))
            if verbose:
                print(
                    f"step={step:6d} cpa_recon={reconstruction_value:9.5f} "
                    f"adv_drug={history['adv_drug'][-1]:8.4f} adv_cell={history['adv_cell'][-1]:8.4f}",
                    flush=True,
                )
        if run_state is not None:
            run_state.after_step(model, optimizers, generator, history, step)
    if run_state is not None:
        run_state.finish(model, history)
    model.eval()
    return model, history


@torch.no_grad()
def generate_cpa(
    model: CPAModel,
    bundle: ConditionalPairBundle,
    split: ConditionalPairSplit,
    *,
    seed: int = 42,
    device: Optional[str] = None,
    batch_size: int = 256,
    library_slopes: Optional[Mapping[Tuple[str, str], float]] = None,
) -> torch.Tensor:
    torch_device = resolve_device(device)
    model = model.to(torch_device).eval()
    generator = torch.Generator(device=torch_device).manual_seed(int(seed) + 2003)
    cell_map, drug_map, lookup = _categorical_layout(bundle)
    outputs: List[torch.Tensor] = []
    for first in range(0, split.n, int(batch_size)):
        last = min(first + int(batch_size), split.n)
        x0 = split.x0[first:last].to(torch_device)
        ids = split.condition_id[first:last]
        treated = torch.ones(last - first, dtype=torch.bool)
        doses, cells = _cpa_batch_metadata(ids, treated, lookup, cell_map, drug_map, device=torch_device)
        mean, variance, _, _ = model(_log_normalize(x0), doses, cells)
        # CPA's public Gaussian decoder parameterizes both mean and variance.
        # Draw from that predictive distribution instead of evaluating only the
        # mean, which would artificially suppress cell-to-cell variability.
        eps = torch.randn(mean.shape, generator=generator, device=torch_device)
        predictive = mean + eps * torch.sqrt(variance)
        # Apply a training-only dose response for total UMI count because CPA is
        # trained on library-normalized expression.
        converted = torch.empty_like(x0)
        for condition_id in map(int, ids.unique().tolist()):
            local = ids == condition_id
            info = lookup[condition_id]
            lib_slope = (_training_library_log_slope(bundle, lookup, info.cell_line_id, info.drug)
                         if library_slopes is None else library_slopes[(info.cell_line_id, info.drug)])
            converted[local.to(torch_device)] = _normalized_to_counts(
                predictive[local.to(torch_device)], x0[local.to(torch_device)],
                library_log_multiplier=lib_slope * float(np.log1p(max(info.dose, 0.0))),
            )
        outputs.append(converted.cpu())
    return torch.cat(outputs, dim=0)


class ConditionalNBVAE(nn.Module):
    """Count-native conditional NB-VAE used as a generic latent count baseline."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        *,
        hidden_dim: int = 512,
        context_hidden_dim: int = 256,
        context_output_dim: int = 256,
        latent_dim: int = 64,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.latent_dim = int(latent_dim)
        self.context_encoder = VectorContextEncoder(context_dim, context_hidden_dim, context_output_dim)
        self.source_encoder = MLP(self.dim, hidden_dim, hidden_dim, depth=2)
        self.target_encoder = MLP(self.dim, hidden_dim, hidden_dim, depth=2)
        self.prior_net = MLP(hidden_dim + context_output_dim, hidden_dim, 2 * latent_dim, depth=2)
        self.posterior_net = MLP(2 * hidden_dim + context_output_dim, hidden_dim, 2 * latent_dim, depth=2)
        self.decoder = MLP(latent_dim + hidden_dim + context_output_dim, hidden_dim, self.dim + 1, depth=depth)
        self.log_inverse_dispersion = nn.Parameter(torch.zeros(self.dim))

    def _features(self, counts: torch.Tensor) -> torch.Tensor:
        return torch.log1p(counts.float())

    def prior(self, x0: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source = self.source_encoder(self._features(x0))
        cond = self.context_encoder(context)
        mean, logvar = self.prior_net(torch.cat([source, cond], dim=-1)).chunk(2, dim=-1)
        return mean, logvar.clamp(-8.0, 8.0), source, cond

    def posterior(self, x0: torch.Tensor, x1: torch.Tensor, context: torch.Tensor):
        prior_mean, prior_logvar, source, cond = self.prior(x0, context)
        target = self.target_encoder(self._features(x1))
        mean, logvar = self.posterior_net(torch.cat([source, target, cond], dim=-1)).chunk(2, dim=-1)
        return mean, logvar.clamp(-8.0, 8.0), prior_mean, prior_logvar, source, cond

    def decode_mean(self, z: torch.Tensor, x0: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        source = self.source_encoder(self._features(x0))
        cond = self.context_encoder(context)
        raw = self.decoder(torch.cat([z, source, cond], dim=-1))
        gene_logits = raw[:, : self.dim]
        library_shift = 2.0 * torch.tanh(raw[:, self.dim :])
        library = x0.float().sum(-1, keepdim=True).clamp_min(1.0) * torch.exp(library_shift)
        mean = torch.softmax(gene_logits, dim=-1) * library
        theta = F.softplus(self.log_inverse_dispersion).clamp_min(1e-3).reshape(1, -1)
        return mean.clamp_min(1e-6), theta


def build_nbvae(bundle: ConditionalPairBundle, cfg: Mapping[str, object]) -> ConditionalNBVAE:
    if bundle.train.context.ndim != 2:
        raise ValueError("Conditional NB-VAE requires vector context.")
    return ConditionalNBVAE(
        bundle.dim,
        int(bundle.train.context.shape[1]),
        hidden_dim=int(cfg.get("hidden_dim", 512)),
        context_hidden_dim=int(cfg.get("context_hidden_dim", 256)),
        context_output_dim=int(cfg.get("context_output_dim", 256)),
        latent_dim=int(cfg.get("latent_dim", 64)),
        depth=int(cfg.get("depth", 3)),
    )


def _normal_kl(q_mean: torch.Tensor, q_logvar: torch.Tensor, p_mean: torch.Tensor, p_logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        p_logvar - q_logvar + (q_logvar.exp() + (q_mean - p_mean).square()) / p_logvar.exp() - 1.0
    ).sum(-1)


def _nb_log_prob(x: torch.Tensor, mean: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    x = x.float()
    theta = theta.expand_as(mean)
    log_theta_mu = torch.log(theta + mean)
    return (
        torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta) - log_theta_mu)
        + x * (torch.log(mean.clamp_min(1e-8)) - log_theta_mu)
    )


def train_nbvae(
    model: ConditionalNBVAE,
    train: ConditionalPairSplit,
    config: PublishedBaselineTrainConfig,
    cfg: Mapping[str, object],
    *,
    device: Optional[str] = None,
    verbose: bool = True,
    run_state=None,
) -> Tuple[ConditionalNBVAE, Dict[str, list]]:
    set_seed(config.seed)
    torch_device = resolve_device(device)
    model = model.to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator(device=train.x0.device).manual_seed(config.seed + 2201)
    kl_warmup = max(1, int(cfg.get("kl_warmup_steps", 10_000)))
    history: Dict[str, list] = {"step": [], "loss": [], "nll": [], "kl": []}
    model.train()
    optimizers = {"main": optimizer}
    start_step = run_state.begin(model, optimizers, generator, history, int(config.steps)) if run_state else 1
    for step in range(start_step, int(config.steps) + 1):
        x0, x1, context, _ = train.sample(config.batch_size, device=torch_device, generator=generator)
        q_mean, q_logvar, p_mean, p_logvar, _, _ = model.posterior(x0, x1, context)
        z = q_mean + torch.randn_like(q_mean) * torch.exp(0.5 * q_logvar)
        mean, theta = model.decode_mean(z, x0, context)
        nll = -_nb_log_prob(x1, mean, theta).sum(-1)
        kl = _normal_kl(q_mean, q_logvar, p_mean, p_logvar)
        beta = min(step / float(kl_warmup), 1.0)
        loss = (nll + beta * kl).mean() / float(model.dim)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            history["step"].append(step)
            history["loss"].append(float(loss.detach().cpu()))
            history["nll"].append(float(nll.mean().detach().cpu() / model.dim))
            history["kl"].append(float(kl.mean().detach().cpu() / model.dim))
            if verbose:
                print(
                    f"step={step:6d} nbvae_loss={history['loss'][-1]:9.5f} "
                    f"nll/gene={history['nll'][-1]:9.5f} kl/gene={history['kl'][-1]:9.5f}",
                    flush=True,
                )
        if run_state is not None:
            run_state.after_step(model, optimizers, generator, history, step)
    if run_state is not None:
        run_state.finish(model, history)
    model.eval()
    return model, history


@torch.no_grad()
def generate_nbvae(
    model: ConditionalNBVAE,
    split: ConditionalPairSplit,
    *,
    seed: int,
    device: Optional[str] = None,
    batch_size: int = 256,
) -> torch.Tensor:
    set_seed(seed)
    torch_device = resolve_device(device)
    model = model.to(torch_device).eval()
    generator = torch.Generator(device=torch_device).manual_seed(seed + 2301)
    outputs: List[torch.Tensor] = []
    for first in range(0, split.n, int(batch_size)):
        last = min(first + int(batch_size), split.n)
        x0 = split.x0[first:last].to(torch_device)
        context = split.context[first:last].to(torch_device)
        p_mean, p_logvar, _, _ = model.prior(x0, context)
        eps = torch.randn(p_mean.shape, generator=generator, device=torch_device)
        z = p_mean + eps * torch.exp(0.5 * p_logvar)
        mean, theta = model.decode_mean(z, x0, context)
        probs = (mean / (mean + theta)).clamp(1e-6, 1.0 - 1e-6)
        draws = torch.distributions.NegativeBinomial(total_count=theta.expand_as(mean), probs=probs).sample()
        outputs.append(draws.long().cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def generate_linear_dose_response(
    bundle: ConditionalPairBundle,
    split: ConditionalPairSplit,
    *,
    ridge: float = 1e-4,
) -> torch.Tensor:
    lookup = _condition_lookup(bundle)
    generated = torch.empty_like(split.x1)
    for condition_id in sorted(map(int, split.condition_id.unique().tolist())):
        info = lookup[condition_id]
        candidates = _train_pair_conditions(bundle, lookup, info.cell_line_id, info.drug)
        xs: List[float] = []
        effects: List[torch.Tensor] = []
        for train_info in candidates:
            mask = bundle.train.condition_id == train_info.condition_id
            if not bool(mask.any()):
                continue
            control = _log_normalize(bundle.train.x0[mask]).mean(0)
            target = _log_normalize(bundle.train.x1[mask]).mean(0)
            xs.append(float(np.log1p(max(train_info.dose, 0.0))))
            effects.append(target - control)
        if not effects:
            raise ValueError(f"Linear baseline found no training response for {info.cell_line_id}/{info.drug}.")
        x = torch.tensor(xs, dtype=torch.float32)
        effect = torch.stack(effects)
        slope = (x[:, None] * effect).sum(0) / (x.square().sum() + float(ridge))
        target_effect = slope * float(np.log1p(max(info.dose, 0.0)))
        mask = split.condition_id == condition_id
        x0 = split.x0[mask]
        prediction = _log_normalize(x0) + target_effect
        lib_slope = _training_library_log_slope(bundle, lookup, info.cell_line_id, info.drug)
        generated[mask] = _normalized_to_counts(
            prediction, x0,
            library_log_multiplier=lib_slope * float(np.log1p(max(info.dose, 0.0))),
        )
    return generated


def _subsample_rows(value: torch.Tensor, maximum: int, seed: int) -> torch.Tensor:
    if value.shape[0] <= int(maximum):
        return value
    generator = torch.Generator().manual_seed(int(seed))
    index = torch.randperm(value.shape[0], generator=generator)[: int(maximum)]
    return value[index]


def _sinkhorn_mapping(
    control_counts: torch.Tensor,
    target_counts: torch.Tensor,
    *,
    epsilon_scale: float,
    iterations: int,
    pca_dim: int,
    max_cells: int,
    seed: int,
) -> Dict[str, torch.Tensor]:
    control_counts = _subsample_rows(control_counts, max_cells, seed)
    target_counts = _subsample_rows(target_counts, max_cells, seed + 1)
    control = _log_normalize(control_counts).double()
    target = _log_normalize(target_counts).double()
    joint = torch.cat([control, target], dim=0)
    mean = joint.mean(0, keepdim=True)
    centered = joint - mean
    q = min(int(pca_dim), centered.shape[0] - 1, centered.shape[1])
    if q < 1:
        raise ValueError("Sinkhorn OT requires at least two training cells.")
    torch.manual_seed(int(seed))
    _, _, v = torch.pca_lowrank(centered.float(), q=q, center=False)
    components = v[:, :q].double()
    z_control = (control - mean) @ components
    z_target = (target - mean) @ components
    cost = torch.cdist(z_control, z_target).square() / float(q)
    positive = cost[cost > 0]
    scale = torch.median(positive) if positive.numel() else torch.tensor(1.0, dtype=cost.dtype)
    epsilon = max(float(epsilon_scale) * float(scale.item()), 1e-4)
    kernel = torch.exp((-cost / epsilon).clamp(min=-60.0, max=0.0)).clamp_min(1e-30)
    a = torch.full((control.shape[0],), 1.0 / control.shape[0], dtype=kernel.dtype)
    b = torch.full((target.shape[0],), 1.0 / target.shape[0], dtype=kernel.dtype)
    u = torch.ones_like(a)
    vscale = torch.ones_like(b)
    for _ in range(int(iterations)):
        u = a / (kernel @ vscale).clamp_min(1e-30)
        vscale = b / (kernel.T @ u).clamp_min(1e-30)
    coupling = u[:, None] * kernel * vscale[None, :]
    row_weights = coupling / coupling.sum(1, keepdim=True).clamp_min(1e-30)
    barycenter = row_weights @ target
    return {
        "control": control.float(),
        "delta": (barycenter - control).float(),
        "mean": mean.float(),
        "components": components.float(),
    }


def _apply_sinkhorn_mapping(mapping: Mapping[str, torch.Tensor], test_counts: torch.Tensor, knn: int) -> torch.Tensor:
    test = _log_normalize(test_counts)
    mean = mapping["mean"]
    components = mapping["components"]
    z_train = (mapping["control"] - mean) @ components
    z_test = (test - mean) @ components
    distance = torch.cdist(z_test.float(), z_train.float())
    k = min(max(1, int(knn)), z_train.shape[0])
    values, index = torch.topk(distance, k=k, largest=False, dim=1)
    weights = 1.0 / values.clamp_min(1e-4)
    weights = weights / weights.sum(1, keepdim=True)
    delta = mapping["delta"][index]
    return (weights[:, :, None] * delta).sum(1)


@torch.no_grad()
def generate_sinkhorn_ot(
    bundle: ConditionalPairBundle,
    split: ConditionalPairSplit,
    *,
    epsilon_scale: float = 0.1,
    iterations: int = 100,
    pca_dim: int = 30,
    max_cells: int = 384,
    knn: int = 8,
    seed: int = 42,
) -> torch.Tensor:
    """Training-only entropic OT maps at observed doses + log-dose interpolation."""
    lookup = _condition_lookup(bundle)
    mappings: Dict[int, Dict[str, torch.Tensor]] = {}
    generated = torch.empty_like(split.x1)
    for condition_id in sorted(map(int, split.condition_id.unique().tolist())):
        info = lookup[condition_id]
        candidates = _train_pair_conditions(bundle, lookup, info.cell_line_id, info.drug)
        effects: List[torch.Tensor] = []
        log_doses: List[float] = []
        mask_test = split.condition_id == condition_id
        x0_test = split.x0[mask_test]
        for train_info in candidates:
            if train_info.condition_id not in mappings:
                mask_train = bundle.train.condition_id == train_info.condition_id
                mappings[train_info.condition_id] = _sinkhorn_mapping(
                    bundle.train.x0[mask_train],
                    bundle.train.x1[mask_train],
                    epsilon_scale=epsilon_scale,
                    iterations=iterations,
                    pca_dim=pca_dim,
                    max_cells=max_cells,
                    seed=seed + 37 * int(train_info.condition_id),
                )
            effects.append(_apply_sinkhorn_mapping(mappings[train_info.condition_id], x0_test, knn))
            log_doses.append(float(np.log1p(max(train_info.dose, 0.0))))
        if not effects:
            raise ValueError(f"Sinkhorn OT found no training response for {info.cell_line_id}/{info.drug}.")
        x = torch.tensor(log_doses, dtype=torch.float32)
        effect_stack = torch.stack(effects, dim=0)
        slope = (x[:, None, None] * effect_stack).sum(0) / x.square().sum().clamp_min(1e-8)
        target_effect = slope * float(np.log1p(max(info.dose, 0.0)))
        prediction = _log_normalize(x0_test) + target_effect
        lib_slope = _training_library_log_slope(bundle, lookup, info.cell_line_id, info.drug)
        generated[mask_test] = _normalized_to_counts(
            prediction, x0_test,
            library_log_multiplier=lib_slope * float(np.log1p(max(info.dose, 0.0))),
        )
    return generated

# ---- Cached response statistics for fair inference-time measurement -----------------

def prepare_scgen_response_state(
    model: ScGenVAE,
    bundle: ConditionalPairBundle,
    *,
    device: Optional[str] = None,
    batch_size: int = 256,
) -> Dict[str, object]:
    """Precompute training-only latent effects for scGen/scVIDR.

    scGen itself is not a continuous-dose model.  For a held-out dose we therefore
    use the latent effect from the nearest observed training dose of the same
    cell-line/drug pair.  scVIDR uses the same VAE but applies its published
    continuous log-dose scaling from a reference treated dose.
    """
    torch_device = resolve_device(device)
    lookup = _condition_lookup(bundle)
    source_centroids, effects = _latent_condition_effects(
        model, bundle, lookup, device=torch_device, batch_size=batch_size
    )
    pair_conditions: Dict[Tuple[str, str], List[ConditionInfo]] = {}
    for info in lookup.values():
        if info.condition_id not in effects:
            continue
        pair_conditions.setdefault((info.cell_line_id, info.drug), []).append(info)
    for pair in pair_conditions:
        pair_conditions[pair] = sorted(pair_conditions[pair], key=lambda x: x.dose)
    library_slopes = {
        pair: _training_library_log_slope(bundle, lookup, pair[0], pair[1])
        for pair in pair_conditions
    }
    return {
        "lookup": lookup,
        "source_centroids": source_centroids,
        "effects": effects,
        "pair_conditions": pair_conditions,
        "library_slopes": library_slopes,
    }


def prepare_scvidr_response_state(
    bundle: ConditionalPairBundle,
    scgen_state: Mapping[str, object],
    *,
    ridge: float = 0.0,
) -> Dict[str, object]:
    """Prepare the continuous-dose scVIDR response state.

    The public multi-dose scVIDR prediction scales a latent treatment delta by
    log1p(target_dose) / log1p(reference_dose).  Because our held-out object is a
    dose for an otherwise observed cell-line/drug pair, we use that pair's highest
    observed training dose as the reference treatment.  No test target expression
    is used.
    """
    del ridge  # kept only for backward-compatible config parsing
    effects = scgen_state["effects"]
    pair_conditions = scgen_state["pair_conditions"]
    references: Dict[Tuple[str, str], Tuple[float, torch.Tensor]] = {}
    assert isinstance(effects, Mapping) and isinstance(pair_conditions, Mapping)
    for pair, conditions in pair_conditions.items():
        usable = [info for info in conditions if info.condition_id in effects and info.dose > 0]
        if not usable:
            continue
        reference = max(usable, key=lambda info: info.dose)
        references[pair] = (float(reference.dose), effects[reference.condition_id])
    return {**dict(scgen_state), "references": references}


@torch.no_grad()
def generate_scgen_cached(
    model: ScGenVAE,
    split: ConditionalPairSplit,
    state: Mapping[str, object],
    *,
    device: Optional[str] = None,
    batch_size: int = 256,
) -> torch.Tensor:
    """Nearest-dose adaptation of scGen for held-out dose prediction."""
    torch_device = resolve_device(device)
    model = model.to(torch_device).eval()
    lookup = state["lookup"]
    effects = state["effects"]
    pair_conditions = state["pair_conditions"]
    library_slopes = state["library_slopes"]
    generated = torch.empty_like(split.x1)
    for condition_id in sorted(map(int, split.condition_id.unique().tolist())):
        info = lookup[condition_id]
        candidates = pair_conditions.get((info.cell_line_id, info.drug), [])
        candidates = [candidate for candidate in candidates if candidate.condition_id in effects]
        if not candidates:
            raise ValueError(f"scGen found no training response for {info.cell_line_id}/{info.drug}.")
        # scGen is a binary perturbation model, not a dose-response model.  Use the
        # nearest observed training dose rather than giving it an unpublished dose
        # regression that would make it artificially similar to scVIDR.
        reference = min(
            candidates,
            key=lambda candidate: abs(
                float(np.log1p(max(candidate.dose, 0.0)))
                - float(np.log1p(max(info.dose, 0.0)))
            ),
        )
        target_delta = effects[reference.condition_id]
        mask = split.condition_id == condition_id
        x0 = split.x0[mask]
        z0 = _encode_counts(model, x0, device=torch_device, batch_size=batch_size).to(torch_device)
        decoded = model.decode(z0 + target_delta.to(torch_device))
        lib_slope = float(library_slopes.get((info.cell_line_id, info.drug), 0.0))
        generated[mask] = _normalized_to_counts(
            decoded,
            x0.to(torch_device),
            library_log_multiplier=lib_slope * float(np.log1p(max(info.dose, 0.0))),
        ).cpu()
    return generated


@torch.no_grad()
def generate_scvidr_cached(
    model: ScGenVAE,
    split: ConditionalPairSplit,
    state: Mapping[str, object],
    *,
    device: Optional[str] = None,
    batch_size: int = 256,
) -> torch.Tensor:
    """Multi-dose scVIDR prediction via published log-dose latent scaling."""
    torch_device = resolve_device(device)
    model = model.to(torch_device).eval()
    lookup = state["lookup"]
    references = state["references"]
    library_slopes = state["library_slopes"]
    generated = torch.empty_like(split.x1)
    for condition_id in sorted(map(int, split.condition_id.unique().tolist())):
        info = lookup[condition_id]
        reference = references.get((info.cell_line_id, info.drug))
        if reference is None:
            raise ValueError(f"scVIDR found no training response for {info.cell_line_id}/{info.drug}.")
        reference_dose, reference_delta = reference
        denom = float(np.log1p(max(reference_dose, 0.0)))
        if denom <= 0.0:
            raise ValueError(f"scVIDR reference dose must be positive for {info.cell_line_id}/{info.drug}.")
        scale = float(np.log1p(max(info.dose, 0.0))) / denom
        target_delta = reference_delta * scale
        mask = split.condition_id == condition_id
        x0 = split.x0[mask]
        z0 = _encode_counts(model, x0, device=torch_device, batch_size=batch_size).to(torch_device)
        decoded = model.decode(z0 + target_delta.to(torch_device))
        lib_slope = float(library_slopes.get((info.cell_line_id, info.drug), 0.0))
        generated[mask] = _normalized_to_counts(
            decoded,
            x0.to(torch_device),
            library_log_multiplier=lib_slope * float(np.log1p(max(info.dose, 0.0))),
        ).cpu()
    return generated


def fit_linear_dose_response(bundle: ConditionalPairBundle, *, ridge: float = 1e-4) -> Dict[str, object]:
    lookup = _condition_lookup(bundle)
    slopes: Dict[Tuple[str, str], torch.Tensor] = {}
    library_slopes: Dict[Tuple[str, str], float] = {}
    for cell, drug in sorted({(info.cell_line_id, info.drug) for info in lookup.values()}):
        candidates = _train_pair_conditions(bundle, lookup, cell, drug)
        xs: List[float] = []
        effects: List[torch.Tensor] = []
        for info in candidates:
            mask = bundle.train.condition_id == info.condition_id
            if not bool(mask.any()):
                continue
            xs.append(float(np.log1p(max(info.dose, 0.0))))
            effects.append(_log_normalize(bundle.train.x1[mask]).mean(0) - _log_normalize(bundle.train.x0[mask]).mean(0))
        if effects:
            x = torch.tensor(xs, dtype=torch.float32)
            slopes[(cell, drug)] = (x[:, None] * torch.stack(effects)).sum(0) / (x.square().sum() + float(ridge))
            library_slopes[(cell, drug)] = _training_library_log_slope(bundle, lookup, cell, drug)
    return {"lookup": lookup, "slopes": slopes, "library_slopes": library_slopes, "ridge": float(ridge)}


@torch.no_grad()
def generate_linear_dose_response_cached(
    split: ConditionalPairSplit,
    state: Mapping[str, object],
) -> torch.Tensor:
    lookup = state["lookup"]
    slopes = state["slopes"]
    library_slopes = state["library_slopes"]
    generated = torch.empty_like(split.x1)
    for condition_id in sorted(map(int, split.condition_id.unique().tolist())):
        info = lookup[condition_id]
        slope = slopes.get((info.cell_line_id, info.drug))
        if slope is None:
            raise ValueError(f"Linear baseline found no training response for {info.cell_line_id}/{info.drug}.")
        mask = split.condition_id == condition_id
        x0 = split.x0[mask]
        effect = slope * float(np.log1p(max(info.dose, 0.0)))
        generated[mask] = _normalized_to_counts(
            _log_normalize(x0) + effect,
            x0,
            library_log_multiplier=float(library_slopes.get((info.cell_line_id, info.drug), 0.0)) * float(np.log1p(max(info.dose, 0.0))),
        )
    return generated


def fit_sinkhorn_ot(
    bundle: ConditionalPairBundle,
    split: ConditionalPairSplit,
    *,
    epsilon_scale: float = 0.1,
    iterations: int = 100,
    pca_dim: int = 30,
    max_cells: int = 384,
    seed: int = 42,
) -> Dict[str, object]:
    lookup = _condition_lookup(bundle)
    needed_pairs = {(lookup[int(cid)].cell_line_id, lookup[int(cid)].drug) for cid in split.condition_id.unique().tolist()}
    mappings: Dict[int, Dict[str, torch.Tensor]] = {}
    for cell, drug in sorted(needed_pairs):
        for info in _train_pair_conditions(bundle, lookup, cell, drug):
            mask = bundle.train.condition_id == info.condition_id
            mappings[info.condition_id] = _sinkhorn_mapping(
                bundle.train.x0[mask],
                bundle.train.x1[mask],
                epsilon_scale=epsilon_scale,
                iterations=iterations,
                pca_dim=pca_dim,
                max_cells=max_cells,
                seed=seed + 37 * int(info.condition_id),
            )
    library_slopes = {
        pair: _training_library_log_slope(bundle, lookup, pair[0], pair[1])
        for pair in needed_pairs
    }
    return {
        "lookup": lookup,
        "mappings": mappings,
        "library_slopes": library_slopes,
        "knn": 8,
    }


@torch.no_grad()
def generate_sinkhorn_ot_cached(
    bundle: ConditionalPairBundle,
    split: ConditionalPairSplit,
    state: Mapping[str, object],
    *,
    knn: Optional[int] = None,
) -> torch.Tensor:
    lookup = state["lookup"]
    mappings = state["mappings"]
    library_slopes = state["library_slopes"]
    use_knn = int(state.get("knn", 8) if knn is None else knn)
    generated = torch.empty_like(split.x1)
    for condition_id in sorted(map(int, split.condition_id.unique().tolist())):
        info = lookup[condition_id]
        candidates = _train_pair_conditions(bundle, lookup, info.cell_line_id, info.drug)
        mask_test = split.condition_id == condition_id
        x0_test = split.x0[mask_test]
        effects: List[torch.Tensor] = []
        log_doses: List[float] = []
        for train_info in candidates:
            mapping = mappings.get(train_info.condition_id)
            if mapping is None:
                continue
            effects.append(_apply_sinkhorn_mapping(mapping, x0_test, use_knn))
            log_doses.append(float(np.log1p(max(train_info.dose, 0.0))))
        if not effects:
            raise ValueError(f"Sinkhorn OT found no training response for {info.cell_line_id}/{info.drug}.")
        x = torch.tensor(log_doses, dtype=torch.float32)
        slope = (x[:, None, None] * torch.stack(effects)).sum(0) / x.square().sum().clamp_min(1e-8)
        prediction = _log_normalize(x0_test) + slope * float(np.log1p(max(info.dose, 0.0)))
        generated[mask_test] = _normalized_to_counts(
            prediction,
            x0_test,
            library_log_multiplier=float(library_slopes.get((info.cell_line_id, info.drug), 0.0)) * float(np.log1p(max(info.dose, 0.0))),
        )
    return generated
