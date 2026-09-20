from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from .application_data import ConditionalPairBundle, ConditionalPairSplit
from .utils import make_ema_copy, resolve_device, set_seed, update_ema


class PoissonGLMForecaster(nn.Module):
    def __init__(self, dim: int, history_bins: int, rate_floor: float = 1e-5) -> None:
        super().__init__()
        self.dim = int(dim)
        self.history_bins = int(history_bins)
        self.rate_floor = float(rate_floor)
        self.linear = nn.Linear(self.dim * self.history_bins, self.dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1:] != (self.history_bins, self.dim):
            raise ValueError("history has the wrong shape.")
        features = torch.log1p(history.float()).reshape(history.shape[0], -1)
        return F.softplus(self.linear(features)) + self.rate_floor


class TransformerPoissonForecaster(nn.Module):
    def __init__(
        self,
        dim: int,
        history_bins: int,
        model_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.0,
        rate_floor: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.history_bins = int(history_bins)
        self.rate_floor = float(rate_floor)
        self.input_projection = nn.Linear(2 * self.dim, model_dim)
        self.position = nn.Parameter(torch.zeros(1, self.history_bins, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=int(n_heads),
            dim_feedforward=4 * model_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(n_layers))
        self.head = nn.Linear(model_dim, self.dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1:] != (self.history_bins, self.dim):
            raise ValueError("history has the wrong shape.")
        x = history.float()
        scale = x.mean().detach().clamp_min(1.0)
        features = torch.cat([torch.log1p(x), x / scale], dim=-1)
        token = self.input_projection(features) + self.position[:, : history.shape[1]]
        encoded = self.encoder(token)
        return F.softplus(self.head(encoded[:, -1])) + self.rate_floor


class MambaPoissonForecaster(nn.Module):
    """Thin wrapper around the official mamba_ssm package.

    Formal neural runs require ``mamba-ssm``. The runner does not silently replace Mamba with another architecture.
    """

    def __init__(
        self,
        dim: int,
        history_bins: int,
        model_dim: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        rate_floor: float = 1e-5,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba  # type: ignore
        except ImportError as error:  # pragma: no cover - optional formal dependency
            raise RuntimeError(
                "Mamba baseline requires the optional mamba-ssm package."
            ) from error
        self.dim = int(dim)
        self.history_bins = int(history_bins)
        self.rate_floor = float(rate_floor)
        self.input_projection = nn.Linear(2 * self.dim, model_dim)
        self.layers = nn.ModuleList(
            [
                Mamba(
                    d_model=model_dim,
                    d_state=int(d_state),
                    d_conv=int(d_conv),
                    expand=int(expand),
                )
                for _ in range(int(n_layers))
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(model_dim) for _ in self.layers])
        self.head = nn.Linear(model_dim, self.dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1:] != (self.history_bins, self.dim):
            raise ValueError("history has the wrong shape.")
        x = history.float()
        scale = x.mean().detach().clamp_min(1.0)
        state = self.input_projection(torch.cat([torch.log1p(x), x / scale], dim=-1))
        for layer, norm in zip(self.layers, self.norms):
            state = state + layer(norm(state))
        return F.softplus(self.head(state[:, -1])) + self.rate_floor


@dataclass
class ForecasterTrainConfig:
    steps: int = 5000
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    ema_decay: float = 0.999
    grad_clip: float = 5.0
    log_every: int = 100
    seed: int = 42


def train_poisson_forecaster(
    model: nn.Module,
    train: ConditionalPairSplit,
    config: ForecasterTrainConfig,
    *,
    device: Optional[str] = None,
    verbose: bool = True,
    progress: bool = False,
    progress_desc: Optional[str] = None,
) -> Tuple[nn.Module, nn.Module, Dict[str, list]]:
    set_seed(config.seed)
    torch_device = resolve_device(device)
    model = model.to(torch_device)
    ema = make_ema_copy(model).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device=train.x0.device).manual_seed(config.seed + 97)
    history: Dict[str, list] = {"step": [], "poisson_nll": []}
    model.train()
    iterator = tqdm(
        range(1, int(config.steps) + 1),
        desc=progress_desc or model.__class__.__name__,
        dynamic_ncols=True,
        leave=True,
        disable=not progress,
    )
    for step in iterator:
        _, target, context, _ = train.sample(
            config.batch_size, device=torch_device, generator=generator
        )
        rate = model(context)
        loss = (rate - target.float() * torch.log(rate.clamp_min(1e-8))).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        update_ema(ema, model, config.ema_decay)
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            history["step"].append(step)
            history["poisson_nll"].append(float(loss.detach().cpu()))
            if progress:
                iterator.set_postfix(nll=f"{history['poisson_nll'][-1]:.3f}", refresh=False)
            elif verbose:
                print(f"step={step:6d} poisson_nll={history['poisson_nll'][-1]:9.4f}")
    ema.eval()
    return model, ema, history


@torch.no_grad()
def sample_poisson_forecaster(
    model: nn.Module,
    context: torch.Tensor,
    *,
    n_draws: int,
    device: Optional[str] = None,
    batch_size: int = 512,
) -> torch.Tensor:
    torch_device = resolve_device(device)
    model = model.to(torch_device).eval()
    outputs = []
    for first in range(0, context.shape[0], int(batch_size)):
        last = min(first + int(batch_size), context.shape[0])
        rate = model(context[first:last].to(torch_device))
        draws = torch.poisson(
            rate[:, None, :].expand(rate.shape[0], int(n_draws), rate.shape[1])
        ).long()
        outputs.append(draws.cpu())
    return torch.cat(outputs, dim=0)


SCRNA_EMPIRICAL_BASELINE_LABELS = {
    "pair_mean_poisson": "Pair mean Poisson (other doses)",
    "nearest_dose_empirical": "Nearest-dose empirical",
    "dose_interpolated_empirical": "Dose-interpolated empirical",
}


def _scrna_condition_metadata(bundle: ConditionalPairBundle) -> Dict[int, Dict[str, object]]:
    raw = bundle.metadata.get("conditions", [])
    lookup: Dict[int, Dict[str, object]] = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping) and "condition_id" in item:
                lookup[int(item["condition_id"])] = dict(item)
    if not lookup:
        raise ValueError("scRNA empirical baselines require condition metadata in the prepared bundle.")
    return lookup


def _training_conditions_for_pair(
    bundle: ConditionalPairBundle,
    lookup: Mapping[int, Mapping[str, object]],
    cell_line_id: str,
    drug: str,
) -> List[Tuple[int, float]]:
    train_ids = set(map(int, bundle.train.condition_id.unique().tolist()))
    matches: List[Tuple[int, float]] = []
    for condition_id in train_ids:
        meta = lookup.get(condition_id)
        if meta is None:
            continue
        if str(meta.get("cell_line_id")) != str(cell_line_id) or str(meta.get("drug")) != str(drug):
            continue
        matches.append((condition_id, float(meta.get("dose", 0.0))))
    matches.sort(key=lambda item: item[1])
    return matches


def _sample_rows(pool: torch.Tensor, n: int, generator: torch.Generator) -> torch.Tensor:
    if pool.ndim != 2 or pool.shape[0] < 1:
        raise ValueError("Cannot sample from an empty empirical baseline pool.")
    index = torch.randint(pool.shape[0], (int(n),), generator=generator)
    return pool[index]


@torch.no_grad()
def generate_scrna_empirical_baseline(
    bundle: ConditionalPairBundle,
    split: ConditionalPairSplit,
    *,
    strategy: str,
    seed: int,
) -> torch.Tensor:
    """Generate count-valued scRNA baselines using training conditions only.

    These baselines are intentionally simple and leakage-free for the dose-holdout
    task: they may use treated cells from *training* doses of the same
    cell-line/drug pair, but never cells from the held-out target condition.
    """
    if strategy not in SCRNA_EMPIRICAL_BASELINE_LABELS:
        raise ValueError(f"Unknown scRNA empirical baseline: {strategy!r}")
    lookup = _scrna_condition_metadata(bundle)
    generator = torch.Generator().manual_seed(int(seed))
    generated = torch.empty_like(split.x1)

    for condition_id in sorted(map(int, split.condition_id.unique().tolist())):
        meta = lookup.get(condition_id)
        if meta is None:
            raise ValueError(f"Missing condition metadata for condition_id={condition_id}.")
        mask = split.condition_id == int(condition_id)
        n = int(mask.sum().item())
        cell = str(meta.get("cell_line_id"))
        drug = str(meta.get("drug"))
        dose = float(meta.get("dose", 0.0))
        candidates = _training_conditions_for_pair(bundle, lookup, cell, drug)
        if not candidates:
            raise ValueError(
                f"No training dose remains for held-out condition {condition_id} ({cell}, {drug}, dose={dose:g})."
            )

        if strategy == "pair_mean_poisson":
            pools = [bundle.train.x1[bundle.train.condition_id == train_id] for train_id, _ in candidates]
            pool = torch.cat(pools, dim=0).float()
            rate = pool.mean(dim=0, keepdim=True).expand(n, -1).clamp_min(0.0)
            draw = torch.poisson(rate, generator=generator).long()
        elif strategy == "nearest_dose_empirical":
            target_log = float(np.log1p(max(dose, 0.0)))
            train_id, _ = min(
                candidates,
                key=lambda item: abs(float(np.log1p(max(item[1], 0.0))) - target_log),
            )
            pool = bundle.train.x1[bundle.train.condition_id == train_id]
            draw = _sample_rows(pool, n, generator)
        else:
            target_log = float(np.log1p(max(dose, 0.0)))
            ordered = sorted(
                [(train_id, train_dose, float(np.log1p(max(train_dose, 0.0)))) for train_id, train_dose in candidates],
                key=lambda item: item[2],
            )
            lower = [item for item in ordered if item[2] <= target_log]
            upper = [item for item in ordered if item[2] >= target_log]
            if lower and upper and lower[-1][0] != upper[0][0]:
                low_id, _, low_log = lower[-1]
                high_id, _, high_log = upper[0]
                weight_high = (target_log - low_log) / max(high_log - low_log, 1e-12)
                choose_high = torch.rand(n, generator=generator) < float(weight_high)
                low_pool = bundle.train.x1[bundle.train.condition_id == low_id]
                high_pool = bundle.train.x1[bundle.train.condition_id == high_id]
                draw = _sample_rows(low_pool, n, generator)
                if bool(choose_high.any()):
                    draw[choose_high] = _sample_rows(high_pool, int(choose_high.sum().item()), generator)
            else:
                nearest_id, _, _ = min(ordered, key=lambda item: abs(item[2] - target_log))
                pool = bundle.train.x1[bundle.train.condition_id == nearest_id]
                draw = _sample_rows(pool, n, generator)
        generated[mask] = draw
    return generated



SCRNA_TRAINABLE_BASELINE_LABELS = {
    "poisson_glm_context": "Poisson GLM (condition only)",
    "poisson_mlp_context": "Poisson MLP (condition only)",
    "poisson_mlp_source_context": "Poisson MLP (DMSO + condition)",
}


class ScrnaConditionalPoisson(nn.Module):
    """Simple conditional Poisson baseline for held-out scRNA conditions.

    This is an application baseline only. It does not share or modify any Count
    Flow Map code. ``use_source=False`` predicts from the experimental condition
    alone; ``use_source=True`` also conditions on the matched DMSO count vector.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        *,
        use_source: bool,
        log_link: bool = False,
        hidden_dim: int = 512,
        depth: int = 2,
        rate_floor: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.context_dim = int(context_dim)
        self.use_source = bool(use_source)
        self.log_link = bool(log_link)
        self.rate_floor = float(rate_floor)
        input_dim = self.context_dim + (self.dim if self.use_source else 0)
        depth = int(depth)
        if depth <= 0:
            self.network = nn.Linear(input_dim, self.dim)
        else:
            layers: List[nn.Module] = [nn.Linear(input_dim, int(hidden_dim)), nn.SiLU()]
            for _ in range(depth - 1):
                layers.extend([nn.Linear(int(hidden_dim), int(hidden_dim)), nn.SiLU()])
            layers.append(nn.Linear(int(hidden_dim), self.dim))
            self.network = nn.Sequential(*layers)

    def forward(self, x0: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if x0.ndim != 2 or x0.shape[1] != self.dim:
            raise ValueError("x0 has the wrong shape for the scRNA Poisson baseline.")
        if context.ndim != 2 or context.shape != (x0.shape[0], self.context_dim):
            raise ValueError("context has the wrong shape for the scRNA Poisson baseline.")
        features = context.float()
        if self.use_source:
            features = torch.cat([torch.log1p(x0.float()), features], dim=-1)
        eta = self.network(features)
        if self.log_link:
            return torch.exp(eta.clamp(min=-12.0, max=8.0)) + self.rate_floor
        return F.softplus(eta) + self.rate_floor


@dataclass
class ScrnaBaselineTrainConfig:
    steps: int = 20_000
    batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    ema_decay: float = 0.9995
    grad_clip: float = 5.0
    log_every: int = 500
    seed: int = 42


def build_scrna_trainable_baseline(
    strategy: str,
    bundle: ConditionalPairBundle,
    cfg: Mapping[str, object],
) -> ScrnaConditionalPoisson:
    if strategy not in SCRNA_TRAINABLE_BASELINE_LABELS:
        raise ValueError(f"Unknown trainable scRNA baseline: {strategy!r}")
    if bundle.train.context.ndim != 2:
        raise ValueError("Trainable scRNA baselines require vector condition context.")
    use_source = strategy == "poisson_mlp_source_context"
    depth = 0 if strategy == "poisson_glm_context" else int(cfg.get("depth", 2))
    return ScrnaConditionalPoisson(
        bundle.dim,
        int(bundle.train.context.shape[1]),
        use_source=use_source,
        log_link=(strategy == "poisson_glm_context"),
        hidden_dim=int(cfg.get("hidden_dim", 512)),
        depth=depth,
        rate_floor=float(cfg.get("rate_floor", 1e-5)),
    )


def scrna_baseline_train_config(cfg: Mapping[str, object], seed: int) -> ScrnaBaselineTrainConfig:
    return ScrnaBaselineTrainConfig(
        steps=int(cfg.get("steps", 20_000)),
        batch_size=int(cfg.get("batch_size", 64)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
        ema_decay=float(cfg.get("ema_decay", 0.9995)),
        grad_clip=float(cfg.get("grad_clip", 5.0)),
        log_every=max(1, int(cfg.get("log_every", 500))),
        seed=int(seed),
    )


def train_scrna_poisson_baseline(
    model: ScrnaConditionalPoisson,
    train: ConditionalPairSplit,
    config: ScrnaBaselineTrainConfig,
    *,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, nn.Module, Dict[str, list]]:
    set_seed(config.seed)
    torch_device = resolve_device(device)
    model = model.to(torch_device)
    ema = make_ema_copy(model).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator(device=train.x0.device).manual_seed(config.seed + 211)
    history: Dict[str, list] = {"step": [], "poisson_nll": []}
    model.train()
    for step in range(1, int(config.steps) + 1):
        x0, target, context, _ = train.sample(
            config.batch_size, device=torch_device, generator=generator
        )
        rate = model(x0, context)
        # Mean over genes keeps the scale stable when the HVG dimension changes.
        loss = (rate - target.float() * torch.log(rate.clamp_min(1e-8))).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        update_ema(ema, model, config.ema_decay)
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            value = float(loss.detach().cpu())
            history["step"].append(step)
            history["poisson_nll"].append(value)
            if verbose:
                print(f"step={step:6d} scrna_poisson_nll={value:9.5f}", flush=True)
    ema.eval()
    return model, ema, history


@torch.no_grad()
def generate_scrna_poisson_baseline(
    model: ScrnaConditionalPoisson,
    split: ConditionalPairSplit,
    *,
    device: Optional[str] = None,
    batch_size: int = 256,
) -> torch.Tensor:
    torch_device = resolve_device(device)
    model = model.to(torch_device).eval()
    outputs: List[torch.Tensor] = []
    for first in range(0, split.n, int(batch_size)):
        last = min(first + int(batch_size), split.n)
        rate = model(split.x0[first:last].to(torch_device), split.context[first:last].to(torch_device))
        outputs.append(torch.poisson(rate).long().cpu())
    return torch.cat(outputs, dim=0)

def save_external_prediction_bundle(
    path: Path,
    *,
    generated: np.ndarray,
    condition_id: np.ndarray,
    metadata: Dict[str, object],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        generated=np.asarray(generated),
        condition_id=np.asarray(condition_id, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def load_external_prediction_bundle(path: Path) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    payload = np.load(Path(path), allow_pickle=False)
    generated = torch.from_numpy(payload["generated"])
    condition_id = torch.from_numpy(payload["condition_id"]).long()
    metadata = json.loads(str(payload["metadata_json"].item()))
    return generated, condition_id, metadata
