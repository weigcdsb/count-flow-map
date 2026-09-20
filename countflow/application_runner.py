from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .application_baselines import (
    ForecasterTrainConfig,
    SCRNA_EMPIRICAL_BASELINE_LABELS,
    SCRNA_TRAINABLE_BASELINE_LABELS,
    MambaPoissonForecaster,
    PoissonGLMForecaster,
    TransformerPoissonForecaster,
    build_scrna_trainable_baseline,
    generate_scrna_empirical_baseline,
    generate_scrna_poisson_baseline,
    load_external_prediction_bundle,
    sample_poisson_forecaster,
    scrna_baseline_train_config,
    train_poisson_forecaster,
    train_scrna_poisson_baseline,
)
from .application_data import (
    ConditionalPairBundle,
    ConditionalPairSplit,
    load_pair_bundle,
    load_spikeprophecy_session,
    make_neural_forecasting_surrogate,
    make_scrna_transport_surrogate,
)
from .application_metrics import (
    evaluate_neural_forecast_samples,
    evaluate_scrna_by_condition,
    fit_count_pca,
    select_best_nfe,
    select_saturated_nfe,
)
from .application_training import (
    ApplicationTrainConfig,
    generate_conditional_count_fm,
    generate_conditional_count_fm_unit_jump,
    generate_conditional_flow_map,
    train_conditional_count_fm,
    train_conditional_flow_map,
)
from .conditional_model import (
    ConditionalCountFlowMap,
    ConditionalCountRateModel,
    HistoryGRUEncoder,
    VectorContextEncoder,
)
from .scrna_response_baselines import (
    SCRNA_RESPONSE_BASELINE_VERSION,
    CPA_LABEL,
    LINEAR_LABEL,
    NBVAE_LABEL,
    SCGEN_LABEL,
    SCVIDR_LABEL,
    SINKHORN_LABEL,
    build_cpa,
    build_nbvae,
    build_scgen,
    generate_cpa,
    fit_linear_dose_response,
    fit_sinkhorn_ot,
    generate_linear_dose_response,
    generate_linear_dose_response_cached,
    generate_nbvae,
    generate_scgen,
    generate_scgen_cached,
    generate_scvidr,
    generate_scvidr_cached,
    generate_sinkhorn_ot,
    generate_sinkhorn_ot_cached,
    prepare_scgen_response_state,
    prepare_scvidr_response_state,
    published_train_config,
    train_cpa,
    train_nbvae,
    train_scgen,
)
from .utils import resolve_device, set_seed


SCRNA_REQUIRED = {"preset", "output_dir", "seeds", "nfe_values", "data", "models"}
NEURAL_REQUIRED = SCRNA_REQUIRED
APPLICATION_CHECKPOINT_VERSION = "application-checkpoint-v2-fingerprint"


def load_application_config(path: Path, *, kind: str) -> Dict[str, object]:
    config = json.loads(Path(path).read_text())
    required = SCRNA_REQUIRED if kind == "scrna" else NEURAL_REQUIRED
    missing = required.difference(config)
    if missing:
        raise ValueError(f"{kind} application config is missing keys: {sorted(missing)}")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str))


def _environment() -> Dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }




def _parameter_count(module: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _summarize_main_table(main: pd.DataFrame, output: Path) -> pd.DataFrame:
    numeric_columns = [
        column for column in main.select_dtypes(include=[np.number]).columns.tolist()
        if column not in {"seed"}
    ]
    rows: List[Dict[str, object]] = []
    for method, group in main.groupby("reported_method", sort=False):
        row: Dict[str, object] = {"reported_method": method, "n_runs": int(len(group))}
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            if values.notna().any():
                row[f"{column}_mean"] = float(values.mean())
                row[f"{column}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output, index=False)
    return summary

def _count_quantile_exact(
    values: torch.Tensor,
    q: float = 0.99,
    *,
    row_chunk_size: int = 512,
) -> float:
    """Exact linear quantile for nonnegative integer counts without sorting one huge tensor.

    ``torch.quantile`` sorts the full flattened input and fails for very large
    scRNA matrices.  Count data are nonnegative integers, so we can accumulate a
    small histogram in row chunks and recover the same linear order-statistic
    quantile without ever materializing/sorting all entries at once.
    """
    values = torch.as_tensor(values)
    if values.ndim == 0 or values.numel() == 0:
        raise ValueError("Cannot estimate count scale from an empty tensor.")
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError("q must lie in [0, 1].")

    histogram = torch.zeros(1, dtype=torch.long)
    total = 0
    n_rows = int(values.shape[0]) if values.ndim > 1 else 1
    row_chunk_size = max(1, int(row_chunk_size))

    if values.ndim == 1:
        chunks = [values]
    else:
        chunks = (values[start : start + row_chunk_size] for start in range(0, n_rows, row_chunk_size))

    for chunk in chunks:
        flat = torch.as_tensor(chunk).detach().cpu().reshape(-1).long()
        if flat.numel() == 0:
            continue
        if bool((flat < 0).any()):
            raise ValueError("Count scale requires nonnegative counts.")
        counts = torch.bincount(flat)
        if counts.numel() > histogram.numel():
            expanded = torch.zeros(counts.numel(), dtype=torch.long)
            expanded[: histogram.numel()] = histogram
            histogram = expanded
        histogram[: counts.numel()] += counts
        total += int(flat.numel())

    if total == 0:
        raise ValueError("Cannot estimate count scale from an empty tensor.")

    # Match torch.quantile's default linear interpolation between the two
    # neighboring order statistics at position q * (N - 1).
    position = float(q) * (total - 1)
    lower_rank = int(np.floor(position))
    upper_rank = int(np.ceil(position))
    weight = position - lower_rank
    cumulative = torch.cumsum(histogram, dim=0)

    def order_statistic(rank: int) -> float:
        target = torch.tensor(rank + 1, dtype=cumulative.dtype)
        return float(torch.searchsorted(cumulative, target, right=False).item())

    lower = order_statistic(lower_rank)
    upper = order_statistic(upper_rank)
    return (1.0 - weight) * lower + weight * upper


def _count_scale(bundle: ConditionalPairBundle) -> float:
    return max(2.0, _count_quantile_exact(bundle.train.x1, 0.99))


def _build_context_encoder(
    split: ConditionalPairSplit,
    model_cfg: Mapping[str, object],
) -> Tuple[torch.nn.Module, int]:
    context_hidden = int(model_cfg.get("context_hidden_dim", model_cfg.get("hidden_dim", 128)))
    context_output = int(model_cfg.get("context_output_dim", context_hidden))
    if split.context.ndim == 2:
        return VectorContextEncoder(split.context.shape[1], context_hidden, context_output), context_output
    if split.context.ndim == 3:
        return (
            HistoryGRUEncoder(
                split.dim,
                context_hidden,
                context_output,
                num_layers=int(model_cfg.get("context_layers", 1)),
                count_scale=float(model_cfg.get("history_count_scale", 4.0)),
            ),
            context_output,
        )
    raise ValueError("Unsupported context rank.")


def _build_flow_map(bundle: ConditionalPairBundle, cfg: Mapping[str, object]) -> ConditionalCountFlowMap:
    encoder, context_dim = _build_context_encoder(bundle.train, cfg)
    return ConditionalCountFlowMap(
        bundle.dim,
        encoder,
        context_dim,
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        depth=int(cfg.get("depth", 3)),
        n_mixtures=int(cfg.get("n_mixtures", 4)),
        count_scale=(
            float(cfg["count_scale"])
            if "count_scale" in cfg
            else _count_scale(bundle)
        ),
        correction_time_scale=str(cfg.get("correction_time_scale", "absolute")),
        correction_scale=float(cfg.get("correction_scale", 3.0)),
        death_chunk_size=int(cfg.get("death_chunk_size", 16)),
    )


def _build_count_fm(bundle: ConditionalPairBundle, cfg: Mapping[str, object]) -> ConditionalCountRateModel:
    encoder, context_dim = _build_context_encoder(bundle.train, cfg)
    return ConditionalCountRateModel(
        bundle.dim,
        encoder,
        context_dim,
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        depth=int(cfg.get("depth", 3)),
        count_scale=(
            float(cfg["count_scale"])
            if "count_scale" in cfg
            else _count_scale(bundle)
        ),
    )


def _train_config(cfg: Mapping[str, object], seed: int) -> ApplicationTrainConfig:
    return ApplicationTrainConfig(
        steps=int(cfg.get("steps", 5000)),
        batch_size=int(cfg.get("batch_size", 128)),
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
        tau=float(cfg.get("tau", 0.98)),
        ck_weight=float(cfg.get("ck_weight", 1.0)),
        ck_warmup_steps=int(cfg.get("ck_warmup_steps", 500)),
        max_span_start=float(cfg.get("max_span_start", 0.1)),
        span_warmup_steps=int(cfg.get("span_warmup_steps", 2500)),
        ema_decay=float(cfg.get("ema_decay", 0.999)),
        grad_clip=float(cfg.get("grad_clip", 5.0)),
        log_every=max(1, int(cfg.get("log_every", max(int(cfg.get("steps", 5000)) // 10, 1)))),
        seed=int(seed),
    )


def _save_model(path: Path, model: torch.nn.Module, history: Dict[str, list], metadata: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "history": history, "metadata": metadata}, path)


def _load_model(path: Path, model: torch.nn.Module, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, list], Dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload.get("history", {}), payload.get("metadata", {})


def _core_checkpoint_signature(
    method: str,
    bundle: ConditionalPairBundle,
    cfg: Mapping[str, object],
    seed: int,
) -> str:
    train_cfg = _train_config(cfg, seed)
    payload = {
        "version": APPLICATION_CHECKPOINT_VERSION,
        "method": str(method),
        "seed": int(seed),
        "bundle_fingerprint": str(bundle.metadata.get("bundle_fingerprint", "")),
        "dataset": str(bundle.metadata.get("dataset", "")),
        "dim": int(bundle.dim),
        "train_shape": list(bundle.train.x1.shape),
        "context_shape": list(bundle.train.context.shape[1:]),
        "train_config": vars(train_cfg),
        "model_config": dict(cfg),
    }
    canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _train_or_load_core(
    method: str,
    bundle: ConditionalPairBundle,
    cfg: Mapping[str, object],
    seed: int,
    output: Path,
    device: torch.device,
    resume: bool,
    progress: bool = False,
    validation_score=None,
) -> Tuple[torch.nn.Module, Dict[str, list], float, Path]:
    checkpoint = output / "checkpoints" / str(seed) / f"{method}.pt"
    set_seed(seed)
    if method == "count_flow_map":
        model = _build_flow_map(bundle, cfg)
    elif method == "count_fm":
        model = _build_count_fm(bundle, cfg)
    else:
        raise ValueError(method)

    compatibility_sha256 = _core_checkpoint_signature(method, bundle, cfg, seed)
    if resume and checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = dict(payload.get("metadata", {}))
        if metadata.get("compatibility_sha256") == compatibility_sha256:
            model.load_state_dict(payload["state_dict"])
            model.to(device).eval()
            return model, payload.get("history", {}), float(metadata.get("train_seconds", 0.0)), checkpoint
        if progress:
            print(
                f"[checkpoint] ignoring stale {method} checkpoint for seed={seed}; "
                "dataset/config fingerprint does not match this run.",
                flush=True,
            )

    train_cfg = _train_config(cfg, seed)
    selector = None
    if cfg.get("validation_selection"):
        from .validation import ValidationSelector
        if validation_score is None:
            raise ValueError("This configuration requires a validation scoring function.")
        selector = ValidationSelector(validation_score,
            every=int(cfg["validation_selection"]["every"]), steps=train_cfg.steps, verbose=progress)
    start = time.perf_counter()
    if method == "count_flow_map":
        _, ema, history = train_conditional_flow_map(
            model, bundle.train, train_cfg, device=str(device), verbose=progress,
            checkpoint_callback=selector,
        )
    else:
        _, ema, history = train_conditional_count_fm(
            model, bundle.train, train_cfg, device=str(device), verbose=progress,
            checkpoint_callback=selector,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - start
    metadata = {
            "method": method,
            "train_seconds": train_seconds,
            "train_config": vars(train_cfg),
            "model_config": dict(cfg),
            "checkpoint_version": APPLICATION_CHECKPOINT_VERSION,
            "compatibility_sha256": compatibility_sha256,
            "bundle_fingerprint": str(bundle.metadata.get("bundle_fingerprint", "")),
        }
    if selector is not None:
        selector.add_history(history)
        # Keep the final EMA too, so selected-versus-final uses the SAME fit.
        _save_model(checkpoint.with_name(f"{method}_final.pt"), ema, history,
                    {**metadata, "checkpoint_kind": "final_ema", "checkpoint_step": train_cfg.steps})
        selector.restore(ema)
        metadata.update(checkpoint_kind="validation_selected", checkpoint_step=selector.best_step)
    _save_model(checkpoint, ema, history, metadata)
    return ema, history, train_seconds, checkpoint



def _scrna_baseline_checkpoint_signature(
    strategy: str,
    bundle: ConditionalPairBundle,
    cfg: Mapping[str, object],
    seed: int,
) -> str:
    train_cfg = scrna_baseline_train_config(cfg, seed)
    payload = {
        "version": APPLICATION_CHECKPOINT_VERSION,
        "kind": "scrna_trainable_baseline",
        "strategy": str(strategy),
        "seed": int(seed),
        "bundle_fingerprint": str(bundle.metadata.get("bundle_fingerprint", "")),
        "dataset": str(bundle.metadata.get("dataset", "")),
        "dim": int(bundle.dim),
        "train_shape": list(bundle.train.x1.shape),
        "context_shape": list(bundle.train.context.shape[1:]),
        "train_config": vars(train_cfg),
        "model_config": dict(cfg),
    }
    canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _train_or_load_scrna_baseline(
    strategy: str,
    bundle: ConditionalPairBundle,
    cfg: Mapping[str, object],
    seed: int,
    output: Path,
    device: torch.device,
    resume: bool,
    progress: bool = False,
) -> Tuple[torch.nn.Module, Dict[str, list], float, Path]:
    model = build_scrna_trainable_baseline(strategy, bundle, cfg)
    checkpoint = output / "checkpoints" / str(seed) / f"{strategy}.pt"
    compatibility_sha256 = _scrna_baseline_checkpoint_signature(strategy, bundle, cfg, seed)
    if resume and checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = dict(payload.get("metadata", {}))
        if metadata.get("compatibility_sha256") == compatibility_sha256:
            model.load_state_dict(payload["state_dict"])
            model.to(device).eval()
            return model, payload.get("history", {}), float(metadata.get("train_seconds", 0.0)), checkpoint
        if progress:
            print(
                f"[checkpoint] ignoring stale {strategy} checkpoint for seed={seed}; "
                "dataset/config fingerprint does not match this run.",
                flush=True,
            )
    train_cfg = scrna_baseline_train_config(cfg, seed)
    start = time.perf_counter()
    _, ema, history = train_scrna_poisson_baseline(
        model, bundle.train, train_cfg, device=str(device), verbose=progress
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - start
    _save_model(
        checkpoint,
        ema,
        history,
        {
            "method": strategy,
            "train_seconds": train_seconds,
            "train_config": vars(train_cfg),
            "model_config": dict(cfg),
            "checkpoint_version": APPLICATION_CHECKPOINT_VERSION,
            "compatibility_sha256": compatibility_sha256,
            "bundle_fingerprint": str(bundle.metadata.get("bundle_fingerprint", "")),
        },
    )
    return ema, history, train_seconds, checkpoint

def _scrna_response_checkpoint_signature(
    kind: str,
    bundle: ConditionalPairBundle,
    cfg: Mapping[str, object],
    seed: int,
) -> str:
    train_cfg = published_train_config(cfg, seed)
    payload = {
        "version": SCRNA_RESPONSE_BASELINE_VERSION,
        "kind": str(kind),
        "seed": int(seed),
        "bundle_fingerprint": str(bundle.metadata.get("bundle_fingerprint", "")),
        "dataset": str(bundle.metadata.get("dataset", "")),
        "dim": int(bundle.dim),
        "train_shape": list(bundle.train.x1.shape),
        "context_shape": list(bundle.train.context.shape[1:]),
        "train_config": vars(train_cfg),
        "model_config": dict(cfg),
    }
    canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _train_or_load_scrna_response_model(
    kind: str,
    bundle: ConditionalPairBundle,
    cfg: Mapping[str, object],
    seed: int,
    output: Path,
    device: torch.device,
    resume: bool,
    progress: bool = False,
) -> Tuple[torch.nn.Module, Dict[str, list], float, Path]:
    if kind == "scgen_vae":
        model = build_scgen(bundle, cfg)
    elif kind == "cpa":
        model = build_cpa(bundle, cfg)
    elif kind == "nbvae":
        model = build_nbvae(bundle, cfg)
    else:
        raise ValueError(f"Unknown scRNA response model: {kind}")
    checkpoint = output / "checkpoints" / str(seed) / f"{kind}.pt"
    compatibility_sha256 = _scrna_response_checkpoint_signature(kind, bundle, cfg, seed)
    if resume and checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = dict(payload.get("metadata", {}))
        if metadata.get("compatibility_sha256") == compatibility_sha256:
            model.load_state_dict(payload["state_dict"])
            model.to(device).eval()
            return model, payload.get("history", {}), float(metadata.get("train_seconds", 0.0)), checkpoint
        if progress:
            print(
                f"[checkpoint] ignoring stale {kind} checkpoint for seed={seed}; "
                "dataset/config fingerprint does not match this run.",
                flush=True,
            )
    train_cfg = published_train_config(cfg, seed)
    start = time.perf_counter()
    if kind == "scgen_vae":
        model, history = train_scgen(model, bundle.train, train_cfg, device=str(device), verbose=progress)
    elif kind == "cpa":
        model, history = train_cpa(model, bundle, train_cfg, cfg, device=str(device), verbose=progress)
    else:
        model, history = train_nbvae(model, bundle.train, train_cfg, cfg, device=str(device), verbose=progress)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - start
    _save_model(
        checkpoint,
        model,
        history,
        {
            "method": kind,
            "train_seconds": train_seconds,
            "train_config": vars(train_cfg),
            "model_config": dict(cfg),
            "checkpoint_version": SCRNA_RESPONSE_BASELINE_VERSION,
            "compatibility_sha256": compatibility_sha256,
            "bundle_fingerprint": str(bundle.metadata.get("bundle_fingerprint", "")),
        },
    )
    return model, history, train_seconds, checkpoint


def _load_scrna_bundle(config: Mapping[str, object]) -> ConditionalPairBundle:
    data = dict(config["data"])
    mode = str(data.get("mode", "prepared_npz"))
    if mode == "synthetic_smoke":
        return make_scrna_transport_surrogate(
            dim=int(data.get("dim", 48)),
            n_cell_lines=int(data.get("n_cell_lines", 3)),
            n_drugs=int(data.get("n_drugs", 4)),
            cells_per_condition=int(data.get("cells_per_condition", 320)),
            latent_rank=int(data.get("latent_rank", 5)),
            seed=int(data.get("seed", 42)),
        )
    if mode == "prepared_npz":
        path = Path(str(data["path"]))
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        return load_pair_bundle(path)
    raise ValueError(f"Unsupported scRNA data mode: {mode}")


def _load_neural_bundles(config: Mapping[str, object]) -> List[Tuple[str, ConditionalPairBundle]]:
    data = dict(config["data"])
    mode = str(data.get("mode", "spikeprophecy"))
    if mode == "synthetic_smoke":
        return [
            (
                "synthetic_session",
                make_neural_forecasting_surrogate(
                    dim=int(data.get("dim", 32)),
                    n_bins=int(data.get("n_bins", 4800)),
                    history_bins=int(data.get("history_bins", 10)),
                    latent_dim=int(data.get("latent_dim", 6)),
                    seed=int(data.get("seed", 42)),
                ),
            )
        ]
    if mode == "spikeprophecy":
        root = Path(str(data["root"]))
        session_indices = [int(value) for value in data["session_indices"]]
        return [
            (
                f"session_{session:03d}",
                load_spikeprophecy_session(
                    root,
                    session,
                    history_bins=int(data.get("history_bins", 10)),
                    max_units=(None if data.get("max_units") is None else int(data["max_units"])),
                ),
            )
            for session in session_indices
        ]
    if mode == "spikeprophecy_multi":
        bundles: List[Tuple[str, ConditionalPairBundle]] = []
        for source in data["sources"]:
            source = dict(source)
            root = Path(str(source["root"]))
            prefix = str(source.get("prefix", root.name))
            for session in map(int, source["session_indices"]):
                bundles.append(
                    (
                        f"{prefix}_session_{session:03d}",
                        load_spikeprophecy_session(
                            root,
                            session,
                            history_bins=int(source.get("history_bins", data.get("history_bins", 10))),
                            max_units=(
                                None
                                if source.get("max_units", data.get("max_units")) is None
                                else int(source.get("max_units", data.get("max_units")))
                            ),
                        ),
                    )
                )
        return bundles
    raise ValueError(f"Unsupported neural data mode: {mode}")


def _time_generation(function, *args, repeats: int = 1, **kwargs) -> Tuple[torch.Tensor, float]:
    value = None
    timings = []
    for _ in range(max(1, int(repeats))):
        start = time.perf_counter()
        value = function(*args, **kwargs)
        timings.append(time.perf_counter() - start)
    assert value is not None
    return value, float(np.median(timings))


def _plot_quality_curve(
    frame: pd.DataFrame,
    *,
    metric: str,
    output: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for method, group in frame.groupby("method"):
        if group["nfe"].notna().sum() < 1:
            continue
        ordered = group.sort_values("nfe")
        ax.plot(ordered["nfe"], ordered[metric], marker="o", label=method)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("NFE")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_runtime_curve(
    frame: pd.DataFrame,
    *,
    metric: str,
    output: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for method, group in frame.groupby("method"):
        ordered = group.sort_values("generation_seconds")
        ax.plot(ordered["generation_seconds"], ordered[metric], marker="o", label=method)
    ax.set_xlabel("generation seconds")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_efficiency_nfe_all(
    frame: pd.DataFrame,
    *,
    metric: str,
    output: Path,
    title: str,
    methods: Optional[Sequence[str]] = None,
) -> None:
    subset = frame[frame["split"] == "test"].copy()
    if methods is not None:
        subset = subset[subset["method"].isin(list(methods))]
    subset["efficiency_nfe"] = pd.to_numeric(subset.get("efficiency_nfe"), errors="coerce")
    subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
    subset = subset[np.isfinite(subset["efficiency_nfe"]) & (subset["efficiency_nfe"] > 0) & np.isfinite(subset[metric])]
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for method, group in subset.groupby("method", sort=False):
        aggregated = group.groupby("efficiency_nfe", as_index=False)[metric].mean().sort_values("efficiency_nfe")
        if len(aggregated) == 1:
            ax.scatter(aggregated["efficiency_nfe"], aggregated[metric], s=42, label=method)
        else:
            ax.plot(aggregated["efficiency_nfe"], aggregated[metric], marker="o", label=method)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Model evaluations / sampling steps (method-specific)")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_runtime_all(
    frame: pd.DataFrame,
    *,
    metric: str,
    output: Path,
    title: str,
    methods: Optional[Sequence[str]] = None,
) -> None:
    subset = frame[frame["split"] == "test"].copy()
    if methods is not None:
        subset = subset[subset["method"].isin(list(methods))]
    subset["generation_seconds"] = pd.to_numeric(subset["generation_seconds"], errors="coerce")
    subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
    subset = subset[np.isfinite(subset["generation_seconds"]) & np.isfinite(subset[metric])]
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for method, group in subset.groupby("method", sort=False):
        grouped = group.groupby(["nfe", "efficiency_nfe"], dropna=False, as_index=False).mean(numeric_only=True)
        grouped = grouped.sort_values("generation_seconds")
        if len(grouped) == 1:
            ax.scatter(grouped["generation_seconds"], grouped[metric], s=42, label=method)
        else:
            ax.plot(grouped["generation_seconds"], grouped[metric], marker="o", label=method)
    ax.set_xlabel("Generation wall-clock seconds")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _scrna_condition_lookup(bundle: ConditionalPairBundle) -> Dict[int, Dict[str, object]]:
    raw = bundle.metadata.get("conditions", [])
    lookup: Dict[int, Dict[str, object]] = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping) and "condition_id" in item:
                lookup[int(item["condition_id"])] = dict(item)
    return lookup


def _validate_scrna_condition_holdout(bundle: ConditionalPairBundle) -> None:
    if bool(bundle.metadata.get("smoke_only", False)):
        return
    train_ids = set(map(int, bundle.train.condition_id.unique().tolist()))
    val_ids = set(map(int, bundle.val.condition_id.unique().tolist()))
    test_ids = set(map(int, bundle.test.condition_id.unique().tolist()))
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("scRNA condition IDs overlap across train/validation/test; complete-condition holdout is violated.")
    if not train_ids or not val_ids or not test_ids:
        raise ValueError("scRNA condition-holdout bundle must have nonempty train/validation/test condition sets.")
    lookup = _scrna_condition_lookup(bundle)
    if lookup:
        for split_name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
            for condition_id in ids:
                if condition_id not in lookup:
                    raise ValueError(f"Missing metadata for condition_id={condition_id}.")
                recorded = str(lookup[condition_id].get("split", ""))
                if recorded and recorded != split_name:
                    raise ValueError(
                        f"Condition metadata split mismatch for {condition_id}: metadata={recorded}, tensor={split_name}."
                    )


def _resolve_external_prediction_paths(config: Mapping[str, object], root: Path) -> Dict[str, List[Path]]:
    resolved: Dict[str, List[Path]] = {}
    for method, value in dict(config.get("external_predictions", {})).items():
        values = value if isinstance(value, (list, tuple)) else [value]
        method_paths: List[Path] = []
        for item in values:
            path = Path(str(item))
            if not path.is_absolute():
                path = root / path
            method_paths.append(path)
        resolved[str(method)] = method_paths
    return resolved


def _preflight_external_predictions(config: Mapping[str, object], root: Path) -> Dict[str, List[Path]]:
    paths = _resolve_external_prediction_paths(config, root)
    required = [str(x) for x in config.get("required_external_predictions", [])]
    if bool(config.get("require_external_predictions", False)):
        missing: List[Tuple[str, Path | None]] = []
        for name in required:
            method_paths = paths.get(name, [])
            if not method_paths:
                missing.append((name, None))
                continue
            missing.extend((name, path) for path in method_paths if not path.exists())
        if missing:
            detail = "\n".join(f"  - {name}: {path}" for name, path in missing)
            raise FileNotFoundError(
                "Formal scRNA run requires the external comparison prediction bundles before GPU training starts.\n"
                f"Missing:\n{detail}\n"
                "Prepare/package these baselines on exactly this Tahoe condition-holdout split before running in strict mode."
            )
    return paths


def _attach_condition_metadata(
    row: Mapping[str, object],
    condition: Mapping[str, float],
    lookup: Mapping[int, Mapping[str, object]],
) -> Dict[str, object]:
    value: Dict[str, object] = {**dict(row), **dict(condition)}
    condition_id = int(round(float(condition["condition_id"])))
    value["condition_id"] = condition_id
    meta = lookup.get(condition_id)
    if meta is not None:
        for key in ("condition_name", "cell_line_id", "drug", "dose", "split"):
            if key in meta:
                value[key] = meta[key]
    return value


def _plot_scrna_method_comparison(summary: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("sliced_w2_mean", "Sliced W2", False),
        ("deg_logfc_pearson_mean", "DEG logFC Pearson", True),
        ("top_response_gene_overlap_mean", "Top-response gene overlap", True),
    ]
    available = [item for item in metrics if item[0] in summary.columns]
    if not available or summary.empty:
        return
    fig, axes = plt.subplots(1, len(available), figsize=(5.2 * len(available), 4.6), squeeze=False)
    labels = summary["reported_method"].astype(str).tolist()
    x = np.arange(len(labels))
    for ax, (metric, label, higher_is_better) in zip(axes[0], available):
        values = pd.to_numeric(summary[metric], errors="coerce").to_numpy(dtype=float)
        std_col = metric.replace("_mean", "_std")
        error = pd.to_numeric(summary[std_col], errors="coerce").fillna(0.0).to_numpy(dtype=float) if std_col in summary else None
        ax.bar(x, values, yerr=error, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title("higher is better" if higher_is_better else "lower is better")
    fig.suptitle("Held-out Tahoe treatment-condition prediction")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _select_representative_condition_ids(bundle: ConditionalPairBundle, n_conditions: int) -> List[int]:
    strengths: List[Tuple[float, int]] = []
    for condition_id in map(int, bundle.test.condition_id.unique().tolist()):
        mask = bundle.test.condition_id == condition_id
        control = bundle.test.x0[mask].double().mean(0)
        target = bundle.test.x1[mask].double().mean(0)
        effect = torch.log1p(target) - torch.log1p(control)
        strengths.append((float(torch.linalg.norm(effect).item()), condition_id))
    strengths.sort(reverse=True)
    return [condition_id for _, condition_id in strengths[: max(1, int(n_conditions))]]


def _plot_scrna_representative_conditions(
    bundle: ConditionalPairBundle,
    generated: torch.Tensor,
    projection,
    *,
    condition_ids: Sequence[int],
    output: Path,
    seed: int,
) -> None:
    lookup = _scrna_condition_lookup(bundle)
    rows = len(condition_ids)
    if rows < 1:
        return
    fig, axes = plt.subplots(rows, 2, figsize=(10.5, 4.1 * rows), squeeze=False)
    rng = np.random.default_rng(seed)
    for row_index, condition_id in enumerate(condition_ids):
        mask = bundle.test.condition_id == int(condition_id)
        control = bundle.test.x0[mask]
        target = bundle.test.x1[mask]
        pred = generated[mask]
        control_pca = projection.transform(control)[:, :2].numpy()
        target_pca = projection.transform(target)[:, :2].numpy()
        pred_pca = projection.transform(pred)[:, :2].numpy()
        ax = axes[row_index, 0]
        for points, label in ((control_pca, "DMSO"), (target_pca, "Observed treated"), (pred_pca, "Count Flow Map")):
            if len(points) > 400:
                points = points[rng.choice(len(points), 400, replace=False)]
            ax.scatter(points[:, 0], points[:, 1], s=8, alpha=0.35, label=label)
        meta = lookup.get(int(condition_id), {})
        name = str(meta.get("condition_name", f"condition {condition_id}"))
        ax.set_title(name)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        if row_index == 0:
            ax.legend(fontsize=8)

        ctl_mean = control.double().mean(0)
        real = (torch.log1p(target.double().mean(0)) - torch.log1p(ctl_mean)).numpy()
        pred_effect = (torch.log1p(pred.double().mean(0)) - torch.log1p(ctl_mean)).numpy()
        ax2 = axes[row_index, 1]
        ax2.scatter(real, pred_effect, s=8, alpha=0.35)
        lo = float(min(real.min(), pred_effect.min())); hi = float(max(real.max(), pred_effect.max()))
        ax2.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        corr = np.corrcoef(real, pred_effect)[0, 1] if np.std(real) > 1e-12 and np.std(pred_effect) > 1e-12 else np.nan
        ax2.set_title(f"Perturbation effect recovery | r={corr:.3f}")
        ax2.set_xlabel("Observed log1p-mean effect")
        ax2.set_ylabel("Generated log1p-mean effect")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_scrna_application(
    config_path: Path | Mapping[str, object],
    *,
    device: Optional[str] = None,
    resume: bool = True,
    progress: bool = False,
) -> Path:
    if isinstance(config_path, Mapping):
        config = copy.deepcopy(dict(config_path))
        missing = SCRNA_REQUIRED.difference(config)
        if missing:
            raise ValueError(f"scrna application config is missing keys: {sorted(missing)}")
        config_sha256 = hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
    else:
        config_path = Path(config_path)
        config = load_application_config(config_path, kind="scrna")
        config_sha256 = _sha256(config_path)

    root = Path(__file__).resolve().parents[1]
    output = root / str(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    _save_json(output / "config_snapshot.json", config)
    _save_json(output / "environment.json", _environment())

    if progress:
        print("[scRNA] loading prepared condition-holdout bundle ...", flush=True)
    stage_start = time.perf_counter()
    bundle = _load_scrna_bundle(config)
    _validate_scrna_condition_holdout(bundle)
    torch_device = resolve_device(device)
    if progress:
        split_conditions = bundle.metadata.get("split_conditions", {})
        print(
            f"[scRNA] bundle loaded in {time.perf_counter()-stage_start:.2f}s | "
            f"train={tuple(bundle.train.x1.shape)} val={tuple(bundle.val.x1.shape)} "
            f"test={tuple(bundle.test.x1.shape)} | conditions={split_conditions} | device={torch_device}",
            flush=True,
        )

    external_paths = _preflight_external_predictions(config, root)
    if progress and not bool(config.get("require_external_predictions", False)):
        missing_external = [
            name for name, paths in external_paths.items()
            if not paths or any(not path.exists() for path in paths)
        ]
        if missing_external:
            print(f"[scRNA] external baselines incomplete; skipping missing bundles: {missing_external}", flush=True)

    pca_cfg = dict(config.get("pca", {}))
    if progress:
        print(
            f"[scRNA] fitting training-only evaluation PCA on DMSO + treated cells: "
            f"n_components={int(pca_cfg.get('n_components', 32))} "
            f"max_cells={int(pca_cfg.get('max_cells', 20000))} ...",
            flush=True,
        )
    stage_start = time.perf_counter()
    pca_training_counts = torch.cat([bundle.train.x0, bundle.train.x1], dim=0)
    projection = fit_count_pca(
        pca_training_counts,
        n_components=int(pca_cfg.get("n_components", 32)),
        max_cells=int(pca_cfg.get("max_cells", 20000)),
        seed=int(pca_cfg.get("seed", 42)),
    )
    if progress:
        print(f"[scRNA] PCA ready in {time.perf_counter()-stage_start:.2f}s", flush=True)

    nfe_values = [int(value) for value in config["nfe_values"]]
    if sorted(set(nfe_values)) != nfe_values or any(value < 1 for value in nfe_values):
        raise ValueError("nfe_values must be unique positive integers in increasing order.")
    rows: List[Dict[str, object]] = []
    per_condition_rows: List[Dict[str, object]] = []
    prediction_audit_rows: List[Dict[str, object]] = []
    timing_repeats = int(config.get("timing_repeats", 1))
    # Every method must be scored with the same PCA projections / metric subsamples.
    # Method-specific metric seeds would add avoidable Monte Carlo noise to the comparison.
    common_metric_seed = int(config.get("evaluation_seed", 314159))
    condition_lookup = _scrna_condition_lookup(bundle)

    def append_evaluation(
        *,
        generated: torch.Tensor,
        split: ConditionalPairSplit,
        split_name: str,
        seed_value: object,
        method: str,
        nfe: object,
        generation_seconds: object,
        train_seconds: object,
        parameters: object,
        metric_seed: int,
        efficiency_nfe: object = np.nan,
        efficiency_kind: str = "unknown",
    ) -> None:
        aggregate, conditions = evaluate_scrna_by_condition(
            generated, split.x0, split.x1, split.condition_id, projection, seed=common_metric_seed
        )
        row: Dict[str, object] = {
            "seed": seed_value,
            "split": split_name,
            "method": method,
            "nfe": nfe,
            "generation_seconds": generation_seconds,
            "train_seconds": train_seconds,
            "parameters": parameters,
            "efficiency_nfe": efficiency_nfe,
            "efficiency_kind": str(efficiency_kind),
            **aggregate,
        }
        rows.append(row)
        for condition in conditions:
            per_condition_rows.append(_attach_condition_metadata(row, condition, condition_lookup))

    for seed in map(int, config["seeds"]):
        set_seed(seed)
        models_cfg = dict(config["models"])
        if progress:
            print(f"\n[scRNA] seed={seed} | Count Flow Map", flush=True)
        flow, _, flow_train_seconds, _ = _train_or_load_core(
            "count_flow_map", bundle, dict(models_cfg["count_flow_map"]), seed,
            output, torch_device, resume, progress=progress,
        )
        if progress:
            print(f"[scRNA] seed={seed} | Count-FM", flush=True)
        count_fm, _, fm_train_seconds, _ = _train_or_load_core(
            "count_fm", bundle, dict(models_cfg["count_fm"]), seed,
            output, torch_device, resume, progress=progress,
        )

        unit_jump_nfe_values = [int(value) for value in config.get("unit_jump_nfe_values", nfe_values)]
        if sorted(set(unit_jump_nfe_values)) != unit_jump_nfe_values or any(value < 1 for value in unit_jump_nfe_values):
            raise ValueError("unit_jump_nfe_values must be unique positive integers in increasing order.")

        for split_name, split in (("val", bundle.val), ("test", bundle.test)):
            for nfe in nfe_values:
                if progress:
                    print(f"[scRNA eval] seed={seed} split={split_name} NFE={nfe}", flush=True)
                generated, seconds = _time_generation(
                    generate_conditional_flow_map, flow, split.x0, split.context,
                    n_steps=nfe, tau=float(config.get("tau", 0.98)), device=str(torch_device),
                    batch_size=int(config.get("generation_batch_size", 512)), repeats=timing_repeats,
                )
                append_evaluation(
                    generated=generated, split=split, split_name=split_name, seed_value=seed,
                    method="Count Flow Map", nfe=nfe, generation_seconds=seconds,
                    train_seconds=flow_train_seconds, parameters=_parameter_count(flow), metric_seed=seed,
                    efficiency_nfe=nfe, efficiency_kind="flow-map-kernel-evaluations",
                )

                generated_fm, seconds_fm = _time_generation(
                    generate_conditional_count_fm, count_fm, split.x0, split.context,
                    n_steps=nfe, tau=float(config.get("tau", 0.98)), device=str(torch_device),
                    batch_size=int(config.get("generation_batch_size", 512)), repeats=timing_repeats,
                )
                append_evaluation(
                    generated=generated_fm, split=split, split_name=split_name, seed_value=seed,
                    method="Count-FM + binomial tau-leap", nfe=nfe, generation_seconds=seconds_fm,
                    train_seconds=fm_train_seconds, parameters=_parameter_count(count_fm), metric_seed=seed + 1000,
                    efficiency_nfe=nfe, efficiency_kind="count-fm-rate-evaluations",
                )

            for nfe in unit_jump_nfe_values:
                if progress:
                    print(f"[scRNA eval] seed={seed} split={split_name} unit-jump steps={nfe}", flush=True)
                generated_unit, seconds_unit = _time_generation(
                    generate_conditional_count_fm_unit_jump, count_fm, split.x0, split.context,
                    n_steps=nfe, tau=float(config.get("tau", 0.98)), device=str(torch_device),
                    batch_size=int(config.get("generation_batch_size", 512)), repeats=timing_repeats,
                )
                append_evaluation(
                    generated=generated_unit, split=split, split_name=split_name, seed_value=seed,
                    method="Count-FM + unit jump", nfe=nfe, generation_seconds=seconds_unit,
                    train_seconds=fm_train_seconds, parameters=_parameter_count(count_fm), metric_seed=seed + 2000,
                    efficiency_nfe=nfe, efficiency_kind="count-fm-unit-jump-steps",
                )

    # Paper-level perturbation-response baselines. These are self-contained and
    # use the same frozen Tahoe split. Only their own new checkpoints are trained;
    # existing Count Flow Map / Count-FM checkpoints above are always reused when compatible.
    published_names = [str(x) for x in config.get("published_baselines", [])]
    valid_published = {"scgen", "scvidr", "cpa", "nbvae", "linear_dose", "sinkhorn_ot"}
    unknown_published = sorted(set(published_names).difference(valid_published))
    if unknown_published:
        raise ValueError(f"Unknown published scRNA baselines: {unknown_published}")
    models_cfg = dict(config["models"])

    published_resume = resume and not bool(config.get("retrain_published_baselines", False))

    if "scgen" in published_names or "scvidr" in published_names:
        scgen_cfg = dict(models_cfg["scgen_vae"])
        for seed in map(int, config["seeds"]):
            if progress:
                print(f"[scRNA paper baseline] seed={seed} | shared scGen/scVIDR VAE", flush=True)
            scgen_model, _, scgen_train_seconds, _ = _train_or_load_scrna_response_model(
                "scgen_vae", bundle, scgen_cfg, seed, output, torch_device, published_resume, progress=progress
            )
            prep_start = time.perf_counter()
            scgen_state = prepare_scgen_response_state(
                scgen_model, bundle, device=str(torch_device),
                batch_size=int(config.get("generation_batch_size", 512)),
            )
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            scgen_prep_seconds = time.perf_counter() - prep_start

            prep_start = time.perf_counter()
            scvidr_state = prepare_scvidr_response_state(
                bundle, scgen_state, ridge=float(scgen_cfg.get("scvidr_ridge", 0.0))
            )
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            scvidr_prep_seconds = time.perf_counter() - prep_start

            for split_name, split in (("val", bundle.val), ("test", bundle.test)):
                generated_scgen = None
                generated_scvidr = None
                if "scgen" in published_names:
                    start = time.perf_counter()
                    generated_scgen = generate_scgen_cached(
                        scgen_model, split, scgen_state, device=str(torch_device),
                        batch_size=int(config.get("generation_batch_size", 512)),
                    )
                    if torch_device.type == "cuda":
                        torch.cuda.synchronize(torch_device)
                    seconds = time.perf_counter() - start
                    append_evaluation(
                        generated=generated_scgen, split=split, split_name=split_name, seed_value=seed,
                        method=SCGEN_LABEL, nfe=np.nan, generation_seconds=seconds,
                        train_seconds=scgen_train_seconds + scgen_prep_seconds,
                        parameters=_parameter_count(scgen_model), metric_seed=common_metric_seed,
                        efficiency_nfe=1, efficiency_kind="one-pass-latent-arithmetic",
                    )
                if "scvidr" in published_names:
                    start = time.perf_counter()
                    generated_scvidr = generate_scvidr_cached(
                        scgen_model, split, scvidr_state, device=str(torch_device),
                        batch_size=int(config.get("generation_batch_size", 512)),
                    )
                    if torch_device.type == "cuda":
                        torch.cuda.synchronize(torch_device)
                    seconds = time.perf_counter() - start
                    append_evaluation(
                        generated=generated_scvidr, split=split, split_name=split_name, seed_value=seed,
                        method=SCVIDR_LABEL, nfe=np.nan, generation_seconds=seconds,
                        train_seconds=scgen_train_seconds + scgen_prep_seconds + scvidr_prep_seconds,
                        parameters=_parameter_count(scgen_model), metric_seed=common_metric_seed,
                        efficiency_nfe=1, efficiency_kind="one-pass-latent-dose-scaling",
                    )
                if generated_scgen is not None and generated_scvidr is not None:
                    a = generated_scgen.to(torch.float32)
                    b = generated_scvidr.to(torch.float32)
                    prediction_audit_rows.append({
                        "seed": seed,
                        "split": split_name,
                        "method_a": SCGEN_LABEL,
                        "method_b": SCVIDR_LABEL,
                        "mean_absolute_difference": float((a - b).abs().mean().item()),
                        "fraction_exactly_equal": float((a == b).float().mean().item()),
                    })

    if "cpa" in published_names:
        cpa_cfg = dict(models_cfg["cpa"])
        for seed in map(int, config["seeds"]):
            if progress:
                print(f"[scRNA paper baseline] seed={seed} | {CPA_LABEL}", flush=True)
            cpa_model, _, cpa_train_seconds, _ = _train_or_load_scrna_response_model(
                "cpa", bundle, cpa_cfg, seed, output, torch_device, published_resume, progress=progress
            )
            for split_name, split in (("val", bundle.val), ("test", bundle.test)):
                start = time.perf_counter()
                generated = generate_cpa(
                    cpa_model, bundle, split, seed=seed + 3300, device=str(torch_device),
                    batch_size=int(config.get("generation_batch_size", 512)),
                )
                if torch_device.type == "cuda":
                    torch.cuda.synchronize(torch_device)
                seconds = time.perf_counter() - start
                append_evaluation(
                    generated=generated, split=split, split_name=split_name, seed_value=seed,
                    method=CPA_LABEL, nfe=np.nan, generation_seconds=seconds,
                    train_seconds=cpa_train_seconds, parameters=_parameter_count(cpa_model),
                    metric_seed=common_metric_seed, efficiency_nfe=1, efficiency_kind="one-pass-compositional-autoencoder",
                )

    if "nbvae" in published_names:
        nbvae_cfg = dict(models_cfg["nbvae"])
        for seed in map(int, config["seeds"]):
            if progress:
                print(f"[scRNA paper baseline] seed={seed} | {NBVAE_LABEL}", flush=True)
            nbvae_model, _, nbvae_train_seconds, _ = _train_or_load_scrna_response_model(
                "nbvae", bundle, nbvae_cfg, seed, output, torch_device, published_resume, progress=progress
            )
            for split_name, split in (("val", bundle.val), ("test", bundle.test)):
                start = time.perf_counter()
                generated = generate_nbvae(
                    nbvae_model, split, seed=seed + 3500, device=str(torch_device),
                    batch_size=int(config.get("generation_batch_size", 512)),
                )
                if torch_device.type == "cuda":
                    torch.cuda.synchronize(torch_device)
                seconds = time.perf_counter() - start
                append_evaluation(
                    generated=generated, split=split, split_name=split_name, seed_value=seed,
                    method=NBVAE_LABEL, nfe=np.nan, generation_seconds=seconds,
                    train_seconds=nbvae_train_seconds, parameters=_parameter_count(nbvae_model),
                    metric_seed=common_metric_seed, efficiency_nfe=1, efficiency_kind="one-pass-latent-count-model",
                )

    if "linear_dose" in published_names:
        linear_cfg = dict(models_cfg.get("linear_dose", {}))
        start = time.perf_counter()
        linear_state = fit_linear_dose_response(bundle, ridge=float(linear_cfg.get("ridge", 1e-4)))
        linear_fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        generated = generate_linear_dose_response_cached(bundle.test, linear_state)
        seconds = time.perf_counter() - start
        append_evaluation(
            generated=generated, split=bundle.test, split_name="test", seed_value=np.nan,
            method=LINEAR_LABEL, nfe=np.nan, generation_seconds=seconds, train_seconds=linear_fit_seconds,
            parameters=0, metric_seed=common_metric_seed, efficiency_nfe=1, efficiency_kind="one-pass-linear-response",
        )

    if "sinkhorn_ot" in published_names:
        sinkhorn_cfg = dict(models_cfg.get("sinkhorn_ot", {}))
        start = time.perf_counter()
        sinkhorn_state = fit_sinkhorn_ot(
            bundle, bundle.test,
            epsilon_scale=float(sinkhorn_cfg.get("epsilon_scale", 0.1)),
            iterations=int(sinkhorn_cfg.get("iterations", 100)),
            pca_dim=int(sinkhorn_cfg.get("pca_dim", 30)),
            max_cells=int(sinkhorn_cfg.get("max_cells", 384)),
            seed=int(sinkhorn_cfg.get("seed", 42)),
        )
        sinkhorn_fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        generated = generate_sinkhorn_ot_cached(
            bundle, bundle.test, sinkhorn_state, knn=int(sinkhorn_cfg.get("knn", 8))
        )
        seconds = time.perf_counter() - start
        append_evaluation(
            generated=generated, split=bundle.test, split_name="test", seed_value=np.nan,
            method=SINKHORN_LABEL, nfe=np.nan, generation_seconds=seconds, train_seconds=sinkhorn_fit_seconds,
            parameters=0, metric_seed=common_metric_seed, efficiency_nfe=1, efficiency_kind="one-pass-transport-interpolation",
        )

    if prediction_audit_rows:
        audit_frame = pd.DataFrame(prediction_audit_rows)
        audit_frame.to_csv(output / "baseline_prediction_audit.csv", index=False)
        suspicious = audit_frame[audit_frame["fraction_exactly_equal"] > 0.999]
        if progress and not suspicious.empty:
            print(
                "[scRNA audit] WARNING: scGen/scVIDR predictions are almost identical for "
                f"{len(suspicious)} seed/split combinations; inspect baseline_prediction_audit.csv",
                flush=True,
            )

    trainable_names = [str(x) for x in config.get("trainable_baselines", [])]
    for seed in map(int, config["seeds"]):
        models_cfg = dict(config["models"])
        for strategy in trainable_names:
            if strategy not in SCRNA_TRAINABLE_BASELINE_LABELS:
                raise ValueError(f"Unknown trainable scRNA baseline {strategy!r}.")
            if strategy not in models_cfg:
                raise ValueError(f"Missing model config for trainable scRNA baseline {strategy!r}.")
            if progress:
                print(f"[scRNA baseline] seed={seed} | {SCRNA_TRAINABLE_BASELINE_LABELS[strategy]}", flush=True)
            baseline, _, baseline_train_seconds, _ = _train_or_load_scrna_baseline(
                strategy, bundle, dict(models_cfg[strategy]), seed,
                output, torch_device, resume, progress=progress,
            )
            for split_name, split in (("val", bundle.val), ("test", bundle.test)):
                start = time.perf_counter()
                generated = generate_scrna_poisson_baseline(
                    baseline, split, device=str(torch_device),
                    batch_size=int(config.get("generation_batch_size", 512)),
                )
                if torch_device.type == "cuda":
                    torch.cuda.synchronize(torch_device)
                seconds = time.perf_counter() - start
                append_evaluation(
                    generated=generated, split=split, split_name=split_name, seed_value=seed,
                    method=SCRNA_TRAINABLE_BASELINE_LABELS[strategy], nfe=np.nan,
                    generation_seconds=seconds, train_seconds=baseline_train_seconds,
                    parameters=_parameter_count(baseline), metric_seed=seed + 5000,
                    efficiency_nfe=1, efficiency_kind="one-pass",
                )

    oracle_metrics, _ = evaluate_scrna_by_condition(
        bundle.test.x1.clone(), bundle.test.x0, bundle.test.x1, bundle.test.condition_id,
        projection, seed=99173,
    )
    if float(oracle_metrics.get("sliced_w2", np.inf)) > 1e-6:
        raise RuntimeError("scRNA evaluation sanity check failed: target-vs-target sliced W2 is not zero.")
    _save_json(output / "sanity_metrics.json", {"oracle_target": oracle_metrics})

    builtin_names = [str(x) for x in config.get(
        "builtin_baselines", list(SCRNA_EMPIRICAL_BASELINE_LABELS.keys())
    )]
    for seed in map(int, config["seeds"]):
        for strategy in builtin_names:
            if strategy not in SCRNA_EMPIRICAL_BASELINE_LABELS:
                raise ValueError(f"Unknown builtin scRNA baseline {strategy!r}.")
            if progress:
                print(f"[scRNA baseline] seed={seed} | {SCRNA_EMPIRICAL_BASELINE_LABELS[strategy]}", flush=True)
            start = time.perf_counter()
            generated = generate_scrna_empirical_baseline(
                bundle, bundle.test, strategy=strategy, seed=seed + 7000
            )
            seconds = time.perf_counter() - start
            append_evaluation(
                generated=generated, split=bundle.test, split_name="test", seed_value=seed,
                method=SCRNA_EMPIRICAL_BASELINE_LABELS[strategy], nfe=np.nan,
                generation_seconds=seconds, train_seconds=0.0, parameters=0,
                metric_seed=seed + 8000, efficiency_nfe=1, efficiency_kind="empirical-one-pass",
            )

    # Deterministic lower-bound sanity baseline: do nothing to the matched DMSO source.
    append_evaluation(
        generated=bundle.test.x0.clone(), split=bundle.test, split_name="test", seed_value=np.nan,
        method="DMSO / no treatment", nfe=np.nan, generation_seconds=0.0,
        train_seconds=0.0, parameters=0, metric_seed=31415,
        efficiency_nfe=1, efficiency_kind="identity-reference",
    )

    expected_fingerprint = str(bundle.metadata.get("bundle_fingerprint", ""))
    for method, method_paths in external_paths.items():
        for path in method_paths:
            if not path.exists():
                continue
            generated, condition_id, metadata = load_external_prediction_bundle(path)
            if generated.shape != bundle.test.x1.shape:
                raise ValueError(
                    f"External predictions for {method} have shape {tuple(generated.shape)}; "
                    f"expected {tuple(bundle.test.x1.shape)}."
                )
            if not torch.equal(condition_id, bundle.test.condition_id):
                raise ValueError(f"External predictions for {method} do not align with the test condition ordering.")
            supplied_fingerprint = str(metadata.get("bundle_fingerprint", ""))
            if expected_fingerprint and supplied_fingerprint != expected_fingerprint:
                raise ValueError(
                    f"External predictions for {method} were packaged for a different Tahoe split. "
                    f"Expected bundle_fingerprint={expected_fingerprint}, got {supplied_fingerprint or '<missing>'}."
                )
            display_method = str(metadata.get("method", method))
            append_evaluation(
                generated=generated, split=bundle.test, split_name="test",
                seed_value=metadata.get("seed", np.nan), method=display_method, nfe=np.nan,
                generation_seconds=metadata.get("generation_seconds", np.nan),
                train_seconds=metadata.get("train_seconds", np.nan),
                parameters=metadata.get("parameters", np.nan), metric_seed=123,
                efficiency_nfe=metadata.get("nfe", metadata.get("sampling_steps", np.nan)),
                efficiency_kind=str(metadata.get("nfe_kind", "external-method-specific")),
            )

    frame = pd.DataFrame(rows)
    condition_frame = pd.DataFrame(per_condition_rows)
    frame.to_csv(output / "metrics.csv", index=False)
    condition_frame.to_csv(output / "condition_metrics.csv", index=False)
    frame[frame["split"] == "val"].to_csv(output / "validation_diagnostics.csv", index=False)
    aggregate_frame = (
        frame.groupby(["split", "method", "nfe"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    aggregate_frame.to_csv(output / "aggregate_metrics.csv", index=False)

    selection_metric = str(config.get("selection_metric", "sliced_w2"))
    if selection_metric not in frame.columns:
        raise ValueError(f"Unknown scRNA selection_metric={selection_metric!r}.")
    selected_nfe: Dict[Tuple[int, str], int] = {}
    all_report_rows: List[Dict[str, object]] = []
    iterative_methods = (
        "Count Flow Map",
        "Count-FM + unit jump",
        "Count-FM + binomial tau-leap",
    )
    for seed in map(int, config["seeds"]):
        val_seed = frame[(frame.seed == seed) & (frame.split == "val")]
        for method in iterative_methods:
            subset = val_seed[val_seed.method == method].sort_values("nfe")
            if subset.empty:
                raise RuntimeError(f"Missing validation rows for {method} seed={seed}.")
            best_nfe = select_best_nfe(
                subset.nfe.astype(int).tolist(), subset[selection_metric].astype(float).tolist()
            )
            selected_nfe[(seed, method)] = best_nfe
            test_seed = frame[
                (frame.seed == seed) & (frame.split == "test") & (frame.method == method)
            ]
            if method == "Count Flow Map":
                one = test_seed[test_seed.nfe == 1].iloc[0].to_dict()
                one["reported_method"] = "Count Flow Map (1 NFE)"
                one["reported_nfe"] = 1
                one["validation_selected_nfe"] = best_nfe
                all_report_rows.append(one)
            selected = test_seed[test_seed.nfe == best_nfe].iloc[0].to_dict()
            selected["reported_method"] = {
                "Count Flow Map": "Count Flow Map (validation-selected)",
                "Count-FM + unit jump": "Count-FM + unit jump (validation-selected)",
                "Count-FM + binomial tau-leap": "Count-FM + binomial tau-leap (validation-selected)",
            }[method]
            selected["reported_nfe"] = best_nfe
            selected["validation_selected_nfe"] = best_nfe
            all_report_rows.append(selected)

    non_nfe_test = frame[(frame.split == "test") & (frame.nfe.isna())]
    for _, row in non_nfe_test.iterrows():
        value = row.to_dict()
        value["reported_method"] = value["method"]
        value["reported_nfe"] = value.get("efficiency_nfe", np.nan)
        value["validation_selected_nfe"] = np.nan
        all_report_rows.append(value)

    all_report = pd.DataFrame(all_report_rows)
    all_report.to_csv(output / "supplementary_quality_table_raw.csv", index=False)
    supplementary_summary = _summarize_main_table(
        all_report, output / "supplementary_quality_table_summary.csv"
    )

    paper_methods = [str(x) for x in config.get("paper_methods", [])]
    if paper_methods:
        main = all_report[all_report["reported_method"].isin(paper_methods)].copy()
        paper_order = {method: index for index, method in enumerate(paper_methods)}
        main["_paper_order"] = main["reported_method"].map(paper_order)
        main = main.sort_values(["_paper_order", "seed"], na_position="last").drop(columns="_paper_order")
    else:
        main = all_report.copy()
    main.to_csv(output / "main_quality_table_raw.csv", index=False)
    summary = _summarize_main_table(main, output / "main_quality_table_summary.csv")

    paper_curve_methods = [str(x) for x in config.get("paper_curve_methods", [])]
    if not paper_curve_methods:
        paper_curve_methods = [
            "Count Flow Map",
            "Count-FM + unit jump",
            "Count-FM + binomial tau-leap",
            *[str(x) for x in config.get("required_external_predictions", [])],
        ]
    _plot_efficiency_nfe_all(
        frame, metric="sliced_w2", output=output / "figures" / "scrna_quality_vs_nfe.png",
        title="Held-out Tahoe conditions: quality vs compute", methods=paper_curve_methods,
    )
    _plot_runtime_all(
        frame, metric="sliced_w2", output=output / "figures" / "scrna_quality_vs_runtime.png",
        title="Held-out Tahoe conditions: quality vs runtime", methods=paper_curve_methods,
    )
    _plot_scrna_method_comparison(summary, output / "figures" / "scrna_method_comparison.png")

    efficiency_manifest = (
        frame[frame.split == "test"]
        .groupby(["method", "efficiency_kind"], dropna=False)
        .agg(
            efficiency_nfe_min=("efficiency_nfe", "min"),
            efficiency_nfe_max=("efficiency_nfe", "max"),
            generation_seconds_mean=("generation_seconds", "mean"),
        )
        .reset_index()
    )
    efficiency_manifest.to_csv(output / "efficiency_method_metadata.csv", index=False)

    figure_seed = int(config.get("figure_seed", list(map(int, config["seeds"]))[0]))
    figure_nfe = selected_nfe[(figure_seed, "Count Flow Map")]
    figure_flow, _, _, _ = _train_or_load_core(
        "count_flow_map", bundle, dict(config["models"]["count_flow_map"]), figure_seed,
        output, torch_device, True, progress=False,
    )
    generated_figure = generate_conditional_flow_map(
        figure_flow, bundle.test.x0, bundle.test.context, n_steps=figure_nfe,
        tau=float(config.get("tau", 0.98)), device=str(torch_device),
        batch_size=int(config.get("generation_batch_size", 512)),
    )
    representative_ids = _select_representative_condition_ids(
        bundle, int(config.get("representative_conditions", 3))
    )
    _plot_scrna_representative_conditions(
        bundle, generated_figure, projection, condition_ids=representative_ids,
        output=output / "figures" / "scrna_representative_conditions.png", seed=figure_seed,
    )
    _save_json(
        output / "figure_metadata.json",
        {
            "figure_seed": figure_seed,
            "count_flow_map_nfe": figure_nfe,
            "representative_condition_ids": representative_ids,
            "selection_metric": selection_metric,
        },
    )

    available_methods = set(main.get("reported_method", pd.Series(dtype=str)).astype(str).tolist())
    missing_paper_methods = [method for method in paper_methods if method not in available_methods]
    nonfinite_paper_methods: List[str] = []
    for method in paper_methods:
        group = main[main["reported_method"] == method]
        if group.empty:
            continue
        values = pd.to_numeric(group[selection_metric], errors="coerce")
        if not np.isfinite(values).any():
            nonfinite_paper_methods.append(method)
    paper_ready = not missing_paper_methods and not nonfinite_paper_methods
    _save_json(
        output / "paper_readiness.json",
        {
            "paper_ready": paper_ready,
            "missing_paper_methods": missing_paper_methods,
            "nonfinite_primary_metric_methods": nonfinite_paper_methods,
            "builtin_baselines": [SCRNA_EMPIRICAL_BASELINE_LABELS[name] for name in builtin_names],
            "trainable_baselines": [SCRNA_TRAINABLE_BASELINE_LABELS[name] for name in trainable_names],
            "published_baselines": published_names,
            "paper_methods": paper_methods,
            "note": (
                "Every configured paper method produced a finite held-out test result."
                if paper_ready
                else "The run is preliminary because one or more configured paper methods are missing or invalid."
            ),
        },
    )
    _save_json(
        output / "completion.json",
        {
            "status": "complete" if paper_ready else "preliminary",
            "paper_ready": paper_ready,
            "config_sha256": config_sha256,
            "dataset_metadata": bundle.metadata,
            "selection_metric": selection_metric,
            "nfe_values": nfe_values,
            "unit_jump_nfe_values": [int(value) for value in config.get("unit_jump_nfe_values", nfe_values)],
            "builtin_baselines": [SCRNA_EMPIRICAL_BASELINE_LABELS[name] for name in builtin_names],
            "trainable_baselines": [SCRNA_TRAINABLE_BASELINE_LABELS[name] for name in trainable_names],
            "published_baselines": published_names,
            "paper_methods": paper_methods,
            "external_prediction_paths": {name: [str(path) for path in paths] for name, paths in external_paths.items()},
            "missing_paper_methods": missing_paper_methods,
            "nonfinite_primary_metric_methods": nonfinite_paper_methods,
            "note": "scRNA condition-holdout run finished; main-paper readiness is verified from the actual comparison rows, not from placeholder external files.",
        },
    )
    if bool(config.get("strict_paper", False)) and not paper_ready:
        raise RuntimeError(
            "Strict paper run is incomplete: "
            f"missing={missing_paper_methods}, nonfinite={nonfinite_paper_methods}. "
            f"Inspect {output / 'paper_readiness.json'}."
        )
    return output


def _build_forecaster(method: str, bundle: ConditionalPairBundle, cfg: Mapping[str, object]) -> torch.nn.Module:
    history_bins = int(bundle.metadata["history_bins"])
    if method == "poisson_glm":
        return PoissonGLMForecaster(bundle.dim, history_bins)
    if method == "transformer":
        return TransformerPoissonForecaster(
            bundle.dim,
            history_bins,
            model_dim=int(cfg.get("model_dim", 256)),
            n_heads=int(cfg.get("n_heads", 8)),
            n_layers=int(cfg.get("n_layers", 4)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    if method == "mamba":
        return MambaPoissonForecaster(
            bundle.dim,
            history_bins,
            model_dim=int(cfg.get("model_dim", 256)),
            n_layers=int(cfg.get("n_layers", 4)),
            d_state=int(cfg.get("d_state", 16)),
            d_conv=int(cfg.get("d_conv", 4)),
            expand=int(cfg.get("expand", 2)),
        )
    raise ValueError(method)


def _train_or_load_forecaster(
    method: str,
    bundle: ConditionalPairBundle,
    cfg: Mapping[str, object],
    seed: int,
    output: Path,
    device: torch.device,
    resume: bool,
) -> Tuple[torch.nn.Module, Dict[str, list], float]:
    model = _build_forecaster(method, bundle, cfg)
    checkpoint = output / "checkpoints" / str(seed) / f"{method}.pt"
    if resume and checkpoint.exists():
        model, history, metadata = _load_model(checkpoint, model, device)
        return model, history, float(metadata.get("train_seconds", 0.0))
    train_cfg = ForecasterTrainConfig(
        steps=int(cfg.get("steps", 5000)),
        batch_size=int(cfg.get("batch_size", 128)),
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
        ema_decay=float(cfg.get("ema_decay", 0.999)),
        grad_clip=float(cfg.get("grad_clip", 5.0)),
        log_every=max(1, int(cfg.get("log_every", max(int(cfg.get("steps", 5000)) // 10, 1)))),
        seed=seed,
    )
    start = time.perf_counter()
    _, ema, history = train_poisson_forecaster(
        model, bundle.train, train_cfg, device=str(device), verbose=False
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    _save_model(
        checkpoint,
        ema,
        history,
        {"method": method, "train_seconds": seconds, "config": dict(cfg)},
    )
    return ema, history, seconds


def _draw_core_forecasts(
    method: str,
    model: torch.nn.Module,
    split: ConditionalPairSplit,
    *,
    n_steps: int,
    n_draws: int,
    tau: float,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    cases = split.n
    x0 = split.x0[:, None, :].expand(cases, n_draws, split.dim).reshape(-1, split.dim)
    if split.context.ndim == 3:
        context = split.context[:, None, :, :].expand(
            cases, n_draws, split.context.shape[1], split.context.shape[2]
        ).reshape(cases * n_draws, split.context.shape[1], split.context.shape[2])
    else:
        context = split.context[:, None, :].expand(cases, n_draws, split.context.shape[1]).reshape(
            cases * n_draws, split.context.shape[1]
        )
    if method == "Count Flow Map":
        flat = generate_conditional_flow_map(
            model,
            x0,
            context,
            n_steps=n_steps,
            tau=tau,
            device=str(device),
            batch_size=batch_size,
        )
    else:
        flat = generate_conditional_count_fm(
            model,
            x0,
            context,
            n_steps=n_steps,
            tau=tau,
            device=str(device),
            batch_size=batch_size,
        )
    return flat.reshape(cases, n_draws, split.dim)


def run_neural_application(
    config_path: Path,
    *,
    device: Optional[str] = None,
    resume: bool = True,
) -> Path:
    config = load_application_config(config_path, kind="neural")
    root = Path(__file__).resolve().parents[1]
    output_root = root / str(config["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    _save_json(output_root / "config_snapshot.json", config)
    _save_json(output_root / "environment.json", _environment())
    bundles = _load_neural_bundles(config)
    torch_device = resolve_device(device)
    nfe_values = [int(value) for value in config["nfe_values"]]
    n_draws = int(config.get("forecast_draws", 32))
    timing_repeats = int(config.get("timing_repeats", 1))
    all_rows: List[Dict[str, object]] = []

    for session_name, bundle in bundles:
        session_output = output_root / session_name
        session_output.mkdir(parents=True, exist_ok=True)
        for seed in map(int, config["seeds"]):
            models_cfg = dict(config["models"])
            flow, _, flow_train_seconds, _ = _train_or_load_core(
                "count_flow_map",
                bundle,
                dict(models_cfg["count_flow_map"]),
                seed,
                session_output,
                torch_device,
                resume,
            )
            count_fm, _, fm_train_seconds, _ = _train_or_load_core(
                "count_fm",
                bundle,
                dict(models_cfg["count_fm"]),
                seed,
                session_output,
                torch_device,
                resume,
            )
            for split_name, split in (("val", bundle.val), ("test", bundle.test)):
                eval_split = split
                max_cases = int(config.get("max_evaluation_cases", split.n))
                if split.n > max_cases:
                    index = torch.arange(max_cases)
                    eval_split = ConditionalPairSplit(
                        split.x0[index], split.x1[index], split.context[index], split.condition_id[index]
                    )
                for nfe in nfe_values:
                    start = time.perf_counter()
                    samples = _draw_core_forecasts(
                        "Count Flow Map",
                        flow,
                        eval_split,
                        n_steps=nfe,
                        n_draws=n_draws,
                        tau=float(config.get("tau", 0.98)),
                        device=torch_device,
                        batch_size=int(config.get("generation_batch_size", 512)),
                    )
                    seconds = time.perf_counter() - start
                    metrics = evaluate_neural_forecast_samples(samples, eval_split.x1, seed=seed)
                    all_rows.append(
                        {
                            "session": session_name,
                            "seed": seed,
                            "split": split_name,
                            "method": "Count Flow Map",
                            "nfe": nfe,
                            "generation_seconds": seconds,
                            "train_seconds": flow_train_seconds,
                            "parameters": _parameter_count(flow),
                            **metrics,
                        }
                    )
                    start = time.perf_counter()
                    samples_fm = _draw_core_forecasts(
                        "Count-FM + binomial tau-leap",
                        count_fm,
                        eval_split,
                        n_steps=nfe,
                        n_draws=n_draws,
                        tau=float(config.get("tau", 0.98)),
                        device=torch_device,
                        batch_size=int(config.get("generation_batch_size", 512)),
                    )
                    seconds_fm = time.perf_counter() - start
                    metrics_fm = evaluate_neural_forecast_samples(samples_fm, eval_split.x1, seed=seed + 1000)
                    all_rows.append(
                        {
                            "session": session_name,
                            "seed": seed,
                            "split": split_name,
                            "method": "Count-FM + binomial tau-leap",
                            "nfe": nfe,
                            "generation_seconds": seconds_fm,
                            "train_seconds": fm_train_seconds,
                            "parameters": _parameter_count(count_fm),
                            **metrics_fm,
                        }
                    )

            for method in config.get("forecast_baselines", []):
                method = str(method)
                baseline_cfg = dict(models_cfg[method])
                try:
                    baseline, _, train_seconds = _train_or_load_forecaster(
                        method,
                        bundle,
                        baseline_cfg,
                        seed,
                        session_output,
                        torch_device,
                        resume,
                    )
                except RuntimeError as error:
                    if method == "mamba" and bool(config.get("allow_missing_mamba", False)):
                        continue
                    raise
                eval_split = bundle.test
                max_cases = int(config.get("max_evaluation_cases", eval_split.n))
                if eval_split.n > max_cases:
                    index = torch.arange(max_cases)
                    eval_split = ConditionalPairSplit(
                        eval_split.x0[index], eval_split.x1[index], eval_split.context[index], eval_split.condition_id[index]
                    )
                start = time.perf_counter()
                samples = sample_poisson_forecaster(
                    baseline,
                    eval_split.context,
                    n_draws=n_draws,
                    device=str(torch_device),
                    batch_size=int(config.get("generation_batch_size", 512)),
                )
                seconds = time.perf_counter() - start
                metrics = evaluate_neural_forecast_samples(samples, eval_split.x1, seed=seed)
                label = {
                    "poisson_glm": "Autoregressive Poisson GLM",
                    "transformer": "Causal Transformer-Poisson",
                    "mamba": "Mamba-Poisson",
                }[method]
                all_rows.append(
                    {
                        "session": session_name,
                        "seed": seed,
                        "split": "test",
                        "method": label,
                        "nfe": np.nan,
                        "generation_seconds": seconds,
                        "train_seconds": train_seconds,
                        "parameters": _parameter_count(baseline),
                        **metrics,
                    }
                )

    frame = pd.DataFrame(all_rows)
    frame.to_csv(output_root / "metrics.csv", index=False)
    aggregate = (
        frame.groupby(["split", "method", "nfe"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    aggregate.to_csv(output_root / "aggregate_metrics.csv", index=False)

    main_rows = []
    for session_name, _ in bundles:
        for seed in map(int, config["seeds"]):
            val = frame[(frame.session == session_name) & (frame.seed == seed) & (frame.split == "val")]
            for method in ("Count Flow Map", "Count-FM + binomial tau-leap"):
                subset = val[val.method == method].sort_values("nfe")
                sat = select_saturated_nfe(
                    subset.nfe.astype(int).tolist(),
                    subset.energy_score.astype(float).tolist(),
                    relative_tolerance=float(config.get("saturation_tolerance", 0.01)),
                )
                test = frame[(frame.session == session_name) & (frame.seed == seed) & (frame.split == "test") & (frame.method == method)]
                if method == "Count Flow Map":
                    row = test[test.nfe == 1].iloc[0].to_dict()
                    row["reported_method"] = "Count Flow Map (1 step)"
                    main_rows.append(row)
                row = test[test.nfe == sat].iloc[0].to_dict()
                row["reported_method"] = f"{method} ({sat} steps; saturated)"
                main_rows.append(row)
            external = frame[(frame.session == session_name) & (frame.seed == seed) & (frame.split == "test") & (frame.nfe.isna())]
            for _, row in external.iterrows():
                value = row.to_dict()
                value["reported_method"] = value["method"]
                main_rows.append(value)
    main = pd.DataFrame(main_rows)
    main.to_csv(output_root / "main_quality_table_raw.csv", index=False)
    _summarize_main_table(main, output_root / "main_quality_table_summary.csv")

    test_core = aggregate[aggregate.split == "test"]
    _plot_quality_curve(
        test_core,
        metric="energy_score",
        output=output_root / "figures" / "neural_quality_vs_nfe.png",
        title="Neural forecasting: quality vs NFE",
    )
    _plot_runtime_curve(
        test_core,
        metric="energy_score",
        output=output_root / "figures" / "neural_quality_vs_runtime.png",
        title="Neural forecasting: quality vs runtime",
    )
    _save_json(
        output_root / "completion.json",
        {
            "status": "complete",
            "config_sha256": _sha256(Path(config_path)),
            "sessions": [name for name, _ in bundles],
            "note": "Synthetic preset is integration-only; formal preset uses processed SpikeProphecy sessions.",
        },
    )
    return output_root
