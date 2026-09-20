from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .baselines import (
    CountRateModel,
    CountsDiffModel,
    D3PMModel,
    DiscreteFlowMapModel,
    GenericTrainConfig,
    sample_binomial_tau_leap,
    sample_original_unit_jump,
    sample_countsdiff,
    sample_d3pm,
    sample_simplex_flow_map,
    train_count_rate_model,
    train_countsdiff,
    train_d3pm,
    train_discrete_flow_map,
)
from .benchmark_data import benchmark_settings, choose_categorical_support
from .bridge import sample_signed_binomial_bridge
from .data import IndependentCoupling
from .exact_2d import Grid2D, gamma_poisson_target_pmf
from .model import CountFlowMap
from .sampling import generate_count_samples
from .simulation_metrics import evaluate_samples
from .training import TrainConfig, train_count_flow_map
from .utils import count_parameters, resolve_device, set_seed


METHOD_ORDER = [
    "count_flow_map",
    "count_fm_unit_jump",
    "count_fm_binomial_tau_leap",
    "countsdiff",
    "d3pm",
    "discrete_flow_map",
]
METHOD_LABELS = {
    "count_flow_map": "Count Flow Map",
    "count_fm_unit_jump": "Count-FM + unit jump",
    "count_fm_binomial_tau_leap": "Count-FM + binomial tau-leap",
    "countsdiff": "CountsDiff",
    "d3pm": "D3PM",
    "discrete_flow_map": "Discrete Flow Maps",
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _total_parameter_count(module: torch.nn.Module) -> int:
    """Count model parameters independent of frozen/EMA requires_grad flags."""
    return sum(parameter.numel() for parameter in module.parameters())


def _active_rate_parameter_count(model: CountFlowMap) -> int:
    return _total_parameter_count(model.state_encoder) + _total_parameter_count(model.rate_net)




def _config_sha256(config: Dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()

def _save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str))


def _save_checkpoint(path: Path, model: torch.nn.Module, history: Dict[str, list], metadata: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "history": history,
            "metadata": metadata,
        },
        path,
    )


def _load_checkpoint(path: Path, model: torch.nn.Module, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, list], Dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        model.load_state_dict(payload["state_dict"])
        history = payload.get("history", {})
        metadata = payload.get("metadata", {})
    else:
        model.load_state_dict(payload)
        history = {}
        metadata = {}
    model.to(device).eval()
    return model, history, metadata


def _exact_grid(setting: object, target_samples: torch.Tensor, preset: str) -> Grid2D:
    if preset == "smoke":
        return Grid2D((48, 48))
    coordinate_q = torch.quantile(
        target_samples.float(),
        0.99999,
        dim=0,
        interpolation="higher",
    ).long()
    max_counts = tuple(
        max(int(coordinate_q[i].item()) + 8, int(setting.source.high[i]) + 2)
        for i in range(2)
    )
    return Grid2D(max_counts=max_counts)


def _build_flow_map(setting: object, model_config: Dict[str, object]) -> CountFlowMap:
    return CountFlowMap(
        dim=setting.dim,
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        depth=int(model_config.get("depth", 3)),
        n_mixtures=int(model_config.get("n_mixtures", 4)),
        count_scale=float(setting.suggested_count_scale),
    )


def _train_or_load_flow_map(
    root: Path,
    output: Path,
    setting: object,
    seed: int,
    config: Dict[str, object],
    device: torch.device,
    resume: bool,
    progress: bool,
) -> Tuple[CountFlowMap, Dict[str, list], float, Path]:
    model_cfg = dict(config["models"]["count_flow_map"])
    model = _build_flow_map(setting, model_cfg)
    checkpoint = output / "checkpoints" / setting.name / str(seed) / "count_flow_map.pt"
    if resume and checkpoint.exists():
        model, history, metadata = _load_checkpoint(checkpoint, model, device)
        return model, history, float(metadata.get("train_seconds", 0.0)), checkpoint

    train_config = TrainConfig(
        steps=int(model_cfg["steps"]),
        batch_size=int(model_cfg["batch_size"]),
        learning_rate=float(model_cfg["learning_rate"]),
        tau=float(config.get("tau", 0.98)),
        ck_weight=float(model_cfg.get("ck_weight", 1.0)),
        seed=int(seed),
        log_every=max(1, int(model_cfg.get("log_every", max(int(model_cfg["steps"]) // 10, 1)))),
        ck_warmup_steps=int(model_cfg.get("ck_warmup_steps", 250)),
        max_span_start=float(model_cfg.get("max_span_start", 0.1)),
        max_span_end=float(config.get("tau", 0.98)),
        span_warmup_steps=int(model_cfg.get("span_warmup_steps", 2000)),
    )
    coupling = IndependentCoupling(setting.source, setting.target)
    start = time.perf_counter()
    _, ema, history = train_count_flow_map(
        model,
        coupling,
        train_config,
        device=str(device),
        verbose=False,
        progress=progress,
        progress_desc=f"{setting.name} | seed {seed} | Count Flow Map",
    )
    _sync(device)
    train_seconds = time.perf_counter() - start
    _save_checkpoint(
        checkpoint,
        ema,
        history,
        {"train_config": vars(train_config), "train_seconds": train_seconds},
    )
    return ema, history, train_seconds, checkpoint



def _train_or_load_count_fm(
    output: Path,
    setting: object,
    seed: int,
    config: Dict[str, object],
    device: torch.device,
    resume: bool,
    progress: bool,
) -> Tuple[CountRateModel, Dict[str, list], float, Path]:
    model_cfg = dict(config["models"]["count_fm"])
    model = CountRateModel(
        dim=setting.dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        depth=int(model_cfg.get("depth", 4)),
        count_scale=float(setting.suggested_count_scale),
    )
    checkpoint = output / "checkpoints" / setting.name / str(seed) / "count_fm.pt"
    if resume and checkpoint.exists():
        model, history, metadata = _load_checkpoint(checkpoint, model, device)
        return model, history, float(metadata.get("train_seconds", 0.0)), checkpoint
    generic = _generic_config(model_cfg, seed)
    coupling = IndependentCoupling(setting.source, setting.target)
    start = time.perf_counter()
    _, ema, history = train_count_rate_model(
        model,
        coupling,
        generic,
        tau=float(config.get("tau", 0.98)),
        device=str(device),
        verbose=False,
        progress=progress,
        progress_desc=f"{setting.name} | seed {seed} | Count-FM",
    )
    _sync(device)
    train_seconds = time.perf_counter() - start
    _save_checkpoint(
        checkpoint,
        ema,
        history,
        {"method": "count_fm", "train_config": vars(generic), "train_seconds": train_seconds},
    )
    return ema, history, train_seconds, checkpoint

def _generic_config(model_cfg: Dict[str, object], seed: int) -> GenericTrainConfig:
    return GenericTrainConfig(
        steps=int(model_cfg["steps"]),
        batch_size=int(model_cfg["batch_size"]),
        learning_rate=float(model_cfg["learning_rate"]),
        weight_decay=float(model_cfg.get("weight_decay", 1e-5)),
        ema_decay=float(model_cfg.get("ema_decay", 0.999)),
        grad_clip=float(model_cfg.get("grad_clip", 5.0)),
        log_every=max(1, int(model_cfg.get("log_every", max(int(model_cfg["steps"]) // 10, 1)))),
        seed=int(seed),
    )


def _train_or_load_external(
    method: str,
    output: Path,
    setting: object,
    seed: int,
    config: Dict[str, object],
    device: torch.device,
    c_max: int,
    resume: bool,
    progress: bool,
) -> Tuple[torch.nn.Module, Dict[str, list], float, Path]:
    model_cfg = dict(config["models"][method])
    n_categories = int(c_max) + 2
    checkpoint = output / "checkpoints" / setting.name / str(seed) / f"{method}.pt"

    if method == "countsdiff":
        model = CountsDiffModel(
            setting.dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            depth=int(model_cfg.get("depth", 3)),
            count_scale=float(setting.suggested_count_scale),
        )
    elif method == "d3pm":
        model = D3PMModel(
            setting.dim,
            n_categories,
            diffusion_steps=int(model_cfg.get("diffusion_steps", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            depth=int(model_cfg.get("depth", 4)),
            beta_start=float(model_cfg.get("beta_start", 1e-4)),
            beta_end=float(model_cfg.get("beta_end", 0.02)),
            transition_bands=model_cfg.get("transition_bands", None),
            rescale_betas=bool(model_cfg.get("rescale_betas", True)),
        )
    elif method == "discrete_flow_map":
        model = DiscreteFlowMapModel(
            setting.dim,
            n_categories,
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            depth=int(model_cfg.get("depth", 4)),
            prior=str(model_cfg.get("prior", "gaussian")),
        )
    else:
        raise ValueError(f"Unsupported external method: {method}")

    if resume and checkpoint.exists():
        model, history, metadata = _load_checkpoint(checkpoint, model, device)
        return model, history, float(metadata.get("train_seconds", 0.0)), checkpoint

    generic = _generic_config(model_cfg, seed)
    start = time.perf_counter()
    if method == "countsdiff":
        _, ema, history = train_countsdiff(
            model, setting.target, generic, device=str(device), verbose=False,
            progress=progress, progress_desc=f"{setting.name} | seed {seed} | CountsDiff"
        )
    elif method == "d3pm":
        _, ema, history = train_d3pm(
            model,
            setting.target,
            generic,
            c_max=c_max,
            device=str(device),
            auxiliary_weight=float(model_cfg.get("auxiliary_weight", 0.1)),
            verbose=False,
            progress=progress,
            progress_desc=f"{setting.name} | seed {seed} | D3PM",
        )
    else:
        _, ema, history = train_discrete_flow_map(
            model,
            setting.target,
            generic,
            c_max=c_max,
            device=str(device),
            consistency_weight=float(model_cfg.get("consistency_weight", 1.0)),
            diagonal_steps=int(model_cfg.get("diagonal_steps", max(1, int(0.8 * generic.steps)))),
            distillation_steps=int(model_cfg.get("distillation_steps", max(1, generic.steps - int(0.8 * generic.steps)))),
            adaptive_r=float(model_cfg.get("adaptive_r", 0.5)),
            adaptive_c=float(model_cfg.get("adaptive_c", 0.01)),
            gradient_surgery=bool(model_cfg.get("gradient_surgery", True)),
            verbose=False,
            progress=progress,
            progress_desc=f"{setting.name} | seed {seed} | Discrete Flow Maps",
        )
    _sync(device)
    train_seconds = time.perf_counter() - start
    _save_checkpoint(
        checkpoint,
        ema,
        history,
        {
            "method": method,
            "train_config": vars(generic),
            "model_config": model_cfg,
            "c_max": c_max,
            "train_seconds": train_seconds,
        },
    )
    return ema, history, train_seconds, checkpoint


def _generate_method(
    method: str,
    model: torch.nn.Module,
    setting: object,
    n_samples: int,
    nfe: int,
    config: Dict[str, object],
    c_max: int,
    device: torch.device,
) -> torch.Tensor:
    if method == "count_flow_map":
        samples, _ = generate_count_samples(
            model,
            setting.source,
            n_samples,
            tau=float(config.get("tau", 0.98)),
            n_steps=int(nfe),
            device=device,
        )
        return samples
    if method == "count_fm_unit_jump":
        return sample_original_unit_jump(
            model,
            setting.source,
            n_samples,
            int(nfe),
            tau=float(config.get("tau", 0.98)),
            device=str(device),
        ).cpu()
    if method == "count_fm_binomial_tau_leap":
        return sample_binomial_tau_leap(
            model,
            setting.source,
            n_samples,
            int(nfe),
            tau=float(config.get("tau", 0.98)),
            device=str(device),
            death_rule="linear",
        ).cpu()
    if method == "countsdiff":
        eta = float(config["models"]["countsdiff"].get("eta_rescale", 0.0))
        return sample_countsdiff(
            model, n_samples, int(nfe), device=str(device), eta_rescale=eta
        ).cpu()
    if method == "d3pm":
        return sample_d3pm(
            model, n_samples, int(nfe), c_max=c_max, device=str(device)
        ).cpu()
    if method == "discrete_flow_map":
        return sample_simplex_flow_map(
            model, n_samples, int(nfe), c_max=c_max, device=str(device)
        ).cpu()
    raise ValueError(method)


def _time_generation(
    function: Callable[[], torch.Tensor],
    device: torch.device,
    repeats: int,
) -> Tuple[torch.Tensor, float, float]:
    # One warm-up to exclude first-call setup from the paper timing.
    _ = function()
    _sync(device)
    times = []
    result = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(max(1, int(repeats))):
        start = time.perf_counter()
        result = function()
        _sync(device)
        times.append(time.perf_counter() - start)
    peak_memory = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
        if device.type == "cuda"
        else float("nan")
    )
    assert result is not None
    return result, float(np.median(times)), peak_memory



@torch.no_grad()
def _evaluate_intermediate_2d(
    output: Path,
    setting: object,
    seed: int,
    flow_model: CountFlowMap,
    count_fm_model: CountRateModel,
    config: Dict[str, object],
    device: torch.device,
) -> List[Dict[str, object]]:
    """Evaluate count-valued intermediate marginals against the known bridge."""
    times = [float(v) for v in config.get("intermediate_times", [])]
    if not times:
        return []
    n_samples = int(config.get("intermediate_samples", 5000))
    full_count_fm_steps = int(config.get("intermediate_count_fm_steps", 256))
    tau = float(config.get("tau", 0.98))
    metric_cfg = dict(config.get("metric_settings", {}))
    rows: List[Dict[str, object]] = []
    sample_dir = output / "intermediate" / setting.name / str(seed)
    sample_dir.mkdir(parents=True, exist_ok=True)

    for time_value in times:
        if not 0.0 < time_value <= tau:
            raise ValueError("intermediate_times must lie in (0,tau].")
        set_seed(seed * 1000 + int(round(time_value * 1000)))
        x0 = setting.source.sample(n_samples, device=device)
        x1 = setting.target.sample(n_samples, device=device)
        time_tensor = torch.full((n_samples,), time_value, device=device)
        reference = sample_signed_binomial_bridge(x0, x1, time_tensor).cpu()

        zeros = torch.zeros(n_samples, device=device)
        flow_model.eval()
        flow_samples = flow_model.sample(x0, zeros, time_tensor).cpu()

        local_steps = max(1, int(round(full_count_fm_steps * time_value / tau)))
        count_fm_samples = sample_binomial_tau_leap(
            count_fm_model,
            setting.source,
            n_samples,
            local_steps,
            tau=time_value,
            device=str(device),
            death_rule="linear",
        ).cpu()

        tag = f"t{time_value:.2f}".replace(".", "p")
        torch.save(reference[: min(5000, n_samples)], sample_dir / f"reference_{tag}.pt")
        torch.save(flow_samples[: min(5000, n_samples)], sample_dir / f"count_flow_map_{tag}.pt")
        torch.save(count_fm_samples[: min(5000, n_samples)], sample_dir / f"count_fm_{tag}.pt")

        for method, generated, nfe in [
            ("count_flow_map", flow_samples, 1),
            ("count_fm_binomial_tau_leap", count_fm_samples, local_steps),
        ]:
            metric = evaluate_samples(
                generated,
                reference,
                seed=seed + int(round(time_value * 10000)),
                mmd_max_points=int(metric_cfg.get("mmd_max_points", 5000)),
                w2_max_points=int(metric_cfg.get("w2_max_points", 1500)),
                w2_repeats=int(metric_cfg.get("w2_repeats", 3)),
                sliced_w2_projections=int(metric_cfg.get("sliced_w2_projections", 128)),
                sliced_w2_max_points=int(metric_cfg.get("sliced_w2_max_points", 10000)),
            )
            rows.append({
                "setting": setting.name,
                "seed": seed,
                "time": time_value,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "nfe": nfe,
                "generated_samples": n_samples,
                **metric,
            })
    return rows

def run_simulation_suite(
    root: Union[str, Path],
    config: Dict[str, object],
    *,
    output_dir: Union[str, Path],
    device: Optional[str] = None,
    resume: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    _ = root  # kept for notebook-call compatibility; all settings are passed explicitly.
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_hash = _config_sha256(config)
    snapshot_path = output / "config_snapshot.json"
    completion_path = output / "completion.json"
    metrics_path = output / "metrics.csv"

    if resume and snapshot_path.exists():
        previous = json.loads(snapshot_path.read_text())
        if _config_sha256(previous) != config_hash:
            raise ValueError(
                f"Existing run at {output} uses a different configuration. "
                "Use resume=False or a new output directory."
            )

    if resume and completion_path.exists() and metrics_path.exists():
        completion = json.loads(completion_path.read_text())
        if completion.get("status") == "complete" and completion.get("config_sha256") == config_hash:
            if progress:
                print(f"cached complete run: {output}")
            return pd.read_csv(metrics_path)

    torch_device = resolve_device(device)
    settings = benchmark_settings(str(config["preset"]))
    _save_json(snapshot_path, config)
    _save_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(torch_device),
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
        },
    )

    all_rows: List[Dict[str, object]] = []
    intermediate_rows: List[Dict[str, object]] = []
    support_records: Dict[str, object] = {}
    method_set = list(config["methods"])
    for setting_name in config["settings"]:
        setting = settings[str(setting_name)]
        pilot = int(config.get("categorical_pilot_samples", 100000))
        c_max, pilot_overflow = choose_categorical_support(
            setting.target,
            tail_probability=float(config.get("categorical_tail_probability", 1e-4)),
            pilot_samples=pilot,
            seed=9081,
        )
        support_records[setting.name] = {
            "c_max": c_max,
            "n_categories_with_overflow": c_max + 2,
            "pilot_overflow": pilot_overflow,
            "pilot_samples": pilot,
        }

        for seed in config["seeds"]:
            seed = int(seed)
            set_seed(seed)
            flow_model, flow_history, flow_train_seconds, flow_checkpoint = _train_or_load_flow_map(
                root, output, setting, seed, config, torch_device, resume, progress
            )
            count_fm_model, count_fm_history, count_fm_train_seconds, count_fm_checkpoint = _train_or_load_count_fm(
                output, setting, seed, config, torch_device, resume, progress
            )
            models: Dict[str, torch.nn.Module] = {
                "count_flow_map": flow_model,
                "count_fm_unit_jump": count_fm_model,
                "count_fm_binomial_tau_leap": count_fm_model,
            }
            histories: Dict[str, Dict[str, list]] = {
                "count_flow_map": flow_history,
                "count_fm_unit_jump": count_fm_history,
                "count_fm_binomial_tau_leap": count_fm_history,
            }
            training_seconds: Dict[str, float] = {
                "count_flow_map": flow_train_seconds,
                "count_fm_unit_jump": count_fm_train_seconds,
                "count_fm_binomial_tau_leap": count_fm_train_seconds,
            }
            checkpoints: Dict[str, Path] = {
                "count_flow_map": flow_checkpoint,
                "count_fm_unit_jump": count_fm_checkpoint,
                "count_fm_binomial_tau_leap": count_fm_checkpoint,
            }
            for method in method_set:
                if method in models:
                    continue
                trained, history, seconds, checkpoint = _train_or_load_external(
                    method, output, setting, seed, config, torch_device, c_max, resume, progress
                )
                models[method] = trained
                histories[method] = history
                training_seconds[method] = seconds
                checkpoints[method] = checkpoint

            target_n = int(
                config.get("exact_2d_target_samples", config.get("target_samples", 50000))
                if setting.exact_2d
                else config.get("target_samples", 50000)
            )
            set_seed(seed + 1111)
            target_samples = setting.target.sample(target_n).cpu()
            grid = _exact_grid(setting, target_samples, str(config["preset"])) if setting.exact_2d else None
            target_pmf = gamma_poisson_target_pmf(setting.target, grid) if setting.exact_2d else None

            eval_bar = tqdm(
                total=len(method_set) * len(config["nfe_values"]),
                desc=f"{setting.name} | seed {seed} | evaluation",
                leave=True,
                dynamic_ncols=True,
                disable=not progress,
            )
            for method in method_set:
                model = models[method]
                if method in {"count_fm_unit_jump", "count_fm_binomial_tau_leap"}:
                    parameters = _total_parameter_count(count_fm_model)
                else:
                    parameters = _total_parameter_count(model)
                for nfe in config["nfe_values"]:
                    nfe = int(nfe)
                    if method == "d3pm" and nfe > model.diffusion_steps:
                        continue
                    generated_n = int(
                        config.get("exact_2d_generated_samples", config.get("generated_samples", 50000))
                        if setting.exact_2d
                        else config.get("generated_samples", 50000)
                    )
                    evaluation_seed = seed * 100000 + nfe * 101 + METHOD_ORDER.index(method)

                    def generate() -> torch.Tensor:
                        set_seed(evaluation_seed)
                        return _generate_method(
                            method,
                            model,
                            setting,
                            generated_n,
                            nfe,
                            config,
                            c_max,
                            torch_device,
                        )

                    generated, runtime, peak_memory = _time_generation(
                        generate,
                        torch_device,
                        int(config.get("timing_repeats", 3)),
                    )
                    metric_cfg = dict(config.get("metric_settings", {}))
                    metric = evaluate_samples(
                        generated,
                        target_samples,
                        exact_grid=grid,
                        exact_target_pmf=target_pmf,
                        seed=evaluation_seed,
                        mmd_max_points=int(metric_cfg.get("mmd_max_points", 5000)),
                        w2_max_points=int(metric_cfg.get("w2_max_points", 1000)),
                        w2_repeats=int(metric_cfg.get("w2_repeats", 3)),
                        sliced_w2_projections=int(metric_cfg.get("sliced_w2_projections", 128)),
                        sliced_w2_max_points=int(metric_cfg.get("sliced_w2_max_points", 10000)),
                    )
                    overflow_fraction = (
                        float((generated == c_max + 1).any(dim=1).float().mean().item())
                        if method in {"d3pm", "discrete_flow_map"}
                        else 0.0
                    )
                    row: Dict[str, object] = {
                        "setting": setting.name,
                        "seed": seed,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "nfe": nfe,
                        "runtime_seconds": runtime,
                        "samples_per_second": generated_n / max(runtime, 1e-12),
                        "peak_memory_mb": peak_memory,
                        "parameters": parameters,
                        "train_seconds": training_seconds[method],
                        "generated_samples": generated_n,
                        "target_samples": target_n,
                        "c_max": c_max,
                        "categorical_overflow_fraction": overflow_fraction,
                        "checkpoint": str(checkpoints[method].relative_to(output)),
                    }
                    row.update(metric)
                    all_rows.append(row)

                    if nfe in {1, 4, max(int(v) for v in config["nfe_values"])}:
                        sample_dir = output / "samples" / setting.name / str(seed)
                        sample_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            generated[: min(5000, generated.shape[0])],
                            sample_dir / f"{method}_nfe{nfe}.pt",
                        )
                    if progress:
                        quality = metric.get("w2", metric.get("sliced_w2", float("nan")))
                        eval_bar.set_postfix(method=METHOD_LABELS[method], nfe=nfe, q=f"{quality:.3f}")
                        eval_bar.update(1)
                    else:
                        print(
                            f"[{setting.name} seed={seed}] {METHOD_LABELS[method]} "
                            f"NFE={nfe}: "
                            + (f"TV={metric['tv']:.4f} " if "tv" in metric else "")
                            + (f"W2={metric['w2']:.4f} " if "w2" in metric else f"SW2={metric['sliced_w2']:.4f} ")
                            + f"MMD2={metric['mmd2_rbf']:.5f} time={runtime:.3f}s"
                        )
            eval_bar.close()
            if setting.exact_2d:
                intermediate_rows.extend(
                    _evaluate_intermediate_2d(
                        output, setting, seed, flow_model, count_fm_model, config, torch_device
                    )
                )

    frame = pd.DataFrame(all_rows)
    frame.to_csv(output / "metrics.csv", index=False)
    numeric = [
        column
        for column in frame.columns
        if column not in {
            "setting", "seed", "method", "method_label", "checkpoint", "nfe"
        }
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    aggregate = frame.groupby(["setting", "method", "method_label", "nfe"])[numeric].agg(["mean", "std"])
    aggregate.columns = ["__".join(column) for column in aggregate.columns]
    aggregate.reset_index().to_csv(output / "aggregate_metrics.csv", index=False)
    if intermediate_rows:
        pd.DataFrame(intermediate_rows).to_csv(output / "intermediate_metrics.csv", index=False)
    _save_json(output / "categorical_support.json", support_records)
    _save_json(
        output / "completion.json",
        {
            "rows": len(frame),
            "settings": list(config["settings"]),
            "seeds": list(config["seeds"]),
            "methods": method_set,
            "status": "complete",
            "config_sha256": config_hash,
        },
    )
    return frame
