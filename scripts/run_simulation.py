from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from countflow.simulation_runner import run_simulation_suite

SEEDS = [42, 123, 2026, 7, 31415]
NFE_VALUES = [1, 2, 4, 16, 64, 128, 256]
TRAIN_STEPS = 10_000
BATCH_SIZE = 512
HIDDEN_DIM = 256
DEPTH = 4
TAU = 0.98

CK_WARMUP_STEPS = max(1, int(0.02 * TRAIN_STEPS))
SPAN_WARMUP_STEPS = max(1, int(0.10 * TRAIN_STEPS))
DFM_DIAGONAL_STEPS = max(1, int(0.80 * TRAIN_STEPS))
DFM_DISTILLATION_STEPS = TRAIN_STEPS - DFM_DIAGONAL_STEPS
LOG_EVERY = max(1, TRAIN_STEPS // 50)

METHODS = [
    "count_flow_map",
    "count_fm_unit_jump",
    "count_fm_binomial_tau_leap",
    "countsdiff",
    "d3pm",
    "discrete_flow_map",
]

COMMON_MODELS = {
    "count_flow_map": {
        "steps": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": 1e-3,
        "hidden_dim": HIDDEN_DIM,
        "depth": DEPTH,
        "n_mixtures": 8,
        "ck_weight": 1.0,
        "ck_warmup_steps": CK_WARMUP_STEPS,
        "span_warmup_steps": SPAN_WARMUP_STEPS,
        "log_every": LOG_EVERY,
    },
    "count_fm": {
        "steps": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": 1e-3,
        "hidden_dim": HIDDEN_DIM,
        "depth": DEPTH,
        "weight_decay": 0.0,
        "ema_decay": 0.999,
        "log_every": LOG_EVERY,
    },
    "countsdiff": {
        "steps": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": 1e-3,
        "hidden_dim": HIDDEN_DIM,
        "depth": DEPTH,
        "weight_decay": 0.0,
        "ema_decay": 0.999,
        "eta_rescale": 0.0,
        "log_every": LOG_EVERY,
    },
    "d3pm": {
        "steps": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": 1e-3,
        "hidden_dim": HIDDEN_DIM,
        "depth": DEPTH,
        "weight_decay": 0.0,
        "ema_decay": 0.999,
        "diffusion_steps": 256,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "rescale_betas": True,
        "transition_bands": None,
        "auxiliary_weight": 0.001,
        "log_every": LOG_EVERY,
    },
    "discrete_flow_map": {
        "steps": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": 1e-3,
        "hidden_dim": HIDDEN_DIM,
        "depth": DEPTH,
        "weight_decay": 0.0,
        "ema_decay": 0.999,
        "consistency_weight": 1.0,
        "prior": "gaussian",
        "diagonal_steps": DFM_DIAGONAL_STEPS,
        "distillation_steps": DFM_DISTILLATION_STEPS,
        "adaptive_r": 0.5,
        "adaptive_c": 0.01,
        "gradient_surgery": True,
        "log_every": LOG_EVERY,
    },
}

METRICS = {
    "mmd_max_points": 5000,
    "w2_max_points": 1500,
    "w2_repeats": 3,
    "sliced_w2_projections": 128,
    "sliced_w2_max_points": 10000,
}


def make_config(setting_name: str) -> dict:
    return {
        "preset": "paper",
        "settings": [setting_name],
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "nfe_values": list(NFE_VALUES),
        "tau": TAU,
        "generated_samples": 5_000,
        "target_samples": 5_000,
        "exact_2d_generated_samples": 30_000,
        "exact_2d_target_samples": 30_000,
        "categorical_pilot_samples": 200_000,
        "categorical_tail_probability": 1e-4,
        "timing_repeats": 3,
        "intermediate_times": [0.25, 0.50, 0.75, 0.98] if setting_name == "exact_2d" else [],
        "intermediate_samples": 5_000,
        "intermediate_count_fm_steps": 256,
        "models": {name: dict(values) for name, values in COMMON_MODELS.items()},
        "metric_settings": dict(METRICS),
    }


PARTS = {
    "1": ("exact_2d", ROOT / "outputs" / "simulation" / "part1_exact_2d"),
    "2": ("scale_32_low", ROOT / "outputs" / "simulation" / "part2_d32_low"),
    "3": ("scale_32_high", ROOT / "outputs" / "simulation" / "part3_d32_high"),
    "4": ("scale_128_high", ROOT / "outputs" / "simulation" / "part4_d128_high"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal Count Flow Map simulations.")
    parser.add_argument("--part", choices=["1", "2", "3", "4", "all"], default="all")
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu; default auto-detect")
    parser.add_argument("--fresh", action="store_true", help="ignore existing checkpoints/results")
    args = parser.parse_args()

    selected = list(PARTS) if args.part == "all" else [args.part]
    for part in selected:
        setting, output = PARTS[part]
        print(f"\nPart {part}: {setting}")
        frame = run_simulation_suite(
            ROOT,
            make_config(setting),
            output_dir=output,
            device=args.device,
            resume=not args.fresh,
            progress=True,
        )
        print(f"saved: {output / 'metrics.csv'} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
