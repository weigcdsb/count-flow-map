from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib
import importlib.machinery
import json
import math
import os
import random
import shutil
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.runtime_env import maybe_reexec_with_conda_libstdcpp as _maybe_reexec_with_conda_libstdcpp

if __name__ == "__main__":
    _maybe_reexec_with_conda_libstdcpp()

import numpy as np
import pandas as pd


SCRNA_PAPER_PIPELINE_VERSION = "scrna-response-baselines-v3-audited"

SEEDS = [42, 123, 2026]
NFE_VALUES = [1, 4, 16, 64, 128, 256]
UNIT_JUMP_NFE_VALUES = [16, 64, 128, 256, 512]
TAU = 0.98
TRAIN_STEPS = 50_000
BASELINE_TRAIN_STEPS = TRAIN_STEPS
BATCH_SIZE = 64
HIDDEN_DIM = 512
DEPTH = 4
LEARNING_RATE = 2e-4
CK_WARMUP_STEPS = max(1, int(0.04 * TRAIN_STEPS))
SPAN_WARMUP_STEPS = max(1, int(0.30 * TRAIN_STEPS))
LOG_EVERY = max(1, TRAIN_STEPS // 100)

# Formal Tahoe condition-generalization panel. Each biological task still has
# exactly two endpoints: DMSO from cell line l -> treated cells for (l, drug, dose).
PANEL_CELL_LINES = 5
PANEL_DRUGS = 8
PANEL_CANDIDATE_CELL_LINES = 12
PANEL_CANDIDATE_DRUGS = 12
MIN_PAPER_CELL_LINES = 4
MIN_PAPER_DRUGS = 6
PANEL_DOSES_PER_DRUG = 3
MIN_DOSES_FOR_HOLDOUT_PAIR = 2
DEFAULT_MAX_CELLS_PER_CONDITION = 600
DEFAULT_CONTROL_CELLS_PER_PLATE = 1_200
VAL_CONDITION_FRACTION = 0.15
TEST_CONDITION_FRACTION = 0.20

TAHOE_DATA = ROOT / "data" / "tahoe_panel" / "tahoe_condition_holdout.npz"
TAHOE_DIR = TAHOE_DATA.parent
PARQUET_CACHE = TAHOE_DIR / "parquet_cache"
SELECTED_PARQUET = PARQUET_CACHE / "selected_expression_condition_holdout.parquet"
SELECTED_MANIFEST = PARQUET_CACHE / "selected_expression_condition_holdout.manifest.json"
VALIDATED_PARQUET = PARQUET_CACHE / "validated_expression_condition_holdout.parquet"
METADATA_CACHE = PARQUET_CACHE / "metadata"
HF_DATASET = "tahoebio/Tahoe-100M"
HF_REPO_PREFIX = f"datasets/{HF_DATASET}"
N_EXPRESSION_SHARDS = 3388
CACHE_VERSION = "tahoe-native-parquet-v9-balanced-condition-holdout"
PREP_VERSION = "tahoe-native-parquet-v9-balanced-condition-holdout"

SCRNA_CONFIG = {
    "preset": "paper_condition_holdout",
    "pipeline_version": SCRNA_PAPER_PIPELINE_VERSION,
    "output_dir": "outputs/scrna_condition_holdout_paper",
    "seeds": list(SEEDS),
    "nfe_values": list(NFE_VALUES),
    "unit_jump_nfe_values": list(UNIT_JUMP_NFE_VALUES),
    "tau": TAU,
    "timing_repeats": 1,
    "generation_batch_size": 256,
    "selection_metric": "sliced_w2",
    "evaluation_seed": 314159,
    "data": {"mode": "prepared_npz", "path": "data/tahoe_panel/tahoe_condition_holdout.npz"},
    "pca": {"n_components": 50, "max_cells": 50_000, "seed": 42},

    # Main paper comparison: methods that are valid for held-out drug-dose response.
    # scGen/scVIDR/CPA are faithful algorithmic reproductions from their public
    # implementations, but run inside this repository to avoid fragile legacy envs.
    "published_baselines": [
        "scgen",
        "scvidr",
        "cpa",
        "nbvae",
        "linear_dose",
        "sinkhorn_ot",
    ],

    # Simple baselines are useful sanity checks but are supplementary only.
    "builtin_baselines": [
        "pair_mean_poisson",
        "nearest_dose_empirical",
        "dose_interpolated_empirical",
    ],
    "trainable_baselines": [],

    "models": {
        "count_flow_map": {
            "steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "hidden_dim": HIDDEN_DIM,
            "context_hidden_dim": 256,
            "context_output_dim": 256,
            "depth": DEPTH,
            "n_mixtures": 4,
            "ck_weight": 1.0,
            "ck_warmup_steps": CK_WARMUP_STEPS,
            "span_warmup_steps": SPAN_WARMUP_STEPS,
            "death_chunk_size": 64,
            "ema_decay": 0.9995,
            "log_every": LOG_EVERY,
        },
        "count_fm": {
            "steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "hidden_dim": HIDDEN_DIM,
            "context_hidden_dim": 256,
            "context_output_dim": 256,
            "depth": DEPTH,
            "ema_decay": 0.9995,
            "log_every": LOG_EVERY,
        },
        # Shared scGen/scVIDR VAE: public architecture/training defaults.
        # scGen is evaluated as nearest-dose latent arithmetic; scVIDR uses
        # its public continuous log-dose scaling.
        "scgen_vae": {
            "steps": 30_000,  # fallback only; epochs below determines the formal run
            "epochs": 100,
            "batch_size": BATCH_SIZE,
            "learning_rate": 1e-3,
            "weight_decay": 1e-6,
            "adam_eps": 1e-2,
            "hidden_dim": 800,
            "latent_dim": 100,
            "depth": 2,
            "dropout": 0.2,
            "kl_weight": 5e-5,
            "scvidr_ridge": 0.0,
            "log_every": 500,
        },
        # CPA public defaults from the released reference implementation.
        "cpa": {
            "steps": 50_000,
            "batch_size": 256,
            "learning_rate": 3e-4,
            "weight_decay": 4e-7,
            "latent_dim": 128,
            "autoencoder_width": 128,
            "autoencoder_depth": 3,
            "adversary_width": 64,
            "adversary_depth": 2,
            "doser_width": 128,
            "doser_depth": 2,
            "autoencoder_lr": 3e-4,
            "adversary_lr": 3e-4,
            "doser_lr": 4e-3,
            "autoencoder_wd": 4e-7,
            "adversary_wd": 4e-7,
            "doser_wd": 1e-7,
            "reg_adversary": 60.0,
            "penalty_adversary": 60.0,
            "adversary_steps": 3,
            "log_every": 500,
        },
        # Count-native latent baseline; deliberately not mislabeled as official scVI.
        "nbvae": {
            "steps": 50_000,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": 1e-5,
            "hidden_dim": 512,
            "context_hidden_dim": 256,
            "context_output_dim": 256,
            "latent_dim": 64,
            "depth": 3,
            "kl_warmup_steps": 10_000,
            "log_every": 500,
        },
        "linear_dose": {"ridge": 1e-4},
        "sinkhorn_ot": {
            "epsilon_scale": 0.1,
            "iterations": 100,
            "pca_dim": 30,
            "max_cells": 384,
            "knn": 8,
            "seed": 42,
        },
    },

    # Optional extra prediction bundles can still be added later, but the formal
    # comparison no longer depends on third-party environments.
    "required_external_predictions": [],
    "external_predictions": {},
    "require_external_predictions": False,

    "paper_methods": [
        "Count Flow Map (1 NFE)",
        "Count Flow Map (validation-selected)",
        "Count-FM + unit jump (validation-selected)",
        "Count-FM + binomial tau-leap (validation-selected)",
        "scGen (nearest-dose)",
        "scVIDR",
        "CPA",
        "Conditional NB-VAE",
        "Sinkhorn OT (dose interpolation)",
        "Linear dose-response",
    ],
    "paper_curve_methods": [
        "Count Flow Map",
        "Count-FM + unit jump",
        "Count-FM + binomial tau-leap",
        "scGen (nearest-dose)",
        "scVIDR",
        "CPA",
        "Conditional NB-VAE",
        "Sinkhorn OT (dose interpolation)",
        "Linear dose-response",
    ],
    "figure_seed": 42,
    "representative_conditions": 3,
}



def _say(message: str) -> None:
    print(message, flush=True)


def _load_countflow_submodule(name: str):
    """Import countflow.<name> without executing countflow/__init__.py."""
    package_dir = ROOT / "countflow"
    if not package_dir.is_dir():
        raise FileNotFoundError(f"Missing countflow package directory: {package_dir}")
    package = sys.modules.get("countflow")
    if package is None:
        package = types.ModuleType("countflow")
        package.__file__ = str(package_dir / "__init__.py")
        package.__package__ = "countflow"
        package.__path__ = [str(package_dir)]
        package.__spec__ = importlib.machinery.ModuleSpec("countflow", loader=None, is_package=True)
        package.__spec__.submodule_search_locations = [str(package_dir)]
        sys.modules["countflow"] = package
    return importlib.import_module(f"countflow.{name}")


def _import_parquet_stack():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError(
            "Tahoe native-Parquet preparation needs the existing pyarrow installation. "
            "Do not install Hugging Face 'datasets'. Verify with: python -c \"import pyarrow; print(pyarrow.__version__)\""
        ) from exc
    try:
        from huggingface_hub import HfFileSystem
    except Exception as exc:
        raise RuntimeError(
            "Tahoe preparation needs huggingface_hub.HfFileSystem. Install only the lightweight Hub client, "
            "not 'datasets': python -m pip install 'huggingface_hub>=0.20'"
        ) from exc
    return pa, pq, HfFileSystem


def _make_hf_fs(HfFileSystem):
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    return HfFileSystem(token=token if token else False, block_size=256 * 1024)


def _remote_metadata_path(name: str) -> str:
    return f"{HF_REPO_PREFIX}/metadata/{name}.parquet"


def _remote_shard_path(index: int) -> str:
    return f"{HF_REPO_PREFIX}/data/train-{index:05d}-of-{N_EXPRESSION_SHARDS:05d}.parquet"


def _copy_remote_file(fs, remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    info = fs.info(remote)
    size = int(info.get("size") or 0)
    _say(f"[metadata] downloading {remote.rsplit('/', 1)[-1]} ({size / 1024:.1f} KiB)")
    copied = 0
    with fs.open(remote, "rb") as src, tmp.open("wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            copied += len(chunk)
    tmp.replace(local)
    _say(f"[metadata] saved {local} ({copied / 1024:.1f} KiB)")


def _load_metadata(fs, pq, name: str) -> pd.DataFrame:
    local = METADATA_CACHE / f"{name}.parquet"
    if not local.exists():
        _copy_remote_file(fs, _remote_metadata_path(name), local)
    frame = pq.read_table(str(local)).to_pandas()
    _say(f"[metadata] {name}: {len(frame):,} rows")
    return frame


def _build_panel(
    sample_meta: pd.DataFrame,
    cell_meta: pd.DataFrame,
    *,
    n_cell_lines: int = PANEL_CANDIDATE_CELL_LINES,
    n_drugs: int = PANEL_CANDIDATE_DRUGS,
    max_doses_per_drug: int = PANEL_DOSES_PER_DRUG,
) -> dict[str, Any]:
    """Construct dose-specific candidate conditions before expression-row validation.

    The sample metadata do not establish that every cell line is present for every
    treatment sample, so this intentionally over-proposes a modest panel and the
    local Parquet validation step removes impossible combinations.
    """
    preferred = ["A549", "MCF7", "HT29", "HCT116", "PC3"]
    cell_lookup = (
        cell_meta[["cell_name", "Cell_ID_Cellosaur"]]
        .dropna()
        .drop_duplicates()
        .set_index("cell_name")["Cell_ID_Cellosaur"]
        .astype(str)
        .to_dict()
    )
    cell_lines = [cell_lookup[name] for name in preferred if name in cell_lookup][: int(n_cell_lines)]
    if len(cell_lines) < int(n_cell_lines):
        fallback = sorted(cell_meta["Cell_ID_Cellosaur"].dropna().astype(str).unique())
        cell_lines += [x for x in fallback if x not in cell_lines][: int(n_cell_lines) - len(cell_lines)]

    meta = sample_meta.copy()
    meta["sample"] = meta["sample"].astype(str)
    meta["drug"] = meta["drug"].astype(str)
    meta["dose"] = [
        _parse_dose_local(str(value)) for value in meta.get("drugname_drugconc", pd.Series([""] * len(meta)))
    ]
    treated = meta.loc[(meta["drug"] != "DMSO_TF") & (meta["dose"] > 0)].copy()
    if treated.empty:
        raise RuntimeError("No positive-dose Tahoe treatment samples were found in sample_metadata.")

    dose_groups = (
        treated.groupby(["drug", "dose"], as_index=False)["sample"]
        .agg(lambda x: sorted(set(map(str, x))))
        .rename(columns={"sample": "samples"})
    )
    summary = (
        dose_groups.groupby("drug")
        .agg(n_doses=("dose", "nunique"), n_samples=("samples", lambda x: sum(len(v) for v in x)))
        .reset_index()
    )
    summary = summary.loc[summary["n_doses"] >= int(MIN_DOSES_FOR_HOLDOUT_PAIR)]
    summary = summary.sort_values(["n_doses", "n_samples", "drug"], ascending=[False, False, True])
    selected_drugs = summary.head(int(n_drugs))["drug"].astype(str).tolist()
    if not selected_drugs:
        raise RuntimeError("Could not find Tahoe drugs with repeated dose levels for condition holdout.")

    conditions: list[dict[str, Any]] = []
    for drug in selected_drugs:
        rows = dose_groups.loc[dose_groups["drug"].astype(str) == drug].sort_values("dose")
        if len(rows) > int(max_doses_per_drug):
            positions = np.linspace(0, len(rows) - 1, int(max_doses_per_drug))
            keep = sorted(set(int(round(x)) for x in positions))
            rows = rows.iloc[keep]
        for cell_line in cell_lines:
            for row in rows.itertuples(index=False):
                conditions.append({
                    "cell_line_id": str(cell_line),
                    "drug": str(drug),
                    "dose": float(row.dose),
                    "samples": list(map(str, row.samples)),
                })

    if not conditions:
        raise RuntimeError("Could not construct a Tahoe dose-specific candidate panel.")
    return {
        "conditions": conditions,
        "protocol": "dose-specific cell-line x drug x dose candidates; validated against expression rows",
        "cell_lines": cell_lines,
        "drugs": selected_drugs,
        "max_doses_per_drug": int(max_doses_per_drug),
    }


def _selection_plan(
    panel: dict[str, Any],
    sample_meta: pd.DataFrame,
    *,
    max_cells_per_condition: int,
    control_cells_per_plate: int,
) -> dict[str, Any]:
    sample_meta = sample_meta.copy()
    sample_meta["sample"] = sample_meta["sample"].astype(str)
    sample_meta["plate"] = sample_meta["plate"].astype(str)
    sample_meta["drug"] = sample_meta["drug"].astype(str)
    sample_info = sample_meta.drop_duplicates("sample").set_index("sample")

    cell_lines = sorted({str(c["cell_line_id"]) for c in panel["conditions"]})
    treated_quota: dict[tuple[str, str], int] = {}
    treated_samples: set[str] = set()
    needed_plates: set[str] = set()
    for condition in panel["conditions"]:
        samples = [str(x) for x in condition["samples"]]
        base, remainder = divmod(int(max_cells_per_condition), max(len(samples), 1))
        for j, sample in enumerate(samples):
            if sample not in sample_info.index:
                raise RuntimeError(f"Selected treatment sample {sample!r} is absent from sample_metadata")
            quota = base + (1 if j < remainder else 0)
            treated_quota[(sample, str(condition["cell_line_id"]))] = max(
                quota, treated_quota.get((sample, str(condition["cell_line_id"])), 0)
            )
            treated_samples.add(sample)
            needed_plates.add(str(sample_info.loc[sample, "plate"]))

    control_rows = sample_meta.loc[
        (sample_meta["drug"] == "DMSO_TF") & sample_meta["plate"].isin(needed_plates)
    ]
    control_samples = sorted(control_rows["sample"].dropna().astype(str).unique())
    if not control_samples:
        raise RuntimeError(
            f"No DMSO_TF control samples were found for treatment plates {sorted(needed_plates)}"
        )
    control_quota = {
        (plate, cell): int(control_cells_per_plate)
        for plate in sorted(needed_plates)
        for cell in cell_lines
    }
    sample_to_plate = sample_info["plate"].astype(str).to_dict()
    selected_samples = sorted(treated_samples | set(control_samples))

    return {
        "cell_lines": cell_lines,
        "treated_samples": sorted(treated_samples),
        "control_samples": control_samples,
        "selected_samples": selected_samples,
        "needed_plates": sorted(needed_plates),
        "treated_quota": treated_quota,
        "control_quota": control_quota,
        "sample_to_plate": sample_to_plate,
    }


def _normalize_stat(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _physical_column_index(pf, name: str) -> int | None:
    schema = pf.schema
    for i in range(pf.metadata.num_columns):
        col = schema.column(i)
        path = getattr(col, "path", None)
        if path is None:
            path = getattr(col, "name", "")
        text = str(path)
        if text == name or text.split(".", 1)[0] == name:
            return i
    return None


def _stats_might_contain(stats, values: list[str]) -> bool:
    if stats is None or not getattr(stats, "has_min_max", False):
        return True
    try:
        lo = _normalize_stat(stats.min)
        hi = _normalize_stat(stats.max)
    except Exception:
        return True
    return any(lo <= value <= hi for value in values)


def _inspect_shard(fs, pq, remote: str, selected_samples: list[str], cell_lines: list[str]) -> list[int]:
    with fs.open(remote, "rb") as handle:
        pf = pq.ParquetFile(handle)
        sample_index = _physical_column_index(pf, "sample")
        cell_index = _physical_column_index(pf, "cell_line_id")
        if sample_index is None or cell_index is None:
            raise RuntimeError(f"Tahoe Parquet schema missing sample/cell_line_id in {remote}")
        candidate: list[int] = []
        for rg in range(pf.metadata.num_row_groups):
            meta = pf.metadata.row_group(rg)
            sample_stats = meta.column(sample_index).statistics
            cell_stats = meta.column(cell_index).statistics
            if not _stats_might_contain(sample_stats, selected_samples):
                continue
            if not _stats_might_contain(cell_stats, cell_lines):
                continue
            candidate.append(rg)
        return candidate


def _scan_candidate_shards(
    pq,
    HfFileSystem,
    *,
    token: str | None,
    selected_samples: list[str],
    cell_lines: list[str],
    workers: int,
    heartbeat_seconds: int,
) -> list[tuple[str, list[int]]]:
    remotes = [_remote_shard_path(i) for i in range(N_EXPRESSION_SHARDS)]
    _say(f"[parquet-index] inspecting {len(remotes):,} native Tahoe Parquet footers")
    _say("[parquet-index] only Parquet metadata/statistics are read in this phase; expression arrays are not downloaded")
    start = time.time()
    thread_state = threading.local()

    def worker(remote: str):
        fs = getattr(thread_state, "fs", None)
        if fs is None:
            thread_state.fs = HfFileSystem(
                token=token if token else False,
                block_size=256 * 1024,
            )
            fs = thread_state.fs
        last: Exception | None = None
        for attempt in range(1, 4):
            try:
                return remote, _inspect_shard(fs, pq, remote, selected_samples, cell_lines)
            except Exception as exc:
                last = exc
                if attempt < 3:
                    time.sleep(float(attempt))
        raise RuntimeError(f"Could not inspect Tahoe shard {remote} after 3 attempts: {last}") from last

    candidates: list[tuple[str, list[int]]] = []
    done = 0
    last_print = time.time()
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(worker, remote): remote for remote in remotes}
        for future in as_completed(futures):
            remote, row_groups = future.result()
            done += 1
            if row_groups:
                candidates.append((remote, row_groups))
                _say(
                    f"[parquet-index] candidate {remote.rsplit('/',1)[-1]} | "
                    f"row_groups={row_groups} | candidates={len(candidates)}"
                )
            now = time.time()
            if done % 50 == 0 or now - last_print >= max(5, heartbeat_seconds):
                _say(
                    f"[parquet-index] progress {done:,}/{len(remotes):,} shards | "
                    f"candidate files={len(candidates)} | elapsed={now-start:.1f}s"
                )
                last_print = now
    candidates.sort(key=lambda x: x[0])
    _say(
        f"[parquet-index] complete: {len(candidates)} candidate files out of {len(remotes)} "
        f"({time.time()-start:.1f}s)"
    )
    return candidates


def _quota_done(plan: dict[str, Any], treated_count, control_count) -> bool:
    return all(treated_count.get(key, 0) >= value for key, value in plan["treated_quota"].items()) and all(
        control_count.get(key, 0) >= value for key, value in plan["control_quota"].items()
    )


def _extract_from_candidates(
    pa,
    pq,
    HfFileSystem,
    candidates: list[tuple[str, list[int]]],
    plan: dict[str, Any],
    *,
    token: str | None,
    output_path: Path,
    heartbeat_seconds: int,
) -> dict[str, Any]:
    selected_samples = set(plan["selected_samples"])
    selected_cells = set(plan["cell_lines"])
    treated_quota = plan["treated_quota"]
    control_quota = plan["control_quota"]
    control_samples = set(plan["control_samples"])
    sample_to_plate = plan["sample_to_plate"]
    treated_count: dict[tuple[str, str], int] = {}
    control_count: dict[tuple[str, str], int] = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    writer = None
    kept_total = 0
    start = time.time()
    status = {"file": 0, "total": len(candidates), "kept": 0, "name": "starting"}
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(max(5, int(heartbeat_seconds))):
            _say(
                f"[parquet heartbeat] candidate {status['file']}/{status['total']} | "
                f"{status['name']} | kept={status['kept']:,} | elapsed={time.time()-start:.1f}s"
            )

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    fs = HfFileSystem(token=token if token else False, block_size=2 * 1024 * 1024)
    try:
        for file_index, (remote, row_groups) in enumerate(candidates, 1):
            status.update(file=file_index, name=remote.rsplit("/", 1)[-1], kept=kept_total)
            with fs.open(remote, "rb") as handle:
                pf = pq.ParquetFile(handle)
                schema_names = list(pf.schema_arrow.names)
                for required in ("sample", "cell_line_id"):
                    if required not in schema_names:
                        raise RuntimeError(f"Required column {required!r} missing from {remote}")
                meta_columns = ["sample", "cell_line_id"]
                for row_group in row_groups:
                    meta_table = pf.read_row_group(row_group, columns=meta_columns, use_threads=False)
                    meta_samples = meta_table["sample"].to_pylist()
                    meta_cells = meta_table["cell_line_id"].to_pylist()
                    could_need = False
                    for sample, cell in zip(meta_samples, meta_cells):
                        sample = str(sample)
                        cell = str(cell)
                        if sample not in selected_samples or cell not in selected_cells:
                            continue
                        treated_key = (sample, cell)
                        if treated_key in treated_quota and treated_count.get(treated_key, 0) < treated_quota[treated_key]:
                            could_need = True
                            break
                        if sample in control_samples:
                            control_key = (str(sample_to_plate[sample]), cell)
                            if control_key in control_quota and control_count.get(control_key, 0) < control_quota[control_key]:
                                could_need = True
                                break
                    if not could_need:
                        continue

                    for batch in pf.iter_batches(
                        batch_size=256,
                        row_groups=[row_group],
                        columns=schema_names,
                        use_threads=False,
                    ):
                        batch_samples = batch.column(batch.schema.get_field_index("sample")).to_pylist()
                        batch_cells = batch.column(batch.schema.get_field_index("cell_line_id")).to_pylist()
                        keep: list[bool] = []
                        for sample, cell in zip(batch_samples, batch_cells):
                            sample = str(sample)
                            cell = str(cell)
                            take = False
                            if sample in selected_samples and cell in selected_cells:
                                treated_key = (sample, cell)
                                if treated_key in treated_quota and treated_count.get(treated_key, 0) < treated_quota[treated_key]:
                                    treated_count[treated_key] = treated_count.get(treated_key, 0) + 1
                                    take = True
                                elif sample in control_samples:
                                    control_key = (str(sample_to_plate[sample]), cell)
                                    if control_key in control_quota and control_count.get(control_key, 0) < control_quota[control_key]:
                                        control_count[control_key] = control_count.get(control_key, 0) + 1
                                        take = True
                            keep.append(take)
                        if not any(keep):
                            continue
                        table = pa.Table.from_batches([batch]).filter(pa.array(keep))
                        if writer is None:
                            writer = pq.ParquetWriter(str(tmp), table.schema, compression="zstd")
                        writer.write_table(table)
                        kept_total += table.num_rows
                        status["kept"] = kept_total
                    if _quota_done(plan, treated_count, control_count):
                        _say("[parquet] all requested treatment/control quotas are satisfied; stopping remote scan early")
                        break
            if file_index % 5 == 0 or row_groups:
                _say(
                    f"[parquet] processed candidate {file_index}/{len(candidates)} | "
                    f"kept={kept_total:,} | elapsed={time.time()-start:.1f}s"
                )
            if _quota_done(plan, treated_count, control_count):
                break
    finally:
        stop.set()
        thread.join(timeout=2)
        if writer is not None:
            writer.close()

    if writer is None or not tmp.exists():
        raise RuntimeError("No Tahoe expression rows matched the selected panel.")
    tmp.replace(output_path)

    missing_treated = {
        f"{sample}|{cell}": [treated_count.get((sample, cell), 0), quota]
        for (sample, cell), quota in treated_quota.items()
        if treated_count.get((sample, cell), 0) < quota
    }
    missing_control = {
        f"{plate}|{cell}": [control_count.get((plate, cell), 0), quota]
        for (plate, cell), quota in control_quota.items()
        if control_count.get((plate, cell), 0) < quota
    }
    _say(f"[parquet] local selected cache: {output_path} | rows={kept_total:,}")
    if missing_treated:
        _say(f"[parquet] note: {len(missing_treated)} treatment sample/cell quotas had fewer available rows")
    if missing_control:
        _say(f"[parquet] note: {len(missing_control)} plate/control quotas had fewer available rows")
    return {
        "rows": kept_total,
        "treated_counts": {f"{a}|{b}": v for (a, b), v in treated_count.items()},
        "control_counts": {f"{a}|{b}": v for (a, b), v in control_count.items()},
        "missing_treated": missing_treated,
        "missing_control": missing_control,
    }


def _cache_signature(panel: dict[str, Any], plan: dict[str, Any], max_cells: int, controls: int) -> dict[str, Any]:
    payload = {
        "version": CACHE_VERSION,
        "panel": panel,
        "selected_samples": plan["selected_samples"],
        "cell_lines": plan["cell_lines"],
        "needed_plates": plan["needed_plates"],
        "max_cells_per_condition": int(max_cells),
        "control_cells_per_plate": int(controls),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _prepare_selected_parquet(
    fs,
    pa,
    pq,
    HfFileSystem,
    panel: dict[str, Any],
    sample_meta: pd.DataFrame,
    *,
    max_cells_per_condition: int,
    control_cells_per_plate: int,
    footer_workers: int,
    heartbeat_seconds: int,
    force_extract: bool,
) -> Path:
    plan = _selection_plan(
        panel,
        sample_meta,
        max_cells_per_condition=max_cells_per_condition,
        control_cells_per_plate=control_cells_per_plate,
    )
    _say("\n=== Selected Tahoe panel ===")
    for i, condition in enumerate(panel["conditions"], 1):
        _say(
            f"  condition {i:02d}/{len(panel['conditions'])}: cell={condition['cell_line_id']} | "
            f"drug={condition['drug']} | samples={condition['samples']}"
        )
    _say(f"[selection] unique treatment samples: {len(plan['treated_samples'])}")
    _say(f"[selection] DMSO control samples on matched plates: {len(plan['control_samples'])}")
    _say(f"[selection] cell lines: {plan['cell_lines']}")
    _say(f"[selection] matched plates: {plan['needed_plates']}")

    signature = _cache_signature(panel, plan, max_cells_per_condition, control_cells_per_plate)
    if not force_extract and SELECTED_PARQUET.exists() and SELECTED_MANIFEST.exists():
        try:
            old = json.loads(SELECTED_MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            old = {}
        if old.get("sha256") == signature["sha256"]:
            _say(f"[cache] reusing selected native-Parquet cache: {SELECTED_PARQUET}")
            return SELECTED_PARQUET

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    candidates = _scan_candidate_shards(
        pq,
        HfFileSystem,
        token=token,
        selected_samples=plan["selected_samples"],
        cell_lines=plan["cell_lines"],
        workers=footer_workers,
        heartbeat_seconds=heartbeat_seconds,
    )
    if not candidates:
        raise RuntimeError("Parquet metadata scan found no Tahoe shards compatible with the selected samples/cell lines.")
    extraction = _extract_from_candidates(
        pa,
        pq,
        HfFileSystem,
        candidates,
        plan,
        token=token,
        output_path=SELECTED_PARQUET,
        heartbeat_seconds=heartbeat_seconds,
    )
    signature["candidate_files"] = len(candidates)
    signature["extraction"] = extraction
    SELECTED_MANIFEST.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    return SELECTED_PARQUET


def _load_extraction_counts(selected_path: Path, pq) -> tuple[dict[str, int], dict[str, int]]:
    """Load observed treatment/control counts from the completed local extraction."""
    treated: dict[str, int] = {}
    controls: dict[str, int] = {}
    if SELECTED_MANIFEST.exists():
        try:
            manifest = json.loads(SELECTED_MANIFEST.read_text(encoding="utf-8"))
            extraction = manifest.get("extraction", {})
            treated = {str(k): int(v) for k, v in extraction.get("treated_counts", {}).items()}
            controls = {str(k): int(v) for k, v in extraction.get("control_counts", {}).items()}
        except Exception as exc:
            _say(f"[panel] warning: could not read extraction counts from manifest: {exc}")
    if treated or controls:
        return treated, controls

    _say("[panel] extraction counts absent from manifest; deriving them from the LOCAL selected cache")
    meta_path = METADATA_CACHE / "sample_metadata.parquet"
    if not meta_path.exists():
        raise RuntimeError("Cannot derive panel availability: sample_metadata cache is missing")
    sample_meta = pq.read_table(str(meta_path)).to_pandas()
    sample_meta = sample_meta.copy()
    sample_meta["sample"] = sample_meta["sample"].astype(str)
    sample_meta["plate"] = sample_meta["plate"].astype(str)
    sample_meta["drug"] = sample_meta["drug"].astype(str)
    sample_info = sample_meta.drop_duplicates("sample").set_index("sample")
    pf = pq.ParquetFile(str(selected_path))
    for batch in pf.iter_batches(batch_size=4096, columns=["sample", "cell_line_id"], use_threads=False):
        samples = batch.column(batch.schema.get_field_index("sample")).to_pylist()
        cells = batch.column(batch.schema.get_field_index("cell_line_id")).to_pylist()
        for sample, cell in zip(samples, cells):
            sample = str(sample)
            cell = str(cell)
            if sample not in sample_info.index:
                continue
            drug = str(sample_info.loc[sample, "drug"])
            plate = str(sample_info.loc[sample, "plate"])
            if drug == "DMSO_TF":
                key = f"{plate}|{cell}"
                controls[key] = controls.get(key, 0) + 1
            else:
                key = f"{sample}|{cell}"
                treated[key] = treated.get(key, 0) + 1
    return treated, controls


def _prune_panel_to_observed_rows(
    panel: dict[str, Any],
    sample_meta: pd.DataFrame,
    selected_path: Path,
    pq,
    *,
    min_treated_cells_per_condition: int,
) -> dict[str, Any]:
    """Remove impossible dose-specific conditions using LOCAL observed counts."""
    treated_counts, control_counts = _load_extraction_counts(selected_path, pq)
    sample_meta = sample_meta.copy()
    sample_meta["sample"] = sample_meta["sample"].astype(str)
    sample_meta["plate"] = sample_meta["plate"].astype(str)
    sample_meta["drug"] = sample_meta["drug"].astype(str)
    sample_meta["dose"] = [
        _parse_dose_local(str(value)) for value in sample_meta.get("drugname_drugconc", pd.Series([""] * len(sample_meta)))
    ]
    sample_info = sample_meta.drop_duplicates("sample").set_index("sample")

    kept_conditions: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    _say("\n=== Validating dose-specific panel against LOCAL extracted rows ===")

    for condition in panel["conditions"]:
        cell = str(condition["cell_line_id"])
        drug = str(condition["drug"])
        dose = float(condition["dose"])
        requested_samples = [str(x) for x in condition["samples"]]
        kept_samples: list[str] = []
        treated_total = 0
        details: list[str] = []
        for sample in requested_samples:
            if sample not in sample_info.index:
                details.append(f"{sample}:metadata-missing")
                continue
            observed_drug = str(sample_info.loc[sample, "drug"])
            observed_dose = float(sample_info.loc[sample, "dose"])
            plate = str(sample_info.loc[sample, "plate"])
            n_treated = int(treated_counts.get(f"{sample}|{cell}", 0))
            n_control = int(control_counts.get(f"{plate}|{cell}", 0))
            if observed_drug != drug:
                details.append(f"{sample}:drug-mismatch({observed_drug})")
            elif not np.isclose(observed_dose, dose, rtol=1e-6, atol=1e-12):
                details.append(f"{sample}:dose-mismatch({observed_dose:g})")
            elif n_treated <= 0:
                details.append(f"{sample}:treated=0")
            elif n_control <= 0:
                details.append(f"{sample}:treated={n_treated},control=0@{plate}")
            else:
                kept_samples.append(sample)
                treated_total += n_treated
                details.append(f"{sample}:treated={n_treated},control={n_control}@{plate}")

        keep_condition = bool(kept_samples) and treated_total >= int(min_treated_cells_per_condition)
        reason = "ok" if keep_condition else (
            "no observed treated sample with plate-matched DMSO control"
            if not kept_samples else f"only {treated_total} treated cells (< {int(min_treated_cells_per_condition)})"
        )
        report_rows.append({
            "cell_line_id": cell,
            "drug": drug,
            "dose": dose,
            "requested_samples": ";".join(requested_samples),
            "kept_samples": ";".join(kept_samples),
            "treated_cells": treated_total,
            "status": "keep" if keep_condition else "drop",
            "reason": reason,
            "sample_details": " | ".join(details),
        })
        _say(
            f"[panel] {'KEEP' if keep_condition else 'DROP':4s} | {cell} | {drug:20s} | "
            f"dose={dose:g} | treated={treated_total:4d} | {reason}"
        )
        if keep_condition:
            kept_conditions.append({
                "cell_line_id": cell, "drug": drug, "dose": dose, "samples": kept_samples
            })

    # Only cell-line x drug pairs with >=2 usable doses can contribute a held-out
    # condition while leaving the same pair represented in training. Single-dose
    # pairs remain usable for training but are never assigned to validation/test.
    report_path = TAHOE_DIR / "condition_holdout_panel_availability.csv"
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    _say(f"[panel] availability report: {report_path}")
    _say(f"[panel] retained {len(kept_conditions)}/{len(panel['conditions'])} dose-specific candidates")
    if not kept_conditions:
        raise RuntimeError(f"No usable Tahoe conditions remain; inspect {report_path}")
    return {
        "conditions": kept_conditions,
        "protocol": panel.get("protocol", "dose-specific condition panel"),
    }


def _select_balanced_observed_panel(
    panel: dict[str, Any],
    *,
    target_cell_lines: int = PANEL_CELL_LINES,
    target_drugs: int = PANEL_DRUGS,
) -> dict[str, Any]:
    """Choose a broad repeated-dose panel after observing actual Tahoe rows.

    The old application selected cell lines and drugs independently *before*
    checking expression support.  That silently collapsed the nominal 5-cell-line
    experiment to only two cell lines.  Here selection is performed after the
    local availability check and favors cell-line/drug pairs with >=2 doses.
    """
    conditions = [dict(c) for c in panel.get("conditions", [])]
    pair_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for condition in conditions:
        key = (str(condition["cell_line_id"]), str(condition["drug"]))
        pair_groups.setdefault(key, []).append(condition)
    pair_groups = {
        key: rows for key, rows in pair_groups.items()
        if len({float(row["dose"]) for row in rows}) >= int(MIN_DOSES_FOR_HOLDOUT_PAIR)
    }
    if not pair_groups:
        raise RuntimeError("No observed cell-line/drug pair has repeated doses for condition holdout.")

    cells = sorted({cell for cell, _ in pair_groups})
    drugs = sorted({drug for _, drug in pair_groups})
    selected_cells = cells
    selected_drugs = drugs
    for _ in range(4):
        cell_scores = []
        for cell in cells:
            matching = [rows for (c, d), rows in pair_groups.items() if c == cell and d in selected_drugs]
            cell_scores.append((len(matching), sum(len(rows) for rows in matching), cell))
        selected_cells = [cell for _, _, cell in sorted(cell_scores, reverse=True)[: int(target_cell_lines)]]

        drug_scores = []
        for drug in drugs:
            matching = [rows for (c, d), rows in pair_groups.items() if d == drug and c in selected_cells]
            drug_scores.append((len(matching), sum(len(rows) for rows in matching), drug))
        selected_drugs = [drug for _, _, drug in sorted(drug_scores, reverse=True)[: int(target_drugs)]]

    kept = [
        condition for (cell, drug), rows in pair_groups.items()
        if cell in selected_cells and drug in selected_drugs
        for condition in rows
    ]
    final_cells = sorted({str(c["cell_line_id"]) for c in kept})
    final_drugs = sorted({str(c["drug"]) for c in kept})
    _say("\n=== Balanced observed Tahoe panel ===")
    _say(f"[panel] cell lines ({len(final_cells)}): {final_cells}")
    _say(f"[panel] drugs ({len(final_drugs)}): {final_drugs}")
    _say(f"[panel] repeated-dose conditions: {len(kept)}")
    if len(final_cells) < int(MIN_PAPER_CELL_LINES) or len(final_drugs) < int(MIN_PAPER_DRUGS):
        raise RuntimeError(
            "Observed Tahoe panel is still too narrow for the intended paper application: "
            f"cell_lines={len(final_cells)} (need >= {MIN_PAPER_CELL_LINES}), "
            f"drugs={len(final_drugs)} (need >= {MIN_PAPER_DRUGS}). "
            "Increase PANEL_CANDIDATE_CELL_LINES/PANEL_CANDIDATE_DRUGS rather than silently running a 2-cell-line experiment."
        )
    return {
        "conditions": kept,
        "protocol": (
            "balanced observed repeated-dose panel selected after local expression-row validation; "
            f"target up to {int(target_cell_lines)} cell lines x {int(target_drugs)} drugs"
        ),
        "cell_lines": final_cells,
        "drugs": final_drugs,
    }


def _write_validated_local_cache(
    selected_path: Path,
    panel: dict[str, Any],
    sample_meta: pd.DataFrame,
    pa,
    pq,
) -> Path:
    """Create a small LOCAL Parquet containing only retained treatment rows + needed controls."""
    output = VALIDATED_PARQUET
    tmp = output.with_suffix(output.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    meta = sample_meta.copy()
    meta["sample"] = meta["sample"].astype(str)
    meta["plate"] = meta["plate"].astype(str)
    meta["drug"] = meta["drug"].astype(str)
    info = meta.drop_duplicates("sample").set_index("sample")

    treated_pairs: set[tuple[str, str]] = set()
    control_pairs: set[tuple[str, str]] = set()
    for condition in panel["conditions"]:
        cell = str(condition["cell_line_id"])
        for sample in condition["samples"]:
            sample = str(sample)
            treated_pairs.add((sample, cell))
            if sample in info.index:
                control_pairs.add((str(info.loc[sample, "plate"]), cell))

    pf = pq.ParquetFile(str(selected_path))
    writer = None
    kept = 0
    try:
        for batch in pf.iter_batches(batch_size=512, use_threads=False):
            samples = batch.column(batch.schema.get_field_index("sample")).to_pylist()
            cells = batch.column(batch.schema.get_field_index("cell_line_id")).to_pylist()
            mask: list[bool] = []
            for sample, cell in zip(samples, cells):
                sample = str(sample)
                cell = str(cell)
                keep = (sample, cell) in treated_pairs
                if not keep and sample in info.index and str(info.loc[sample, "drug"]) == "DMSO_TF":
                    keep = (str(info.loc[sample, "plate"]), cell) in control_pairs
                mask.append(keep)
            if not any(mask):
                continue
            table = pa.Table.from_batches([batch]).filter(pa.array(mask))
            if writer is None:
                writer = pq.ParquetWriter(str(tmp), table.schema, compression="zstd")
            writer.write_table(table)
            kept += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None or not tmp.exists():
        raise RuntimeError("Validated panel produced zero local expression rows")
    tmp.replace(output)
    _say(f"[panel] validated LOCAL expression cache: {output} | rows={kept:,}")
    return output


class LocalParquetDataset:
    """Small HuggingFace-IterableDataset-like wrapper over our selected local Parquet cache."""

    def __init__(self, path: Path, pq, transforms=None):
        self.path = Path(path)
        self.pq = pq
        self.transforms = list(transforms or [])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        pf = self.pq.ParquetFile(str(self.path))
        iterator: Iterable[dict[str, Any]] = (
            row
            for batch in pf.iter_batches(batch_size=256, use_threads=False)
            for row in batch.to_pylist()
        )
        for transform in self.transforms:
            iterator = transform(iterator)
        yield from iterator

    def __len__(self) -> int:
        return int(self.pq.ParquetFile(str(self.path)).metadata.num_rows)

    def take(self, n: int):
        import itertools
        return itertools.islice(iter(self), int(n))

    def shuffle(self, seed: int | None = None, buffer_size: int | None = None):
        size = max(1000, int(buffer_size or 10_000))
        def transform(iterator):
            rng = random.Random(seed)
            buffer: list[dict[str, Any]] = []
            for row in iterator:
                if len(buffer) < size:
                    buffer.append(row)
                    continue
                j = rng.randrange(len(buffer))
                yield buffer[j]
                buffer[j] = row
            rng.shuffle(buffer)
            yield from buffer
        return LocalParquetDataset(self.path, self.pq, self.transforms + [transform])

    def filter(self, function: Callable[..., bool], *args, **kwargs):
        def transform(iterator):
            for row in iterator:
                if function(row):
                    yield row
        return LocalParquetDataset(self.path, self.pq, self.transforms + [transform])

    def map(self, function: Callable[..., dict[str, Any]], *args, **kwargs):
        def transform(iterator):
            for row in iterator:
                yield function(row)
        return LocalParquetDataset(self.path, self.pq, self.transforms + [transform])

    def iter(self, batch_size: int | None = None):
        if batch_size is None:
            yield from self
            return
        batch: list[dict[str, Any]] = []
        for row in self:
            batch.append(row)
            if len(batch) >= batch_size:
                keys = batch[0].keys()
                yield {key: [item.get(key) for item in batch] for key in keys}
                batch = []
        if batch:
            keys = batch[0].keys()
            yield {key: [item.get(key) for item in batch] for key in keys}

    def with_format(self, *args, **kwargs):
        return self


class FrameDataset:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def to_pandas(self) -> pd.DataFrame:
        return self.frame.copy()

    def __iter__(self):
        yield from self.frame.to_dict(orient="records")

    def __len__(self):
        return len(self.frame)

    def take(self, n: int):
        return iter(self.frame.head(int(n)).to_dict(orient="records"))



def _parse_dose_local(value: str) -> float:
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list) and parsed:
            item = parsed[0]
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                return float(item[1])
    except (SyntaxError, ValueError, TypeError):
        pass
    return 0.0


def _cell_split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < 3:
        raise RuntimeError(f"Need at least 3 cells for train/val/test, got {n}.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = max(1, int(round(0.15 * n)))
    n_test = max(1, int(round(0.15 * n)))
    if n_val + n_test >= n:
        n_val = 1
        n_test = 1
    n_train = n - n_val - n_test
    if n_train < 1:
        raise RuntimeError(f"Could not make nonempty train/val/test split for n={n}.")
    return order[:n_train], order[n_train:n_train+n_val], order[n_train+n_val:]


def _replicate_assignment(samples: list[str], seed: int) -> dict[str, str]:
    """Replicate-aware split with at least one replicate in every split.

    The original application_data implementation accidentally maps all 3 replicates
    to train because j/3 is always < 0.7. Here n=3 gives exactly 1/1/1.
    """
    if len(samples) < 3:
        raise ValueError("replicate assignment requires at least 3 samples")
    samples = list(samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(samples)
    n = len(samples)
    n_val = max(1, int(round(0.15 * n)))
    n_test = max(1, int(round(0.15 * n)))
    if n_val + n_test >= n:
        n_val = 1
        n_test = 1
    n_train = n - n_val - n_test
    if n_train < 1:
        raise RuntimeError(f"Could not make replicate split for {n} samples")
    out: dict[str, str] = {}
    for sample in samples[:n_train]:
        out[sample] = "train"
    for sample in samples[n_train:n_train+n_val]:
        out[sample] = "val"
    for sample in samples[n_train+n_val:]:
        out[sample] = "test"
    return out


def _condition_holdout_assignment(
    condition_specs: list[dict[str, Any]],
    *,
    seed: int,
    val_fraction: float = VAL_CONDITION_FRACTION,
    test_fraction: float = TEST_CONDITION_FRACTION,
) -> dict[int, str]:
    """Deterministic complete-condition holdout with train coverage constraints.

    A validation/test condition may be held out only when its cell line, drug, and
    exact cell-line x drug pair all remain represented by at least one *other dose*
    in training. This makes the task unseen (cell line, drug, dose) prediction, not
    unseen-category extrapolation.
    """
    from collections import Counter

    n = len(condition_specs)
    if n < 5:
        raise RuntimeError(f"Need at least 5 observed conditions for condition holdout, got {n}.")
    assignment = {i: "train" for i in range(n)}
    cell_count = Counter(str(c["cell_line_id"]) for c in condition_specs)
    drug_count = Counter(str(c["drug"]) for c in condition_specs)
    pair_count = Counter((str(c["cell_line_id"]), str(c["drug"])) for c in condition_specs)
    eligible = [
        i for i, c in enumerate(condition_specs)
        if pair_count[(str(c["cell_line_id"]), str(c["drug"]))] >= int(MIN_DOSES_FOR_HOLDOUT_PAIR)
    ]
    if len(eligible) < 2:
        raise RuntimeError(
            "Need at least two dose-specific conditions from repeated cell-line x drug pairs "
            "for validation/test condition holdout."
        )

    rng = np.random.default_rng(seed)
    jitter = {i: float(rng.random()) for i in eligible}
    held_cell: dict[str, int] = {}
    held_drug: dict[str, int] = {}
    held_pair: dict[tuple[str, str], int] = {}

    def can_hold(i: int) -> bool:
        c = condition_specs[i]
        cell = str(c["cell_line_id"])
        drug = str(c["drug"])
        pair = (cell, drug)
        return cell_count[cell] > 1 and drug_count[drug] > 1 and pair_count[pair] > 1

    def choose_one() -> int | None:
        candidates = [i for i in eligible if assignment[i] == "train" and can_hold(i)]
        if not candidates:
            return None
        def key(i: int):
            c = condition_specs[i]
            cell = str(c["cell_line_id"]); drug = str(c["drug"]); pair = (cell, drug)
            return (
                held_pair.get(pair, 0),
                held_cell.get(cell, 0) + held_drug.get(drug, 0),
                jitter[i],
            )
        return min(candidates, key=key)

    target_val = max(1, int(round(float(val_fraction) * len(eligible))))
    target_test = max(1, int(round(float(test_fraction) * len(eligible))))
    if target_val + target_test >= len(eligible):
        target_val = 1
        target_test = max(1, len(eligible) - 2)

    # Test first because it is the primary scientific evaluation; validation uses
    # the remaining eligible conditions for NFE/model selection.
    for split, target in (("test", target_test), ("val", target_val)):
        for _ in range(target):
            i = choose_one()
            if i is None:
                break
            c = condition_specs[i]
            cell = str(c["cell_line_id"]); drug = str(c["drug"]); pair = (cell, drug)
            assignment[i] = split
            cell_count[cell] -= 1; drug_count[drug] -= 1; pair_count[pair] -= 1
            held_cell[cell] = held_cell.get(cell, 0) + 1
            held_drug[drug] = held_drug.get(drug, 0) + 1
            held_pair[pair] = held_pair.get(pair, 0) + 1

    if not any(v == "val" for v in assignment.values()) or not any(v == "test" for v in assignment.values()):
        raise RuntimeError("Could not construct nonempty validation/test condition holdouts with train coverage.")

    train_specs = [condition_specs[i] for i, v in assignment.items() if v == "train"]
    train_cells = {str(c["cell_line_id"]) for c in train_specs}
    train_drugs = {str(c["drug"]) for c in train_specs}
    train_pairs = {(str(c["cell_line_id"]), str(c["drug"])) for c in train_specs}
    for i, split in assignment.items():
        if split == "train":
            continue
        c = condition_specs[i]
        cell = str(c["cell_line_id"]); drug = str(c["drug"]); pair = (cell, drug)
        if cell not in train_cells or drug not in train_drugs or pair not in train_pairs:
            raise RuntimeError(f"Holdout coverage failure for condition {i}: {c}")
    return assignment


def _build_tahoe_npz_from_rows(
    rows: Iterable[dict[str, Any]],
    *,
    panel: dict[str, Any],
    sample_meta: pd.DataFrame,
    output_path: Path,
    max_cells_per_condition: int = DEFAULT_MAX_CELLS_PER_CONDITION,
    n_hvg: int = 2_000,
    pilot_cells: int = 50_000,
    seed: int = 42,
) -> Path:
    """Build a two-endpoint Tahoe bundle with *complete condition* holdout.

    Each condition is (cell line, drug, dose). Target cells from validation/test
    conditions are never used in training or HVG selection. DMSO sources are
    matched to the same cell line and plate. The model context explicitly contains
    cell line, drug, and standardized log-dose.
    """
    conditions = [dict(c) for c in panel.get("conditions", [])]
    if not conditions:
        raise ValueError("Validated Tahoe panel contains no conditions.")

    meta = sample_meta.copy()
    meta["sample"] = meta["sample"].astype(str)
    meta["plate"] = meta["plate"].astype(str)
    meta["drug"] = meta["drug"].astype(str)
    meta["dose"] = [
        _parse_dose_local(str(value)) for value in meta.get("drugname_drugconc", pd.Series([""] * len(meta)))
    ]
    sample_info = meta.drop_duplicates("sample").set_index("sample")

    sample_condition: dict[tuple[str, str], int] = {}
    selected_cells: set[str] = set()
    for condition_id, condition in enumerate(conditions):
        cell = str(condition["cell_line_id"]); selected_cells.add(cell)
        for sample in map(str, condition.get("samples", [])):
            key = (sample, cell)
            if key in sample_condition and sample_condition[key] != condition_id:
                raise RuntimeError(f"Treatment sample/cell pair appears in multiple dose conditions: {key}")
            sample_condition[key] = condition_id

    rng = np.random.default_rng(seed)
    target_reservoirs: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(conditions))}
    target_seen: dict[int, int] = {i: 0 for i in range(len(conditions))}
    controls: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for row in rows:
        cell = str(row["cell_line_id"]); sample = str(row["sample"]); drug = str(row["drug"]); plate = str(row["plate"])
        if cell not in selected_cells:
            continue
        genes_raw = list(row["genes"]); expr_raw = list(row["expressions"])
        record = {
            "genes": list(map(int, genes_raw[1:])),
            "expressions": list(map(float, expr_raw[1:])),
            "sample": sample,
            "plate": plate,
        }
        if drug == "DMSO_TF":
            controls.setdefault((cell, plate), []).append(record)
            continue
        condition_id = sample_condition.get((sample, cell))
        if condition_id is None:
            continue
        expected_drug = str(conditions[condition_id]["drug"])
        if drug != expected_drug:
            raise RuntimeError(
                f"Treatment row drug mismatch for sample={sample}, cell_line={cell}: "
                f"row={drug}, condition={expected_drug}"
            )
        if sample in sample_info.index:
            sample_dose = float(sample_info.loc[sample, "dose"])
            expected_dose = float(conditions[condition_id]["dose"])
            if not np.isfinite(sample_dose) or not np.isclose(sample_dose, expected_dose, rtol=1e-6, atol=1e-10):
                raise RuntimeError(
                    f"Treatment row dose mismatch for sample={sample}, cell_line={cell}: "
                    f"metadata={sample_dose}, condition={expected_dose}"
                )
        target_seen[condition_id] += 1
        bucket = target_reservoirs[condition_id]
        if len(bucket) < int(max_cells_per_condition):
            bucket.append(record)
        else:
            j = int(rng.integers(target_seen[condition_id]))
            if j < int(max_cells_per_condition):
                bucket[j] = record

    # Drop any condition that became empty after the final local filter.
    observed_old_ids = [i for i in range(len(conditions)) if target_reservoirs[i]]
    conditions = [conditions[i] for i in observed_old_ids]
    target_reservoirs = {new: target_reservoirs[old] for new, old in enumerate(observed_old_ids)}
    if len(conditions) < 5:
        raise RuntimeError(f"Only {len(conditions)} observed dose conditions remain after local filtering.")

    assignment = _condition_holdout_assignment(conditions, seed=seed)
    cell_lines = sorted({str(c["cell_line_id"]) for c in conditions})
    drugs = sorted({str(c["drug"]) for c in conditions})
    cell_to_idx = {name: i for i, name in enumerate(cell_lines)}
    drug_to_idx = {name: i for i, name in enumerate(drugs)}

    train_doses = np.asarray([
        np.log1p(float(conditions[i]["dose"]))
        for i, split in assignment.items() if split == "train"
    ], dtype=np.float64)
    dose_mean = float(train_doses.mean())
    dose_std = float(train_doses.std())
    if dose_std < 1e-8:
        dose_std = 1.0

    def context_for(condition: dict[str, Any]) -> np.ndarray:
        cell = str(condition["cell_line_id"]); drug = str(condition["drug"]); dose = float(condition["dose"])
        cell_vec = np.zeros(len(cell_lines), dtype=np.float32); cell_vec[cell_to_idx[cell]] = 1.0
        drug_vec = np.zeros(len(drugs), dtype=np.float32); drug_vec[drug_to_idx[drug]] = 1.0
        dose_z = (np.log1p(dose) - dose_mean) / dose_std
        return np.concatenate([cell_vec, drug_vec, np.asarray([dose_z], dtype=np.float32)])

    sparse_pairs: list[tuple[dict[str, Any], dict[str, Any], np.ndarray, int]] = []
    split_rows: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    split_report: list[dict[str, Any]] = []
    condition_names: list[str] = []

    for condition_id, condition in enumerate(conditions):
        cell = str(condition["cell_line_id"]); drug = str(condition["drug"]); dose = float(condition["dose"])
        targets = target_reservoirs[condition_id]
        context = context_for(condition)
        condition_name = f"{cell}:{drug}:dose={dose:g}"
        condition_names.append(condition_name)
        local_indices: list[int] = []
        for j, target in enumerate(targets):
            plate = str(target["plate"])
            matched_controls = controls.get((cell, plate), [])
            if not matched_controls:
                raise RuntimeError(
                    f"No plate-matched DMSO source for retained condition {condition_name} on plate {plate}."
                )
            source = matched_controls[int(rng.integers(len(matched_controls)))]
            sparse_pairs.append((source, target, context, condition_id))
            local_indices.append(len(sparse_pairs) - 1)
        split = assignment[condition_id]
        split_rows[split].extend(local_indices)
        split_report.append({
            "condition_id": condition_id,
            "condition": condition_name,
            "split": split,
            "cell_line_id": cell,
            "drug": drug,
            "dose": dose,
            "n_rows": len(local_indices),
            "samples": ";".join(map(str, condition.get("samples", []))),
        })
        _say(f"[condition split] {split:5s} | {condition_name:55s} | rows={len(local_indices):4d}")

    for name in ("train", "val", "test"):
        if not split_rows[name]:
            raise RuntimeError(f"Global Tahoe {name} split is empty after complete-condition holdout.")

    split_report_path = Path(output_path).parent / "condition_holdout_split_report.csv"
    pd.DataFrame(split_report).to_csv(split_report_path, index=False)
    _say(
        "[condition split] rows train/val/test = "
        f"{len(split_rows['train']):,}/{len(split_rows['val']):,}/{len(split_rows['test']):,}"
    )

    # HVGs use training conditions only. Repeated DMSO record objects are de-duplicated.
    training_records: list[dict[str, Any]] = []
    used_records: set[int] = set()
    for i in split_rows["train"]:
        for record in (sparse_pairs[i][0], sparse_pairs[i][1]):
            key = id(record)
            if key not in used_records:
                used_records.add(key); training_records.append(record)
            if len(training_records) >= int(pilot_cells):
                break
        if len(training_records) >= int(pilot_cells):
            break
    if not training_records:
        raise RuntimeError("No training cells available for HVG selection.")

    gene_sum: dict[int, float] = {}; gene_sq: dict[int, float] = {}
    for record in training_records:
        for gene, value in zip(record["genes"], record["expressions"]):
            gene_sum[gene] = gene_sum.get(gene, 0.0) + value
            gene_sq[gene] = gene_sq.get(gene, 0.0) + value * value
    n_pilot = float(len(training_records))
    dispersion: list[tuple[float, int]] = []
    for gene, total in gene_sum.items():
        mean = total / n_pilot
        variance = max(gene_sq[gene] / n_pilot - mean * mean, 0.0)
        dispersion.append((variance / (mean + 1e-3), gene))
    token_ids = [gene for _, gene in sorted(dispersion, reverse=True)[: int(n_hvg)]]
    if not token_ids:
        raise RuntimeError("HVG selection returned no genes.")
    token_to_col = {token: i for i, token in enumerate(token_ids)}
    _say(f"[HVG] selected {len(token_ids):,} genes from {len(training_records):,} training-only cells")

    def dense(record: dict[str, Any]) -> np.ndarray:
        output = np.zeros(len(token_ids), dtype=np.int32)
        for gene, value in zip(record["genes"], record["expressions"]):
            col = token_to_col.get(int(gene))
            if col is not None:
                output[col] = int(round(value))
        return output

    arrays: dict[str, np.ndarray] = {}
    for name in ("train", "val", "test"):
        idx = split_rows[name]
        arrays[f"x0_{name}"] = np.stack([dense(sparse_pairs[i][0]) for i in idx])
        arrays[f"x1_{name}"] = np.stack([dense(sparse_pairs[i][1]) for i in idx])
        arrays[f"context_{name}"] = np.stack([sparse_pairs[i][2] for i in idx]).astype(np.float32)
        arrays[f"condition_{name}"] = np.asarray([sparse_pairs[i][3] for i in idx], dtype=np.int64)
        arrays[f"pair_group_{name}"] = np.asarray([
            f"{sparse_pairs[i][3]}:{sparse_pairs[i][1]['plate']}" for i in idx
        ])

    condition_metadata = []
    for condition_id, condition in enumerate(conditions):
        condition_metadata.append({
            "condition_id": condition_id,
            "condition_name": condition_names[condition_id],
            "split": assignment[condition_id],
            "cell_line_id": str(condition["cell_line_id"]),
            "drug": str(condition["drug"]),
            "dose": float(condition["dose"]),
            "samples": list(map(str, condition.get("samples", []))),
        })
    fingerprint_payload = {
        "prep_version": PREP_VERSION,
        "gene_token_ids": token_ids,
        "conditions": condition_metadata,
        "context_cell_lines": cell_lines,
        "context_drugs": drugs,
        "dose_mean": dose_mean,
        "dose_std": dose_std,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "dataset": "Tahoe-100M",
        "task": "DMSO-to-treated complete-condition generalization",
        "source_distribution": "DMSO conditional on cell line and matched plate",
        "target_distribution": "treated conditional on cell line, drug, dose",
        "coupling_protocol": "independent empirical DMSO draw per treated cell within matched cell-line/plate; no biological cell pairing is assumed",
        "panel": panel,
        "gene_token_ids": token_ids,
        "condition_names": condition_names,
        "conditions": condition_metadata,
        "cell_lines": cell_lines,
        "drugs": drugs,
        "context_dim": len(cell_lines) + len(drugs) + 1,
        "context_layout": {
            "cell_line_one_hot": [0, len(cell_lines)],
            "drug_one_hot": [len(cell_lines), len(cell_lines) + len(drugs)],
            "standardized_log1p_dose": len(cell_lines) + len(drugs),
        },
        "dose_log1p_train_mean": dose_mean,
        "dose_log1p_train_std": dose_std,
        "split_protocol": (
            "complete (cell_line,drug,dose) holdout; val/test condition targets are absent from training; "
            "each held-out cell line, drug, and cell_line x drug pair remains represented by another dose in train"
        ),
        "split_rows": {name: len(split_rows[name]) for name in ("train", "val", "test")},
        "split_conditions": {
            name: sum(1 for value in assignment.values() if value == name)
            for name in ("train", "val", "test")
        },
        "hvg_protocol": "training conditions only",
        "prep_version": PREP_VERSION,
        "bundle_fingerprint": fingerprint,
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    _say(
        f"[NPZ] wrote {output_path} | dim={len(token_ids)} | "
        f"conditions train/val/test={metadata['split_conditions']['train']}/"
        f"{metadata['split_conditions']['val']}/{metadata['split_conditions']['test']}"
    )
    return output_path


def _build_tahoe_npz_from_local_parquet(
    parquet_path: Path,
    *,
    panel: dict[str, Any],
    sample_meta: pd.DataFrame,
    pq,
    output_path: Path,
    max_cells_per_condition: int,
    n_hvg: int = 2_000,
    pilot_cells: int = 50_000,
    seed: int = 42,
) -> Path:
    pf = pq.ParquetFile(str(parquet_path))
    rows = (
        row
        for batch in pf.iter_batches(batch_size=256, use_threads=False)
        for row in batch.to_pylist()
    )
    return _build_tahoe_npz_from_rows(
        rows,
        panel=panel,
        sample_meta=sample_meta,
        output_path=output_path,
        max_cells_per_condition=max_cells_per_condition,
        n_hvg=n_hvg,
        pilot_cells=pilot_cells,
        seed=seed,
    )


@contextlib.contextmanager
def _fake_datasets_environment(fs, pq, expression_dataset: LocalParquetDataset, metadata_frames: dict[str, pd.DataFrame]):
    cache = {key: value.copy() for key, value in metadata_frames.items()}

    def fake_load_dataset(path, name=None, split=None, streaming=False, **kwargs):
        config = name or "expression_data"
        if str(path).lower() not in {HF_DATASET.lower(), "tahoebio/tahoe-100m"}:
            raise RuntimeError(f"Unexpected dataset requested during Tahoe preparation: {path}")
        if config == "expression_data":
            dataset = expression_dataset
        else:
            if config not in cache:
                cache[config] = _load_metadata(fs, pq, config)
            dataset = FrameDataset(cache[config])
        if split is None:
            return {"train": dataset}
        return dataset

    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = fake_load_dataset
    old = sys.modules.get("datasets")
    sys.modules["datasets"] = fake_module
    try:
        yield fake_load_dataset
    finally:
        if old is None:
            sys.modules.pop("datasets", None)
        else:
            sys.modules["datasets"] = old


def _prepared_bundle_is_current(path: Path) -> bool:
    try:
        with np.load(Path(path), allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
        return str(metadata.get("prep_version", "")) == PREP_VERSION
    except Exception:
        return False


def prepare_tahoe_if_missing(
    *,
    force: bool = False,
    force_extract: bool = False,
    max_cells_per_condition: int = DEFAULT_MAX_CELLS_PER_CONDITION,
    control_cells_per_plate: int = DEFAULT_CONTROL_CELLS_PER_PLATE,
    footer_workers: int = 6,
    heartbeat_seconds: int = 20,
    min_treated_cells_per_condition: int = 100,
) -> Path:
    if TAHOE_DATA.exists() and not force:
        if _prepared_bundle_is_current(TAHOE_DATA):
            _say(f"Tahoe data found and current: {TAHOE_DATA}")
            return TAHOE_DATA
        _say(f"Tahoe data is stale for prep_version={PREP_VERSION}; rebuilding the NPZ/panel automatically.")
        force = True
    if force and TAHOE_DATA.exists():
        TAHOE_DATA.unlink()

    TAHOE_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_CACHE.mkdir(parents=True, exist_ok=True)
    pa, pq, HfFileSystem = _import_parquet_stack()
    fs = _make_hf_fs(HfFileSystem)

    _say(f"Tahoe prep version: {PREP_VERSION}")
    _say(f"Selected-Parquet cache compatibility: {CACHE_VERSION}")
    _say("Backend: native Hugging Face Parquet + HfFileSystem; NO Dataset Viewer /filter API; NO datasets package; NO SciPy")
    _say(f"pyarrow: {getattr(pa, '__version__', 'unknown')}")
    _say(f"Target output: {TAHOE_DATA}")
    _say(f"Parquet cache: {PARQUET_CACHE}")
    _say("The first run must inspect Parquet footers for 3,388 shards; live progress is printed. Only candidate row groups download expression arrays.")

    sample_meta = _load_metadata(fs, pq, "sample_metadata")
    cell_meta = _load_metadata(fs, pq, "cell_line_metadata")
    panel = _build_panel(sample_meta, cell_meta)
    requested_panel_path = TAHOE_DIR / "condition_holdout_panel_requested.json"
    requested_panel_path.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    _say(f"Requested panel written to: {requested_panel_path}")

    selected = _prepare_selected_parquet(
        fs,
        pa,
        pq,
        HfFileSystem,
        panel,
        sample_meta,
        max_cells_per_condition=max_cells_per_condition,
        control_cells_per_plate=control_cells_per_plate,
        footer_workers=footer_workers,
        heartbeat_seconds=heartbeat_seconds,
        force_extract=force_extract,
    )

    panel = _prune_panel_to_observed_rows(
        panel, sample_meta, selected, pq,
        min_treated_cells_per_condition=int(min_treated_cells_per_condition),
    )
    panel = _select_balanced_observed_panel(panel)
    panel_path = TAHOE_DIR / "condition_holdout_panel.json"
    panel_path.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    _say(f"Validated panel written to: {panel_path}")

    validated_selected = _write_validated_local_cache(selected, panel, sample_meta, pa, pq)
    local_row_count = int(pq.ParquetFile(str(validated_selected)).metadata.num_rows)
    _say(f"\n=== Local Tahoe expression cache ready: {local_row_count:,} rows ===")
    _say("Building the condition-holdout NPZ DIRECTLY from the LOCAL cache.")
    _say("Validation/test hold out COMPLETE (cell line, drug, dose) target conditions; HVGs use training conditions only.")
    start = time.time()
    _build_tahoe_npz_from_local_parquet(
        validated_selected,
        panel=panel,
        sample_meta=sample_meta,
        pq=pq,
        output_path=TAHOE_DATA,
        max_cells_per_condition=int(max_cells_per_condition),
        n_hvg=2_000,
        pilot_cells=min(50_000, max(local_row_count, 1)),
        seed=42,
    )
    _say(f"Local Tahoe NPZ formatter finished in {time.time()-start:.1f}s")

    if not TAHOE_DATA.exists():
        raise RuntimeError("Tahoe formatter returned but did not create tahoe_panel.npz")
    _say(f"saved: {TAHOE_DATA} ({TAHOE_DATA.stat().st_size / 1024**2:.1f} MiB)")
    return TAHOE_DATA


def run_training(
    *,
    device: str | None = None,
    resume: bool = True,
    strict_paper: bool = False,
    retrain_baselines: bool = False,
):
    if not TAHOE_DATA.exists():
        raise FileNotFoundError(f"Missing prepared Tahoe data: {TAHOE_DATA}")
    _say(f"[scRNA paper] pipeline={SCRNA_PAPER_PIPELINE_VERSION} | self-contained response baselines | no external CPA/CondOT bundles required")
    _say("Loading Count Flow Map application runner only after Tahoe data are ready ...")
    application_runner = _load_countflow_submodule("application_runner")
    config = json.loads(json.dumps(SCRNA_CONFIG))
    config["strict_paper"] = bool(strict_paper)
    config["retrain_published_baselines"] = bool(retrain_baselines)
    return application_runner.run_scrna_application(
        config, device=device, resume=resume, progress=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tahoe DMSO-to-treated complete-condition generalization.")
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu; default auto-detect")
    parser.add_argument("--fresh", action="store_true", help="ignore existing model checkpoints/results")
    parser.add_argument(
        "--strict-paper", action="store_true",
        help="fail unless every configured main paper comparison produced a valid test result",
    )
    parser.add_argument(
        "--retrain-baselines", action="store_true",
        help="retrain only scGen/scVIDR shared VAE, CPA, and NB-VAE; keep Count Flow Map/Count-FM checkpoints",
    )
    parser.add_argument("--prepare-only", action="store_true", help="prepare Tahoe data and stop")
    parser.add_argument("--reprepare-data", action="store_true", help="rebuild the condition-holdout NPZ; compatible selected-Parquet cache is reused")
    parser.add_argument("--reextract-parquet", action="store_true", help="force a new native-Parquet remote extraction")
    parser.add_argument("--max-cells-per-condition", type=int, default=DEFAULT_MAX_CELLS_PER_CONDITION)
    parser.add_argument("--control-cells-per-plate", type=int, default=DEFAULT_CONTROL_CELLS_PER_PLATE)
    parser.add_argument("--footer-workers", type=int, default=6)
    parser.add_argument("--heartbeat-seconds", type=int, default=20)
    parser.add_argument(
        "--min-treated-cells-per-condition", type=int, default=100,
        help="drop requested cell-line/drug/dose conditions with fewer observed treated cells after local extraction",
    )
    # Backward compatibility with the previous notebook; intentionally ignored.
    parser.add_argument("--viewer-page-size", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    prepare_tahoe_if_missing(
        force=args.reprepare_data,
        force_extract=args.reextract_parquet,
        max_cells_per_condition=args.max_cells_per_condition,
        control_cells_per_plate=args.control_cells_per_plate,
        footer_workers=args.footer_workers,
        heartbeat_seconds=args.heartbeat_seconds,
        min_treated_cells_per_condition=args.min_treated_cells_per_condition,
    )
    if args.prepare_only:
        return
    output = run_training(
        device=args.device, resume=not args.fresh, strict_paper=args.strict_paper,
        retrain_baselines=args.retrain_baselines,
    )
    _say(f"saved: {output}")


if __name__ == "__main__":
    main()
