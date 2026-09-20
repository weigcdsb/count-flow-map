"""Read-only checks and report assembly for the baseline-only scRNA update."""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from countflow.application_data import ConditionalPairSplit
from countflow.application_metrics import PCAProjection
from countflow.scrna_matched_baselines import atomic_json, sha256
from countflow import scrna_response_baselines as base
from scripts.scrna_paper_v2 import (
    PAIRINGS, CORE_ROWS, METRICS, ITERATIVE, CFM, FM_UNIT, FM_TAU,
    summarize_table, write_tables,
)

FOLDERS = {"Non-OT": "non_ot", "OT": "ot"}
FIXED = (base.LINEAR_LABEL, base.SINKHORN_LABEL)


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message + " No count-model work will be performed.")


def read_reference(root, data, config):
    """Validate existing exported evidence without opening any core checkpoint."""
    root = Path(root)
    with np.load(data, allow_pickle=False) as arrays:
        fingerprint = json.loads(str(arrays["metadata_json"].item()))["bundle_fingerprint"]
    paper = read_json(root / "paper_v2_status.json")
    require(paper.get("status") == "completed", "The reference paper export is incomplete.")
    statuses, metrics, raw, condition_frames, projections, indices, paths = {}, [], [], [], {}, {}, []
    architecture = config["architecture"]
    seeds = config["seeds"]
    for pairing, folder in FOLDERS.items():
        directory = root / folder
        status = read_json(directory / "paper_run_status.json")
        statuses[pairing] = status
        require(status.get("status") == "completed" and status["evaluation_fingerprint"] == fingerprint,
                "Missing completed reference export or different data: " + pairing)
        require(paper["source_export_run_ids"][pairing] == status["run_id"], "Stale reference report: " + pairing)
        settings = status["config"]
        require(settings["seeds"] == seeds, "Reference training seeds differ from the baseline configuration.")
        require(settings["nfe"] == [1, 4, 16, 64, 128, 256] and settings["unit_nfe"] == [16, 64, 128, 256, 512],
                "The reference inference grids differ from the paper protocol.")
        require(settings["max_cells"] == 400 and settings["batch_size"] == 256
                and status["timing_repeats"] == 3 and status["untimed_warmups"] == 1
                and status["metric_seed"] == 314159, "Reference evaluation protocol differs from the paper protocol.")
        for kind in ("count_flow_map", "count_fm"):
            models = [m for m in status["models"] if m["method"] == kind]
            require(len(models) == len(seeds), "Incomplete exported core model metadata: " + kind)
            for m in models:
                cfg = m["model_config"]
                rule = cfg.get("validation_selection", {})
                require(cfg["hidden_dim"] == architecture["hidden_dim"]
                        and cfg["depth"] - 1 == architecture["hidden_layers"]
                        and cfg["steps"] == 50000 and rule.get("every") == 5000
                        and rule.get("nfe") == (1 if kind == "count_flow_map" else 128)
                        and rule.get("evaluation_fingerprint") == fingerprint,
                        "Cached core architecture or validation settings differ from the agreed comparison.")
        frame = pd.read_csv(directory / "metrics.csv")
        rows = pd.read_csv(directory / "main_quality_table_raw.csv")
        conditions = pd.read_csv(directory / "condition_metrics.csv")
        for table in (frame, rows, conditions):
            require(set(table.run_id) == {status["run_id"]}, "Mixed reference run IDs: " + pairing)
        # Old neural baseline rows are deliberately irrelevant to this update.
        keep = list(ITERATIVE) + (list(FIXED) if pairing == "OT" else [])
        frame = frame[frame.method.isin(keep)].copy()
        rows = rows[rows.method.isin(keep)].copy()
        conditions = conditions[conditions.method.isin(keep)].copy()
        for method in ITERATIVE:
            grid = settings["unit_nfe"] if method == FM_UNIT else settings["nfe"]
            subset = frame[frame.method == method]
            require(set(subset.seed) == set(seeds), "Incomplete core seeds: " + method)
            for seed in seeds:
                for split in ("val", "test"):
                    actual = subset[(subset.seed == seed) & (subset.split == split)]
                    require(sorted(actual.nfe.tolist()) == grid, f"Incomplete core export: {pairing}, {method}, {seed}, {split}")
        for label in CORE_ROWS:
            selected = rows[rows.reported_method == label]
            require(len(selected) == len(seeds) and set(selected.seed) == set(seeds), "Incomplete cached table row: " + label)
        if pairing == "OT":
            for label in FIXED:
                selected = rows[rows.reported_method == label]
                require(len(selected) == 1 and selected.seed.isna().all(), "Missing cached one-off baseline: " + label)
        require(rows.split.eq("test").all(), "Reference main table must use test data.")
        for row in frame.itertuples():
            sub = conditions[(conditions.method == row.method) & (conditions.split == row.split)]
            sub = sub[sub.seed.isna()] if pd.isna(row.seed) else sub[sub.seed == row.seed]
            sub = sub[sub.nfe.isna()] if pd.isna(row.nfe) else sub[sub.nfe == row.nfe]
            require(len(sub) == status["evaluation_rows"][row.split]["conditions"]
                    and not sub.condition_id.duplicated().any(), "Incomplete condition metrics: " + row.method)
            for key in METRICS[:-1]:
                require(np.isfinite(sub[key]).all() and np.isclose(sub[key].mean(), getattr(row, key)),
                        "Inconsistent condition-averaged reference metric: " + key)
        # Selected core numbers must equal their exported operating-point rows.
        for row in rows.itertuples():
            sub = frame[(frame.method == row.method) & (frame.split == "test")]
            sub = sub[sub.seed.isna()] if pd.isna(row.seed) else sub[sub.seed == row.seed]
            sub = sub[sub.nfe.isna()] if pd.isna(row.nfe) else sub[sub.nfe == row.nfe]
            require(len(sub) == 1, "Reference table refers to a missing operating point.")
            for key in METRICS:
                require(np.isfinite(getattr(row, key)) and np.isclose(sub.iloc[0][key], getattr(row, key)),
                        "Reference main table disagrees with its metrics: " + key)
        frame["pairing"] = pairing
        conditions["pairing"] = pairing
        rows["pairing"] = np.where(rows.method.isin(ITERATIVE), pairing, "—")
        metrics.append(frame); raw.append(rows); condition_frames.append(conditions)
        with np.load(directory / "paper_projections.npz", allow_pickle=False) as array:
            require(str(array["run_id"].item()) == status["run_id"], "Stale reference PCA.")
            projections[pairing] = {key: array[key].copy() for key in
                                   ("count_mean", "count_components", "effect_mean", "effect_components")}
        indices[pairing] = {split: np.load(directory / f"{split}_row_indices.npy", allow_pickle=False)
                            for split in ("val", "test")}
        paths.extend(directory / name for name in (
            "paper_run_status.json", "metrics.csv", "main_quality_table_raw.csv", "condition_metrics.csv",
            "paper_projections.npz", "val_row_indices.npy", "test_row_indices.npy", "validation_selected_nfe.csv"))
    a, b = (statuses[p] for p in PAIRINGS)
    for key in ("evaluation_fingerprint", "pca_config", "metric_seed", "evaluation_rows", "environment"):
        require(a[key] == b[key], "Reference OT/non-OT mismatch: " + key)
    for split in ("val", "test"):
        require(np.array_equal(indices["Non-OT"][split], indices["OT"][split]), "Different reference evaluation cells.")
    for key in projections["Non-OT"]:
        require(np.allclose(projections["Non-OT"][key], projections["OT"][key], rtol=1e-6, atol=1e-7),
                "Different reference PCA arrays.")
    paths.extend(root / name for name in ("paper_v2_status.json", "checkpoint_selection.csv"))
    for name in ("paper_scrna_main", "paper_scrna_appendix"):
        for extension in ("pdf", "png"):
            paths.append(root / "figures" / f"{name}.{extension}")
    missing = [str(path) for path in paths if not path.is_file()]
    require(not missing, "Missing reference files: " + ", ".join(missing))
    return {"root": root, "statuses": statuses, "fingerprint": fingerprint,
            "raw": pd.concat(raw, ignore_index=True), "metrics": pd.concat(metrics, ignore_index=True),
            "conditions": pd.concat(condition_frames, ignore_index=True), "indices": indices["Non-OT"],
            "projections": projections["Non-OT"], "hashes": {str(p.resolve()): sha256(p) for p in paths},
            "paper": paper}


def frozen_evaluation(reference, bundle):
    splits = {}
    for name in ("val", "test"):
        split = getattr(bundle, name)
        index = reference["indices"][name]
        require(index.ndim == 1 and np.issubdtype(index.dtype, np.integer)
                and len(np.unique(index)) == len(index) and index.min() >= 0 and index.max() < split.n,
                "Invalid saved evaluation row indices: " + name)
        idx = torch.from_numpy(index.astype(np.int64))
        chosen = ConditionalPairSplit(split.x0[idx], split.x1[idx], split.context[idx], split.condition_id[idx])
        expected = reference["statuses"]["Non-OT"]["evaluation_rows"][name]
        require(chosen.n == expected["cells"] and chosen.condition_id.unique().numel() == expected["conditions"],
                "Saved indices do not match the data bundle: " + name)
        for cid in split.condition_id.unique():
            require(int((chosen.condition_id == cid).sum()) == min(400, int((split.condition_id == cid).sum())),
                    "Unexpected number of frozen cells per condition.")
        splits[name] = chosen
    arrays = reference["projections"]
    require(arrays["count_components"].shape[1] == bundle.dim, "PCA gene dimension does not match the data.")
    projection = PCAProjection(torch.from_numpy(arrays["count_mean"]), torch.from_numpy(arrays["count_components"]))
    return splits, projection


def check_reference_unchanged(reference):
    for filename, digest in reference["hashes"].items():
        require(Path(filename).is_file() and sha256(filename) == digest, "Reference changed during update: " + filename)


def write_report(output, reference, baseline_rows, condition_rows, selections, architecture_rows, seeds):
    check_reference_unchanged(reference)
    output = Path(output)
    fresh = pd.DataFrame(baseline_rows)
    fresh["pairing"] = "—"
    fresh["reported_method"] = fresh.method
    fresh["reported_nfe"] = np.nan
    fresh["nfe"] = np.nan
    raw = pd.concat([reference["raw"], fresh[fresh.split == "test"]], ignore_index=True)
    summary = summarize_table(raw)
    old_selection = pd.read_csv(reference["root"] / "checkpoint_selection.csv")
    all_selection = pd.concat([old_selection, pd.DataFrame(selections)], ignore_index=True)
    write_tables(output, raw, summary, all_selection, seeds)
    # The existing writer's numbers/layout are retained. Clarify new baseline selection.
    table = output / "paper_scrna_table.tex"
    table.write_text(table.read_text().replace("Time is median warm generation time", 
        "Neural baseline checkpoints are selected by validation sliced $W_2$ on the same frozen cells and PCA. "
        "Time is median warm generation time"))
    pd.concat([reference["metrics"], fresh], ignore_index=True).to_csv(output / "metrics.csv", index=False)
    fresh_conditions = pd.DataFrame(condition_rows).assign(pairing="—")
    fresh_conditions["nfe"] = np.nan
    pd.concat([reference["conditions"], fresh_conditions], ignore_index=True).to_csv(output / "condition_metrics.csv", index=False)
    pd.DataFrame(architecture_rows).to_csv(output / "architecture_and_parameters.csv", index=False)
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name in ("paper_scrna_main", "paper_scrna_appendix"):
        for extension in ("pdf", "png"):
            shutil.copy2(reference["root"] / "figures" / f"{name}.{extension}", figure_dir)
    atomic_json(output / "paper_v3_status.json", {
        "status": "completed", "source_export_run_ids": reference["paper"]["source_export_run_ids"],
        "reference_file_sha256": reference["hashes"], "core_retrained": False,
        "core_resampled": False, "core_retimed": False, "core_figures": "copied unchanged",
        "baseline_methods": sorted(fresh.method.unique().tolist()), "seeds": seeds,
        "architecture": "Matched main MLP depth, width and activation. Method-specific components are retained.",
    })
    return summary
