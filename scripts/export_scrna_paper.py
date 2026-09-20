"""Fresh paper evaluation from saved models; never trains neural networks.

Reuses the original generators and metrics, including both Count-FM samplers
and all six paper baselines. Every method uses the same frozen evaluation rows
and training-only PCA. Generation time excludes model/state setup and metrics.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.runtime_env import maybe_reexec_with_conda_libstdcpp

if __name__ == "__main__":
    maybe_reexec_with_conda_libstdcpp()

import numpy as np
import pandas as pd
import torch
from countflow.application_data import ConditionalPairSplit, load_pair_bundle
from countflow.application_metrics import evaluate_scrna_by_condition, fit_count_pca
from countflow import application_runner as app
from countflow import scrna_response_baselines as baselines
from countflow.utils import set_seed
from scripts.run_scrna_drug_transport import SCRNA_CONFIG

CFM = "Count Flow Map"
FM_TAU = "Count-FM + binomial tau-leap"
FM_UNIT = "Count-FM + unit jump"
ITERATIVE = (CFM, FM_UNIT, FM_TAU)


def subset_rows(split, split_name, max_cells):
    """Use exactly the audit's deterministic within-condition row selection."""
    si = {"val": 1, "test": 2}[split_name]
    selected = []
    for cid in sorted(split.condition_id.unique().tolist()):
        idx = torch.where(split.condition_id == cid)[0]
        generator = torch.Generator().manual_seed(10000 + si * 1000 + cid)
        idx = idx[torch.randperm(len(idx), generator=generator)]
        selected.append(idx[:max_cells] if max_cells else idx)
    idx = torch.cat(selected)
    return ConditionalPairSplit(split.x0[idx], split.x1[idx], split.context[idx], split.condition_id[idx]), idx


def checkpoint_model(path, kind, bundle, expected_fingerprint, device):
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}. No training is performed by this exporter.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload.get("metadata", {})
    if meta.get("bundle_fingerprint") != expected_fingerprint:
        raise ValueError(f"Checkpoint/data fingerprint mismatch: {path}")
    if meta.get("method", kind) != kind or "model_config" not in meta:
        raise ValueError(f"Checkpoint kind/configuration mismatch: {path}")
    builders = {"count_flow_map": app._build_flow_map, "count_fm": app._build_count_fm,
                "scgen_vae": baselines.build_scgen, "cpa": baselines.build_cpa,
                "nbvae": baselines.build_nbvae}
    model = builders[kind](bundle, meta["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    provenance = {"path": str(path.resolve()), "sha256": digest.hexdigest(),
                  "method": kind, "training_fingerprint": expected_fingerprint,
                  "model_config": meta["model_config"]}
    return model, meta, provenance


def time_generation(draw, seed, device, repeats):
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    # Warm-up is untimed. Reset RNG so repeat count does not change predictions.
    set_seed(seed)
    draw()
    sync()
    timings = []
    for _ in range(repeats):
        set_seed(seed)
        sync()
        start = time.perf_counter()
        generated = draw()
        sync()
        timings.append(time.perf_counter() - start)
    return generated, float(np.median(timings)), timings


def choose_budgets(metrics):
    """Choose NFE on validation separately by method/seed; ties favor less compute."""
    rows = []
    for (method, seed), frame in metrics[metrics.method.isin(ITERATIVE)].groupby(["method", "seed"]):
        val = frame[frame.split == "val"].dropna(subset=["sliced_w2", "nfe"])
        if val.empty:
            raise ValueError(f"Missing validation scores: {method}, seed {seed}")
        best = val.sort_values(["sliced_w2", "nfe"]).iloc[0]
        rows.append({"method": method, "seed": int(seed), "selected_nfe": int(best.nfe),
                     "validation_sliced_w2": float(best.sliced_w2)})
    return pd.DataFrame(rows)


def main_table_rows(metrics, choices):
    rows = []
    for seed in sorted(metrics.loc[metrics.method == CFM, "seed"].dropna().unique()):
        for nfe in (1, 16):
            match = metrics[(metrics.method == CFM) & (metrics.seed == seed) &
                            (metrics.split == "test") & (metrics.nfe == nfe)]
            if len(match) != 1:
                raise ValueError(f"Expected one genuine fixed-{nfe} CFM row for seed {seed}.")
            rows.append({**match.iloc[0].to_dict(), "reported_method": f"{CFM} ({nfe} NFE)", "reported_nfe": nfe})
    for row in choices.to_dict("records"):
        if row["method"] == CFM:
            continue
        match = metrics[(metrics.method == row["method"]) & (metrics.seed == row["seed"]) &
                        (metrics.split == "test") & (metrics.nfe == row["selected_nfe"])]
        if len(match) != 1:
            raise ValueError(f"Selected test score missing: {row}")
        rows.append({**match.iloc[0].to_dict(), "reported_method": row["method"] + " (validation-selected)",
                     "reported_nfe": row["selected_nfe"]})
    for row in metrics[(metrics.split == "test") & ~metrics.method.isin(ITERATIVE)].to_dict("records"):
        rows.append({**row, "reported_method": row["method"], "reported_nfe": np.nan})
    return pd.DataFrame(rows)


def effect_projection(training_split):
    effects = []
    for cid in sorted(training_split.condition_id.unique().tolist()):
        mask = training_split.condition_id == cid
        effects.append(torch.log1p(training_split.x1[mask].double().mean(0)) -
                       torch.log1p(training_split.x0[mask].double().mean(0)))
    effects = torch.stack(effects)
    mean = effects.mean(0)
    _, _, vh = torch.linalg.svd(effects - mean, full_matrices=False)
    components = torch.zeros((2, effects.shape[1]), dtype=effects.dtype)
    components[:min(2, len(vh))] = vh[:2]
    return mean.numpy(), components.numpy()


def save_effect_snapshot(path, split, generated, projection, lookup, metadata):
    ids = sorted(split.condition_id.unique().tolist())
    observed, predicted = [], []
    for cid in ids:
        mask = split.condition_id == cid
        ctl = torch.log1p(split.x0[mask].double().mean(0))
        observed.append((torch.log1p(split.x1[mask].double().mean(0)) - ctl).numpy())
        predicted.append((torch.log1p(generated[mask].double().mean(0)) - ctl).numpy())
    np.savez_compressed(path, condition_ids=np.asarray(ids),
                        condition_names=np.asarray([lookup.get(cid, str(cid)) for cid in ids]),
                        cell_condition_ids=split.condition_id.numpy(),
                        observed_effect=np.stack(observed), generated_effect=np.stack(predicted),
                        control_pca=projection.transform(split.x0)[:, :2].numpy(),
                        target_pca=projection.transform(split.x1)[:, :2].numpy(),
                        generated_pca=projection.transform(generated)[:, :2].numpy(),
                        metadata_json=np.asarray(json.dumps(metadata)))


def export(args):
    bundle = load_pair_bundle(args.data)
    fingerprint = bundle.metadata.get("bundle_fingerprint")
    audit = json.loads((args.audit_output / "run_status.json").read_text())
    if audit.get("status") != "completed" or audit.get("evaluation_fingerprint") != fingerprint:
        raise ValueError("Choose a completed audit run evaluated on this original dataset.")
    core_paths = {}
    for seed in args.seeds:
        for kind in ("count_flow_map", "count_fm"):
            matches = [m for m in audit["models_used"] if m["method"] == kind and m["seed"] == seed]
            if len(matches) != 1:
                raise ValueError(f"Run the audit for {kind}, seed {seed} first; this exporter never trains.")
            local = args.audit_output / "checkpoints" / str(seed) / f"{kind}.pt"
            core_paths[(seed, kind)] = local if local.exists() else Path(matches[0]["checkpoint"])
    needed = list(core_paths.values())
    if not args.core_only:
        needed += [args.baseline_checkpoint_dir / str(seed) / f"{kind}.pt"
                   for seed in args.seeds for kind in ("scgen_vae", "cpa", "nbvae")]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required saved models missing:\n" + "\n".join(missing) +
                                "\nPoint --baseline-checkpoint-dir to the old paper checkpoints, or use --core-only for an explicitly incomplete comparison.")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "snapshots").mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex
    device = torch.device(args.device)
    status = {"status": "running", "run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
              "source_audit_run_id": audit["run_id"], "source_audit_output": str(args.audit_output.resolve()),
              "evaluation_fingerprint": fingerprint, "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "pca_config": SCRNA_CONFIG["pca"], "models": [], "neural_training_performed": False,
              "effect_definition": "log1p(mean treated counts) - log1p(mean matched control counts)",
              "effect_pca_fit": "training-condition observed effects only",
              "metric_seed": 314159, "timing_scope": "whole selected split, including generation transfers; excludes loading, state preparation, PCA and metrics",
              "timing_repeats": args.timing_repeats, "untimed_warmups": 1,
              "environment": {"python": sys.version, "torch": torch.__version__, "device": str(device),
                              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
              "complete_baseline_set": not args.core_only, "seed_count": len(args.seeds)}
    status_path = args.output / "paper_run_status.json"
    status_path.write_text(json.dumps(status, indent=2))
    projection = fit_count_pca(torch.cat([bundle.train.x0, bundle.train.x1]), **SCRNA_CONFIG["pca"])
    effect_mean, effect_components = effect_projection(bundle.train)
    np.savez_compressed(args.output / "paper_projections.npz", count_mean=projection.mean.numpy(),
                        count_components=projection.components.numpy(), effect_mean=effect_mean,
                        effect_components=effect_components, run_id=np.asarray(run_id))
    splits = {}
    for name in ("val", "test"):
        splits[name], idx = subset_rows(getattr(bundle, name), name, args.max_cells)
        np.save(args.output / f"{name}_row_indices.npy", idx.numpy())
    status["evaluation_rows"] = {name: {"cells": split.n, "conditions": len(split.condition_id.unique())}
                                 for name, split in splits.items()}
    lookup = {int(c["condition_id"]): c.get("condition_name", str(c["condition_id"]))
              for c in bundle.metadata.get("conditions", [])}
    aggregate_rows, condition_rows, timing_rows = [], [], []

    def load(path, kind, expected):
        model, meta, record = checkpoint_model(path, kind, bundle, expected, device)
        status["models"].append(record)
        status_path.write_text(json.dumps(status, indent=2))
        return model, meta

    def evaluate(draw, method, seed, split_name, nfe, rng_seed):
        split = splits[split_name]
        generated, seconds, repeats = time_generation(draw, rng_seed, device, args.timing_repeats)
        aggregate, conditions = evaluate_scrna_by_condition(generated, split.x0, split.x1, split.condition_id,
                                                           projection, seed=314159)
        row = {"run_id": run_id, "method": method, "seed": seed, "split": split_name,
               "nfe": nfe, "generation_seconds": seconds, "n_cells": split.n,
               "n_conditions": len(conditions), **aggregate}
        aggregate_rows.append(row)
        for c in conditions:
            condition_rows.append({**row, **c, "condition_name": lookup.get(int(c["condition_id"]), str(int(c["condition_id"])))})
        timing_rows.extend({"run_id": run_id, "method": method, "seed": seed, "split": split_name,
                            "nfe": nfe, "repeat": i, "generation_seconds": value}
                           for i, value in enumerate(repeats))
        if method == CFM and split_name == "test":
            save_effect_snapshot(args.output / "snapshots" / f"cfm_seed{seed}_nfe{nfe}.npz",
                                 split, generated, projection, lookup, row)
        pd.DataFrame(aggregate_rows).to_csv(args.output / "metrics.csv", index=False)
        print(f"[paper] {method} seed={seed} {split_name} NFE={nfe}: SW2={aggregate['sliced_w2']:.4f}, generation={seconds:.3f}s", flush=True)

    for seed in args.seeds:
        flow, flow_meta = load(core_paths[(seed, "count_flow_map")], "count_flow_map", audit["training_fingerprint"])
        fm, fm_meta = load(core_paths[(seed, "count_fm")], "count_fm", audit["training_fingerprint"])
        for split_name, split in splits.items():
            si = {"val": 1, "test": 2}[split_name]
            for method, model, meta, generator, grid, offset in (
                (CFM, flow, flow_meta, app.generate_conditional_flow_map, args.nfe, 0),
                (FM_TAU, fm, fm_meta, app.generate_conditional_count_fm, args.nfe, 0),
                (FM_UNIT, fm, fm_meta, app.generate_conditional_count_fm_unit_jump, args.unit_nfe, 50000),
            ):
                for nfe in grid:
                    draw = lambda: generator(model, split.x0, split.context, n_steps=nfe,
                                             tau=float(meta.get("train_config", {}).get("tau", SCRNA_CONFIG["tau"])),
                                             device=str(device), batch_size=args.batch_size)
                    evaluate(draw, method, seed, split_name, nfe, seed + 100000 + si * 1000 + nfe + offset)
        del flow, fm
        if args.core_only:
            continue
        model, meta = load(args.baseline_checkpoint_dir / str(seed) / "scgen_vae.pt", "scgen_vae", fingerprint)
        set_seed(seed)
        scgen_state = baselines.prepare_scgen_response_state(model, bundle, device=str(device), batch_size=args.batch_size)
        scvidr_state = baselines.prepare_scvidr_response_state(bundle, scgen_state, ridge=float(meta["model_config"].get("scvidr_ridge", 0.0)))
        for split_name, split in splits.items():
            for method, generator, state in ((baselines.SCGEN_LABEL, baselines.generate_scgen_cached, scgen_state),
                                             (baselines.SCVIDR_LABEL, baselines.generate_scvidr_cached, scvidr_state)):
                evaluate(lambda: generator(model, split, state, device=str(device), batch_size=args.batch_size),
                         method, seed, split_name, np.nan, seed + 3000)
        del model, scgen_state, scvidr_state
        for kind, method in (("cpa", baselines.CPA_LABEL), ("nbvae", baselines.NBVAE_LABEL)):
            model, _ = load(args.baseline_checkpoint_dir / str(seed) / f"{kind}.pt", kind, fingerprint)
            for split_name, split in splits.items():
                if kind == "cpa":
                    draw = lambda: baselines.generate_cpa(model, bundle, split, seed=seed + 3300, device=str(device), batch_size=args.batch_size)
                else:
                    draw = lambda: baselines.generate_nbvae(model, split, seed=seed + 3500, device=str(device), batch_size=args.batch_size)
                evaluate(draw, method, seed, split_name, np.nan, seed + 3000)
            del model
    if not args.core_only:
        # Original one-off baselines: training-only response statistics, excluded
        # from inference timing. Report one run, not fabricated seed replicates.
        state = baselines.fit_linear_dose_response(bundle, **SCRNA_CONFIG["models"]["linear_dose"])
        evaluate(lambda: baselines.generate_linear_dose_response_cached(splits["test"], state),
                 baselines.LINEAR_LABEL, np.nan, "test", np.nan, 42)
        cfg = dict(SCRNA_CONFIG["models"]["sinkhorn_ot"])
        knn = cfg.pop("knn", 8)
        set_seed(int(cfg.get("seed", 42)))
        state = baselines.fit_sinkhorn_ot(bundle, splits["test"], **cfg)
        evaluate(lambda: baselines.generate_sinkhorn_ot_cached(bundle, splits["test"], state, knn=knn),
                 baselines.SINKHORN_LABEL, np.nan, "test", np.nan, 42)
    metrics = pd.DataFrame(aggregate_rows)
    choices = choose_budgets(metrics)
    table = main_table_rows(metrics, choices)
    choices.to_csv(args.output / "validation_selected_nfe.csv", index=False)
    table.to_csv(args.output / "main_quality_table_raw.csv", index=False)
    app._summarize_main_table(table, args.output / "main_quality_table_summary.csv")
    pd.DataFrame(condition_rows).to_csv(args.output / "condition_metrics.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(args.output / "generation_timings.csv", index=False)
    status.update(status="completed", completed_at=datetime.now(timezone.utc).isoformat())
    status_path.write_text(json.dumps(status, indent=2))
    print(f"[paper] Completed fresh evaluation in {args.output}. No neural training was performed.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--nfe", type=int, nargs="+", default=[1, 4, 16, 64, 128, 256])
    parser.add_argument("--unit-nfe", type=int, nargs="+", default=[16, 64, 128, 256, 512])
    parser.add_argument("--max-cells", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--core-only", action="store_true")
    args = parser.parse_args(argv)
    if not {1, 16}.issubset(args.nfe) or any(n < 1 for n in args.nfe + args.unit_nfe):
        parser.error("Positive NFE grids required; the paper table requires fixed CFM budgets 1 and 16.")
    if args.max_cells < 0 or args.batch_size < 1 or args.timing_repeats < 1:
        parser.error("Invalid cell limit, batch size or timing repeat count.")
    args.nfe = sorted(set(args.nfe)); args.unit_nfe = sorted(set(args.unit_nfe)); args.seeds = sorted(set(args.seeds))
    if args.output.resolve() in (args.audit_output.resolve(), args.baseline_checkpoint_dir.parent.resolve()):
        parser.error("Use a separate paper-export directory.")
    export(args)


if __name__ == "__main__":
    main()
