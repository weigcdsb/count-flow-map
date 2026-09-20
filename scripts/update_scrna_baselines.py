"""Update only application baselines. There is no core training/sampling path."""
import argparse
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.runtime_env import maybe_reexec_with_conda_libstdcpp
if __name__ == "__main__":
    maybe_reexec_with_conda_libstdcpp()

import numpy as np
import torch
from countflow.application_data import load_pair_bundle
from countflow.application_metrics import evaluate_scrna_by_condition
from countflow.simulation_metrics import sliced_w2
from countflow import scrna_response_baselines as base
from countflow.scrna_matched_baselines import (
    VERSION, KINDS, LABELS, STEMS, BaselineRun, build_matched, total_steps,
    atomic_json, atomic_torch, sha256, signature, scoring_mode,
)
from countflow.utils import set_seed
from scripts.export_scrna_paper import time_generation
from scripts.scrna_cached_reference import (
    read_json, require, read_reference, frozen_evaluation, write_report, check_reference_unchanged,
)


@contextmanager
def exclusive_run(directory):
    """An OS lock is released automatically on interruption, including process death."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".update.lock").open("a+b") as handle:
        if sys.platform == "win32":
            import msvcrt
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b"0"); handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("Another update is using " + str(directory)) from error
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("Another update is using " + str(directory)) from error
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def protect_paths(args):
    paths = [args.reference.resolve(), args.baseline_output.resolve(), args.output.resolve()]
    for i, left in enumerate(paths):
        for right in paths[i + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError("Reference, baseline output, and report output must be separate, non-nested directories.")
    for destination in paths[1:]:
        if destination == ROOT or destination in ROOT.parents:
            raise ValueError("Choose a dedicated output directory.")


def code_hashes():
    names = ["countflow/scrna_response_baselines.py", "countflow/scrna_matched_baselines.py",
             "countflow/model.py", "countflow/conditional_model.py", "countflow/application_data.py",
             "countflow/application_metrics.py", "countflow/simulation_metrics.py", "countflow/utils.py",
             "scripts/update_scrna_baselines.py", "scripts/scrna_cached_reference.py",
             "scripts/export_scrna_paper.py", "scripts/scrna_paper_v2.py"]
    return {name: sha256(ROOT / name) for name in names}


def make_identity(kind, seed, config, reference, data_hash, sources):
    return {"version": VERSION, "kind": kind, "seed": seed, "architecture": config["architecture"],
            "train_config": config["models"][kind], "validation_checkpoints": config["validation_checkpoints"],
            "data_sha256": data_hash, "bundle_fingerprint": reference["fingerprint"],
            # Relative names allow moving an intact experiment to a new machine/folder.
            "reference_sha256": {str(Path(p).relative_to(reference["root"].resolve())): v
                                  for p, v in reference["hashes"].items()}, "code_sha256": sources}


def runtime_device(reference, requested):
    expected = reference["statuses"]["Non-OT"]["environment"]
    device = torch.device(requested or expected["device"])
    require(device.type == torch.device(expected["device"]).type,
            "Use the same device type as the saved count-model timings.")
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA is required for these cached GPU timing comparisons. Activate the original environment on the GPU machine.")
        require(torch.cuda.get_device_name(device) == expected["gpu"],
                "Use the original GPU model for comparable generation timings: " + expected["gpu"])
    require(str(torch.__version__) == expected["torch"],
            "Use the original PyTorch environment for comparable timings (" + expected["torch"] + ").")
    return device


def draw_factory(kind, label, model, bundle, split, seed, device, prepared=None):
    options = {"device": str(device), "batch_size": 256}
    if kind == "nbvae":
        return lambda: base.generate_nbvae(model, split, seed=seed + 3500, **options)
    if kind == "cpa":
        lookup = base._condition_lookup(bundle)
        pairs = {(lookup[int(cid)].cell_line_id, lookup[int(cid)].drug)
                 for cid in split.condition_id.unique().tolist()}
        slopes = {pair: base._training_library_log_slope(bundle, lookup, *pair) for pair in pairs}
        return lambda: base.generate_cpa(model, bundle, split, seed=seed + 3300, library_slopes=slopes, **options)
    state = prepared
    if state is None:
        state = base.prepare_scgen_response_state(model, bundle, **options)
    if label == base.SCVIDR_LABEL:
        state = base.prepare_scvidr_response_state(bundle, state)
        return lambda: base.generate_scvidr_cached(model, split, state, **options)
    return lambda: base.generate_scgen_cached(model, split, state, **options)


def validation_scorer(kind, bundle, split, projection, seed, device):
    # Precompute observed PCA once. Fit nothing on validation cells.
    observed = projection.transform(split.x1)
    masks = [(int(cid), split.condition_id == cid) for cid in sorted(split.condition_id.unique().tolist())]

    def score(model):
        prepared = (base.prepare_scgen_response_state(model, bundle, device=str(device), batch_size=256)
                    if kind == "scgen_vae" else None)
        scores = {}
        for label in LABELS[kind]:
            set_seed(seed + 3000)
            draw = draw_factory(kind, label, model, bundle, split, seed, device, prepared)
            generated = projection.transform(draw())
            scores[label] = float(np.mean([
                sliced_w2(generated[mask], observed[mask], n_projections=128,
                          max_points=5000, seed=314159 + cid) for cid, mask in masks]))
        return scores
    return score


def train_one(kind, model, bundle, config, seed, device, run):
    cfg = config["models"][kind]
    options = {"device": str(device), "run_state": run}
    train_config = base.published_train_config(cfg, seed)
    if kind == "scgen_vae":
        base.train_scgen(model, bundle.train, train_config, **options)
    elif kind == "nbvae":
        base.train_nbvae(model, bundle.train, train_config, cfg, **options)
    elif kind == "cpa":
        base.train_cpa(model, bundle, train_config, cfg, **options)
    else:
        raise ValueError("Only revised application baselines can train here.")


def evaluation_paths(directory, label, split):
    stem = STEMS[label] + "_" + split
    return directory / (stem + "_samples.pt"), directory / (stem + "_metrics.json")


def evaluation_identity(run, label, split_name):
    selected = run.directory / (STEMS[label] + "_selected.pt")
    return signature({"training_signature": run.signature, "checkpoint_sha256": sha256(selected),
                      "method": label, "split": split_name, "metric_seed": 314159,
                      "batch_size": 256, "timing_repeats": 3})


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def cached_evaluation(run, label, split_name):
    samples, metrics = evaluation_paths(run.directory, label, split_name)
    if not metrics.is_file():
        return None
    result = read_json(metrics)
    require(result["signature"] == evaluation_identity(run, label, split_name),
            "Incompatible baseline evaluation cache: " + str(metrics))
    require(samples.is_file() and sha256(samples) == result["samples_sha256"],
            "Saved baseline samples are missing or changed: " + str(samples))
    for key in ("sliced_w2", "deg_logfc_pearson", "top_response_gene_overlap", "generation_seconds"):
        require(result["row"][key] is not None and np.isfinite(result["row"][key]), "Non-finite baseline paper metric: " + key)
    return result


def evaluate_one(run, kind, label, split_name, model, bundle, split, projection, seed, device):
    saved = cached_evaluation(run, label, split_name)
    if saved is not None:
        print(f"[reuse] {label}, seed {seed}, {split_name}: saved samples, metrics and timings.", flush=True)
        return saved
    samples, metrics = evaluation_paths(run.directory, label, split_name)
    identity = evaluation_identity(run, label, split_name)
    if samples.is_file():
        payload = torch.load(samples, map_location="cpu", weights_only=False)
        require(payload["signature"] == identity, "Incompatible saved baseline samples: " + str(samples))
        print(f"[reuse] {label}, seed {seed}, {split_name}: compute missing metrics from saved samples.", flush=True)
    else:
        print(f"[evaluate] {label}, seed {seed}, {split_name}.", flush=True)
        with scoring_mode(model):
            draw = draw_factory(kind, label, model, bundle, split, seed, device)
            generated, elapsed, times = time_generation(draw, seed + 3000, device, 3)
        payload = {"signature": identity, "generated": generated.cpu(), "generation_seconds": elapsed,
                   "generation_timings": times, "torch": str(torch.__version__), "device": str(device),
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
        # Commit samples/timing before metrics so interrupted metric work never resamples.
        atomic_torch(samples, payload)
    aggregate, conditions = evaluate_scrna_by_condition(
        payload["generated"], split.x0, split.x1, split.condition_id, projection, seed=314159)
    common = {"run_id": run.signature, "method": label, "seed": seed, "split": split_name, "nfe": None}
    row = {**common, "generation_seconds": payload["generation_seconds"], "n_cells": split.n,
           "n_conditions": int(split.condition_id.unique().numel()), **aggregate}
    result = finite_json({"signature": identity, "samples_sha256": sha256(samples), "row": row,
                          "conditions": [{**common, **condition} for condition in conditions],
                          "generation_timings": payload["generation_timings"]})
    atomic_json(metrics, result)
    return cached_evaluation(run, label, split_name)


def architecture_report(bundle, config, reference):
    """Inspect model shapes on the meta device, with no real core weights or data forward pass."""
    from countflow.application_runner import _build_flow_map, _build_count_fm
    records = []
    with torch.device("meta"):
        for kind, name, builder in (("count_flow_map", "Count Flow Map", _build_flow_map),
                                    ("count_fm", "Count-FM (both samplers)", _build_count_fm)):
            cfg = next(m["model_config"] for m in reference["statuses"]["Non-OT"]["models"] if m["method"] == kind)
            model = builder(bundle, {**cfg, "count_scale": 1.0})
            records.append({"method": name, "main_hidden_layers": cfg["depth"] - 1,
                            "main_hidden_width": cfg["hidden_dim"], "main_activation": "SiLU",
                            "main_batch_norm": False, "main_dropout": 0.0, "latent_dim": None,
                            "parameters": sum(p.numel() for p in model.parameters()),
                            "components": "Original state/context encoders and method-specific count heads"})
        for kind in KINDS:
            model = build_matched(kind, bundle, config["architecture"], config["models"][kind])
            a = config["architecture"]
            for label in LABELS[kind]:
                records.append({"method": label, "main_hidden_layers": a["hidden_layers"],
                                "main_hidden_width": a["hidden_dim"], "main_activation": "SiLU",
                                "main_batch_norm": False, "main_dropout": 0.0, "latent_dim": a["latent_dim"],
                                "parameters": sum(p.numel() for p in model.parameters()),
                                "components": {"nbvae": "Matched prior/posterior/decoder. Native source/target and context encoders, NB head",
                                               "cpa": (f"Matched encoder/decoder. Native embeddings, {config['models']['cpa']['adversary_depth']} "
                                                       f"adversary layers of width {config['models']['cpa']['adversary_width']} and "
                                                       f"{config['models']['cpa']['doser_depth']} doser layers of width {config['models']['cpa']['doser_width']}"),
                                               "scgen_vae": "Matched encoder/decoder, separate mean/logvar heads. One shared training trajectory per seed"}[kind]})
    records += [{"method": label, "parameters": 0, "components": "No neural network"}
                for label in (base.LINEAR_LABEL, base.SINKHORN_LABEL)]
    return records


def run_update(args):
    protect_paths(args)
    config = read_json(args.config)
    require(config["version"] == VERSION and set(config["models"]) == set(KINDS), "Unknown baseline configuration.")
    print("[check] Reading existing core exports. Core retraining/resampling/retiming is disabled.", flush=True)
    reference = read_reference(args.reference, args.data, config)
    bundle = load_pair_bundle(args.data)
    splits, projection = frozen_evaluation(reference, bundle)
    sources, data_hash = code_hashes(), sha256(args.data)
    jobs = []
    for seed in config["seeds"]:
        for kind in KINDS:
            identity = make_identity(kind, seed, config, reference, data_hash, sources)
            directory = args.baseline_output / str(seed) / kind
            run = BaselineRun(directory, identity, None, config["validation_checkpoints"])
            complete = run.completed()
            # Inspect incomplete state now, before any earlier model can start training.
            if not complete and run.resume_path.is_file():
                payload = torch.load(run.resume_path, map_location="cpu", weights_only=False)
                run._check_signature(payload)
                del payload
            pending = [(label, name) for label in LABELS[kind] for name in ("val", "test")
                       if not complete or cached_evaluation(run, label, name) is None]
            jobs.append((seed, kind, run, complete, pending))
            print(f"[check] seed {seed}, {kind}: {'reuse fit' if complete else 'resume fit' if run.resume_path.is_file() else 'new fit'}, "
                  f"{len(pending)} evaluation items pending.", flush=True)
    if args.check_only:
        print("[check] Passed. No training, sampling, timing, or report writes were performed.", flush=True)
        return
    if args.report_only:
        require(all(complete and not pending for _, _, _, complete, pending in jobs),
                "Revised baseline results are incomplete. Run once without --report-only to train/resume the baselines.")
    needs_compute = any(not complete or pending for _, _, _, complete, pending in jobs)
    device = runtime_device(reference, args.device) if needs_compute else torch.device("cpu")
    rows, conditions, selections = [], [], []
    with ExitStack() as locks:
        locks.enter_context(exclusive_run(args.baseline_output))
        locks.enter_context(exclusive_run(args.output))
        for seed, kind, run, complete, pending in jobs:
            # Another process may have completed between preflight and acquiring the lock.
            complete = run.completed()
            pending = [(label, name) for label in LABELS[kind] for name in ("val", "test")
                       if not complete or cached_evaluation(run, label, name) is None]
            model = None
            if not complete:
                set_seed(seed)  # Seed initialization as well as optimization.
                model = build_matched(kind, bundle, config["architecture"], config["models"][kind]).to(device)
                run.score = validation_scorer(kind, bundle, splits["val"], projection, seed, device)
                print(f"[train] {kind}, seed {seed}, {total_steps(kind, bundle.train, config['models'][kind])} updates.", flush=True)
                train_one(kind, model, bundle, config, seed, device, run)
            for label in LABELS[kind]:
                selected_path = run.directory / (STEMS[label] + "_selected.pt")
                selected = None
                if any(item[0] == label for item in pending):
                    if model is None:
                        set_seed(seed)
                        model = build_matched(kind, bundle, config["architecture"], config["models"][kind]).to(device)
                    selected = torch.load(selected_path, map_location="cpu", weights_only=False)
                    model.load_state_dict(selected["state_dict"])
                    model.eval()
                for name in ("val", "test"):
                    result = evaluate_one(run, kind, label, name, model, bundle, splits[name], projection, seed, device)
                    rows.append(result["row"]); conditions.extend(result["conditions"])
                info = read_json(run.directory / "training_complete.json")
                choice = info["selections"][label]
                selections.append({"pairing": "—", "seed": seed, "method": label,
                                   "selected_update": choice["step"], "training_updates": info["steps"],
                                   "validation_sliced_w2": choice["score"], "selection_nfe": 1,
                                   "checkpoint": str(selected_path.resolve())})
                del selected
            del model
            run.best.clear()
            run.score = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
        architecture = architecture_report(bundle, config, reference)
        write_report(args.output, reference, rows, conditions, selections, architecture, config["seeds"])
        atomic_json(args.output / "baseline_config.json", config)
        check_reference_unchanged(reference)
    print("[done] Updated table and unchanged core figures: " + str(args.output.resolve()), flush=True)
    print("[done] Count Flow Map / Count-FM: 0 training runs, 0 generated samples, 0 timing runs.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/tahoe_panel/tahoe_condition_holdout.npz")
    parser.add_argument("--reference", type=Path, default=ROOT / "outputs/scrna_paper_v2")
    parser.add_argument("--baseline-output", type=Path, default=ROOT / "outputs/scrna_baselines_matched")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/scrna_paper_v3")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/scrna_baselines_matched.json")
    parser.add_argument("--device", default=None, help="Defaults to the cached core device. Must match the original timing environment.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true", help="Read-only cache/data preflight.")
    modes.add_argument("--report-only", action="store_true", help="Require complete revised baseline caches. Never fit or generate.")
    args = parser.parse_args(argv)
    try:
        run_update(args)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(1, "ERROR: " + str(error) + "\nExisting count-model outputs were not regenerated.\n")


if __name__ == "__main__":
    main()
