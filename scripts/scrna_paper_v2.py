"""Combine matched OT/non-OT exports into the three scRNA paper items.

Training, sampling, metric definitions, and NFE selection are reused from the
existing scripts. This module only checks provenance, summarizes, and plots.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

from scripts.export_scrna_paper import CFM, FM_TAU, FM_UNIT, ITERATIVE
from countflow import scrna_response_baselines as baselines

PAIRINGS = ("Non-OT", "OT")
CORE_ROWS = (CFM + " (1 NFE)", CFM + " (16 NFE)",
             FM_UNIT + " (validation-selected)", FM_TAU + " (validation-selected)")
BASELINES = (baselines.NBVAE_LABEL, baselines.CPA_LABEL, baselines.SCGEN_LABEL,
             baselines.SCVIDR_LABEL, baselines.SINKHORN_LABEL, baselines.LINEAR_LABEL)
METRICS = ("sliced_w2", "deg_logfc_pearson", "top_response_gene_overlap", "generation_seconds")


def read_json(path):
    return json.loads(Path(path).read_text())


def data_metadata(path):
    with np.load(path, allow_pickle=False) as arrays:
        return json.loads(str(arrays["metadata_json"].item()))


def check_baselines(data, checkpoint_dir, seeds):
    """Fail before expensive core fitting if the existing baseline fits are absent."""
    fingerprint = data_metadata(data)["bundle_fingerprint"]
    paths = [Path(checkpoint_dir) / str(seed) / (kind + ".pt")
             for seed in seeds for kind in ("scgen_vae", "cpa", "nbvae")]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Set BASELINE_CHECKPOINTS to the existing three-seed fits. Missing:\n"
                                + "\n".join(missing))
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("metadata", {}).get("bundle_fingerprint") != fingerprint:
            raise ValueError("Baseline was trained on a different data bundle: " + str(path))
        del payload


def check_audits(audit_dirs, data, seeds, max_cells, steps):
    """Require the same selection/training settings across pairings for each model."""
    fingerprint = data_metadata(data)["bundle_fingerprint"]
    statuses, configs, selections = {}, {}, []
    for pairing in PAIRINGS:
        folder = Path(audit_dirs[pairing])
        status = read_json(folder / "run_status.json")
        if status.get("status") != "completed" or status.get("evaluation_fingerprint") != fingerprint:
            raise ValueError("Incomplete audit or wrong evaluation data: " + pairing)
        settings = status["evaluation_settings"]
        if sorted(settings["seeds"]) != sorted(seeds) or settings["max_cells"] != max_cells:
            raise ValueError("Audit seeds/cell limit differ from notebook settings: " + pairing)
        if pairing == "Non-OT" and status["training_fingerprint"] != fingerprint:
            raise ValueError("The Non-OT audit must use the original training bundle.")
        if pairing == "OT" and status["training_fingerprint"] == fingerprint:
            raise ValueError("The OT audit must use the recoupled training bundle.")
        for seed in seeds:
            for method, budget in (("count_flow_map", 1), ("count_fm", 128)):
                records = [r for r in status["models_used"] if r["method"] == method and r["seed"] == seed]
                if len(records) != 1:
                    raise ValueError(f"Missing or duplicate audit model: {pairing}, {method}, seed {seed}")
                path = Path(records[0]["checkpoint"])
                payload = torch.load(path, map_location="cpu", weights_only=False)
                meta = payload["metadata"]
                cfg, train = meta["model_config"], meta["train_config"]
                rule = cfg.get("validation_selection", {})
                if (meta.get("checkpoint_kind") != "validation_selected" or rule.get("nfe") != budget
                        or rule.get("evaluation_fingerprint") != fingerprint
                        or rule.get("max_cells") != max_cells or int(train["steps"]) != steps
                        or meta["bundle_fingerprint"] != status["training_fingerprint"]):
                    raise ValueError(f"Run matched --fit --select-checkpoint audits: {pairing}, {method}, seed {seed}")
                if records[0].get("checkpoint_modified_ns") != path.stat().st_mtime_ns:
                    raise ValueError("Checkpoint changed after its audit; rerun that audit: " + str(path))
                configs[pairing, seed, method] = (cfg, train)
                selections.append({"pairing": pairing, "seed": seed, "method": method,
                                   "selected_update": meta["checkpoint_step"], "training_updates": train["steps"],
                                   "selection_nfe": budget, "checkpoint": str(path.resolve())})
                del payload
        statuses[pairing] = status
    if statuses["Non-OT"]["evaluation_settings"] != statuses["OT"]["evaluation_settings"]:
        raise ValueError("The OT and Non-OT audits use different evaluation protocols.")
    for seed in seeds:
        for method in ("count_flow_map", "count_fm"):
            if configs["Non-OT", seed, method] != configs["OT", seed, method]:
                raise ValueError(f"Training/selection settings differ between pairings: {method}, seed {seed}")
    return pd.DataFrame(selections)


def read_exports(export_dirs, audit_dirs, expected):
    """Read only complete, matched exports and require every requested seed/NFE."""
    statuses, frames, raw_frames = {}, [], []
    for pairing in PAIRINGS:
        folder = Path(export_dirs[pairing])
        status = read_json(folder / "paper_run_status.json")
        audit = read_json(Path(audit_dirs[pairing]) / "run_status.json")
        if status.get("status") != "completed" or audit.get("status") != "completed":
            raise ValueError("Incomplete export/audit: " + pairing)
        if (status["source_audit_run_id"] != audit["run_id"] or
                Path(status["source_audit_output"]).resolve() != Path(audit_dirs[pairing]).resolve()):
            raise ValueError("The source audit changed; set REFRESH_EVALUATION=True: " + pairing)
        for key, value in expected.items():
            if status["config"].get(key) != value:
                raise ValueError(f"Export setting {key} changed; set REFRESH_EVALUATION=True: {pairing}")
        if status["complete_baseline_set"] != (pairing == "OT"):
            raise ValueError("Export application baselines once, in the OT export.")
        metrics = pd.read_csv(folder / "metrics.csv")
        raw = pd.read_csv(folder / "main_quality_table_raw.csv")
        conditions = pd.read_csv(folder / "condition_metrics.csv")
        for frame in (metrics, raw, conditions):
            if set(frame.run_id) != {status["run_id"]}:
                raise ValueError("Mixed/stale export files: " + pairing)
        for method in ITERATIVE:
            grid = expected["unit_nfe"] if method == FM_UNIT else expected["nfe"]
            subset = metrics[metrics.method == method]
            if set(subset.seed) != set(expected["seeds"]):
                raise ValueError("Missing training seeds: " + method)
            for seed in expected["seeds"]:
                for split in ("val", "test"):
                    actual = subset[(subset.seed == seed) & (subset.split == split)]
                    if sorted(actual.nfe.tolist()) != grid:
                        raise ValueError(f"Missing/duplicate NFE rows: {pairing}, {method}, {seed}, {split}")
        if pairing == "OT":
            for method in BASELINES:
                rows = raw[raw.reported_method == method]
                if method in (baselines.LINEAR_LABEL, baselines.SINKHORN_LABEL):
                    valid = len(rows) == 1 and rows.seed.isna().all()
                else:
                    valid = len(rows) == len(expected["seeds"]) and set(rows.seed) == set(expected["seeds"])
                if not valid:
                    raise ValueError("Missing/duplicate baseline runs: " + method)
        # Verify the metric is a condition average, not an average weighted by cell count.
        for row in metrics.itertuples():
            sub = conditions[(conditions.method == row.method) & (conditions.split == row.split)]
            sub = sub[sub.seed.isna()] if pd.isna(row.seed) else sub[sub.seed == row.seed]
            sub = sub[sub.nfe.isna()] if pd.isna(row.nfe) else sub[sub.nfe == row.nfe]
            if len(sub) != status["evaluation_rows"][row.split]["conditions"] or sub.condition_id.duplicated().any():
                raise ValueError("Incomplete per-condition evaluation: " + row.method)
            for key in METRICS[:-1]:
                values = sub[key].to_numpy(float)
                if not np.isfinite(values).all() or not np.isclose(values.mean(), getattr(row, key)):
                    raise ValueError("Invalid condition-averaged metric: " + key + ", " + row.method)
        metrics["pairing"] = pairing
        raw["pairing"] = np.where(raw.method.isin(ITERATIVE), pairing, "—")
        frames.append(metrics)
        raw_frames.append(raw)
        statuses[pairing] = status
    left, right = statuses["Non-OT"], statuses["OT"]
    for key in ("evaluation_fingerprint", "pca_config", "metric_seed", "evaluation_rows"):
        if left[key] != right[key]:
            raise ValueError("OT/non-OT evaluation mismatch: " + key)
    for split in ("val", "test"):
        arrays = [np.load(Path(export_dirs[p]) / f"{split}_row_indices.npy") for p in PAIRINGS]
        if not np.array_equal(*arrays):
            raise ValueError("OT/non-OT evaluated different cells: " + split)
    with np.load(Path(export_dirs["Non-OT"]) / "paper_projections.npz") as a, \
            np.load(Path(export_dirs["OT"]) / "paper_projections.npz") as b:
        for key in ("count_mean", "count_components", "effect_mean", "effect_components"):
            if not np.allclose(a[key], b[key], rtol=1e-6, atol=1e-7):
                raise ValueError("OT/non-OT used different training-only projections: " + key)
    return pd.concat(frames, ignore_index=True), pd.concat(raw_frames, ignore_index=True), statuses


def summarize_table(raw):
    """One metric per training seed enters the sample SD (ddof=1)."""
    rows = []
    order = [(p, method) for p in PAIRINGS for method in CORE_ROWS]
    order += [("—", method) for method in BASELINES]
    for pairing, method in order:
        frame = raw[(raw.pairing == pairing) & (raw.reported_method == method)]
        if frame.empty or frame.seed.dropna().duplicated().any():
            raise ValueError("Missing/duplicate table runs: " + pairing + ", " + method)
        row = {"pairing": pairing, "reported_method": method, "n_runs": len(frame)}
        budgets = frame.reported_nfe.dropna().astype(int)
        row["budget"] = "1-pass" if budgets.empty else (
            str(budgets.min()) if budgets.min() == budgets.max() else f"{budgets.min()}–{budgets.max()}")
        if method.endswith("(validation-selected)"):
            row["budget"] += " (val.)"
        for key in METRICS:
            if not np.isfinite(frame[key]).all():
                raise ValueError("Non-finite table metric: " + method + ", " + key)
            row[key + "_mean"] = frame[key].mean()
            row[key + "_std"] = frame[key].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def formatted_table(summary, latex=False):
    rows = []
    for record in summary.to_dict("records"):
        name = record["reported_method"]
        if name.startswith(CFM):
            name = CFM
        else:
            name = name.replace(" (validation-selected)", "").replace(" (nearest-dose)", "")
            name = name.replace(" (dose interpolation)", "")
            name = name.replace(FM_UNIT, "Count-FM (unit jump)").replace(FM_TAU, "Count-FM (τ-leap)")
        if latex:
            name = name.replace("τ", r"$\tau$")
        row = {"Method": name, "Pairing": record["pairing"], "NFE / budget": record["budget"]}
        for key, label, digits in zip(METRICS, ("SW2 ↓", "DEG r ↑", "Top overlap ↑", "Time (s) ↓"), (3, 3, 3, 2)):
            mean, sd = record[key + "_mean"], record[key + "_std"]
            value = f"{mean:.{digits}f}"
            if record["n_runs"] > 1 and np.isfinite(sd):
                value += (r" $\pm$ " if latex else " ± ") + f"{sd:.{digits}f}"
            row[label] = value
        rows.append(row)
    result = pd.DataFrame(rows)
    if latex:
        result.columns = ["Method", "Pairing", "NFE", r"SW$_2\downarrow$", r"DEG $r\uparrow$",
                          r"Top overlap$\uparrow$", r"Time (s)$\downarrow$"]
        result = result.replace({"—": "--"}, regex=False)
        result["NFE"] = result["NFE"].str.replace("–", "--", regex=False).str.replace(" (val.)", "", regex=False)
    return result


def load_snapshot(folder, seed, nfe):
    folder = Path(folder)
    status = read_json(folder / "paper_run_status.json")
    with np.load(folder / "snapshots" / f"cfm_seed{seed}_nfe{nfe}.npz", allow_pickle=False) as arrays:
        snapshot = {key: arrays[key].copy() for key in arrays.files}
    if json.loads(str(snapshot["metadata_json"].item()))["run_id"] != status["run_id"]:
        raise ValueError("Snapshot does not belong to the current export.")
    return snapshot


def correlation(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else np.nan


def slope(x, y):
    denominator = np.dot(x, x)
    return float(np.dot(x, y) / denominator) if denominator > 1e-12 else np.nan


def main_figure(metrics, export_dirs, figure_seed):
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.7), constrained_layout=True)
    colors = {CFM: "#0072B2", FM_UNIT: "#D55E00", FM_TAU: "#009E73"}
    labels = {CFM: CFM, FM_UNIT: "Count-FM, unit jump", FM_TAU: "Count-FM, τ-leap"}
    styles = {"Non-OT": "--", "OT": "-"}
    curves = metrics[(metrics.split == "test") & metrics.method.isin(ITERATIVE)]
    for (pairing, method), group in curves.groupby(["pairing", "method"]):
        curve = group.groupby("nfe").agg(score=("sliced_w2", "mean"), sd=("sliced_w2", "std"),
                                         seconds=("generation_seconds", "mean")).sort_index()
        err = curve.sd.to_numpy() if curve.sd.notna().all() else None
        for ax, x in zip(axes[:2], (curve.index, curve.seconds)):
            ax.errorbar(x, curve.score, yerr=err, color=colors[method], linestyle=styles[pairing],
                        marker="o", markersize=3.5, linewidth=1.4, capsize=2)
    axes[0].set(xscale="log", xlabel="NFE", ylabel=r"Sliced $W_2$ ↓", title="A  Quality versus NFE")
    ticks = sorted(curves.nfe.unique())
    axes[0].set_xticks(ticks, [str(int(n)) for n in ticks])
    axes[0].tick_params(axis="x", labelsize=7)
    axes[1].set(xscale="log", xlabel="Generation time (s)", ylabel=r"Sliced $W_2$ ↓",
                title="B  Quality versus time")
    handles = [Line2D([0], [0], color=colors[m], label=labels[m]) for m in ITERATIVE]
    handles += [Line2D([0], [0], color="black", linestyle=styles[p], label=p) for p in PAIRINGS]
    axes[0].legend(handles=handles, fontsize=7, loc="best")
    for ax in axes[:2]:
        ax.grid(alpha=.2)
    snapshots = {p: load_snapshot(export_dirs[p], figure_seed, 1) for p in PAIRINGS}
    a, b = (snapshots[p] for p in PAIRINGS)
    if not np.array_equal(a["condition_ids"], b["condition_ids"]) or not np.array_equal(a["observed_effect"], b["observed_effect"]):
        raise ValueError("Effect comparisons use different observed conditions.")
    upper = 0.
    for pairing, color, marker in (("Non-OT", "#777777", "o"), ("OT", "#0072B2", "^")):
        snapshot = snapshots[pairing]
        observed = np.linalg.norm(snapshot["observed_effect"], axis=1)
        generated = np.linalg.norm(snapshot["generated_effect"], axis=1)
        label = f"{pairing}: r={correlation(observed, generated):.2f}, slope={slope(observed, generated):.2f}"
        axes[2].scatter(observed, generated, color=color, marker=marker, s=25, alpha=.8, label=label)
        upper = max(upper, observed.max(), generated.max())
    upper = max(upper * 1.06, 1e-6)
    axes[2].plot([0, upper], [0, upper], "k--", linewidth=1)
    axes[2].set(xlim=(0, upper), ylim=(0, upper), xlabel="Observed effect magnitude",
                ylabel="Generated effect magnitude", title="C  One-step effect recovery")
    axes[2].legend(fontsize=7, loc="upper left")
    return fig


def appendix_figure(ot_export, figure_seed):
    snapshots = {n: load_snapshot(ot_export, figure_seed, n) for n in (1, 16)}
    first = snapshots[1]
    second = snapshots[16]
    if not np.array_equal(first["condition_ids"], second["condition_ids"]) or not np.array_equal(first["observed_effect"], second["observed_effect"]):
        raise ValueError("Appendix panels must use the same observed cells and condition order.")
    status = read_json(Path(ot_export) / "paper_run_status.json")
    with np.load(Path(ot_export) / "paper_projections.npz") as projection:
        if str(projection["run_id"].item()) != status["run_id"]:
            raise ValueError("Projection does not belong to the current export.")
        mean, components = projection["effect_mean"], projection["effect_components"]
    order = np.argsort(-np.linalg.norm(first["observed_effect"], axis=1), kind="stable")
    names = [str(first["condition_names"][i]).replace(":dose=", " · ").replace(":", " · ") for i in order]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, max(7., 3.4 + .16 * len(order))),
                             gridspec_kw={"height_ratios": [1, 1.15]}, constrained_layout=True)
    points, all_correlations = [], []
    for column, nfe in enumerate((1, 16)):
        snapshot = snapshots[nfe]
        observed, generated = snapshot["observed_effect"], snapshot["generated_effect"]
        a, b = (observed - mean) @ components.T, (generated - mean) @ components.T
        points.extend([a, b])
        ax = axes[0, column]
        for x, y in zip(a, b):
            ax.plot([x[0], y[0]], [x[1], y[1]], color="gray", alpha=.35, linewidth=.7)
        ax.scatter(a[:, 0], a[:, 1], s=25, color="#0072B2", label="Observed")
        ax.scatter(b[:, 0], b[:, 1], s=28, marker="^", color="#D55E00", label="Generated")
        ax.set(xlabel="Effect PC1", ylabel="Effect PC2", title=f"OT · {nfe} NFE")
        ax.legend(fontsize=8)
        values = np.asarray([correlation(x, y) for x, y in zip(observed, generated)])
        all_correlations.extend(values[np.isfinite(values)])
        ax = axes[1, column]
        ax.barh(np.arange(len(order)), values[order], color="#0072B2")
        ax.set_yticks(np.arange(len(order)), names if column == 0 else [""] * len(order), fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Gene-effect Pearson r")
        ax.grid(axis="x", alpha=.15)
    points = np.concatenate(points)
    low, high = points.min(0), points.max(0)
    pad = np.maximum(.07 * (high - low), 1e-6)
    corr_low = min(0., min(all_correlations, default=0.) - .05)
    for ax in axes[0]:
        ax.set(xlim=(low[0] - pad[0], high[0] + pad[0]), ylim=(low[1] - pad[1], high[1] + pad[1]))
    for ax in axes[1]:
        ax.set_xlim(corr_low, 1.)
    fig.suptitle(f"Count Flow Map with OT pairing · seed {figure_seed}", fontsize=12)
    return fig


def save_figure(fig, output, name):
    folder = Path(output) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(folder / (name + "." + extension), dpi=250, bbox_inches="tight")


def write_tables(output, raw, summary, selections, seeds):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "paper_main_table_per_seed.csv", index=False)
    summary.to_csv(output / "paper_main_table_summary.csv", index=False)
    formatted_table(summary).to_csv(output / "paper_main_table.csv", index=False)
    selections.to_csv(output / "checkpoint_selection.csv", index=False)
    uncertainty = (f"Mean $\\pm$ sample standard deviation over {len(seeds)} training seeds."
                   if len(seeds) > 1 else "One training seed; across-training-seed uncertainty is unavailable.")
    caption = ("Held-out dose-condition prediction. " + uncertainty +
               " Metrics are first averaged equally across test conditions within each seed. "
               "Both count models use the stated pairing. Checkpoints are selected on validation at 1 NFE "
               "for Count Flow Map and 128 binomial tau-leap steps for Count-FM, separately for each pairing and seed. "
               "Count-FM sampling budgets are then selected on validation for each sampler; its two samplers share weights. "
               "Time is median warm generation time for the full evaluated test split within each seed, "
               "excluding setup and metrics. Linear and Sinkhorn baselines are each evaluated once and have no training-seed SD.")
    latex = formatted_table(summary, latex=True).to_latex(index=False, escape=False)
    (output / "paper_scrna_table.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\resizebox{\\linewidth}{!}{%\n" + latex + "}\n\\caption{" + caption
        + "}\n\\label{tab:scrna-endpoint-app}\n\\end{table}\n")


def write_figure_captions(output, seeds, figure_seed):
    error = f"Error bars show one sample standard deviation over {len(seeds)} training seeds." if len(seeds) > 1 else "Curves use one training seed and have no across-seed error bars."
    captions = [
        ("paper_scrna_main", "fig:scrna-efficiency",
         "Held-out single-cell quality, generation cost, and effect recovery. "
         "A--B: color identifies model/sampler and solid/dashed lines indicate OT/non-OT training pairing. "
         + error + " C: Count Flow Map at fixed 1 NFE for both pairings, "
         + f"seed {figure_seed}; every test condition is shown. "
         "Effect magnitudes are norms of log1p mean-count differences relative to matched controls; "
         "the dashed identity line indicates equality of the estimated magnitudes."),
        ("paper_scrna_appendix", "fig:scrna-all-heldout-effects",
         f"Count Flow Map with OT pairing at fixed 1 and 16 NFE, seed {figure_seed}. "
         "Top: observed and generated condition effects in a PCA fitted only to training-condition effects, "
         "with common axes. Bottom: all-gene effect correlations for every test condition in the same observed-strength order. "
         "These correlations differ from the top-200-gene logFC correlations reported in the main table.")]
    blocks = ["\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/"
              + name + ".pdf}\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\end{figure}"
              for name, label, caption in captions]
    (Path(output) / "paper_scrna_figures.tex").write_text("\n\n".join(blocks) + "\n")
