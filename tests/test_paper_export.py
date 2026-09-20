import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from countflow.application_data import make_scrna_transport_surrogate, save_pair_bundle
from countflow import application_runner as app
from countflow import scrna_response_baselines as baselines
from scripts.export_scrna_paper import CFM, FM_TAU, FM_UNIT, choose_budgets, main_table_rows, main


def test_paper_selection_uses_validation_and_preserves_fixed_budgets():
    rows = []
    for method in (CFM, FM_TAU, FM_UNIT):
        for split, values in (("val", [3., 1., 1.]), ("test", [.1, 9., 2.])):
            rows.extend(dict(method=method, seed=42, split=split, nfe=nfe, sliced_w2=value)
                        for nfe, value in zip([1, 16, 128], values))
    frame = pd.DataFrame(rows)
    choices = choose_budgets(frame)
    assert set(choices.selected_nfe) == {16}  # tied validation scores favor less compute
    table = main_table_rows(frame, choices)
    assert table.set_index("reported_method").loc[CFM + " (1 NFE)", "sliced_w2"] == .1
    assert table.set_index("reported_method").loc[CFM + " (16 NFE)", "sliced_w2"] == 9.
    assert table.set_index("reported_method").loc[FM_TAU + " (validation-selected)", "reported_nfe"] == 16
    frame.loc[frame.split == "test", "sliced_w2"] = -999.
    pd.testing.assert_frame_equal(choose_budgets(frame), choices)


def make_paper_fixture(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    bundle = make_scrna_transport_surrogate(dim=12, n_cell_lines=1, n_drugs=2,
                                            cells_per_condition=40, seed=73)
    bundle.metadata["bundle_fingerprint"] = "paper-export-fixture"
    data = folder / "data.npz"
    save_pair_bundle(data, bundle)
    cfg = {"hidden_dim": 16, "context_hidden_dim": 16, "context_output_dim": 16,
           "depth": 2, "n_mixtures": 2, "latent_dim": 4, "autoencoder_width": 16,
           "autoencoder_depth": 2, "adversary_width": 8, "adversary_depth": 1,
           "doser_width": 8, "doser_depth": 1}
    audit = folder / "audit"
    baseline = folder / "old_paper/checkpoints"
    records = []
    builders = {"count_flow_map": app._build_flow_map, "count_fm": app._build_count_fm,
                "scgen_vae": baselines.build_scgen, "cpa": baselines.build_cpa, "nbvae": baselines.build_nbvae}
    paths = []
    for kind, builder in builders.items():
        set_directory = audit / "checkpoints" if kind in ("count_flow_map", "count_fm") else baseline
        path = set_directory / "42" / f"{kind}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(42)
        model = builder(bundle, cfg)
        torch.save({"state_dict": model.state_dict(), "metadata": {
            "method": kind, "model_config": cfg, "bundle_fingerprint": "paper-export-fixture",
            "train_config": {"tau": .98}}}, path)
        paths.append(path)
        if kind in ("count_flow_map", "count_fm"):
            records.append({"method": kind, "seed": 42, "checkpoint": str(path)})
    (audit / "run_status.json").write_text(json.dumps({"status": "completed", "run_id": "fixture-audit",
        "evaluation_fingerprint": "paper-export-fixture", "training_fingerprint": "paper-export-fixture",
        "models_used": records}))
    return data, audit, baseline, paths


def test_complete_paper_export_reuses_models_and_saves_scored_effects(tmp_path):
    data, audit, baseline, paths = make_paper_fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in paths}
    output = tmp_path / "paper"
    main(["--data", str(data), "--audit-output", str(audit), "--baseline-checkpoint-dir", str(baseline),
          "--output", str(output), "--device", "cpu", "--seeds", "42", "--nfe", "1", "16",
          "--unit-nfe", "1", "16", "--max-cells", "6", "--timing-repeats", "1", "--batch-size", "32"])
    assert before == {str(p): p.read_bytes() for p in paths}
    status = json.loads((output / "paper_run_status.json").read_text())
    assert status["status"] == "completed" and status["neural_training_performed"] is False
    assert status["complete_baseline_set"] and len(status["models"]) == 5
    raw = pd.read_csv(output / "main_quality_table_raw.csv")
    summary = pd.read_csv(output / "main_quality_table_summary.csv")
    assert len(raw) == len(summary) == 10
    assert summary.n_runs.eq(1).all()
    assert summary.sliced_w2_std.isna().all()
    assert np.isfinite(raw.sliced_w2).all()
    assert (raw.generation_seconds > 0).all()
    conditions = pd.read_csv(output / "condition_metrics.csv")
    with np.load(output / "snapshots/cfm_seed42_nfe16.npz") as p:
        assert str(json.loads(str(p["metadata_json"].item()))["run_id"]) == status["run_id"]
        expected_ids = conditions[(conditions.method == CFM) & (conditions.split == "test") & (conditions.nfe == 16)].condition_id
        assert set(p["condition_ids"]) == set(expected_ids)
        for cid, observed, generated in zip(p["condition_ids"], p["observed_effect"], p["generated_effect"]):
            row = conditions[(conditions.method == CFM) & (conditions.split == "test") &
                             (conditions.nfe == 16) & (conditions.condition_id == cid)].iloc[0]
            slope = np.dot(observed, generated) / np.dot(observed, observed)
            assert np.isclose(slope, row.log1p_effect_slope)
    assert set(raw.run_id) == {status["run_id"]}
