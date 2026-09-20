import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

from countflow import scrna_response_baselines as baselines
from scripts.scrna_paper_v2 import (
    PAIRINGS,
    CORE_ROWS,
    BASELINES,
    METRICS,
    summarize_table,
    formatted_table,
)


def test_summary_uses_training_seed_sd_and_preserves_one_off_baselines():
    rows = []
    for pairing in PAIRINGS:
        for method in CORE_ROWS:
            for seed, score in zip([42, 123, 2026], [1.0, 3.0, 5.0]):
                rows.append(
                    {
                        "pairing": pairing,
                        "reported_method": method,
                        "seed": seed,
                        "reported_nfe": 1 if "(1 NFE)" in method else 16,
                        **{key: score for key in METRICS},
                    }
                )
    for method in BASELINES:
        one_off = method in (baselines.LINEAR_LABEL, baselines.SINKHORN_LABEL)
        values = [(np.nan, 7.0)] if one_off else [(42, 1.0), (123, 3.0), (2026, 5.0)]
        for seed, score in values:
            rows.append(
                {
                    "pairing": "—",
                    "reported_method": method,
                    "seed": seed,
                    "reported_nfe": np.nan,
                    **{key: score for key in METRICS},
                }
            )
    raw = pd.DataFrame(rows)
    summary = summarize_table(raw)
    assert len(summary) == 14
    repeated = summary[summary.n_runs == 3]
    assert len(repeated) == 12
    assert np.allclose(repeated.sliced_w2_mean, 3.0)
    assert np.allclose(repeated.sliced_w2_std, 2.0)
    assert summary.loc[summary.n_runs == 1, "sliced_w2_std"].isna().all()
    table = formatted_table(summary)
    assert (table.iloc[:12]["SW2 ↓"].str.contains("±")).sum() == 12
    assert "±" not in table.iloc[-1]["SW2 ↓"]
    with pytest.raises(ValueError, match="duplicate"):
        summarize_table(pd.concat([raw, raw.iloc[:1]], ignore_index=True))


def test_baseline_update_notebook_is_clean_and_points_to_matched_config():
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "notebooks" / "2_scrna_drug_transport_v2.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    assert all(
        not cell.outputs and cell.execution_count is None
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    sources = "\n".join(cell.source for cell in notebook.cells)
    assert "update_scrna_baselines.py" in sources
    assert "configs/scrna_baselines_matched.json" in sources
    assert "Count Flow Map and Count-FM are read-only" in sources


def test_matched_baseline_config_matches_manuscript_architecture():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs" / "scrna_baselines_matched.json").read_text())
    architecture = config["architecture"]
    assert architecture["hidden_layers"] == 3
    assert architecture["hidden_dim"] == 512
    assert architecture["activation"].lower() == "silu"
    assert architecture["batch_norm"] is False
    assert architecture["dropout"] == 0.0
    assert architecture["latent_dim"] == 128
    assert config["seeds"] == [42, 123, 2026]
