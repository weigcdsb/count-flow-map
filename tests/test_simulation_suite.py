from __future__ import annotations

import ast
from pathlib import Path

import torch

from countflow.baselines import (
    CountsDiffModel,
    D3PMModel,
    DiscreteFlowMapModel,
    sample_binomial_tau_leap,
    sample_original_unit_jump,
    sample_countsdiff,
    sample_d3pm,
    sample_simplex_flow_map,
)
from countflow.benchmark_data import benchmark_settings, choose_categorical_support
from countflow.model import CountFlowMap


def test_factor_benchmark_and_support_are_valid() -> None:
    settings = benchmark_settings("smoke")
    target = settings["scale_8_low"].target
    samples = target.sample(128)
    assert samples.shape == (128, 8)
    assert samples.dtype == torch.long
    assert (samples >= 0).all()
    c_max, overflow = choose_categorical_support(
        target, tail_probability=0.05, pilot_samples=1000, seed=11
    )
    assert c_max >= 1
    assert 0.0 <= overflow <= 0.08


def test_original_unit_jump_is_nonnegative() -> None:
    setting = benchmark_settings("smoke")["exact_2d"]
    model = CountFlowMap(dim=2, hidden_dim=16, depth=3, n_mixtures=2, count_scale=24.0)
    samples = sample_original_unit_jump(
        model, setting.source, 64, 3, tau=0.9, device="cpu"
    )
    assert samples.shape == (64, 2)
    assert (samples >= 0).all()


def test_binomial_tau_leap_is_nonnegative() -> None:
    setting = benchmark_settings("smoke")["exact_2d"]
    model = CountFlowMap(dim=2, hidden_dim=16, depth=3, n_mixtures=2, count_scale=24.0)
    samples = sample_binomial_tau_leap(
        model, setting.source, 64, 3, tau=0.9, device="cpu"
    )
    assert samples.shape == (64, 2)
    assert (samples >= 0).all()


def test_countsdiff_reverse_sampler_shapes() -> None:
    model = CountsDiffModel(dim=3, hidden_dim=16, depth=3, count_scale=8.0)
    samples = sample_countsdiff(model, 32, 4, device="cpu")
    assert samples.shape == (32, 3)
    assert samples.dtype == torch.long
    assert (samples >= 0).all()


def test_d3pm_gaussian_transitions_and_skip_posteriors_are_valid() -> None:
    model = D3PMModel(
        dim=2,
        n_categories=17,
        diffusion_steps=32,
        hidden_dim=16,
        depth=3,
    )
    assert torch.all(model.q_onestep >= 0)
    assert torch.allclose(
        model.q_onestep.sum(-1),
        torch.ones_like(model.q_onestep.sum(-1)),
        atol=1e-5,
    )
    xt = torch.randint(0, 17, (7, 2))
    p_x0 = torch.softmax(torch.randn(7, 2, 17), dim=-1)
    posterior = model.posterior_skip(xt, p_x0, s=8, t=32)
    assert torch.allclose(posterior.sum(-1), torch.ones(7, 2), atol=1e-6)
    samples = sample_d3pm(model, 16, 4, c_max=15, device="cpu")
    assert samples.shape == (16, 2)
    assert (samples >= 0).all()


def test_d3pm_first_reverse_step_uses_predicted_x0_distribution() -> None:
    model = D3PMModel(dim=2, n_categories=11, diffusion_steps=16, hidden_dim=16, depth=3)
    xt = torch.randint(0, 11, (5, 2))
    p_x0 = torch.softmax(torch.randn(5, 2, 11), dim=-1)
    t = torch.ones(5, dtype=torch.long)
    posterior = model.posterior_one_step(xt, p_x0, t)
    assert torch.allclose(posterior, p_x0, atol=1e-7)


def test_discrete_flow_map_preserves_identity_and_generates_counts() -> None:
    model = DiscreteFlowMapModel(dim=2, n_categories=8, hidden_dim=16, depth=3)
    x = model.sample_prior(10, torch.device("cpu"))
    t = torch.rand(10) * 0.8
    identity = model.flow(x, t, t)
    assert torch.allclose(identity, x, atol=1e-6)
    samples = sample_simplex_flow_map(model, 10, 2, c_max=6, device="cpu")
    assert samples.shape == (10, 2)
    assert (samples >= 0).all()


def test_new_python_files_parse_with_python39_grammar() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "countflow" / "baselines").glob("*.py")) + [
        root / "countflow" / "benchmark_data.py",
        root / "countflow" / "simulation_metrics.py",
        root / "countflow" / "simulation_runner.py",
    ]
    for path in paths:
        ast.parse(path.read_text(), filename=str(path), feature_version=(3, 9))


def test_submission_simulation_notebook_has_paper_settings_and_inline_config() -> None:
    import json

    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks" / "1_simulation.ipynb").read_text())
    sources = "\n".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source", []), list)
        else cell.get("source", "")
        for cell in notebook["cells"]
    )
    assert "Part 1" in sources
    assert "Part 2" in sources
    assert 'make_config("exact_2d")' in sources
    assert 'make_config("scale_32_high")' in sources
    assert "RUN_MODE" not in sources
    assert "simulation_paper.json" not in sources
    assert "TRAIN_STEPS = 10_000" in sources
    assert "NFE_VALUES = [1, 2, 4, 16, 64, 128, 256]" in sources
    assert "categorical_flow_map" not in sources
    assert '"auxiliary_weight": 0.001' in sources
    assert '"intermediate_times": [0.25, 0.50, 0.75, 0.98]' in sources
