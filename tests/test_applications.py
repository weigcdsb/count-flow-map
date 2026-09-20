from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from countflow.application_baselines import generate_scrna_empirical_baseline
from countflow.application_data import (
    build_forecasting_bundle_from_counts,
    make_neural_forecasting_surrogate,
    make_scrna_transport_surrogate,
    ConditionalPairBundle,
    ConditionalPairSplit,
)
from countflow.application_metrics import (
    evaluate_neural_forecast_samples,
    evaluate_scrna_by_condition,
    fit_count_pca,
    select_saturated_nfe,
)
from countflow.application_training import (
    ApplicationTrainConfig,
    generate_conditional_count_fm,
    generate_conditional_count_fm_unit_jump,
    generate_conditional_flow_map,
    train_conditional_count_fm,
    train_conditional_flow_map,
)
from countflow.conditional_model import (
    ConditionalCountFlowMap,
    ConditionalCountRateModel,
    HistoryGRUEncoder,
    VectorContextEncoder,
)
from countflow.distributions import PoissonBinomialResidualMixture, ResidualMixtureParams
from countflow.scrna_response_baselines import (
    PublishedBaselineTrainConfig,
    build_cpa,
    build_nbvae,
    build_scgen,
    fit_linear_dose_response,
    fit_sinkhorn_ot,
    generate_cpa,
    generate_linear_dose_response_cached,
    generate_nbvae,
    generate_scgen_cached,
    generate_scvidr_cached,
    generate_sinkhorn_ot_cached,
    prepare_scgen_response_state,
    prepare_scvidr_response_state,
    train_cpa,
    train_nbvae,
    train_scgen,
)


def _full_log_prob(
    y: torch.Tensor,
    x: torch.Tensor,
    params: ResidualMixtureParams,
    eps: float = 1e-8,
) -> torch.Tensor:
    x_float = x.float()
    y_float = y.float()
    max_deaths = int(x.max().item())
    d = torch.arange(max_deaths + 1, dtype=torch.float32).view(1, 1, 1, -1)
    x4 = x_float[:, None, :, None]
    y4 = y_float[:, None, :, None]
    mean4 = params.birth_mean.clamp_min(eps)[:, :, :, None]
    prob4 = params.death_prob.clamp(eps, 1.0 - eps)[:, :, :, None]
    births = y4 - x4 + d
    valid = (d <= x4) & (births >= 0.0)
    log_binom = (
        torch.lgamma(x4 + 1.0)
        - torch.lgamma(d + 1.0)
        - torch.lgamma(x4 - d + 1.0)
        + d * torch.log(prob4)
        + (x4 - d) * torch.log1p(-prob4)
    )
    safe_births = births.clamp_min(0.0)
    log_poisson = (
        safe_births * torch.log(mean4)
        - mean4
        - torch.lgamma(safe_births + 1.0)
    )
    terms = torch.where(valid, log_binom + log_poisson, torch.full_like(log_binom, -torch.inf))
    coordinate = torch.logsumexp(terms, dim=-1)
    component = coordinate.sum(-1)
    result = torch.logsumexp(F.log_softmax(params.mixture_logits, dim=-1) + component, dim=-1)
    identity = params.delta <= 1e-12
    if identity.any():
        exact = torch.where(
            (x == y).all(-1),
            torch.zeros_like(result),
            torch.full_like(result, -torch.inf),
        )
        result = torch.where(identity, exact, result)
    return result


def _vector_map(dim: int = 4, context_dim: int = 3) -> ConditionalCountFlowMap:
    encoder = VectorContextEncoder(context_dim, hidden_dim=16, output_dim=12)
    return ConditionalCountFlowMap(
        dim,
        encoder,
        12,
        hidden_dim=24,
        depth=2,
        n_mixtures=3,
        count_scale=8.0,
        death_chunk_size=2,
    )


def _vector_rate(dim: int = 4, context_dim: int = 3) -> ConditionalCountRateModel:
    encoder = VectorContextEncoder(context_dim, hidden_dim=16, output_dim=12)
    return ConditionalCountRateModel(
        dim,
        encoder,
        12,
        hidden_dim=24,
        depth=2,
        count_scale=8.0,
    )


def test_chunked_residual_log_prob_matches_full_vectorization() -> None:
    torch.manual_seed(3)
    x = torch.tensor([[0, 2, 4], [3, 1, 2]], dtype=torch.long)
    y = torch.tensor([[1, 1, 5], [2, 3, 0]], dtype=torch.long)
    params = ResidualMixtureParams(
        mixture_logits=torch.randn(2, 3),
        birth_mean=torch.rand(2, 3, 3) * 2.0 + 0.05,
        death_prob=torch.rand(2, 3, 3) * 0.8 + 0.05,
        delta=torch.tensor([0.4, 0.7]),
    )
    chunked = PoissonBinomialResidualMixture(death_chunk_size=2).log_prob(y, x, params)
    full = _full_log_prob(y, x, params)
    torch.testing.assert_close(chunked, full, rtol=1e-6, atol=1e-6)


def test_conditional_flow_map_identity_and_diagonal_tangent() -> None:
    torch.manual_seed(7)
    model = _vector_map()
    x = torch.tensor([[0, 2, 5, 1], [3, 1, 0, 4]], dtype=torch.long)
    context = torch.randn(2, 3)
    s = torch.tensor([0.31, 0.57])
    same = model.sample(x, s, s, context)
    assert torch.equal(same, x)
    assert torch.all(model.log_prob(x, x, s, s, context) == 0)

    h = 1e-4
    t = s + h
    params = model.kernel_params(x, s, t, context)
    weights = torch.softmax(params.mixture_logits, dim=-1)
    actual_h = params.delta[:, None]
    mean_birth_per_time = (weights[:, :, None] * params.birth_mean).sum(1) / actual_h
    mean_death_per_time = (
        weights[:, :, None] * (x[:, None, :].float() * params.death_prob)
    ).sum(1) / actual_h
    birth, death, _ = model.local_rates(x, s, context)
    torch.testing.assert_close(mean_birth_per_time, birth, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(mean_death_per_time, death, rtol=1e-3, atol=1e-3)


def test_conditional_models_support_vector_and_history_contexts() -> None:
    torch.manual_seed(11)
    x = torch.poisson(torch.ones(5, 6) * 1.2).long()
    times = torch.rand(5)

    vector_encoder = VectorContextEncoder(4, 12, 10)
    vector_model = ConditionalCountRateModel(6, vector_encoder, 10, hidden_dim=20, depth=2)
    vector_context = torch.randn(5, 4)
    birth, death, coefficient = vector_model(x, times, vector_context)
    assert birth.shape == death.shape == coefficient.shape == x.shape
    assert torch.all(birth > 0) and torch.all(coefficient > 0)
    assert torch.all(death[x == 0] == 0)

    history_encoder = HistoryGRUEncoder(6, 12, 10)
    history_map = ConditionalCountFlowMap(
        6, history_encoder, 10, hidden_dim=20, depth=2, n_mixtures=2
    )
    history = torch.poisson(torch.ones(5, 8, 6) * 0.7)
    params = history_map(x, times * 0.5, times * 0.5 + 0.1, history)
    assert params.birth_mean.shape == (5, 2, 6)
    assert params.death_prob.shape == (5, 2, 6)
    assert torch.isfinite(params.birth_mean).all()


def test_forecasting_splits_do_not_cross_boundaries() -> None:
    counts = np.arange(2 * 40, dtype=np.int64).reshape(2, 40)
    bundle = build_forecasting_bundle_from_counts(
        counts,
        history_bins=4,
        train_end=20,
        val_end=30,
        metadata={"dataset": "boundary_test"},
    )
    time_major = counts.T
    # Validation targets begin at 20 + history_bins, so all history bins are inside validation.
    np.testing.assert_array_equal(bundle.val.context[0].numpy(), time_major[20:24])
    np.testing.assert_array_equal(bundle.val.x1[0].numpy(), time_major[24])
    np.testing.assert_array_equal(bundle.test.context[0].numpy(), time_major[30:34])
    np.testing.assert_array_equal(bundle.test.x1[0].numpy(), time_major[34])


def test_application_metrics_and_saturated_selection_are_finite() -> None:
    scrna = make_scrna_transport_surrogate(
        dim=12, n_cell_lines=2, n_drugs=2, cells_per_condition=40, latent_rank=3
    )
    projection = fit_count_pca(scrna.train.x1, n_components=6)
    aggregate, rows = evaluate_scrna_by_condition(
        scrna.test.x1.clone(),
        scrna.test.x0,
        scrna.test.x1,
        scrna.test.condition_id,
        projection,
    )
    assert rows and aggregate
    for key, value in aggregate.items():
        if key == "condition_id":
            continue
        assert math.isfinite(value)
    assert aggregate["sliced_w2"] < 1e-6
    assert aggregate["logfc_pearson"] > 0.999
    assert aggregate["top_response_gene_overlap"] > 0.999
    assert select_saturated_nfe([1, 2, 4, 8], [0.8, 0.42, 0.405, 0.404]) == 4

    neural = make_neural_forecasting_surrogate(dim=8, n_bins=240, history_bins=5)
    observations = neural.test.x1[:20]
    samples = observations[:, None, :].repeat(1, 6, 1)
    neural_metrics = evaluate_neural_forecast_samples(samples, observations)
    assert math.isfinite(neural_metrics["energy_score"])
    assert neural_metrics["energy_score"] == 0.0


def test_short_conditional_training_and_generation_integration() -> None:
    torch.manual_seed(13)
    bundle = make_scrna_transport_surrogate(
        dim=8, n_cell_lines=1, n_drugs=2, cells_per_condition=32, latent_rank=2
    )
    flow = _vector_map(dim=8, context_dim=bundle.train.context.shape[1])
    rate = _vector_rate(dim=8, context_dim=bundle.train.context.shape[1])
    cfg = ApplicationTrainConfig(
        steps=2,
        batch_size=16,
        learning_rate=5e-4,
        ck_warmup_steps=1,
        span_warmup_steps=1,
        log_every=1,
        seed=13,
    )
    _, flow_ema, flow_history = train_conditional_flow_map(
        flow, bundle.train, cfg, device="cpu", verbose=False
    )
    _, rate_ema, rate_history = train_conditional_count_fm(
        rate, bundle.train, cfg, device="cpu", verbose=False
    )
    assert len(flow_history["step"]) == 2
    assert len(rate_history["step"]) == 2
    x0 = bundle.test.x0[:10]
    context = bundle.test.context[:10]
    flow_samples = generate_conditional_flow_map(
        flow_ema, x0, context, n_steps=2, device="cpu"
    )
    rate_samples = generate_conditional_count_fm(
        rate_ema, x0, context, n_steps=2, device="cpu"
    )
    unit_samples = generate_conditional_count_fm_unit_jump(
        rate_ema, x0, context, n_steps=2, device="cpu"
    )
    assert flow_samples.shape == rate_samples.shape == unit_samples.shape == x0.shape
    assert torch.all(flow_samples >= 0) and torch.all(rate_samples >= 0)
    assert torch.all(unit_samples >= 0)


def test_scrna_published_response_baselines_short_run() -> None:
    bundle = make_scrna_transport_surrogate(
        dim=10, n_cell_lines=3, n_drugs=2, cells_per_condition=18, latent_rank=2, seed=19
    )
    tiny = PublishedBaselineTrainConfig(
        steps=2, batch_size=8, learning_rate=1e-3, weight_decay=0.0, log_every=1, seed=19
    )

    scgen = build_scgen(bundle, {"hidden_dim": 24, "latent_dim": 6, "depth": 2, "dropout": 0.0})
    scgen, _ = train_scgen(scgen, bundle.train, tiny, device="cpu", verbose=False)
    state = prepare_scgen_response_state(scgen, bundle, device="cpu", batch_size=32)
    scgen_draw = generate_scgen_cached(scgen, bundle.test, state, device="cpu", batch_size=32)
    scvidr_state = prepare_scvidr_response_state(bundle, state, ridge=0.0)
    scvidr_draw = generate_scvidr_cached(scgen, bundle.test, scvidr_state, device="cpu", batch_size=32)

    cpa = build_cpa(bundle, {
        "latent_dim": 8, "autoencoder_width": 16, "autoencoder_depth": 1,
        "adversary_width": 8, "adversary_depth": 1, "doser_width": 8, "doser_depth": 1,
    })
    cpa_cfg = PublishedBaselineTrainConfig(steps=4, batch_size=8, learning_rate=1e-3, log_every=1, seed=19)
    cpa, _ = train_cpa(
        cpa, bundle, cpa_cfg,
        {"adversary_steps": 3, "reg_adversary": 1.0, "penalty_adversary": 1.0,
         "autoencoder_lr": 1e-3, "adversary_lr": 1e-3, "doser_lr": 1e-3,
         "autoencoder_wd": 0.0, "adversary_wd": 0.0, "doser_wd": 0.0},
        device="cpu", verbose=False,
    )
    cpa_draw = generate_cpa(cpa, bundle, bundle.test, device="cpu", batch_size=32)

    nbvae = build_nbvae(bundle, {
        "hidden_dim": 16, "context_hidden_dim": 8, "context_output_dim": 8,
        "latent_dim": 4, "depth": 2,
    })
    nbvae, _ = train_nbvae(
        nbvae, bundle.train, tiny, {"kl_warmup_steps": 1}, device="cpu", verbose=False
    )
    nbvae_draw = generate_nbvae(nbvae, bundle.test, seed=19, device="cpu", batch_size=32)

    linear_state = fit_linear_dose_response(bundle)
    linear_draw = generate_linear_dose_response_cached(bundle.test, linear_state)
    sinkhorn_state = fit_sinkhorn_ot(
        bundle, bundle.test, epsilon_scale=0.2, iterations=8, pca_dim=4, max_cells=10, seed=19
    )
    sinkhorn_draw = generate_sinkhorn_ot_cached(bundle, bundle.test, sinkhorn_state, knn=3)

    for draw in (scgen_draw, scvidr_draw, cpa_draw, nbvae_draw, linear_draw, sinkhorn_draw):
        assert draw.shape == bundle.test.x1.shape
        assert draw.dtype == torch.long
        assert torch.all(draw >= 0)


def test_pair_bundle_and_external_prediction_roundtrip(tmp_path) -> None:
    from countflow.application_baselines import (
        load_external_prediction_bundle,
        save_external_prediction_bundle,
    )
    from countflow.application_data import load_pair_bundle, save_pair_bundle

    bundle = make_scrna_transport_surrogate(
        dim=6, n_cell_lines=1, n_drugs=2, cells_per_condition=24, latent_rank=2
    )
    path = tmp_path / "bundle.npz"
    save_pair_bundle(path, bundle)
    loaded = load_pair_bundle(path)
    assert loaded.metadata["dataset"] == bundle.metadata["dataset"]
    torch.testing.assert_close(loaded.test.x1, bundle.test.x1)
    torch.testing.assert_close(loaded.test.context, bundle.test.context)

    prediction_path = tmp_path / "external.npz"
    save_external_prediction_bundle(
        prediction_path,
        generated=bundle.test.x1.numpy(),
        condition_id=bundle.test.condition_id.numpy(),
        metadata={"method": "test", "seed": 1},
    )
    generated, condition_id, metadata = load_external_prediction_bundle(prediction_path)
    torch.testing.assert_close(generated, bundle.test.x1)
    torch.testing.assert_close(condition_id, bundle.test.condition_id)
    assert metadata["method"] == "test"


def test_spikeprophecy_loader_contract(tmp_path) -> None:
    from countflow.application_data import load_spikeprophecy_session
    import json

    counts = np.arange(5 * 80, dtype=np.int64).reshape(5, 80) % 4
    np.save(tmp_path / "session_000.npy", counts)
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "history_bins": 5,
                "bin_width_ms": 50,
                "sessions": [
                    {
                        "split_boundaries": {"train_end": 45, "val_end": 62},
                        "name": "dummy",
                    }
                ],
            }
        )
    )
    bundle = load_spikeprophecy_session(tmp_path, 0, max_units=4)
    assert bundle.dim == 4
    assert bundle.metadata["bin_width_ms"] == 50
    assert bundle.train.context.shape[1:] == (5, 4)
    assert bundle.val.n > 0 and bundle.test.n > 0



def test_scrna_empirical_baselines_use_training_doses_only() -> None:
    dim = 3
    train = ConditionalPairSplit(
        x0=torch.zeros(8, dim, dtype=torch.long),
        x1=torch.tensor([[1, 0, 0]] * 4 + [[5, 0, 0]] * 4, dtype=torch.long),
        context=torch.zeros(8, 1),
        condition_id=torch.tensor([0] * 4 + [1] * 4),
    )
    val = ConditionalPairSplit(
        x0=torch.zeros(2, dim, dtype=torch.long),
        x1=torch.tensor([[3, 0, 0]] * 2, dtype=torch.long),
        context=torch.zeros(2, 1),
        condition_id=torch.tensor([2, 2]),
    )
    test = ConditionalPairSplit(
        x0=torch.zeros(20, dim, dtype=torch.long),
        x1=torch.tensor([[3, 0, 0]] * 20, dtype=torch.long),
        context=torch.zeros(20, 1),
        condition_id=torch.tensor([2] * 20),
    )
    bundle = ConditionalPairBundle(
        train, val, test,
        {
            "conditions": [
                {"condition_id": 0, "split": "train", "cell_line_id": "c", "drug": "d", "dose": 0.1},
                {"condition_id": 1, "split": "train", "cell_line_id": "c", "drug": "d", "dose": 10.0},
                {"condition_id": 2, "split": "test", "cell_line_id": "c", "drug": "d", "dose": 1.0},
            ]
        },
    )
    nearest = generate_scrna_empirical_baseline(bundle, test, strategy="nearest_dose_empirical", seed=1)
    assert set(nearest[:, 0].tolist()).issubset({1, 5})
    interpolated = generate_scrna_empirical_baseline(bundle, test, strategy="dose_interpolated_empirical", seed=2)
    assert set(interpolated[:, 0].tolist()).issubset({1, 5})
    assert 1 in interpolated[:, 0].tolist() and 5 in interpolated[:, 0].tolist()


def test_balanced_panel_selection_does_not_silently_collapse() -> None:
    from scripts.run_scrna_drug_transport import _select_balanced_observed_panel

    conditions = []
    for cell in [f"c{i}" for i in range(5)]:
        for drug in [f"d{j}" for j in range(8)]:
            for dose in (0.1, 1.0, 10.0):
                conditions.append({"cell_line_id": cell, "drug": drug, "dose": dose, "samples": ["s"]})
    selected = _select_balanced_observed_panel({"conditions": conditions})
    assert len(set(c["cell_line_id"] for c in selected["conditions"])) == 5
    assert len(set(c["drug"] for c in selected["conditions"])) == 8


def test_application_notebooks_keep_config_contracts_visible() -> None:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    scrna = json.loads((root / "notebooks" / "2_scrna_drug_transport.ipynb").read_text())
    scrna_sources = "\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else cell.get("source", "")
        for cell in scrna["cells"]
    )
    assert "SCRNA_CONFIG = scrna_runner.SCRNA_CONFIG" in scrna_sources
    assert "[1, 4, 16, 64, 128, 256]" not in scrna_sources or "NFE_VALUES" in scrna_sources
    assert "builtin_baselines" in scrna_sources
    assert "published_baselines" in scrna_sources
    assert "scGen" in scrna_sources and "CPA" in scrna_sources and "Sinkhorn" in scrna_sources
    assert "paper_ready" in scrna_sources
    assert "na_rep=\"—\"" in scrna_sources

    neural = json.loads((root / "notebooks" / "3_neural_forecasting.ipynb").read_text())
    neural_sources = "\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else cell.get("source", "")
        for cell in neural["cells"]
    )
    assert "NEURAL_CONFIG" in neural_sources


def test_scrna_condition_holdout_assignment_preserves_training_coverage() -> None:
    from scripts.run_scrna_drug_transport import _condition_holdout_assignment

    conditions = []
    for cell in ("c0", "c1", "c2"):
        for drug in ("d0", "d1", "d2"):
            for dose in (0.1, 1.0, 10.0):
                conditions.append({"cell_line_id": cell, "drug": drug, "dose": dose, "samples": []})
    assignment = _condition_holdout_assignment(conditions, seed=7)
    assert {"train", "val", "test"}.issubset(set(assignment.values()))

    train = [conditions[i] for i, split in assignment.items() if split == "train"]
    train_cells = {c["cell_line_id"] for c in train}
    train_drugs = {c["drug"] for c in train}
    train_pairs = {(c["cell_line_id"], c["drug"]) for c in train}
    for i, split in assignment.items():
        if split == "train":
            continue
        condition = conditions[i]
        assert condition["cell_line_id"] in train_cells
        assert condition["drug"] in train_drugs
        assert (condition["cell_line_id"], condition["drug"]) in train_pairs

