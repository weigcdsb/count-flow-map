from __future__ import annotations

import torch

from countflow.bridge import conditional_birth_death_rates, sample_signed_binomial_bridge
from countflow.model import CountFlowMap


def test_bridge_endpoints() -> None:
    x0 = torch.tensor([[0, 3], [5, 1]])
    x1 = torch.tensor([[4, 1], [2, 7]])
    assert torch.equal(sample_signed_binomial_bridge(x0, x1, torch.zeros(2)), x0)
    assert torch.equal(sample_signed_binomial_bridge(x0, x1, torch.ones(2)), x1)


def test_conditional_rates_point_toward_endpoint() -> None:
    xt = torch.tensor([[2, 5]])
    x1 = torch.tensor([[7, 1]])
    birth, death = conditional_birth_death_rates(xt, x1, torch.tensor([0.5]))
    assert torch.allclose(birth, torch.tensor([[10.0, 0.0]]))
    assert torch.allclose(death, torch.tensor([[0.0, 8.0]]))


def test_kernel_identity_and_nonnegative_samples() -> None:
    model = CountFlowMap(dim=2, hidden_dim=32, n_mixtures=2, count_scale=10.0)
    x = torch.tensor([[0, 2], [4, 1], [3, 3]])
    t = torch.tensor([0.2, 0.5, 0.8])
    assert torch.equal(model.sample(x, t, t), x)
    assert torch.allclose(model.log_prob(x, x, t, t), torch.zeros(3))
    assert (model.sample(x, torch.zeros(3), torch.ones(3) * 0.9) >= 0).all()


def test_diagonal_tangent() -> None:
    torch.manual_seed(0)
    model = CountFlowMap(dim=2, hidden_dim=32, n_mixtures=3, count_scale=10.0)
    x = torch.tensor([[3, 2]], dtype=torch.long)
    t = torch.tensor([0.4])
    h = 1e-5
    birth, death, _ = model.local_rates(x, t)
    plus = x.clone()
    plus[:, 0] += 1
    minus = x.clone()
    minus[:, 1] -= 1
    birth_estimate = model.log_prob(plus, x, t, t + h).exp() / h
    death_estimate = model.log_prob(minus, x, t, t + h).exp() / h
    assert torch.allclose(birth_estimate, birth[:, 0], rtol=3e-2, atol=3e-2)
    assert torch.allclose(death_estimate, death[:, 1], rtol=3e-2, atol=3e-2)


def test_ck_time_sampler_is_uniform_span_midpoint() -> None:
    from countflow.training import _sample_time_triples

    torch.manual_seed(0)
    tau = 0.98
    s, u, t = _sample_time_triples(20_000, tau, torch.device("cpu"))
    assert ((0.0 <= s) & (s <= u) & (u <= t) & (t <= tau)).all()
    assert torch.allclose(u, 0.5 * (s + t))
    assert abs(float((t - s).mean()) - tau / 2.0) < 0.01
    assert abs(float(((t - s) > 0.8 * tau).float().mean()) - 0.2) < 0.02
