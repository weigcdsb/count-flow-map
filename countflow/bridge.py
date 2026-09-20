from __future__ import annotations

from typing import Tuple

import torch


def _expand_time(t: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    t = torch.as_tensor(t, dtype=torch.float32, device=device)
    if t.ndim == 0:
        t = t.expand(batch_size)
    if t.ndim != 1 or t.shape[0] != batch_size:
        raise ValueError("t must be scalar or have shape [batch].")
    return t


def sample_signed_binomial_bridge(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Sample X_t = x0 + sign(x1-x0) Binomial(|x1-x0|, t)."""
    if x0.shape != x1.shape or x0.ndim != 2:
        raise ValueError("x0 and x1 must have the same shape [batch, dim].")
    if (x0 < 0).any() or (x1 < 0).any():
        raise ValueError("Counts must be nonnegative.")
    batch_size = x0.shape[0]
    t = _expand_time(t, batch_size, x0.device)
    if ((t < 0.0) | (t > 1.0)).any():
        raise ValueError("Bridge times must lie in [0, 1].")
    displacement = x1.long() - x0.long()
    total_count = displacement.abs().float()
    probs = t[:, None].expand_as(total_count)
    completed = torch.distributions.Binomial(total_count=total_count, probs=probs).sample().long()
    return x0.long() + displacement.sign() * completed


def conditional_birth_death_rates(
    xt: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Closed-form endpoint-conditioned rates for the signed-binomial bridge."""
    if xt.shape != x1.shape or xt.ndim != 2:
        raise ValueError("xt and x1 must have the same shape [batch, dim].")
    t = _expand_time(t, xt.shape[0], xt.device)
    denominator = (1.0 - t).clamp_min(eps)[:, None]
    birth = (x1.float() - xt.float()).clamp_min(0.0) / denominator
    death = (xt.float() - x1.float()).clamp_min(0.0) / denominator
    return birth, death
