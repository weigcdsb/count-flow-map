from __future__ import annotations

from typing import Tuple

import torch

from .model import CountFlowMap


def generalized_kl_rate_loss(
    target_birth: torch.Tensor,
    target_death: torch.Tensor,
    model_birth: torch.Tensor,
    model_death: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    birth_loss = model_birth - target_birth * torch.log(model_birth + eps)
    death_loss = model_death - target_death * torch.log(model_death + eps)
    return (birth_loss + death_loss).sum(dim=-1).mean()


def diagonal_loss(
    model: CountFlowMap,
    xt: torch.Tensor,
    t: torch.Tensor,
    target_birth: torch.Tensor,
    target_death: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model_birth, model_death, _ = model.local_rates(xt, t)
    loss = generalized_kl_rate_loss(
        target_birth, target_death, model_birth, model_death
    )
    return loss, model_birth, model_death


def consistency_nll(
    model: CountFlowMap,
    x_s: torch.Tensor,
    y_target: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    return -model.log_prob(y_target, x_s, s, t).mean()
