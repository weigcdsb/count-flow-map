from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..utils import make_ema_copy, resolve_device, set_seed
from .common import CountBackbone, GenericTrainConfig, append_history, make_optimizer, optimizer_step, progress_steps


def survival_probability(t: torch.Tensor) -> torch.Tensor:
    return torch.cos(0.5 * math.pi * t).pow(2)


def counts_diff_weight(t: torch.Tensor) -> torch.Tensor:
    return 0.5 * math.pi * torch.sin(math.pi * t)


class CountsDiffModel(nn.Module):
    """CountsDiff remaining-count predictor on N_0."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        count_scale: float = 20.0,
        output_floor: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.output_floor = float(output_floor)
        self.backbone = CountBackbone(
            dim=dim,
            output_dim=dim,
            hidden_dim=hidden_dim,
            depth=depth,
            count_scale=count_scale,
        )

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.backbone(xt, t)) + self.output_floor


def train_countsdiff(
    model: CountsDiffModel,
    target: object,
    config: GenericTrainConfig,
    *,
    device: Optional[str] = None,
    time_epsilon: float = 1e-4,
    verbose: bool = True,
    progress: bool = False,
    progress_desc: str = "CountsDiff",
) -> Tuple[CountsDiffModel, CountsDiffModel, Dict[str, list]]:
    """Train the continuous-time CountsDiff objective from the paper."""
    set_seed(config.seed)
    torch_device = resolve_device(device)
    model = model.to(torch_device)
    ema = make_ema_copy(model).to(torch_device)
    optimizer = make_optimizer(model, config)
    history: Dict[str, list] = {"step": [], "loss": []}
    model.train()
    iterator = progress_steps(config.steps, enabled=progress, desc=progress_desc)
    for step in iterator:
        x0 = target.sample(config.batch_size, device=torch_device)
        t = time_epsilon + (1.0 - 2.0 * time_epsilon) * torch.rand(
            config.batch_size, device=torch_device
        )
        p = survival_probability(t)[:, None]
        xt = torch.binomial(x0.float(), p.expand_as(x0.float())).long()
        removed = (x0 - xt).float()
        prediction = model(xt, t)
        weight = counts_diff_weight(t)[:, None]
        loss = (weight * (prediction - removed * torch.log(prediction))).sum(-1).mean()
        optimizer_step(loss, model, optimizer, ema, config)
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            append_history(history, step=step, loss=float(loss.detach().cpu()))
            if progress:
                iterator.set_postfix(loss=f"{history['loss'][-1]:.3f}")
            elif verbose:
                print(f"[CountsDiff] step={step:6d} loss={history['loss'][-1]:10.4f}")
    ema.eval()
    return model, ema, history


@torch.no_grad()
def randomized_round(
    value: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    value = value.clamp_min(0.0)
    lower = torch.floor(value)
    fraction = value - lower
    bernoulli = torch.bernoulli(fraction, generator=generator)
    return (lower + bernoulli).long()


@torch.no_grad()
def sample_countsdiff(
    model: CountsDiffModel,
    n_samples: int,
    n_steps: int,
    *,
    device: Optional[str] = None,
    eta_rescale: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """CountsDiff reverse process with optional attrition/remasking."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if eta_rescale < 0:
        raise ValueError("eta_rescale must be nonnegative.")
    torch_device = resolve_device(device)
    model = model.to(torch_device).eval()
    x = torch.zeros(n_samples, model.dim, device=torch_device, dtype=torch.long)
    grid = torch.linspace(1.0, 0.0, int(n_steps) + 1, device=torch_device)
    eps = 1e-8
    for index in range(int(n_steps)):
        t_scalar = grid[index]
        s_scalar = grid[index + 1]
        t = t_scalar.expand(n_samples)
        predicted_missing = randomized_round(model(x, t), generator=generator)
        p_t = survival_probability(t_scalar).clamp(0.0, 1.0)
        p_s = survival_probability(s_scalar).clamp(0.0, 1.0)
        if float(p_t) <= eps:
            sigma_max = torch.tensor(1.0, device=torch_device)
        else:
            sigma_max = torch.minimum(
                torch.tensor(1.0, device=torch_device),
                (1.0 - p_s) / p_t.clamp_min(eps),
            )
        sigma = (float(eta_rescale) * sigma_max).clamp(0.0, sigma_max)
        beta = (p_s - (1.0 - sigma) * p_t) / (1.0 - p_t).clamp_min(eps)
        beta = beta.clamp(0.0, 1.0)
        survivors = torch.binomial(
            x.float(), torch.full_like(x.float(), float(1.0 - sigma)), generator=generator
        ).long()
        births = torch.binomial(
            predicted_missing.float(),
            torch.full_like(predicted_missing.float(), float(beta)),
            generator=generator,
        ).long()
        x = survivors + births
    return x
