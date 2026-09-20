from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..bridge import conditional_birth_death_rates, sample_signed_binomial_bridge
from ..losses import generalized_kl_rate_loss
from ..model import FourierTimeEmbedding, MLP
from ..utils import make_ema_copy, resolve_device, set_seed
from .common import GenericTrainConfig, append_history, make_optimizer, optimizer_step, progress_steps


class CountRateModel(nn.Module):
    """Standalone Count-FM model with the same local-rate architecture as Count Flow Map."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        count_scale: float = 20.0,
        rate_floor: float = 1e-5,
        n_time_frequencies: int = 8,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.count_scale = float(count_scale)
        self.rate_floor = float(rate_floor)
        self.time_embedding = FourierTimeEmbedding(n_time_frequencies)
        time_dim = self.time_embedding.output_dim
        self.state_encoder = MLP(2 * self.dim, hidden_dim, hidden_dim, depth=2)
        self.rate_net = MLP(hidden_dim + time_dim, hidden_dim, 2 * self.dim, depth=depth)

    def _encode_state(self, x: torch.Tensor) -> torch.Tensor:
        import math
        x_float = x.float()
        features = torch.cat(
            [
                torch.log1p(x_float) / math.log1p(self.count_scale),
                x_float / self.count_scale,
            ],
            dim=-1,
        )
        return self.state_encoder(features)

    def local_rates(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must have shape [batch,{self.dim}].")
        t = torch.as_tensor(t, dtype=torch.float32, device=x.device)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        state = self._encode_state(x)
        raw = self.rate_net(torch.cat([state, self.time_embedding(t)], dim=-1))
        birth_raw, death_coefficient_raw = raw.chunk(2, dim=-1)
        birth = F.softplus(birth_raw) + self.rate_floor
        death_coefficient = F.softplus(death_coefficient_raw) + self.rate_floor
        death = x.float() * death_coefficient
        return birth, death, death_coefficient

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        birth, death, _ = self.local_rates(x, t)
        return birth, death


def train_count_rate_model(
    model: CountRateModel,
    coupling: object,
    config: GenericTrainConfig,
    *,
    tau: float = 0.98,
    device: Optional[str] = None,
    verbose: bool = True,
    progress: bool = False,
    progress_desc: str = "Count-FM",
) -> Tuple[CountRateModel, CountRateModel, Dict[str, list]]:
    set_seed(config.seed)
    torch_device = resolve_device(device)
    model = model.to(torch_device)
    ema = make_ema_copy(model).to(torch_device)
    optimizer = make_optimizer(model, config)
    history: Dict[str, list] = {"step": [], "loss": []}
    model.train()
    iterator = progress_steps(config.steps, enabled=progress, desc=progress_desc)
    for step in iterator:
        x0, x1 = coupling.sample(config.batch_size, device=torch_device)
        t = torch.rand(config.batch_size, device=torch_device) * float(tau)
        xt = sample_signed_binomial_bridge(x0, x1, t)
        target_birth, target_death = conditional_birth_death_rates(xt, x1, t)
        birth, death, _ = model.local_rates(xt, t)
        loss = generalized_kl_rate_loss(target_birth, target_death, birth, death)
        optimizer_step(loss, model, optimizer, ema, config)
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            append_history(history, step=step, loss=float(loss.detach().cpu()))
            if progress:
                iterator.set_postfix(loss=f"{history['loss'][-1]:.3f}")
            elif verbose:
                print(f"[Count-FM] step={step:6d} loss={history['loss'][-1]:10.4f}")
    ema.eval()
    return model, ema, history




@torch.no_grad()
def sample_original_unit_jump(
    rate_model: object,
    source: object,
    n_samples: int,
    n_steps: int,
    *,
    tau: float = 0.98,
    device: Optional[str] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Original Count-FM at-most-one-jump-per-coordinate sampler."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    torch_device = resolve_device(device)
    try:
        model_device = next(rate_model.parameters()).device
        if model_device != torch_device:
            rate_model = rate_model.to(torch_device)
    except (AttributeError, StopIteration):
        pass
    rate_model.eval()
    x = source.sample(n_samples, device=torch_device, generator=generator).long()
    h = float(tau) / int(n_steps)
    for step in range(int(n_steps)):
        time = torch.full(
            (n_samples,), step * h, dtype=torch.float32, device=torch_device
        )
        birth, death, _ = rate_model.local_rates(x, time)
        total = birth + death
        jump_probability = 1.0 - torch.exp(-h * total)
        birth_probability = torch.where(
            total > 0, jump_probability * birth / total, torch.zeros_like(total)
        )
        death_probability = torch.where(
            total > 0, jump_probability * death / total, torch.zeros_like(total)
        )
        uniforms = torch.rand(
            total.shape, device=torch_device, generator=generator, dtype=total.dtype
        )
        update = torch.zeros_like(x)
        update[uniforms < death_probability] = -1
        update[uniforms > 1.0 - birth_probability] = 1
        x = (x + update).clamp_min(0)
    return x


@torch.no_grad()
def sample_binomial_tau_leap(
    rate_model: object,
    source: object,
    n_samples: int,
    n_steps: int,
    *,
    tau: float = 0.98,
    device: Optional[str] = None,
    generator: Optional[torch.Generator] = None,
    death_rule: str = "linear",
) -> torch.Tensor:
    """Sample Count-FM using a nonnegative binomial tau-leap.

    Births use a Poisson leap. Deaths use a binomial leap and therefore cannot
    exceed the available count. The default linear probability

        p_i = min(h * mu_i / max(x_i,1), 1)

    is the standard binomial tau-leap construction. The optional exponential
    rule uses p_i = 1-exp(-h*mu_i/x_i).
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    torch_device = resolve_device(device)
    try:
        model_device = next(rate_model.parameters()).device
        if model_device != torch_device:
            rate_model = rate_model.to(torch_device)
    except (AttributeError, StopIteration):
        pass
    rate_model.eval()
    x = source.sample(n_samples, device=torch_device, generator=generator).long()
    h = float(tau) / int(n_steps)
    for step in range(int(n_steps)):
        time = torch.full(
            (n_samples,), step * h, dtype=torch.float32, device=torch_device
        )
        birth, death, death_coefficient = rate_model.local_rates(x, time)
        births = torch.poisson((h * birth).clamp_min(0.0), generator=generator).long()
        if death_rule == "linear":
            death_prob = (h * death_coefficient).clamp(0.0, 1.0)
        elif death_rule == "exponential":
            death_prob = (-torch.expm1(-h * death_coefficient)).clamp(0.0, 1.0)
        else:
            raise ValueError("death_rule must be 'linear' or 'exponential'.")
        deaths = torch.binomial(x.float(), death_prob, generator=generator).long()
        x = x - deaths + births
    return x
