from __future__ import annotations

from dataclasses import dataclass
import time as walltime
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn

from .application_data import ConditionalPairSplit
from .bridge import conditional_birth_death_rates, sample_signed_binomial_bridge
from .conditional_model import ConditionalCountFlowMap, ConditionalCountRateModel
from .losses import generalized_kl_rate_loss
from .utils import make_ema_copy, resolve_device, set_seed, update_ema


@dataclass
class ApplicationTrainConfig:
    steps: int = 5000
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    tau: float = 0.98
    ck_weight: float = 1.0
    ck_warmup_steps: int = 500
    max_span_start: float = 0.1
    span_warmup_steps: int = 2500
    ema_decay: float = 0.999
    grad_clip: float = 5.0
    log_every: int = 100
    seed: int = 42


def _sample_time_triples(
    batch_size: int,
    tau: float,
    device: torch.device,
    max_span: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta = torch.rand(batch_size, device=device) * min(float(tau), float(max_span))
    s = torch.rand(batch_size, device=device) * (float(tau) - delta)
    t = s + delta
    return s, 0.5 * (s + t), t


def train_conditional_flow_map(
    model: ConditionalCountFlowMap,
    train: ConditionalPairSplit,
    config: ApplicationTrainConfig,
    *,
    device: Optional[str] = None,
    verbose: bool = True,
    checkpoint_callback: Optional[Callable[[nn.Module, int], None]] = None,
) -> Tuple[ConditionalCountFlowMap, ConditionalCountFlowMap, Dict[str, list]]:
    set_seed(config.seed)
    torch_device = resolve_device(device)
    model = model.to(torch_device)
    ema = make_ema_copy(model).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device=train.x0.device).manual_seed(config.seed + 17)
    history: Dict[str, list] = {
        "step": [],
        "total": [],
        "diag": [],
        "ck": [],
        "ck_weight": [],
    }

    model.train()
    train_start = walltime.perf_counter()
    if verbose:
        print(
            f"[CFM train] device={torch_device} dim={train.dim} "
            f"batch={config.batch_size} steps={config.steps} "
            f"death_chunk_max={model.residual_distribution.death_chunk_size} "
            f"small_count_threshold={model.residual_distribution.small_count_threshold}",
            flush=True,
        )
    for step in range(1, int(config.steps) + 1):
        step_start = walltime.perf_counter()
        x0, x1, context, _ = train.sample(
            config.batch_size, device=torch_device, generator=generator
        )
        bridge_time = torch.rand(config.batch_size, device=torch_device) * config.tau
        x_t = sample_signed_binomial_bridge(x0, x1, bridge_time)
        target_birth, target_death = conditional_birth_death_rates(
            x_t, x1, bridge_time
        )
        model_birth, model_death, _ = model.local_rates(x_t, bridge_time, context)
        loss_diag = generalized_kl_rate_loss(
            target_birth, target_death, model_birth, model_death
        )

        x0_ck, x1_ck, context_ck, _ = train.sample(
            config.batch_size, device=torch_device, generator=generator
        )
        progress = min(step / max(int(config.span_warmup_steps), 1), 1.0)
        max_span = config.max_span_start + progress * (
            config.tau - config.max_span_start
        )
        s, u, t = _sample_time_triples(
            config.batch_size, config.tau, torch_device, max_span
        )
        x_s = sample_signed_binomial_bridge(x0_ck, x1_ck, s)
        with torch.no_grad():
            z = ema.sample(x_s, s, u, context_ck)
            y = ema.sample(z, u, t, context_ck)
        loss_ck = -model.log_prob(y, x_s, s, t, context_ck).mean()
        alpha = config.ck_weight * min(
            step / max(int(config.ck_warmup_steps), 1), 1.0
        )
        loss = loss_diag + alpha * loss_ck

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        update_ema(ema, model, config.ema_decay)
        if checkpoint_callback is not None:
            checkpoint_callback(ema, step)

        if step == 1 or step % config.log_every == 0 or step == config.steps:
            history["step"].append(step)
            history["total"].append(float(loss.detach().cpu()))
            history["diag"].append(float(loss_diag.detach().cpu()))
            history["ck"].append(float(loss_ck.detach().cpu()))
            history["ck_weight"].append(float(alpha))
            if verbose:
                print(
                    f"step={step:6d} total={history['total'][-1]:9.4f} "
                    f"diag={history['diag'][-1]:9.4f} "
                    f"ck={history['ck'][-1]:9.4f} alpha={alpha:.3f} "
                    f"step_s={walltime.perf_counter()-step_start:.2f} "
                    f"elapsed_s={walltime.perf_counter()-train_start:.1f}",
                    flush=True,
                )
    ema.eval()
    return model, ema, history


def train_conditional_count_fm(
    model: ConditionalCountRateModel,
    train: ConditionalPairSplit,
    config: ApplicationTrainConfig,
    *,
    device: Optional[str] = None,
    verbose: bool = True,
    checkpoint_callback: Optional[Callable[[nn.Module, int], None]] = None,
) -> Tuple[ConditionalCountRateModel, ConditionalCountRateModel, Dict[str, list]]:
    set_seed(config.seed)
    torch_device = resolve_device(device)
    model = model.to(torch_device)
    ema = make_ema_copy(model).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device=train.x0.device).manual_seed(config.seed + 29)
    history: Dict[str, list] = {"step": [], "loss": []}
    model.train()
    train_start = walltime.perf_counter()
    if verbose:
        print(
            f"[Count-FM train] device={torch_device} dim={train.dim} "
            f"batch={config.batch_size} steps={config.steps}",
            flush=True,
        )
    for step in range(1, int(config.steps) + 1):
        step_start = walltime.perf_counter()
        x0, x1, context, _ = train.sample(
            config.batch_size, device=torch_device, generator=generator
        )
        time = torch.rand(config.batch_size, device=torch_device) * config.tau
        x_t = sample_signed_binomial_bridge(x0, x1, time)
        target_birth, target_death = conditional_birth_death_rates(x_t, x1, time)
        birth, death, _ = model.local_rates(x_t, time, context)
        loss = generalized_kl_rate_loss(target_birth, target_death, birth, death)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        update_ema(ema, model, config.ema_decay)
        if checkpoint_callback is not None:
            checkpoint_callback(ema, step)
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            history["step"].append(step)
            history["loss"].append(float(loss.detach().cpu()))
            if verbose:
                print(
                    f"step={step:6d} loss={history['loss'][-1]:9.4f} "
                    f"step_s={walltime.perf_counter()-step_start:.2f} "
                    f"elapsed_s={walltime.perf_counter()-train_start:.1f}",
                    flush=True,
                )
    ema.eval()
    return model, ema, history


@torch.no_grad()
def generate_conditional_flow_map(
    model: ConditionalCountFlowMap,
    x0: torch.Tensor,
    context: torch.Tensor,
    *,
    n_steps: int,
    tau: float = 0.98,
    device: Optional[str] = None,
    batch_size: int = 512,
) -> torch.Tensor:
    if n_steps < 1:
        raise ValueError("n_steps must be positive.")
    torch_device = resolve_device(device)
    model = model.to(torch_device).eval()
    output = []
    grid = torch.linspace(0.0, float(tau), int(n_steps) + 1, device=torch_device)
    for first in range(0, x0.shape[0], int(batch_size)):
        last = min(first + int(batch_size), x0.shape[0])
        x = x0[first:last].to(torch_device)
        c = context[first:last].to(torch_device)
        for step in range(int(n_steps)):
            s = grid[step].expand(x.shape[0])
            t = grid[step + 1].expand(x.shape[0])
            x = model.sample(x, s, t, c)
        output.append(x.cpu())
    return torch.cat(output, dim=0)


@torch.no_grad()
def generate_conditional_count_fm_unit_jump(
    model: ConditionalCountRateModel,
    x0: torch.Tensor,
    context: torch.Tensor,
    *,
    n_steps: int,
    tau: float = 0.98,
    device: Optional[str] = None,
    batch_size: int = 512,
) -> torch.Tensor:
    """Original Count-FM at-most-one-unit-jump-per-coordinate sampler.

    This uses the same trained Count-FM rate model as the binomial tau-leap
    sampler.  It changes only the numerical CTMC sampler: at each small step,
    each coordinate can make at most one +1 or -1 jump with probabilities
    induced by the learned birth/death rates.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be positive.")
    torch_device = resolve_device(device)
    model = model.to(torch_device).eval()
    output = []
    h = float(tau) / float(n_steps)
    for first in range(0, x0.shape[0], int(batch_size)):
        last = min(first + int(batch_size), x0.shape[0])
        x = x0[first:last].to(torch_device).long()
        c = context[first:last].to(torch_device)
        for step in range(int(n_steps)):
            time = torch.full(
                (x.shape[0],),
                float(step) * h,
                device=torch_device,
                dtype=torch.float32,
            )
            birth, death, _ = model.local_rates(x, time, c)
            total = birth + death
            jump_probability = 1.0 - torch.exp(-h * total)
            birth_probability = torch.where(
                total > 0.0,
                jump_probability * birth / total.clamp_min(1e-12),
                torch.zeros_like(total),
            )
            death_probability = torch.where(
                total > 0.0,
                jump_probability * death / total.clamp_min(1e-12),
                torch.zeros_like(total),
            )
            uniforms = torch.rand_like(total)
            update = torch.zeros_like(x)
            update[uniforms < death_probability] = -1
            update[uniforms > 1.0 - birth_probability] = 1
            x = (x + update).clamp_min(0)
        output.append(x.cpu())
    return torch.cat(output, dim=0)


@torch.no_grad()
def generate_conditional_count_fm(
    model: ConditionalCountRateModel,
    x0: torch.Tensor,
    context: torch.Tensor,
    *,
    n_steps: int,
    tau: float = 0.98,
    device: Optional[str] = None,
    batch_size: int = 512,
) -> torch.Tensor:
    """Binomial tau-leap generation from learned Count-FM rates."""
    if n_steps < 1:
        raise ValueError("n_steps must be positive.")
    torch_device = resolve_device(device)
    model = model.to(torch_device).eval()
    output = []
    grid = torch.linspace(0.0, float(tau), int(n_steps) + 1, device=torch_device)
    for first in range(0, x0.shape[0], int(batch_size)):
        last = min(first + int(batch_size), x0.shape[0])
        x = x0[first:last].to(torch_device).long()
        c = context[first:last].to(torch_device)
        for step in range(int(n_steps)):
            time = grid[step].expand(x.shape[0])
            h = float(grid[step + 1] - grid[step])
            birth, _, death_coefficient = model.local_rates(x, time, c)
            births = torch.poisson((h * birth).clamp_min(0.0)).long()
            death_probability = -torch.expm1(-h * death_coefficient)
            death_probability = death_probability.clamp(0.0, 1.0 - 1e-7)
            deaths = torch.distributions.Binomial(
                total_count=x.float(), probs=death_probability
            ).sample().long()
            x = x - deaths + births
        output.append(x.cpu())
    return torch.cat(output, dim=0)
