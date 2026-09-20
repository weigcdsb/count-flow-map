from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from tqdm import trange

import torch
from torch import nn

from .bridge import conditional_birth_death_rates, sample_signed_binomial_bridge
from .losses import consistency_nll, diagonal_loss
from .model import CountFlowMap
from .utils import make_ema_copy, resolve_device, set_seed, update_ema


@dataclass
class TrainConfig:
    steps: int = 4000
    batch_size: int = 256
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    tau: float = 0.98
    ck_weight: float = 1.0
    ema_decay: float = 0.995
    grad_clip: float = 5.0
    log_every: int = 100
    seed: int = 42
    ck_warmup_steps: int = 250
    max_span_start: float = 0.1
    max_span_end: float = 0.98
    span_warmup_steps: int = 2000


def _sample_time_triples(
    batch_size: int,
    tau: float,
    device: torch.device,
    max_span: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample CK times uniformly over map duration.

    Draw Delta uniformly from [0,tau], then draw a valid start time uniformly
    from [0,tau-Delta].  This gives equal training mass to short, medium, and
    long maps while remaining translation-neutral in the start time.
    """
    upper = float(tau) if max_span is None else min(float(tau), float(max_span))
    delta = torch.rand(batch_size, device=device) * upper
    s = torch.rand(batch_size, device=device) * (float(tau) - delta)
    t = s + delta
    u = 0.5 * (s + t)
    return s, u, t


def train_count_flow_map(
    model: CountFlowMap,
    coupling: object,
    config: TrainConfig,
    device: Optional[str] = None,
    verbose: bool = True,
    progress: bool = False,
    progress_desc: str = "Count Flow Map",
) -> Tuple[CountFlowMap, CountFlowMap, Dict[str, list]]:
    """Joint one-stage optimization of diagonal and CK losses."""
    set_seed(config.seed)
    torch_device = resolve_device(device)
    model = model.to(torch_device)
    ema_model = make_ema_copy(model).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: Dict[str, list] = {
        "step": [],
        "total": [],
        "diag": [],
        "ck": [],
        "ck_weight": [],
    }

    model.train()
    iterator = trange(1, config.steps + 1, desc=progress_desc, leave=True, dynamic_ncols=True) if progress else range(1, config.steps + 1)
    for step in iterator:
        warmup = min(step / max(config.ck_warmup_steps, 1), 1.0)
        alpha = config.ck_weight * warmup

        x0, x1 = coupling.sample(config.batch_size, device=torch_device)
        r = torch.rand(config.batch_size, device=torch_device) * config.tau
        x_r = sample_signed_binomial_bridge(x0, x1, r)
        target_birth, target_death = conditional_birth_death_rates(x_r, x1, r)
        loss_diag, _, _ = diagonal_loss(
            model, x_r, r, target_birth, target_death
        )

        x0_ck, x1_ck = coupling.sample(config.batch_size, device=torch_device)
        span_progress = min(step / max(config.span_warmup_steps, 1), 1.0)
        max_span = config.max_span_start + span_progress * (
            config.max_span_end - config.max_span_start
        )
        s, u, t = _sample_time_triples(
            config.batch_size,
            config.tau,
            torch_device,
            max_span=max_span,
        )
        x_s = sample_signed_binomial_bridge(x0_ck, x1_ck, s)
        with torch.no_grad():
            z = ema_model.sample(x_s, s, u)
            y = ema_model.sample(z, u, t)
        loss_ck = consistency_nll(model, x_s, y, s, t)

        loss = loss_diag + alpha * loss_ck
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        update_ema(ema_model, model, config.ema_decay)

        if step == 1 or step % config.log_every == 0 or step == config.steps:
            history["step"].append(step)
            history["total"].append(float(loss.detach().cpu()))
            history["diag"].append(float(loss_diag.detach().cpu()))
            history["ck"].append(float(loss_ck.detach().cpu()))
            history["ck_weight"].append(float(alpha))
            if progress:
                iterator.set_postfix(total=f"{history['total'][-1]:.3f}", diag=f"{history['diag'][-1]:.3f}")
            elif verbose:
                print(
                    "step={:5d} total={:9.4f} diag={:9.4f} ck={:9.4f} "
                    "alpha={:.3f}".format(
                        step,
                        history["total"][-1],
                        history["diag"][-1],
                        history["ck"][-1],
                        alpha,
                    )
                )
    ema_model.eval()
    return model, ema_model, history
