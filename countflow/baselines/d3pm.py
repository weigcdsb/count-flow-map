from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..utils import make_ema_copy, resolve_device, set_seed
from .common import CategoricalBackbone, GenericTrainConfig, append_history, make_optimizer, optimizer_step, progress_steps


def linear_betas(
    n_steps: int,
    start: float = 1e-4,
    stop: float = 0.02,
    *,
    rescale_to_1000: bool = True,
) -> torch.Tensor:
    """Linear D3PM/DDPM schedule.

    The official Gaussian D3PM image configuration uses 1000 steps with
    beta in [1e-4, 0.02].  When a shorter native horizon is used here, the
    optional 1000/T rescaling preserves approximately the same total noise.
    """
    n_steps = int(n_steps)
    scale = 1000.0 / float(n_steps) if rescale_to_1000 else 1.0
    betas = torch.linspace(float(start) * scale, float(stop) * scale, n_steps, dtype=torch.float64)
    return betas.clamp(min=1e-8, max=0.999)


def gaussian_transition_matrix(
    n_categories: int,
    beta: float,
    transition_bands: Optional[int] = None,
) -> torch.Tensor:
    """Gaussian-like ordinal D3PM transition from Austin et al. (2021).

    This follows the official D3PM construction: off-diagonal probability
    decays with squared ordinal distance and the diagonal is chosen so rows
    and columns sum to one.  The resulting chain has a uniform stationary
    distribution while respecting count ordering.
    """
    k = int(n_categories)
    bands = k - 1 if transition_bands is None else min(int(transition_bands), k - 1)
    beta = float(beta)
    # Count categories are ordinal levels 0,...,K-1. Normalize their
    # distance to [0,2], the Gaussian-D3PM analogue of a continuous
    # coordinate scale, rather than importing the image-specific 0..255 scale.
    values = torch.arange(k, dtype=torch.float64) * (2.0 / float(k - 1))
    values = values[: bands + 1]
    log_weight = -(values.square()) / beta
    symmetric = torch.cat([torch.flip(log_weight[1:], dims=[0]), log_weight], dim=0)
    probs = torch.softmax(symmetric, dim=0)[bands:]

    mat = torch.zeros((k, k), dtype=torch.float64)
    for offset in range(1, bands + 1):
        value = probs[offset]
        idx = torch.arange(k - offset)
        mat[idx, idx + offset] = value
        mat[idx + offset, idx] = value
    diag = 1.0 - mat.sum(dim=1)
    mat[torch.arange(k), torch.arange(k)] = diag
    return mat


class D3PMModel(nn.Module):
    """Ordinal Gaussian D3PM with the hybrid variational + x0 CE loss."""

    def __init__(
        self,
        dim: int,
        n_categories: int,
        diffusion_steps: int = 256,
        hidden_dim: int = 256,
        depth: int = 4,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        transition_bands: Optional[int] = None,
        rescale_betas: bool = True,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_categories = int(n_categories)
        self.diffusion_steps = int(diffusion_steps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.transition_bands = None if transition_bands is None else int(transition_bands)
        self.rescale_betas = bool(rescale_betas)
        self.backbone = CategoricalBackbone(
            dim=dim,
            n_categories=n_categories,
            hidden_dim=hidden_dim,
            depth=depth,
            two_time=False,
        )

        betas = linear_betas(
            self.diffusion_steps,
            self.beta_start,
            self.beta_end,
            rescale_to_1000=self.rescale_betas,
        )
        one_step = torch.stack(
            [
                gaussian_transition_matrix(
                    self.n_categories,
                    float(beta),
                    transition_bands=self.transition_bands,
                )
                for beta in betas
            ],
            dim=0,
        )
        cumulative = [torch.eye(self.n_categories, dtype=torch.float64)]
        current = cumulative[0]
        for transition in one_step:
            current = current @ transition
            cumulative.append(current)

        self.register_buffer("betas", betas.float(), persistent=True)
        self.register_buffer("q_onestep", one_step.float(), persistent=True)
        self.register_buffer("q_cumulative", torch.stack(cumulative, dim=0).float(), persistent=True)
        self._segment_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}

    def forward(self, xt: torch.Tensor, t_index: torch.Tensor) -> torch.Tensor:
        time = t_index.float() / float(self.diffusion_steps)
        return self.backbone(xt, time)

    def q_probs(self, x0: torch.Tensor, t_index: torch.Tensor) -> torch.Tensor:
        """q(x_t | x_0) probabilities, shape [B,D,K]."""
        batch = x0.shape[0]
        qbar = self.q_cumulative[t_index.long()].to(x0.device)  # [B,K,K]
        rows = qbar[
            torch.arange(batch, device=x0.device)[:, None],
            x0.long(),
        ]
        return rows

    def sample_forward(
        self,
        x0: torch.Tensor,
        t_index: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        probs = self.q_probs(x0, t_index)
        return torch.multinomial(
            probs.reshape(-1, self.n_categories),
            1,
            generator=generator,
        ).reshape_as(x0)

    def posterior_one_step(
        self,
        xt: torch.Tensor,
        p_x0: torch.Tensor,
        t_index: torch.Tensor,
    ) -> torch.Tensor:
        """q/p(x_{t-1}|x_t) after marginalizing over p(x_0)."""
        if p_x0.shape != (*xt.shape, self.n_categories):
            raise ValueError("p_x0 has incompatible shape.")
        batch = xt.shape[0]
        s_index = t_index.long() - 1
        qbar_s = self.q_cumulative[s_index].to(xt.device)  # [B,K,K]
        q_s = torch.einsum("bdk,bkj->bdj", p_x0, qbar_s)

        q_step = self.q_onestep[t_index.long() - 1].to(xt.device)  # [B,K,K]
        likelihood = q_step.transpose(1, 2)[
            torch.arange(batch, device=xt.device)[:, None],
            xt.long(),
        ]
        posterior = q_s * likelihood
        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        # D3PM predicts x_start directly at the first noisy timestep.
        first = (t_index.long() == 1)[:, None, None]
        return torch.where(first, p_x0, posterior)

    def segment_transition(self, s: int, t: int, device: torch.device) -> torch.Tensor:
        """Return Q_{s+1}...Q_t for 0 <= s < t <= T."""
        s = int(s)
        t = int(t)
        if not (0 <= s < t <= self.diffusion_steps):
            raise ValueError("Require 0 <= s < t <= diffusion_steps.")
        if t == s + 1:
            return self.q_onestep[s].to(device)
        key = (s, t, str(device))
        cached = self._segment_cache.get(key)
        if cached is not None:
            return cached
        segment = torch.eye(self.n_categories, dtype=torch.float64)
        one_step = self.q_onestep.detach().cpu().double()
        for idx in range(s, t):
            segment = segment @ one_step[idx]
        segment = segment.float().to(device)
        self._segment_cache[key] = segment
        return segment

    def posterior_skip(
        self,
        xt: torch.Tensor,
        p_x0: torch.Tensor,
        *,
        s: int,
        t: int,
    ) -> torch.Tensor:
        """Structured posterior for an arbitrary skipped reverse interval."""
        if int(s) == 0:
            return p_x0
        qbar_s = self.q_cumulative[int(s)].to(xt.device)
        q_s = torch.einsum("bdk,kj->bdj", p_x0, qbar_s)
        segment = self.segment_transition(int(s), int(t), xt.device)
        likelihood = segment.transpose(0, 1)[xt.long()]  # [B,D,K]
        posterior = q_s * likelihood
        return posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def terminal_prior_error(self) -> float:
        uniform = torch.full(
            (self.n_categories,),
            1.0 / float(self.n_categories),
            dtype=self.q_cumulative.dtype,
            device=self.q_cumulative.device,
        )
        qbar = self.q_cumulative[-1]
        return float((0.5 * (qbar - uniform).abs().sum(dim=-1)).mean().item())


def train_d3pm(
    model: D3PMModel,
    target: object,
    config: GenericTrainConfig,
    *,
    c_max: int,
    device: Optional[str] = None,
    auxiliary_weight: float = 0.001,
    verbose: bool = True,
    progress: bool = False,
    progress_desc: str = "D3PM",
) -> Tuple[D3PMModel, D3PMModel, Dict[str, list]]:
    from ..benchmark_data import encode_categories

    set_seed(config.seed)
    torch_device = resolve_device(device)
    model = model.to(torch_device)
    ema = make_ema_copy(model).to(torch_device)
    optimizer = make_optimizer(model, config)
    history: Dict[str, list] = {"step": [], "loss": [], "vb": [], "ce": []}
    model.train()
    iterator = progress_steps(config.steps, enabled=progress, desc=progress_desc)
    for step in iterator:
        x0_count = target.sample(config.batch_size, device=torch_device)
        x0 = encode_categories(x0_count, c_max).to(torch_device)
        t_index = torch.randint(
            1,
            model.diffusion_steps + 1,
            (config.batch_size,),
            device=torch_device,
        )
        xt = model.sample_forward(x0, t_index)
        logits = model(xt, t_index)
        p_x0 = torch.softmax(logits, dim=-1)
        true_x0 = F.one_hot(x0, num_classes=model.n_categories).float()
        true_posterior = model.posterior_one_step(xt, true_x0, t_index)
        model_posterior = model.posterior_one_step(xt, p_x0, t_index)
        vb = (
            true_posterior
            * (
                torch.log(true_posterior.clamp_min(1e-12))
                - torch.log(model_posterior.clamp_min(1e-12))
            )
        ).sum(-1).sum(-1).mean()
        ce = F.cross_entropy(
            logits.reshape(-1, model.n_categories),
            x0.reshape(-1),
            reduction="none",
        ).view(config.batch_size, model.dim).sum(dim=-1).mean()
        loss = vb + float(auxiliary_weight) * ce
        optimizer_step(loss, model, optimizer, ema, config)
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            append_history(
                history,
                step=step,
                loss=float(loss.detach().cpu()),
                vb=float(vb.detach().cpu()),
                ce=float(ce.detach().cpu()),
            )
            if progress:
                iterator.set_postfix(loss=f"{history['loss'][-1]:.3f}")
            elif verbose:
                print(
                    f"[D3PM] step={step:6d} loss={history['loss'][-1]:9.4f} "
                    f"vb={history['vb'][-1]:9.4f} ce={history['ce'][-1]:9.4f}"
                )
    ema.eval()
    return model, ema, history


@torch.no_grad()
def sample_d3pm(
    model: D3PMModel,
    n_samples: int,
    n_steps: int,
    *,
    c_max: int,
    device: Optional[str] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    from ..benchmark_data import decode_categories

    if n_steps <= 0 or n_steps > model.diffusion_steps:
        raise ValueError("n_steps must lie in [1,diffusion_steps].")
    torch_device = resolve_device(device)
    model = model.to(torch_device).eval()
    x = torch.randint(
        0,
        model.n_categories,
        (n_samples, model.dim),
        device=torch_device,
        generator=generator,
    )
    grid = torch.linspace(
        model.diffusion_steps,
        0,
        int(n_steps) + 1,
        device=torch_device,
    ).round().long()
    for index in range(int(n_steps)):
        t_value = int(grid[index].item())
        s_value = int(grid[index + 1].item())
        t_index = torch.full((n_samples,), t_value, device=torch_device, dtype=torch.long)
        logits = model(x, t_index)
        p_x0 = torch.softmax(logits, dim=-1)
        if s_value == 0:
            # Match the original D3PM decoder: the final clean state is the
            # mode of the predicted x_start distribution (no added noise).
            x = p_x0.argmax(dim=-1)
        else:
            posterior = model.posterior_skip(x, p_x0, s=s_value, t=t_value)
            x = torch.multinomial(
                posterior.reshape(-1, model.n_categories),
                1,
                generator=generator,
            ).reshape(n_samples, model.dim)
    return decode_categories(x, c_max)
