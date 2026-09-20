from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..benchmark_data import decode_categories, encode_categories
from ..utils import make_ema_copy, resolve_device, set_seed, update_ema
from .common import CategoricalBackbone, GenericTrainConfig, append_history, make_optimizer, optimizer_step, progress_steps


class _SimplexFlowMap(nn.Module):
    def __init__(
        self,
        dim: int,
        n_categories: int,
        hidden_dim: int = 256,
        depth: int = 4,
        prior: str = "gaussian",
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_categories = int(n_categories)
        self.prior = str(prior)
        if self.prior not in {"gaussian", "discrete_uniform"}:
            raise ValueError("prior must be 'gaussian' or 'discrete_uniform'.")
        self.backbone = CategoricalBackbone(
            dim=dim,
            n_categories=n_categories,
            hidden_dim=hidden_dim,
            depth=depth,
            two_time=True,
        )

    def logits(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.backbone(x, s, t)

    def endpoint(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(x, s, t), dim=-1)

    def flow(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        endpoint: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = x.shape[0]
        s = torch.as_tensor(s, dtype=torch.float32, device=x.device)
        t = torch.as_tensor(t, dtype=torch.float32, device=x.device)
        if s.ndim == 0:
            s = s.expand(batch)
        if t.ndim == 0:
            t = t.expand(batch)
        endpoint = self.endpoint(x, s, t) if endpoint is None else endpoint
        gamma = ((t - s) / (1.0 - s).clamp_min(1e-6)).clamp(0.0, 1.0)
        return (1.0 - gamma[:, None, None]) * x + gamma[:, None, None] * endpoint

    def sample_prior(
        self,
        batch_size: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.prior == "gaussian":
            return torch.randn(
                batch_size,
                self.dim,
                self.n_categories,
                device=device,
                generator=generator,
            )
        index = torch.randint(
            0,
            self.n_categories,
            (batch_size, self.dim),
            device=device,
            generator=generator,
        )
        return F.one_hot(index, num_classes=self.n_categories).float()


class DiscreteFlowMapModel(_SimplexFlowMap):
    """Discrete Flow Maps using the mean-denoiser semigroup (PSD) objective."""


def _sample_diagonal_batch(
    model: _SimplexFlowMap,
    target: object,
    batch_size: int,
    c_max: int,
    device: torch.device,
    time_epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x1_count = target.sample(batch_size, device=device)
    x1 = encode_categories(x1_count, c_max).to(device)
    x1_one_hot = F.one_hot(x1, num_classes=model.n_categories).float()
    x0 = model.sample_prior(batch_size, device)
    t = torch.rand(batch_size, device=device) * (1.0 - time_epsilon)
    xt = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1_one_hot
    return x0, x1, x1_one_hot, xt, t


def _categorical_nll_per_sample(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch, dim, n_categories = logits.shape
    return F.cross_entropy(
        logits.reshape(-1, n_categories),
        target.reshape(-1),
        reduction="none",
    ).view(batch, dim).sum(dim=-1)


def _adaptive_kl_weight(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    r: float,
    c: float,
) -> torch.Tensor:
    mismatch = (student - teacher).pow(2).sum(dim=-1).mean(dim=-1)
    return (mismatch.detach() + float(c)).pow(-float(r))


def _priority_gradient_step(
    diagonal_loss: torch.Tensor,
    distillation_loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: nn.Module,
    config: GenericTrainConfig,
    distillation_weight: float,
) -> None:
    """Give diagonal gradients priority and project conflicting PSD gradients."""
    parameters: List[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    diagonal_gradients = torch.autograd.grad(
        diagonal_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    distillation_gradients = torch.autograd.grad(
        float(distillation_weight) * distillation_loss,
        parameters,
        allow_unused=True,
    )
    dot = torch.zeros((), device=diagonal_loss.device)
    diagonal_norm = torch.zeros((), device=diagonal_loss.device)
    for g_diag, g_distill in zip(diagonal_gradients, distillation_gradients):
        if g_diag is not None:
            diagonal_norm = diagonal_norm + g_diag.detach().pow(2).sum()
        if g_diag is not None and g_distill is not None:
            dot = dot + (g_diag.detach() * g_distill.detach()).sum()
    projection = torch.where(
        dot < 0,
        dot / diagonal_norm.clamp_min(1e-12),
        torch.zeros_like(dot),
    )
    for parameter, g_diag, g_distill in zip(
        parameters, diagonal_gradients, distillation_gradients
    ):
        if g_diag is None and g_distill is None:
            parameter.grad = None
            continue
        if g_diag is None:
            combined = g_distill
        elif g_distill is None:
            combined = g_diag
        else:
            combined = g_diag + g_distill - projection * g_diag
        parameter.grad = combined.detach().clone()
    nn.utils.clip_grad_norm_(parameters, config.grad_clip)
    optimizer.step()
    update_ema(ema, model, config.ema_decay)


def train_discrete_flow_map(
    model: DiscreteFlowMapModel,
    target: object,
    config: GenericTrainConfig,
    *,
    c_max: int,
    device: Optional[str] = None,
    consistency_weight: float = 1.0,
    diagonal_steps: Optional[int] = None,
    distillation_steps: Optional[int] = None,
    adaptive_r: float = 0.5,
    adaptive_c: float = 0.01,
    gradient_surgery: bool = True,
    time_epsilon: float = 1e-3,
    verbose: bool = True,
    progress: bool = False,
    progress_desc: str = "Discrete Flow Maps",
) -> Tuple[DiscreteFlowMapModel, DiscreteFlowMapModel, Dict[str, list]]:
    """Train DFM using diagonal pretraining followed by PSD distillation.

    This follows the paper's staged protocol. During PSD training, the diagonal
    objective remains active and (by default) receives gradient priority via a
    PCGrad-style projection, preventing the semigroup objective from destroying
    the learned diagonal denoiser.
    """
    if diagonal_steps is None and distillation_steps is None:
        diagonal_steps = max(1, int(round(0.8 * config.steps)))
        distillation_steps = max(1, int(config.steps) - int(diagonal_steps))
    elif diagonal_steps is None or distillation_steps is None:
        raise ValueError("Specify both diagonal_steps and distillation_steps, or neither.")
    diagonal_steps = int(diagonal_steps)
    distillation_steps = int(distillation_steps)
    if diagonal_steps <= 0 or distillation_steps <= 0:
        raise ValueError("Both DFM stages must have positive update counts.")

    set_seed(config.seed)
    torch_device = resolve_device(device)
    model = model.to(torch_device)
    ema = make_ema_copy(model).to(torch_device)
    optimizer = make_optimizer(model, config)
    history: Dict[str, list] = {
        "step": [], "loss": [], "diagonal": [], "psd": [], "stage": []
    }
    model.train()
    total_steps = diagonal_steps + distillation_steps

    iterator = progress_steps(total_steps, enabled=progress, desc=progress_desc)
    for step in iterator:
        x0, x1, x1_one_hot, xt, diagonal_t = _sample_diagonal_batch(
            model, target, config.batch_size, c_max, torch_device, time_epsilon
        )
        diagonal_logits = model.logits(xt, diagonal_t, diagonal_t)
        diagonal_probabilities = torch.softmax(diagonal_logits, dim=-1)
        one_hot = F.one_hot(x1, num_classes=model.n_categories).float()
        diagonal_per = _categorical_nll_per_sample(diagonal_logits, x1)
        adaptive_weight = _adaptive_kl_weight(
            diagonal_probabilities,
            one_hot,
            r=adaptive_r,
            c=adaptive_c,
        )
        diagonal_loss = (adaptive_weight * diagonal_per).mean()

        psd = torch.zeros((), device=torch_device)
        if step <= diagonal_steps:
            loss = diagonal_loss
            optimizer_step(loss, model, optimizer, ema, config)
            stage = 0.0
        else:
            s = torch.rand(config.batch_size, device=torch_device) * (1.0 - time_epsilon)
            t = s + torch.rand(config.batch_size, device=torch_device) * (
                1.0 - time_epsilon - s
            )
            u = 0.5 * (s + t)
            xs = (1.0 - s[:, None, None]) * x0 + s[:, None, None] * x1_one_hot
            psi_st = model.endpoint(xs, s, t)
            with torch.no_grad():
                psi_su = ema.endpoint(xs, s, u)
                x_su = ema.flow(xs, s, u, endpoint=psi_su)
                psi_ut = ema.endpoint(x_su, u, t)
                denominator = ((1.0 - u) * (t - s)).clamp_min(1e-8)
                alpha = (1.0 - t) * (u - s) / denominator
                beta = (t - u) * (1.0 - s) / denominator
                target_psi = (
                    alpha[:, None, None] * psi_su + beta[:, None, None] * psi_ut
                )
                target_psi = target_psi / target_psi.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-10)
            psd_per = (
                target_psi
                * (
                    torch.log(target_psi.clamp_min(1e-10))
                    - torch.log(psi_st.clamp_min(1e-10))
                )
            ).sum(dim=-1).sum(dim=-1)
            psd_weight = _adaptive_kl_weight(
                psi_st,
                target_psi,
                r=adaptive_r,
                c=adaptive_c,
            )
            psd = (psd_weight * psd_per).mean()
            loss = diagonal_loss + float(consistency_weight) * psd
            if gradient_surgery:
                _priority_gradient_step(
                    diagonal_loss,
                    psd,
                    model,
                    optimizer,
                    ema,
                    config,
                    consistency_weight,
                )
            else:
                optimizer_step(loss, model, optimizer, ema, config)
            stage = 1.0

        if step == 1 or step % config.log_every == 0 or step in {diagonal_steps, total_steps}:
            append_history(
                history,
                step=step,
                loss=float(loss.detach().cpu()),
                diagonal=float(diagonal_loss.detach().cpu()),
                psd=float(psd.detach().cpu()),
                stage=stage,
            )
            if progress:
                label = "diag" if step <= diagonal_steps else "psd"
                iterator.set_postfix(stage=label, loss=f"{history['loss'][-1]:.3f}")
            elif verbose:
                label = "diag" if step <= diagonal_steps else "psd"
                print(
                    f"[Discrete FM:{label}] step={step:6d}/{total_steps} "
                    f"loss={history['loss'][-1]:9.4f} "
                    f"diag={history['diagonal'][-1]:8.4f} psd={history['psd'][-1]:8.4f}"
                )
    ema.eval()
    return model, ema, history


@torch.no_grad()
def sample_simplex_flow_map(
    model: _SimplexFlowMap,
    n_samples: int,
    n_steps: int,
    *,
    c_max: int,
    device: Optional[str] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    torch_device = resolve_device(device)
    model = model.to(torch_device).eval()
    x = model.sample_prior(n_samples, torch_device, generator=generator)
    grid = torch.linspace(0.0, 1.0, int(n_steps) + 1, device=torch_device)
    for index in range(int(n_steps)):
        s = grid[index].expand(n_samples)
        t = grid[index + 1].expand(n_samples)
        x = model.flow(x, s, t)
    category = x.argmax(dim=-1)
    return decode_categories(category, c_max)
