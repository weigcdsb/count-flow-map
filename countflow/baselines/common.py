from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

from tqdm import trange

import torch
from torch import nn

from ..model import FourierTimeEmbedding, MLP
from ..utils import count_parameters, make_ema_copy, update_ema


class CountBackbone(nn.Module):
    """Shared time-conditioned MLP for nonnegative count vectors."""

    def __init__(
        self,
        dim: int,
        output_dim: Optional[int] = None,
        hidden_dim: int = 128,
        depth: int = 3,
        count_scale: float = 20.0,
        n_time_frequencies: int = 8,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.output_dim = int(output_dim or dim)
        self.count_scale = float(count_scale)
        self.time_embedding = FourierTimeEmbedding(n_time_frequencies)
        input_dim = 2 * self.dim + self.time_embedding.output_dim
        self.net = MLP(input_dim, hidden_dim, self.output_dim, depth=depth)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must have shape [batch,{self.dim}].")
        t = torch.as_tensor(t, dtype=torch.float32, device=x.device)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        x_float = x.float()
        state = torch.cat(
            [
                torch.log1p(x_float) / math.log1p(self.count_scale),
                x_float / self.count_scale,
            ],
            dim=-1,
        )
        return self.net(torch.cat([state, self.time_embedding(t)], dim=-1))


class CategoricalBackbone(nn.Module):
    """Matched MLP used by the categorical D3PM and Discrete Flow Map baselines.

    Inputs can be category indices [B,D] or simplex-valued states [B,D,K].
    The explicit D*K input/output scaling is deliberate: it exposes the cost of
    treating an unbounded count coordinate as a finite categorical variable.
    """

    def __init__(
        self,
        dim: int,
        n_categories: int,
        hidden_dim: int = 256,
        depth: int = 4,
        two_time: bool = False,
        n_time_frequencies: int = 8,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_categories = int(n_categories)
        self.two_time = bool(two_time)
        self.time_embedding = FourierTimeEmbedding(n_time_frequencies)
        n_time = 2 if self.two_time else 1
        input_dim = self.dim * self.n_categories + n_time * self.time_embedding.output_dim
        self.net = MLP(
            input_dim,
            hidden_dim,
            self.dim * self.n_categories,
            depth=depth,
        )

    def _state_to_float(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 2:
            return torch.nn.functional.one_hot(
                state.long(), num_classes=self.n_categories
            ).float()
        if state.ndim != 3 or state.shape[1:] != (self.dim, self.n_categories):
            raise ValueError(
                f"state must have shape [B,{self.dim}] or [B,{self.dim},{self.n_categories}]."
            )
        return state.float()

    def forward(
        self,
        state: torch.Tensor,
        s: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        state_f = self._state_to_float(state)
        batch = state_f.shape[0]
        s = torch.as_tensor(s, dtype=torch.float32, device=state_f.device)
        if s.ndim == 0:
            s = s.expand(batch)
        pieces = [state_f.reshape(batch, -1), self.time_embedding(s)]
        if self.two_time:
            if t is None:
                raise ValueError("two_time=True requires a target time t.")
            t = torch.as_tensor(t, dtype=torch.float32, device=state_f.device)
            if t.ndim == 0:
                t = t.expand(batch)
            pieces.append(self.time_embedding(t))
        logits = self.net(torch.cat(pieces, dim=-1))
        return logits.view(batch, self.dim, self.n_categories)


@dataclass
class GenericTrainConfig:
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float = 1e-5
    ema_decay: float = 0.999
    grad_clip: float = 5.0
    log_every: int = 500
    seed: int = 42



def progress_steps(steps: int, *, enabled: bool, desc: str):
    if enabled:
        return trange(1, int(steps) + 1, desc=desc, leave=True, dynamic_ncols=True)
    return range(1, int(steps) + 1)


def make_optimizer(model: nn.Module, config: GenericTrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def append_history(history: Dict[str, list], **values: float) -> None:
    for key, value in values.items():
        history.setdefault(key, []).append(float(value))


def optimizer_step(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: nn.Module,
    config: GenericTrainConfig,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
    update_ema(ema, model, config.ema_decay)


def model_metadata(model: nn.Module) -> Dict[str, int]:
    return {"trainable_parameters": int(count_parameters(model))}


def checkpoint_payload(
    model: nn.Module,
    ema: nn.Module,
    history: Dict[str, list],
    config: object,
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    return {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "history": history,
        "config": vars(config) if hasattr(config, "__dict__") else config,
        "metadata": dict(metadata or {}),
    }


def load_ema_checkpoint(model: nn.Module, checkpoint: Dict[str, object]) -> nn.Module:
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    return model
