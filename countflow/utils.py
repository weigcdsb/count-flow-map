from __future__ import annotations

import copy
import random
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def make_ema_copy(module: nn.Module) -> nn.Module:
    ema = copy.deepcopy(module)
    ema.eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(ema: nn.Module, online: nn.Module, decay: float) -> None:
    ema_params = dict(ema.named_parameters())
    online_params = dict(online.named_parameters())
    for name, ema_parameter in ema_params.items():
        ema_parameter.mul_(decay).add_(online_params[name], alpha=1.0 - decay)
    ema_buffers = dict(ema.named_buffers())
    online_buffers = dict(online.named_buffers())
    for name, ema_buffer in ema_buffers.items():
        ema_buffer.copy_(online_buffers[name])


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_float_time(value: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(value, dtype=torch.float32, device=device)
    if value.ndim == 0:
        value = value.expand(batch_size)
    if value.ndim != 1 or value.shape[0] != batch_size:
        raise ValueError("Time tensors must be scalar or have shape [batch].")
    return value


def history_to_numpy(history: Dict[str, Iterable[float]]) -> Dict[str, np.ndarray]:
    return {key: np.asarray(list(values), dtype=np.float64) for key, values in history.items()}
