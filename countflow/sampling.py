from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .model import CountFlowMap


@torch.no_grad()
def generate_count_samples(
    model: CountFlowMap,
    source: object,
    n_samples: int,
    tau: float = 0.98,
    n_steps: int = 1,
    device: Optional[torch.device] = None,
    return_path: bool = False,
) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    if n_steps < 1:
        raise ValueError("n_steps must be positive.")
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must lie in (0, 1].")
    if device is None:
        device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    x = source.sample(n_samples, device=device)
    path = [x.detach().cpu()] if return_path else None
    grid = torch.linspace(0.0, tau, n_steps + 1, device=device)
    for k in range(n_steps):
        s = grid[k].expand(n_samples)
        t = grid[k + 1].expand(n_samples)
        x = model.sample(x, s, t)
        if path is not None:
            path.append(x.detach().cpu())
    if was_training:
        model.train()
    return x.detach().cpu(), path
