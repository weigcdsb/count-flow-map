from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_sample_panels(
    panels: Sequence[torch.Tensor],
    titles: Sequence[str],
    max_points: int = 3000,
    limits: Optional[Sequence[float]] = None,
    figure_size: Sequence[float] = (16, 4),
):
    if len(panels) != len(titles):
        raise ValueError("panels and titles must have the same length.")
    fig, axes = plt.subplots(1, len(panels), figsize=figure_size, squeeze=False)
    for axis, samples, title in zip(axes[0], panels, titles):
        array = torch.as_tensor(samples).detach().cpu().numpy()
        if array.shape[0] > max_points:
            index = np.random.choice(array.shape[0], max_points, replace=False)
            array = array[index]
        axis.scatter(array[:, 0], array[:, 1], s=8, alpha=0.35)
        axis.set_title(title)
        axis.set_xlabel("count 1")
        axis.set_ylabel("count 2")
        if limits is not None:
            axis.set_xlim(limits[0], limits[1])
            axis.set_ylim(limits[2], limits[3])
        axis.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def plot_training_history(history: Dict[str, Iterable[float]]):
    steps = np.asarray(list(history["step"]))
    fig, axis = plt.subplots(figsize=(8, 4))
    for key in ("total", "diag", "ck"):
        axis.plot(steps, np.asarray(list(history[key])), label=key)
    axis.set_xlabel("training step")
    axis.set_ylabel("loss")
    axis.set_yscale("symlog", linthresh=1.0)
    axis.legend()
    fig.tight_layout()
    return fig


def plot_generation_path(path: List[torch.Tensor], max_trajectories: int = 80):
    stack = torch.stack(path, dim=0).float().numpy()
    n = min(stack.shape[1], max_trajectories)
    fig, axis = plt.subplots(figsize=(6, 6))
    for j in range(n):
        axis.plot(stack[:, j, 0], stack[:, j, 1], alpha=0.3)
        axis.scatter(stack[-1, j, 0], stack[-1, j, 1], s=10)
    axis.set_xlabel("count 1")
    axis.set_ylabel("count 2")
    axis.set_title("Few-step count-flow trajectories")
    axis.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig
