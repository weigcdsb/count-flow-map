from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.special import gammaln

from .data import GammaPoissonMixture2D


@dataclass(frozen=True)
class Grid2D:
    max_counts: Tuple[int, int]

    def __post_init__(self) -> None:
        if len(self.max_counts) != 2 or any(int(v) < 1 for v in self.max_counts):
            raise ValueError("max_counts must contain two positive integers.")
        object.__setattr__(self, "max_counts", tuple(int(v) for v in self.max_counts))

    @property
    def shape(self) -> Tuple[int, int]:
        return self.max_counts[0] + 1, self.max_counts[1] + 1

    @property
    def size(self) -> int:
        h, w = self.shape
        return h * w

    def states(self, device: Optional[torch.device] = None) -> torch.Tensor:
        h, w = self.shape
        device = torch.device("cpu") if device is None else device
        return torch.cartesian_prod(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
        ).long()


@dataclass
class PMFSolution:
    pmf: np.ndarray
    overflow_mass: float = 0.0

    def __post_init__(self) -> None:
        self.pmf = np.asarray(self.pmf, dtype=np.float64)
        self.overflow_mass = float(self.overflow_mass)

    @property
    def total_mass(self) -> float:
        return float(self.pmf.sum() + self.overflow_mass)


def _negative_binomial_pmf(max_count: int, mean: float, concentration: float) -> np.ndarray:
    k = np.arange(max_count + 1, dtype=np.float64)
    r = float(concentration)
    p = r / (r + float(mean))
    log_pmf = (
        gammaln(k + r)
        - gammaln(r)
        - gammaln(k + 1.0)
        + r * math.log(p)
        + k * math.log1p(-p)
    )
    return np.exp(log_pmf)


def gamma_poisson_target_pmf(target: GammaPoissonMixture2D, grid: Grid2D) -> PMFSolution:
    h, w = grid.shape
    pmf = np.zeros((h, w), dtype=np.float64)
    means = np.asarray(target.means, dtype=np.float64)
    concentrations = np.asarray(target.concentrations, dtype=np.float64)
    weights = np.asarray(target.weights, dtype=np.float64)
    weights = weights / weights.sum()
    for component in range(2):
        p0 = _negative_binomial_pmf(h - 1, means[component, 0], concentrations[component, 0])
        p1 = _negative_binomial_pmf(w - 1, means[component, 1], concentrations[component, 1])
        pmf += weights[component] * np.outer(p0, p1)
    return PMFSolution(pmf, max(1.0 - float(pmf.sum()), 0.0))


def total_variation(left: PMFSolution, right: PMFSolution) -> float:
    if left.pmf.shape != right.pmf.shape:
        raise ValueError("PMF shapes must agree.")
    return 0.5 * float(
        np.abs(left.pmf - right.pmf).sum()
        + abs(left.overflow_mass - right.overflow_mass)
    )
