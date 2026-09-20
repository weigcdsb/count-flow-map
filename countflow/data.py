from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch


@dataclass
class DiscreteUniformBox:
    low: Sequence[int]
    high: Sequence[int]

    def __post_init__(self) -> None:
        self.low = tuple(int(v) for v in self.low)
        self.high = tuple(int(v) for v in self.high)
        if len(self.low) != len(self.high):
            raise ValueError("low and high must have the same length.")
        if any(lo < 0 or hi < lo for lo, hi in zip(self.low, self.high)):
            raise ValueError("Require 0 <= low <= high in each coordinate.")

    @property
    def dim(self) -> int:
        return len(self.low)

    def sample(
        self,
        n: int,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        device = torch.device("cpu") if device is None else device
        columns = [
            torch.randint(lo, hi + 1, (n,), device=device, generator=generator)
            for lo, hi in zip(self.low, self.high)
        ]
        return torch.stack(columns, dim=1).long()


@dataclass
class GammaPoissonMixture2D:
    """Two-component Gamma-Poisson target for the 2D demo."""

    means: Tuple[Tuple[float, float], Tuple[float, float]] = ((18.0, 4.0), (18.0, 15.0))
    concentrations: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (120.0, 80.0),
        (120.0, 120.0),
    )
    weights: Tuple[float, float] = (0.5, 0.5)

    def __post_init__(self) -> None:
        self._means = torch.tensor(self.means, dtype=torch.float32)
        self._concentrations = torch.tensor(self.concentrations, dtype=torch.float32)
        self._weights = torch.tensor(self.weights, dtype=torch.float32)
        if self._means.shape != (2, 2) or self._concentrations.shape != (2, 2):
            raise ValueError("means and concentrations must each have shape [2, 2].")
        if (self._means <= 0).any() or (self._concentrations <= 0).any():
            raise ValueError("Means and concentrations must be positive.")
        if (self._weights < 0).any() or self._weights.sum() <= 0:
            raise ValueError("Mixture weights must be nonnegative with positive sum.")
        self._weights = self._weights / self._weights.sum()

    @classmethod
    def manuscript_scale(cls) -> "GammaPoissonMixture2D":
        return cls(
            means=((60.0, 5.0), (60.0, 40.0)),
            concentrations=((160.0, 80.0), (160.0, 140.0)),
            weights=(0.5, 0.5),
        )

    @property
    def dim(self) -> int:
        return 2

    def sample(
        self,
        n: int,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        device = torch.device("cpu") if device is None else device
        component = torch.multinomial(
            self._weights.to(device), n, replacement=True, generator=generator
        )
        means = self._means.to(device)[component]
        concentration = self._concentrations.to(device)[component]
        try:
            standard_gamma = torch._standard_gamma(concentration, generator=generator)
        except TypeError:  # pragma: no cover - compatibility with older torch
            standard_gamma = torch._standard_gamma(concentration)
        latent_rate = standard_gamma / (concentration / means)
        return torch.poisson(latent_rate, generator=generator).long()


@dataclass
class IndependentCoupling:
    source: object
    target: object

    def sample(
        self,
        n: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.source.sample(n, device=device), self.target.sample(n, device=device)
