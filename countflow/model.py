from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .distributions import PoissonBinomialResidualMixture, ResidualMixtureParams
from .utils import to_float_time
from .parameterization import correction_duration


class FourierTimeEmbedding(nn.Module):
    def __init__(self, n_frequencies: int = 8) -> None:
        super().__init__()
        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(n_frequencies, dtype=torch.float32),
            persistent=False,
        )

    @property
    def output_dim(self) -> int:
        return 2 * int(self.frequencies.numel()) + 1

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * t[:, None] * self.frequencies[None, :]
        return torch.cat([t[:, None], torch.sin(angles), torch.cos(angles)], dim=-1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int = 3) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2.")
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CountFlowMap(nn.Module):
    """One sampleable two-time count kernel with the correct diagonal tangent."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        n_mixtures: int = 4,
        n_time_frequencies: int = 8,
        count_scale: float = 20.0,
        correction_scale: float = 3.0,
        rate_floor: float = 1e-5,
        correction_time_scale: str = "absolute",
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_mixtures = int(n_mixtures)
        self.count_scale = float(count_scale)
        self.correction_scale = float(correction_scale)
        if correction_time_scale not in {"absolute", "remaining"}:
            raise ValueError("correction_time_scale must be absolute or remaining.")
        self.correction_time_scale = correction_time_scale
        self.rate_floor = float(rate_floor)

        self.time_embedding = FourierTimeEmbedding(n_time_frequencies)
        time_dim = self.time_embedding.output_dim
        state_feature_dim = 2 * self.dim
        self.state_encoder = MLP(state_feature_dim, hidden_dim, hidden_dim, depth=2)
        self.rate_net = MLP(hidden_dim + time_dim, hidden_dim, 2 * self.dim, depth=depth)

        map_input_dim = hidden_dim + 3 * time_dim + 1
        map_output_dim = self.n_mixtures + self.n_mixtures * 2 * self.dim
        self.map_net = MLP(map_input_dim, hidden_dim, map_output_dim, depth=depth)
        self.residual_distribution = PoissonBinomialResidualMixture()

    def _encode_state(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        features = torch.cat(
            [
                torch.log1p(x_float) / math.log1p(self.count_scale),
                x_float / self.count_scale,
            ],
            dim=-1,
        )
        return self.state_encoder(features)

    def _local_rates_from_state(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.rate_net(torch.cat([state, self.time_embedding(t)], dim=-1))
        birth_raw, death_coefficient_raw = raw.chunk(2, dim=-1)
        birth_rate = F.softplus(birth_raw) + self.rate_floor
        death_coefficient = F.softplus(death_coefficient_raw) + self.rate_floor
        death_rate = x.float() * death_coefficient
        return birth_rate, death_rate, death_coefficient

    def local_rates(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return birth rate, death rate, and per-particle death coefficient."""
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError("x must have shape [batch, dim].")
        t = to_float_time(t, x.shape[0], x.device)
        state = self._encode_state(x)
        return self._local_rates_from_state(x, t, state)

    def kernel_params(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
    ) -> ResidualMixtureParams:
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError("x must have shape [batch, dim].")
        batch_size = x.shape[0]
        s = to_float_time(s, batch_size, x.device)
        t = to_float_time(t, batch_size, x.device)
        if (t < s).any():
            raise ValueError("Require s <= t.")
        delta = t - s

        state = self._encode_state(x)
        map_input = torch.cat(
            [
                state,
                self.time_embedding(s),
                self.time_embedding(t),
                self.time_embedding(delta),
                delta[:, None],
            ],
            dim=-1,
        )
        raw = self.map_net(map_input)
        mixture_logits = raw[:, : self.n_mixtures]
        corrections = raw[:, self.n_mixtures :].view(
            batch_size, self.n_mixtures, 2, self.dim
        )
        corrections = self.correction_scale * torch.tanh(corrections)
        birth_correction = corrections[:, :, 0, :]
        death_correction = corrections[:, :, 1, :]

        base_birth, _, base_death_coefficient = self._local_rates_from_state(x, s, state)
        delta3 = delta[:, None, None]
        correction_delta3 = correction_duration(s, t, self.correction_time_scale)[:, None, None]
        component_birth_rate = base_birth[:, None, :] * torch.exp(
            correction_delta3 * birth_correction
        )
        component_death_coefficient = base_death_coefficient[:, None, :] * torch.exp(
            correction_delta3 * death_correction
        )
        birth_mean = delta3 * component_birth_rate
        death_prob = -torch.expm1(-delta3 * component_death_coefficient)
        death_prob = death_prob.clamp(0.0, 1.0 - 1e-7)
        return ResidualMixtureParams(mixture_logits, birth_mean, death_prob, delta)

    @torch.no_grad()
    def sample(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.residual_distribution.sample(x, self.kernel_params(x, s, t))

    def log_prob(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return self.residual_distribution.log_prob(y, x, self.kernel_params(x, s, t))

    def forward(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> ResidualMixtureParams:
        return self.kernel_params(x, s, t)
