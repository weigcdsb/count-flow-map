from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .distributions import PoissonBinomialResidualMixture, ResidualMixtureParams
from .model import FourierTimeEmbedding, MLP
from .utils import to_float_time
from .parameterization import correction_duration


class VectorContextEncoder(nn.Module):
    """Encode a dense context vector such as cell-line/drug/dose features."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: Optional[int] = None) -> None:
        super().__init__()
        output_dim = hidden_dim if output_dim is None else int(output_dim)
        self.input_dim = int(input_dim)
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU(),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[1] != self.input_dim:
            raise ValueError(f"context must have shape [batch,{self.input_dim}].")
        return self.net(context.float())


class HistoryGRUEncoder(nn.Module):
    """Causal encoder for recent spike-count history [batch, history, neurons]."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        num_layers: int = 1,
        count_scale: float = 4.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.output_dim = hidden_dim if output_dim is None else int(output_dim)
        self.count_scale = float(count_scale)
        self.input_projection = nn.Linear(2 * self.dim, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=int(num_layers),
            batch_first=True,
        )
        self.output_projection = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 3 or context.shape[2] != self.dim:
            raise ValueError(
                f"history context must have shape [batch,history,{self.dim}]."
            )
        x = context.float()
        features = torch.cat(
            [
                torch.log1p(x) / math.log1p(self.count_scale),
                x / self.count_scale,
            ],
            dim=-1,
        )
        encoded = F.silu(self.input_projection(features))
        _, hidden = self.gru(encoded)
        return F.silu(self.output_projection(hidden[-1]))


class ConditionalCountRateModel(nn.Module):
    """Context-conditioned birth/death rates for Count-FM sampling."""

    def __init__(
        self,
        dim: int,
        context_encoder: nn.Module,
        context_dim: int,
        hidden_dim: int = 256,
        depth: int = 3,
        n_time_frequencies: int = 8,
        count_scale: float = 20.0,
        rate_floor: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.context_encoder = context_encoder
        self.context_dim = int(context_dim)
        self.count_scale = float(count_scale)
        self.rate_floor = float(rate_floor)
        self.time_embedding = FourierTimeEmbedding(n_time_frequencies)
        time_dim = self.time_embedding.output_dim
        self.state_encoder = MLP(2 * self.dim, hidden_dim, hidden_dim, depth=2)
        self.rate_net = MLP(
            hidden_dim + self.context_dim + time_dim,
            hidden_dim,
            2 * self.dim,
            depth=depth,
        )

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

    def local_rates(
        self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError("x must have shape [batch, dim].")
        t = to_float_time(t, x.shape[0], x.device)
        state = self._encode_state(x)
        encoded_context = self.context_encoder(context)
        if encoded_context.shape != (x.shape[0], self.context_dim):
            raise ValueError("context encoder returned the wrong shape.")
        raw = self.rate_net(
            torch.cat([state, encoded_context, self.time_embedding(t)], dim=-1)
        )
        birth_raw, death_coefficient_raw = raw.chunk(2, dim=-1)
        birth_rate = F.softplus(birth_raw) + self.rate_floor
        death_coefficient = F.softplus(death_coefficient_raw) + self.rate_floor
        death_rate = x.float() * death_coefficient
        return birth_rate, death_rate, death_coefficient

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.local_rates(x, t, context)


class ConditionalCountFlowMap(nn.Module):
    """Two-time count kernel conditioned on static or historical context."""

    def __init__(
        self,
        dim: int,
        context_encoder: nn.Module,
        context_dim: int,
        hidden_dim: int = 256,
        depth: int = 3,
        n_mixtures: int = 4,
        n_time_frequencies: int = 8,
        count_scale: float = 20.0,
        correction_scale: float = 3.0,
        rate_floor: float = 1e-5,
        death_chunk_size: int = 32,
        correction_time_scale: str = "absolute",
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.context_encoder = context_encoder
        self.context_dim = int(context_dim)
        self.n_mixtures = int(n_mixtures)
        self.count_scale = float(count_scale)
        self.correction_scale = float(correction_scale)
        if correction_time_scale not in {"absolute", "remaining"}:
            raise ValueError("correction_time_scale must be absolute or remaining.")
        self.correction_time_scale = correction_time_scale
        self.rate_floor = float(rate_floor)

        self.time_embedding = FourierTimeEmbedding(n_time_frequencies)
        time_dim = self.time_embedding.output_dim
        self.state_encoder = MLP(2 * self.dim, hidden_dim, hidden_dim, depth=2)
        self.rate_net = MLP(
            hidden_dim + self.context_dim + time_dim,
            hidden_dim,
            2 * self.dim,
            depth=depth,
        )
        map_input_dim = hidden_dim + self.context_dim + 3 * time_dim + 1
        map_output_dim = self.n_mixtures + self.n_mixtures * 2 * self.dim
        self.map_net = MLP(map_input_dim, hidden_dim, map_output_dim, depth=depth)
        self.residual_distribution = PoissonBinomialResidualMixture(
            death_chunk_size=death_chunk_size
        )

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

    def _encode_context(self, context: torch.Tensor, batch_size: int) -> torch.Tensor:
        encoded = self.context_encoder(context)
        if encoded.shape != (batch_size, self.context_dim):
            raise ValueError("context encoder returned the wrong shape.")
        return encoded

    def _local_rates_from_features(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        state: torch.Tensor,
        encoded_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.rate_net(
            torch.cat([state, encoded_context, self.time_embedding(t)], dim=-1)
        )
        birth_raw, death_coefficient_raw = raw.chunk(2, dim=-1)
        birth_rate = F.softplus(birth_raw) + self.rate_floor
        death_coefficient = F.softplus(death_coefficient_raw) + self.rate_floor
        death_rate = x.float() * death_coefficient
        return birth_rate, death_rate, death_coefficient

    def local_rates(
        self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError("x must have shape [batch, dim].")
        t = to_float_time(t, x.shape[0], x.device)
        state = self._encode_state(x)
        encoded_context = self._encode_context(context, x.shape[0])
        return self._local_rates_from_features(x, t, state, encoded_context)

    def kernel_params(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
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
        encoded_context = self._encode_context(context, batch_size)
        map_input = torch.cat(
            [
                state,
                encoded_context,
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
        base_birth, _, base_death_coefficient = self._local_rates_from_features(
            x, s, state, encoded_context
        )
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
    def sample(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        return self.residual_distribution.sample(
            x, self.kernel_params(x, s, t, context)
        )

    def log_prob(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        return self.residual_distribution.log_prob(
            y, x, self.kernel_params(x, s, t, context)
        )

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> ResidualMixtureParams:
        return self.kernel_params(x, s, t, context)
