from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ResidualMixtureParams:
    mixture_logits: torch.Tensor  # [B, M]
    birth_mean: torch.Tensor      # [B, M, D]
    death_prob: torch.Tensor      # [B, M, D]
    delta: torch.Tensor           # [B]


class PoissonBinomialResidualMixture:
    """Distribution of Y = X - D + B.

    Given mixture component m,
      B_i ~ Poisson(birth_mean[m, i]),
      D_i ~ Binomial(X_i, death_prob[m, i]).

    ``log_prob`` is exact.  The implementation is sparse-count aware:

    * x_i == 0 is evaluated analytically with one Poisson term;
    * small positive counts are evaluated in one compact exact block;
    * only coordinates whose death support intersects a high-count chunk are
      materialized for that chunk;
    * log-factorials are computed once per call and gathered by integer index.

    This avoids making every coordinate pay the cost of the largest count in
    the minibatch, which is critical for sparse scRNA matrices with rare large
    UMI counts.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        death_chunk_size: int = 32,
        small_count_threshold: int = 8,
        max_sparse_terms: int = 8_000_000,
    ) -> None:
        self.eps = float(eps)
        self.death_chunk_size = int(death_chunk_size)
        self.small_count_threshold = int(small_count_threshold)
        self.max_sparse_terms = int(max_sparse_terms)
        if self.death_chunk_size < 1:
            raise ValueError("death_chunk_size must be positive.")
        if self.small_count_threshold < 0:
            raise ValueError("small_count_threshold must be nonnegative.")
        if self.max_sparse_terms < 1:
            raise ValueError("max_sparse_terms must be positive.")

    def sample(self, x: torch.Tensor, params: ResidualMixtureParams) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape [batch, dim].")
        batch_size, dim = x.shape
        component = torch.distributions.Categorical(logits=params.mixture_logits).sample()
        gather_index = component[:, None, None].expand(batch_size, 1, dim)
        birth_mean = params.birth_mean.gather(1, gather_index).squeeze(1)
        death_prob = params.death_prob.gather(1, gather_index).squeeze(1)
        births = torch.poisson(birth_mean).long()
        deaths = torch.distributions.Binomial(
            total_count=x.float(), probs=death_prob
        ).sample().long()
        y = x.long() - deaths + births
        identity_rows = params.delta <= 1e-12
        if identity_rows.any():
            y = torch.where(identity_rows[:, None], x.long(), y)
        return y

    @staticmethod
    def _flat_coordinate_view(value: torch.Tensor) -> torch.Tensor:
        # [B,M,D] -> [B*D,M]
        return value.permute(0, 2, 1).reshape(-1, value.shape[1])

    def _exact_terms_for_indices(
        self,
        indices: torch.Tensor,
        d_first: int,
        d_last: int,
        *,
        x_flat: torch.Tensor,
        y_flat: torch.Tensor,
        mean_flat: torch.Tensor,
        log_mean_flat: torch.Tensor,
        log_death_flat: torch.Tensor,
        log_survival_flat: torch.Tensor,
        log_factorial: torch.Tensor,
        max_factorial: int,
    ) -> torch.Tensor:
        """Exact coordinate/component log-prob contribution for one d interval."""
        x = x_flat.index_select(0, indices)
        y = y_flat.index_select(0, indices)
        mean = mean_flat.index_select(0, indices)
        log_mean = log_mean_flat.index_select(0, indices)
        log_death = log_death_flat.index_select(0, indices)
        log_survival = log_survival_flat.index_select(0, indices)

        d = torch.arange(
            d_first,
            d_last,
            device=x.device,
            dtype=torch.long,
        )
        births = y[:, None] - x[:, None] + d[None, :]
        valid = (d[None, :] <= x[:, None]) & (births >= 0)

        # Invalid entries are clipped only for safe table lookup and masked to
        # -inf before logsumexp, so this does not alter the exact probability.
        x_minus_d = (x[:, None] - d[None, :]).clamp(0, max_factorial)
        safe_births = births.clamp(0, max_factorial)

        safe_d = d.clamp(0, max_factorial)
        log_combination = (
            log_factorial.index_select(0, x)[:, None]
            - log_factorial.index_select(0, safe_d)[None, :]
            - log_factorial[x_minus_d]
        )
        dtype = mean.dtype
        log_binomial = (
            log_combination[:, None, :]
            + d.to(dtype)[None, None, :] * log_death[:, :, None]
            + x_minus_d.to(dtype)[:, None, :] * log_survival[:, :, None]
        )
        log_poisson = (
            safe_births.to(dtype)[:, None, :] * log_mean[:, :, None]
            - mean[:, :, None]
            - log_factorial[safe_births][:, None, :]
        )
        terms = torch.where(
            valid[:, None, :],
            log_binomial + log_poisson,
            torch.full_like(log_binomial, -torch.inf),
        )
        return torch.logsumexp(terms, dim=-1)  # [active coordinates, M]

    def log_prob(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        params: ResidualMixtureParams,
    ) -> torch.Tensor:
        """Exact sparse-aware log probability.

        The result is mathematically the same finite death-count sum as the
        dense/chunked implementation.  Only impossible coordinates/death values
        are omitted before tensor materialization.
        """
        if y.shape != x.shape or x.ndim != 2:
            raise ValueError("x and y must have the same shape [batch, dim].")
        if (x < 0).any() or (y < 0).any():
            raise ValueError("x and y must be nonnegative counts.")

        x_long = x.long()
        y_long = y.long()
        mean = params.birth_mean.clamp_min(self.eps)
        prob = params.death_prob.clamp(self.eps, 1.0 - self.eps)
        batch_size, n_mixtures, dim = mean.shape
        if x_long.shape != (batch_size, dim):
            raise ValueError("parameter shapes do not match x/y.")

        mean_flat = self._flat_coordinate_view(mean)
        prob_flat = self._flat_coordinate_view(prob)
        x_flat = x_long.reshape(-1)
        y_flat = y_long.reshape(-1)
        coordinate_log_prob = torch.full_like(mean_flat, -torch.inf)

        # In a valid term, d <= x and birth = y - x + d <= y.  Therefore a
        # factorial table through max(max(x), max(y)) covers every valid term.
        if x_flat.numel() == 0:
            raise ValueError("x and y must be non-empty.")
        max_factorial = int(torch.maximum(x_long.max(), y_long.max()).item())
        factorial_argument = torch.arange(
            max_factorial + 1,
            device=x.device,
            dtype=mean.dtype,
        )
        log_factorial = torch.lgamma(factorial_argument + 1.0)

        log_mean_flat = torch.log(mean_flat)
        log_death_flat = torch.log(prob_flat)
        log_survival_flat = torch.log1p(-prob_flat)

        # x == 0 -> D == 0 exactly, so Y is simply Poisson(mean).
        zero_indices = torch.nonzero(x_flat == 0, as_tuple=False).flatten()
        if zero_indices.numel() > 0:
            y_zero = y_flat.index_select(0, zero_indices)
            zero_log_prob = (
                y_zero.to(mean.dtype)[:, None]
                * log_mean_flat.index_select(0, zero_indices)
                - mean_flat.index_select(0, zero_indices)
                - log_factorial.index_select(0, y_zero)[:, None]
            )
            coordinate_log_prob = torch.index_copy(
                coordinate_log_prob, 0, zero_indices, zero_log_prob
            )

        # Most nonzero scRNA coordinates are tiny.  Evaluate all d values for
        # x <= threshold in one compact block, independent of rare huge counts.
        small_threshold = self.small_count_threshold
        if small_threshold > 0:
            small_indices = torch.nonzero(
                (x_flat > 0) & (x_flat <= small_threshold),
                as_tuple=False,
            ).flatten()
            if small_indices.numel() > 0:
                small_log_prob = self._exact_terms_for_indices(
                    small_indices,
                    0,
                    small_threshold + 1,
                    x_flat=x_flat,
                    y_flat=y_flat,
                    mean_flat=mean_flat,
                    log_mean_flat=log_mean_flat,
                    log_death_flat=log_death_flat,
                    log_survival_flat=log_survival_flat,
                    log_factorial=log_factorial,
                    max_factorial=max_factorial,
                )
                coordinate_log_prob = torch.index_copy(
                    coordinate_log_prob, 0, small_indices, small_log_prob
                )

        # Rare larger-count coordinates use exact chunking, but a coordinate is
        # included in a chunk only if [max(0,x-y), x] intersects that chunk.
        large_indices = torch.nonzero(
            x_flat > small_threshold, as_tuple=False
        ).flatten()
        if large_indices.numel() > 0:
            x_large = x_flat.index_select(0, large_indices)
            y_large = y_flat.index_select(0, large_indices)
            d_min_large = (x_large - y_large).clamp_min(0)
            max_deaths = int(x_large.max().item())

            # death_chunk_size is an upper bound.  On GPU, automatically reduce
            # it if the active large-count set would create an oversized tensor.
            max_chunk = self.death_chunk_size
            terms_per_d = max(1, int(large_indices.numel()) * n_mixtures)
            memory_chunk = max(1, self.max_sparse_terms // terms_per_d)
            chunk_size = max(1, min(max_chunk, memory_chunk))

            for first in range(0, max_deaths + 1, chunk_size):
                last = min(first + chunk_size, max_deaths + 1)
                active_local = torch.nonzero(
                    (x_large >= first) & (d_min_large < last),
                    as_tuple=False,
                ).flatten()
                if active_local.numel() == 0:
                    continue
                active_indices = large_indices.index_select(0, active_local)
                chunk_log_prob = self._exact_terms_for_indices(
                    active_indices,
                    first,
                    last,
                    x_flat=x_flat,
                    y_flat=y_flat,
                    mean_flat=mean_flat,
                    log_mean_flat=log_mean_flat,
                    log_death_flat=log_death_flat,
                    log_survival_flat=log_survival_flat,
                    log_factorial=log_factorial,
                    max_factorial=max_factorial,
                )
                previous = coordinate_log_prob.index_select(0, active_indices)
                coordinate_log_prob = torch.index_copy(
                    coordinate_log_prob,
                    0,
                    active_indices,
                    torch.logaddexp(previous, chunk_log_prob),
                )

        coordinate_log_prob = coordinate_log_prob.reshape(
            batch_size, dim, n_mixtures
        ).permute(0, 2, 1)
        component_log_prob = coordinate_log_prob.sum(dim=-1)
        log_prob = torch.logsumexp(
            F.log_softmax(params.mixture_logits, dim=-1) + component_log_prob,
            dim=-1,
        )

        identity_rows = params.delta <= 1e-12
        if identity_rows.any():
            identity_log_prob = torch.where(
                (y_long == x_long).all(dim=-1),
                torch.zeros_like(log_prob),
                torch.full_like(log_prob, -torch.inf),
            )
            log_prob = torch.where(identity_rows, identity_log_prob, log_prob)
        return log_prob
