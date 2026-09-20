from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch

from .data import DiscreteUniformBox, GammaPoissonMixture2D


def _sample_gamma(
    shape: torch.Tensor,
    rate: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Gamma sampler that honors a torch.Generator when supported."""
    try:
        standard = torch._standard_gamma(shape, generator=generator)
    except TypeError:  # pragma: no cover - compatibility with older torch
        standard = torch._standard_gamma(shape)
    return standard / rate


@dataclass
class GammaPoissonFactorMixture:
    """Correlated multimodal count distribution used by the paper benchmark.

    Given a mixture component C and nonnegative latent factors H,

        H_l | C=c ~ Gamma(shape[c,l], rate[c,l]),
        X_i | C,H ~ Poisson(base[c,i] + sum_l loadings[c,i,l] H_l).

    The family exposes count scale, dimension, overdispersion, multimodality,
    and cross-coordinate dependence through one common construction.
    """

    weights: torch.Tensor
    base_rates: torch.Tensor
    loadings: torch.Tensor
    factor_shapes: torch.Tensor
    factor_rates: torch.Tensor
    name: str = "gamma_poisson_factor_mixture"

    def __post_init__(self) -> None:
        self.weights = torch.as_tensor(self.weights, dtype=torch.float32)
        self.base_rates = torch.as_tensor(self.base_rates, dtype=torch.float32)
        self.loadings = torch.as_tensor(self.loadings, dtype=torch.float32)
        self.factor_shapes = torch.as_tensor(self.factor_shapes, dtype=torch.float32)
        self.factor_rates = torch.as_tensor(self.factor_rates, dtype=torch.float32)
        if self.base_rates.ndim != 2:
            raise ValueError("base_rates must have shape [components, dim].")
        if self.loadings.ndim != 3:
            raise ValueError("loadings must have shape [components, dim, rank].")
        n_components, dim = self.base_rates.shape
        if self.loadings.shape[:2] != (n_components, dim):
            raise ValueError("base_rates and loadings dimensions do not match.")
        rank = self.loadings.shape[2]
        if self.factor_shapes.shape != (n_components, rank):
            raise ValueError("factor_shapes must have shape [components, rank].")
        if self.factor_rates.shape != (n_components, rank):
            raise ValueError("factor_rates must have shape [components, rank].")
        if self.weights.shape != (n_components,):
            raise ValueError("weights must have shape [components].")
        if (self.weights < 0).any() or self.weights.sum() <= 0:
            raise ValueError("weights must be nonnegative with positive sum.")
        if (self.base_rates < 0).any() or (self.loadings < 0).any():
            raise ValueError("base rates and loadings must be nonnegative.")
        if (self.factor_shapes <= 0).any() or (self.factor_rates <= 0).any():
            raise ValueError("Gamma shapes and rates must be positive.")
        self.weights = self.weights / self.weights.sum()

    @property
    def dim(self) -> int:
        return int(self.base_rates.shape[1])

    @property
    def rank(self) -> int:
        return int(self.loadings.shape[2])

    @property
    def n_components(self) -> int:
        return int(self.weights.numel())

    def component_means(self) -> torch.Tensor:
        factor_mean = self.factor_shapes / self.factor_rates
        return self.base_rates + torch.einsum("cdr,cr->cd", self.loadings, factor_mean)

    def mean(self) -> torch.Tensor:
        return torch.einsum("c,cd->d", self.weights, self.component_means())

    def sample(
        self,
        n: int,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        device = torch.device("cpu") if device is None else torch.device(device)
        weights = self.weights.to(device)
        component = torch.multinomial(weights, n, replacement=True, generator=generator)
        shapes = self.factor_shapes.to(device)[component]
        rates = self.factor_rates.to(device)[component]
        factors = _sample_gamma(shapes, rates, generator=generator)
        base = self.base_rates.to(device)[component]
        loading = self.loadings.to(device)[component]
        poisson_rate = base + torch.einsum("ndr,nr->nd", loading, factors)
        return torch.poisson(poisson_rate.clamp_min(0.0), generator=generator).long()


@dataclass(frozen=True)
class BenchmarkSetting:
    name: str
    source: object
    target: object
    dim: int
    nominal_mean: float
    exact_2d: bool
    suggested_count_scale: float
    description: str


def make_factor_mixture(
    dim: int,
    mean_count: float,
    *,
    n_components: int = 3,
    rank: int = 4,
    seed: int = 2027,
    name: Optional[str] = None,
) -> GammaPoissonFactorMixture:
    """Create one deterministic benchmark member with structured dependence."""
    if dim <= 0 or mean_count <= 0:
        raise ValueError("dim and mean_count must be positive.")
    rank = min(int(rank), int(dim))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    weights = torch.tensor([0.34, 0.33, 0.33], dtype=torch.float32)[:n_components]
    weights = weights / weights.sum()

    coordinate = torch.linspace(0.0, 1.0, dim)
    base_rates = []
    all_loadings = []
    factor_shapes = []
    factor_rates = []
    for component in range(n_components):
        phase = 2.0 * torch.pi * component / max(n_components, 1)
        modulation = 0.65 + 0.35 * (
            1.0 + torch.sin(2.0 * torch.pi * coordinate + phase)
        ) / 2.0
        base = 0.08 * mean_count * modulation

        raw = 0.15 + torch.rand(dim, rank, generator=generator)
        for factor in range(rank):
            center = (factor + 0.5 + 0.4 * component) / rank
            width = 0.12 + 0.03 * ((factor + component) % 2)
            bump = torch.exp(-0.5 * ((coordinate - center) / width).pow(2))
            raw[:, factor] *= 0.25 + bump
        shape = torch.linspace(1.5, 4.5, rank) + 0.35 * component
        rate = shape.clone()  # unit-mean latent factors
        current = base + raw.sum(dim=1)
        scale = ((mean_count * modulation - base).clamp_min(1e-3) / raw.sum(dim=1).clamp_min(1e-6))
        loading = raw * scale[:, None]
        # A mild component-wide scale shift makes modes distinct without
        # changing the overall nominal count scale dramatically.
        loading = loading * (0.78 + 0.22 * component)
        base_rates.append(base)
        all_loadings.append(loading)
        factor_shapes.append(shape)
        factor_rates.append(rate)

    return GammaPoissonFactorMixture(
        weights=weights,
        base_rates=torch.stack(base_rates),
        loadings=torch.stack(all_loadings),
        factor_shapes=torch.stack(factor_shapes),
        factor_rates=torch.stack(factor_rates),
        name=name or f"factor_d{dim}_mean{mean_count:g}",
    )


def benchmark_settings(preset: str = "paper") -> Dict[str, BenchmarkSetting]:
    """Return the exact and scalable settings agreed for the paper."""
    if preset not in {"paper", "smoke"}:
        raise ValueError("preset must be 'paper' or 'smoke'.")

    if preset == "smoke":
        exact_target = GammaPoissonMixture2D()
        exact_source = DiscreteUniformBox(low=(0, 0), high=(24, 24))
        scalable = {
            "scale_8_low": (8, 2.0),
        }
    else:
        exact_target = GammaPoissonMixture2D.manuscript_scale()
        exact_source = DiscreteUniformBox(low=(0, 0), high=(72, 72))
        scalable = {
            "scale_32_low": (32, 2.0),
            "scale_32_high": (32, 20.0),
            "scale_128_high": (128, 20.0),
        }

    settings: Dict[str, BenchmarkSetting] = {
        "exact_2d": BenchmarkSetting(
            name="exact_2d",
            source=exact_source,
            target=exact_target,
            dim=2,
            nominal_mean=float(torch.tensor(exact_target.means).mean()),
            exact_2d=True,
            suggested_count_scale=80.0 if preset == "paper" else 24.0,
            description=(
                "Correlated multimodal two-component Gamma-Poisson target with "
                "an accurately evaluable finite-grid PMF."
            ),
        )
    }
    for index, (setting_name, (dim, mean_count)) in enumerate(scalable.items()):
        target = make_factor_mixture(
            dim,
            mean_count,
            seed=2027 + index,
            name=setting_name,
        )
        high = max(4, int(round(2.5 * mean_count)))
        source = DiscreteUniformBox(low=(0,) * dim, high=(high,) * dim)
        settings[setting_name] = BenchmarkSetting(
            name=setting_name,
            source=source,
            target=target,
            dim=dim,
            nominal_mean=mean_count,
            exact_2d=False,
            suggested_count_scale=max(8.0, 3.0 * mean_count),
            description=(
                f"Gamma-Poisson factor mixture with d={dim} and average count "
                f"approximately {mean_count:g}."
            ),
        )
    return settings


def choose_categorical_support(
    target: object,
    *,
    tail_probability: float = 1e-4,
    pilot_samples: int = 1_000_000,
    seed: int = 12345,
    max_categories: Optional[int] = None,
    batch_size: int = 10000,
) -> Tuple[int, float]:
    """Choose C_max and return the empirical overflow probability.

    The categorical baselines use categories 0,...,C_max plus one overflow
    category. C_max is selected from the per-vector maximum so that the chance
    that any coordinate overflows is approximately below tail_probability.
    """
    if not 0 < tail_probability < 1:
        raise ValueError("tail_probability must lie in (0,1).")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    maxima_blocks = []
    remaining = int(pilot_samples)
    while remaining > 0:
        current = min(int(batch_size), remaining)
        samples = target.sample(current, generator=generator).cpu()
        maxima_blocks.append(samples.max(dim=1).values.float())
        remaining -= current
    maxima = torch.cat(maxima_blocks, dim=0)
    quantile = min(1.0, max(0.0, 1.0 - tail_probability))
    c_max = int(torch.quantile(maxima, quantile, interpolation="higher").item())
    if max_categories is not None:
        c_max = min(c_max, int(max_categories) - 2)
    overflow = float((maxima > c_max).float().mean().item())
    return max(c_max, 1), overflow


def encode_categories(x: torch.Tensor, c_max: int) -> torch.Tensor:
    """Map counts to 0,...,C_max plus overflow category C_max+1."""
    x = torch.as_tensor(x).long()
    overflow = torch.full_like(x, int(c_max) + 1)
    return torch.where(x <= int(c_max), x, overflow)


def decode_categories(categories: torch.Tensor, c_max: int) -> torch.Tensor:
    """Decode overflow as C_max+1; its empirical mass is reported separately."""
    categories = torch.as_tensor(categories).long()
    return categories.clamp(min=0, max=int(c_max) + 1)
