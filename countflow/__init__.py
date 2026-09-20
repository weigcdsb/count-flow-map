"""Stochastic flow-map matching for count-valued data."""

from .bridge import conditional_birth_death_rates, sample_signed_binomial_bridge
from .benchmark_data import BenchmarkSetting, GammaPoissonFactorMixture, benchmark_settings
from .data import DiscreteUniformBox, GammaPoissonMixture2D, IndependentCoupling
from .exact_2d import Grid2D, PMFSolution, gamma_poisson_target_pmf, total_variation
from .model import CountFlowMap
from .sampling import generate_count_samples
from .training import TrainConfig, train_count_flow_map
from .simulation_runner import run_simulation_suite

__all__ = [
    "CountFlowMap",
    "BenchmarkSetting",
    "GammaPoissonFactorMixture",
    "benchmark_settings",
    "DiscreteUniformBox",
    "GammaPoissonMixture2D",
    "IndependentCoupling",
    "Grid2D",
    "PMFSolution",
    "gamma_poisson_target_pmf",
    "total_variation",
    "TrainConfig",
    "conditional_birth_death_rates",
    "generate_count_samples",
    "sample_signed_binomial_bridge",
    "train_count_flow_map",
    "run_simulation_suite",
]

from .application_data import (
    ConditionalPairBundle,
    ConditionalPairSplit,
    load_pair_bundle,
    load_spikeprophecy_session,
    make_neural_forecasting_surrogate,
    make_scrna_transport_surrogate,
    prepare_tahoe_panel_from_huggingface,
    save_pair_bundle,
)
from .application_runner import run_neural_application, run_scrna_application
from .application_training import (
    ApplicationTrainConfig,
    generate_conditional_count_fm,
    generate_conditional_flow_map,
    train_conditional_count_fm,
    train_conditional_flow_map,
)
from .conditional_model import (
    ConditionalCountFlowMap,
    ConditionalCountRateModel,
    HistoryGRUEncoder,
    VectorContextEncoder,
)

__all__ += [
    "ConditionalPairBundle",
    "ConditionalPairSplit",
    "load_pair_bundle",
    "load_spikeprophecy_session",
    "make_neural_forecasting_surrogate",
    "make_scrna_transport_surrogate",
    "prepare_tahoe_panel_from_huggingface",
    "save_pair_bundle",
    "run_neural_application",
    "run_scrna_application",
    "ApplicationTrainConfig",
    "generate_conditional_count_fm",
    "generate_conditional_flow_map",
    "train_conditional_count_fm",
    "train_conditional_flow_map",
    "ConditionalCountFlowMap",
    "ConditionalCountRateModel",
    "HistoryGRUEncoder",
    "VectorContextEncoder",
]
