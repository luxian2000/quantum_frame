"""Quantum architecture search and state-synthesis utilities."""

from __future__ import annotations

from ._types import (
    ArchitectureScore,
    ArchitectureSpec,
    MetricDefinition,
    MetricGroupScore,
    SearchConfig,
    SearchResult,
)
from .architecture_candidates import build_common_architectures, common_architecture_names
from .architecture_search import ArchitectureSearch, NoiseAdaptiveQAS
from .evaluator import ArchitectureEvaluator, evaluate_architectures, metric_catalog
from .multi_objective_reward import (
    ExpressibilityScore,
    HardwareEfficiencyScore,
    MultiObjectiveReward,
    NoiseRobustnessScore,
    QASRewardWrapper,
    TrainabilityScore,
)
from .reward import RewardComposer, RewardWeights
from .search_env import NoisyQASEnv, QASState
from ..metrics.expressibility import KL_Haar_divergence, KL_Haar_relative, MMD_relative
from ..metrics.noisy_expressibility import (
    KL_Haar_noisy,
    MMD_noisy,
    comparative_expressibility,
    expressibility_score,
)
from ..channel.noise import (
    IonTrapNoiseConfig,
    NoiseSensitivityResult,
    load_default_ion_trap_noise_config,
    load_ion_trap_noise_config,
    noise_sensitivity,
)
from ..channel.noise.metrics import ion_trap_error_budget_proxy

_OPTIONAL_RL_EXPORTS: list[str] = []
try:
    from . import config
    from .CRLQAS import AdamSPSAConfig, CRLQASConfig, CRLQASResult, crlqas, train_crlqas
    from .PPR_DQL import PPRDQLConfig, PPRDQLPolicy, PPRDQLResult, ppr_dql_state_to_circuit, train_ppr_dql
    from .PPO_RB import PPORollbackConfig, ppo_rb_qas
    from .runner import QASRunConfig, available_qas_methods, default_qas_config, run
    from .supernet import (
        Architecture,
        LayerArchitecture,
        Supernet,
        SupernetConfig,
        SupernetResult,
        classification_supernet,
        h2_vqe_supernet,
        supernet_qas,
        train_supernet,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    _OPTIONAL_RL_EXPORTS.extend(
        [
            "AdamSPSAConfig",
            "Architecture",
            "CRLQASConfig",
            "CRLQASResult",
            "LayerArchitecture",
            "PPRDQLConfig",
            "PPRDQLPolicy",
            "PPRDQLResult",
            "PPORollbackConfig",
            "QASRunConfig",
            "Supernet",
            "SupernetConfig",
            "SupernetResult",
            "available_qas_methods",
            "classification_supernet",
            "config",
            "crlqas",
            "default_qas_config",
            "h2_vqe_supernet",
            "ppr_dql_state_to_circuit",
            "ppo_rb_qas",
            "run",
            "supernet_qas",
            "train_crlqas",
            "train_ppr_dql",
            "train_supernet",
        ]
    )

__all__ = [
    "ArchitectureEvaluator",
    "ArchitectureScore",
    "ArchitectureSearch",
    "ArchitectureSpec",
    "ExpressibilityScore",
    "HardwareEfficiencyScore",
    "IonTrapNoiseConfig",
    "KL_Haar_divergence",
    "KL_Haar_noisy",
    "KL_Haar_relative",
    "MMD_noisy",
    "MMD_relative",
    "MetricDefinition",
    "MetricGroupScore",
    "MultiObjectiveReward",
    "NoiseAdaptiveQAS",
    "NoiseRobustnessScore",
    "NoiseSensitivityResult",
    "NoisyQASEnv",
    "QASRewardWrapper",
    "QASState",
    "RewardComposer",
    "RewardWeights",
    "SearchConfig",
    "SearchResult",
    "TrainabilityScore",
    "build_common_architectures",
    "common_architecture_names",
    "comparative_expressibility",
    "evaluate_architectures",
    "expressibility_score",
    "ion_trap_error_budget_proxy",
    "load_default_ion_trap_noise_config",
    "load_ion_trap_noise_config",
    "metric_catalog",
    "noise_sensitivity",
] + _OPTIONAL_RL_EXPORTS
