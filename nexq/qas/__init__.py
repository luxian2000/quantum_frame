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
from .task_evaluation import (
    OptimizerConfig,
    TaskEvaluationResult,
    bind_parameters,
    evaluate_task_objective,
    optimize_task_parameters,
)
from .experiments import MultiSeedValidationReport, ValidationReport, run_multi_seed_validation_experiment, run_validation_experiment
from .problems import (
    MaxCutInstance,
    ProblemInstance,
    ResourceAllocationInstance,
    maxcut_line,
    maxcut_ring,
    small_resource_allocation,
)
from ..metrics.expressibility import KL_Haar_divergence, KL_Haar_relative, MMD_relative
from ..metrics.hardware import HardwareProfile
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
    from .CRLQAS import AdamSPSAConfig, CRLQASConfig, CRLQASResult, crlqas, train_crlqas
    from .PPR_DQL import PPRDQLConfig, PPRDQLPolicy, PPRDQLResult, ppr_dql_state_to_circuit, train_ppr_dql
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    _OPTIONAL_RL_EXPORTS.extend(
        [
            "AdamSPSAConfig",
            "CRLQASConfig",
            "CRLQASResult",
            "PPRDQLConfig",
            "PPRDQLPolicy",
            "PPRDQLResult",
            "crlqas",
            "ppr_dql_state_to_circuit",
            "train_crlqas",
            "train_ppr_dql",
        ]
    )

__all__ = [
    "ArchitectureEvaluator",
    "ArchitectureScore",
    "ArchitectureSearch",
    "ArchitectureSpec",
    "ExpressibilityScore",
    "HardwareEfficiencyScore",
    "HardwareProfile",
    "IonTrapNoiseConfig",
    "KL_Haar_divergence",
    "KL_Haar_noisy",
    "KL_Haar_relative",
    "MMD_noisy",
    "MMD_relative",
    "MetricDefinition",
    "MetricGroupScore",
    "MultiSeedValidationReport",
    "MultiObjectiveReward",
    "NoiseAdaptiveQAS",
    "NoiseRobustnessScore",
    "NoiseSensitivityResult",
    "NoisyQASEnv",
    "OptimizerConfig",
    "ProblemInstance",
    "QASRewardWrapper",
    "QASState",
    "ResourceAllocationInstance",
    "RewardComposer",
    "RewardWeights",
    "SearchConfig",
    "SearchResult",
    "TaskEvaluationResult",
    "TrainabilityScore",
    "ValidationReport",
    "MaxCutInstance",
    "bind_parameters",
    "build_common_architectures",
    "common_architecture_names",
    "comparative_expressibility",
    "evaluate_task_objective",
    "evaluate_architectures",
    "expressibility_score",
    "ion_trap_error_budget_proxy",
    "load_default_ion_trap_noise_config",
    "load_ion_trap_noise_config",
    "maxcut_line",
    "maxcut_ring",
    "metric_catalog",
    "noise_sensitivity",
    "optimize_task_parameters",
    "run_validation_experiment",
    "run_multi_seed_validation_experiment",
    "small_resource_allocation",
] + _OPTIONAL_RL_EXPORTS
