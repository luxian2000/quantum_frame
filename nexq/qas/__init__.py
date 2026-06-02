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
from .vqe_hea_demo import (
    H2_HAMILTONIAN,
    H2_REFERENCE_ENERGY,
    HEAMask,
    ISING4_HAMILTONIAN,
    SABudgetSweepRow,
    SAMultiStartRow,
    VQEFitnessCorrelationReport,
    VQEFitnessCorrelationRow,
    VQEHEADemoReport,
    VQEBudgetSweepReport,
    VQEDemoProblem,
    VQEMultiStartSAReport,
    architecture_from_hea_mask,
    enumerate_hea_masks,
    evaluate_h2_energy,
    evaluate_vqe_energy,
    exact_ground_energy,
    h2_demo_problem,
    hamiltonian_matrix,
    ising4_demo_problem,
    mutate_hea_mask,
    optimize_h2_energy,
    optimize_vqe_energy,
    run_ising4_budget_sweep,
    run_ising4_fitness_correlation,
    run_ising4_multistart_sa,
    run_sa_search,
    run_vqe_hea_demo,
    run_vqe_ising4_demo,
    zero_cost_guardrail,
)
from .experiments import (
    MultiSeedValidationReport,
    RandomProxyValidationReport,
    RandomProxyValidationRow,
    StrategyComparisonReport,
    ValidationReport,
    run_hybrid_qas_validation_experiment,
    run_multi_seed_validation_experiment,
    run_random_proxy_validation_experiment,
    run_search_strategy_comparison,
    run_task_feedback_validation_experiment,
    run_validation_experiment,
)
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
    "H2_HAMILTONIAN",
    "H2_REFERENCE_ENERGY",
    "HEAMask",
    "ISING4_HAMILTONIAN",
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
    "RandomProxyValidationReport",
    "RandomProxyValidationRow",
    "ResourceAllocationInstance",
    "RewardComposer",
    "RewardWeights",
    "SABudgetSweepRow",
    "SAMultiStartRow",
    "SearchConfig",
    "SearchResult",
    "StrategyComparisonReport",
    "TaskEvaluationResult",
    "TrainabilityScore",
    "ValidationReport",
    "VQEBudgetSweepReport",
    "VQEFitnessCorrelationReport",
    "VQEFitnessCorrelationRow",
    "VQEHEADemoReport",
    "VQEDemoProblem",
    "VQEMultiStartSAReport",
    "MaxCutInstance",
    "architecture_from_hea_mask",
    "bind_parameters",
    "build_common_architectures",
    "common_architecture_names",
    "comparative_expressibility",
    "evaluate_task_objective",
    "evaluate_architectures",
    "evaluate_h2_energy",
    "evaluate_vqe_energy",
    "enumerate_hea_masks",
    "exact_ground_energy",
    "expressibility_score",
    "h2_demo_problem",
    "hamiltonian_matrix",
    "ion_trap_error_budget_proxy",
    "ising4_demo_problem",
    "load_default_ion_trap_noise_config",
    "load_ion_trap_noise_config",
    "maxcut_line",
    "maxcut_ring",
    "metric_catalog",
    "mutate_hea_mask",
    "noise_sensitivity",
    "optimize_task_parameters",
    "optimize_h2_energy",
    "optimize_vqe_energy",
    "run_ising4_budget_sweep",
    "run_ising4_fitness_correlation",
    "run_ising4_multistart_sa",
    "run_sa_search",
    "run_validation_experiment",
    "run_hybrid_qas_validation_experiment",
    "run_multi_seed_validation_experiment",
    "run_random_proxy_validation_experiment",
    "run_search_strategy_comparison",
    "run_task_feedback_validation_experiment",
    "run_vqe_hea_demo",
    "run_vqe_ising4_demo",
    "small_resource_allocation",
    "zero_cost_guardrail",
] + _OPTIONAL_RL_EXPORTS
