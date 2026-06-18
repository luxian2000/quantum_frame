"""Quantum architecture search and state-synthesis utilities."""

from __future__ import annotations

from .core._types import (
    ArchitectureScore,
    ArchitectureSpec,
    MetricDefinition,
    MetricGroupScore,
    SearchConfig,
    SearchResult,
)
from .library.architectures import build_common_architectures, common_architecture_names
from .core.architecture_search import ArchitectureSearch, NoiseAdaptiveQAS
from .core.evaluator import ArchitectureEvaluator, evaluate_architectures, metric_catalog
from .algorithms.MoG_VQE import (
    MOGVQEBlock,
    MOGVQECandidate,
    MOGVQEConfig,
    MOGVQEIndividual,
    MOGVQEResult,
    block_hardware_efficient_ansatz,
    count_cnot_gates,
    extract_blocks_from_circuit,
    mutate_individual,
    non_dominated_sort,
    nsga_ii_select,
    pareto_front,
    run_mog_vqe,
)
from .core.reward import RewardComposer, RewardWeights
from .core.search_env import NoisyQASEnv, QASState
from .primitives.ansatz import (
    HEAMask,
    LayerwiseAnsatzGene,
    architecture_from_hea_mask,
    architecture_from_layerwise_gene,
    enumerate_hea_masks,
    sample_layerwise_genes,
)
from .primitives.backend_utils import backend_runtime_metadata, resolve_qas_backend
from .vqe_loop.fair_vqe import (
    THETA_INIT_RANDOM_UNIFORM_PI,
    THETA_INIT_ZERO_DIAGNOSTIC,
    COBYLA_RHOBEG,
    COBYLA_TOL,
    VQEOptimizationResult,
    adaptive_fair_n_starts,
    evaluate_h2_energy,
    evaluate_vqe_energy,
    is_b1_improvement_valid,
    optimize_h2_energy,
    optimize_vqe_energy,
    fair_vqe_final_maxfev,
    fair_vqe_screening_maxfev,
    fair_vqe_top_k,
)
from .vqe_loop import ClosedLoopConfig, ClosedLoopResult, run_vqe_qas_closed_loop, stamp_literal_hamiltonian_terms
from .problems.hamiltonians import (
    H2_HAMILTONIAN,
    H2_REFERENCE_ENERGY,
    ISING4_HAMILTONIAN,
    VQEDemoProblem,
    VQEProblem,
    exact_ground_energy,
    h2_demo_problem,
    h2_hamiltonian_matrix,
    hamiltonian_matrix,
    ising4_demo_problem,
    tfim_chain_demo_problem,
    tfim_chain_hamiltonian,
)
from ..metrics.expressibility import KL_Haar_divergence, KL_Haar_relative, MMD_relative
from ..metrics.hardware import HardwareProfile
from ..metrics.noisy_expressibility import (
    KL_Haar_noisy,
    MMD_noisy,
    comparative_expressibility,
    expressibility_score,
)
from ..noise import (
    IonTrapNoiseConfig,
    NoiseSensitivityResult,
    load_default_ion_trap_noise_config,
    load_ion_trap_noise_config,
    noise_sensitivity,
)
from ..noise.metrics import ion_trap_error_budget_proxy
from .core import config

_OPTIONAL_RL_EXPORTS: list[str] = []
try:
    from .algorithms.CRLQAS import AdamSPSAConfig, CRLQASConfig, CRLQASResult, crlqas, train_crlqas
    from .algorithms.PPR_DQL import PPRDQLConfig, PPRDQLPolicy, PPRDQLResult, ppr_dql_state_to_circuit, train_ppr_dql
    from .algorithms.PPO_RB import PPORollbackConfig, ppo_rb_qas
    from .core.runner import QASRunConfig, available_qas_methods, default_qas_config, run
    from .algorithms.supernet import (
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
    "H2_HAMILTONIAN",
    "H2_REFERENCE_ENERGY",
    "HEAMask",
    "ISING4_HAMILTONIAN",
    "HardwareProfile",
    "IonTrapNoiseConfig",
    "KL_Haar_divergence",
    "KL_Haar_noisy",
    "KL_Haar_relative",
    "MMD_noisy",
    "MMD_relative",
    "MOGVQEBlock",
    "MOGVQECandidate",
    "MOGVQEConfig",
    "MOGVQEIndividual",
    "MOGVQEResult",
    "MetricDefinition",
    "MetricGroupScore",
    "NoiseAdaptiveQAS",
    "NoiseSensitivityResult",
    "NoisyQASEnv",
    "QASState",
    "RewardComposer",
    "RewardWeights",
    "SearchConfig",
    "SearchResult",
    "THETA_INIT_RANDOM_UNIFORM_PI",
    "THETA_INIT_ZERO_DIAGNOSTIC",
    "COBYLA_RHOBEG",
    "COBYLA_TOL",
    "ClosedLoopConfig",
    "ClosedLoopResult",
    "VQEDemoProblem",
    "VQEProblem",
    "VQEOptimizationResult",
    "architecture_from_hea_mask",
    "architecture_from_layerwise_gene",
    "adaptive_fair_n_starts",
    "build_common_architectures",
    "block_hardware_efficient_ansatz",
    "common_architecture_names",
    "config",
    "comparative_expressibility",
    "count_cnot_gates",
    "evaluate_architectures",
    "evaluate_h2_energy",
    "evaluate_vqe_energy",
    "enumerate_hea_masks",
    "exact_ground_energy",
    "extract_blocks_from_circuit",
    "expressibility_score",
    "h2_demo_problem",
    "h2_hamiltonian_matrix",
    "hamiltonian_matrix",
    "is_b1_improvement_valid",
    "ion_trap_error_budget_proxy",
    "ising4_demo_problem",
    "load_default_ion_trap_noise_config",
    "load_ion_trap_noise_config",
    "metric_catalog",
    "mutate_individual",
    "non_dominated_sort",
    "noise_sensitivity",
    "nsga_ii_select",
    "optimize_h2_energy",
    "optimize_vqe_energy",
    "pareto_front",
    "resolve_qas_backend",
    "backend_runtime_metadata",
    "run_mog_vqe",
    "run_vqe_qas_closed_loop",
    "sample_layerwise_genes",
    "stamp_literal_hamiltonian_terms",
    "LayerwiseAnsatzGene",
    "tfim_chain_demo_problem",
    "tfim_chain_hamiltonian",
    "fair_vqe_final_maxfev",
    "fair_vqe_screening_maxfev",
    "fair_vqe_top_k",
] + _OPTIONAL_RL_EXPORTS
