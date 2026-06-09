"""Minimal VQE-QAS demo utilities for a 2-qubit H2 toy Hamiltonian.

The goal of this module is not chemical accuracy. It provides a small, fully
self-contained demo for the pipeline:

zero-cost guardrail -> simulated annealing with short-step VQE energy -> final VQE validation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from pathlib import Path
from math import exp
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

try:  # pragma: no cover - fallback is tested implicitly when scipy is absent.
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None

from ..channel.backends.base import Backend
from ..channel.backends.numpy_backend import NumpyBackend
from ..measure.measure import Measure
from ._types import ArchitectureScore, ArchitectureSpec
from .architecture_candidates import build_common_architectures, qaoa_ansatz
from .evaluator import evaluate_architectures
from .task_evaluation import bind_parameters, parameter_count

try:  # pragma: no cover - optional accelerator backend.
    from ..channel.backends.npu_backend import NPUBackend
except Exception:  # pragma: no cover
    NPUBackend = None

try:  # pragma: no cover - optional torch backend.
    from ..channel.backends.torch_backend import TorchBackend
except Exception:  # pragma: no cover
    TorchBackend = None


H2_HAMILTONIAN = (
    (-1.0523732, "II"),
    (0.3979374, "ZI"),
    (-0.3979374, "IZ"),
    (-0.0112801, "ZZ"),
    (0.1809312, "XX"),
)
H2_REFERENCE_ENERGY = -1.8572
ISING4_HAMILTONIAN = (
    (-1.0, "ZZII"),
    (-1.0, "IZZI"),
    (-1.0, "IIZZ"),
    (-0.5, "XIII"),
    (-0.5, "IXII"),
    (-0.5, "IIXI"),
    (-0.5, "IIIX"),
)

ROTATION_BLOCKS = ("ry", "ry_rz", "rx_ry_rz")
ENTANGLERS = ("cx", "cz", "rzz")
FINAL_ROTATIONS = ("ry", "ry_rz")
ENTANGLE_PATTERNS = ("linear", "ring")
LAYER_CHOICES = (1, 2, 3)
THETA_INIT_RANDOM_UNIFORM_PI = "random_uniform_pi"
THETA_INIT_ZERO_DIAGNOSTIC = "zero"
V3_COBYLA_RHOBEG = 1.0
V3_COBYLA_TOL = 1e-6


def v3_screening_maxfev(n_params: int) -> int:
    """V3 low-fidelity proxy budget, counted as selection cost."""
    return max(500, 80 * max(1, int(n_params)))


def v3_final_maxfev(n_params: int) -> int:
    """V3 final-label COBYLA budget for fair VQE."""
    return max(1000, 200 * max(1, int(n_params)))


def v3_top_k(candidate_count: int) -> int:
    """V3 deployment top-K rule."""
    return min(max(0, int(candidate_count)), max(10, int(np.ceil(0.1 * max(0, int(candidate_count))))))


def resolve_qas_backend(
    kind: Optional[str] = None,
    fallback_to_cpu: bool = True,
    dtype: Optional[str] = None,
) -> Backend:
    """Resolve the QAS demo backend. Defaults to CPU/NumPy unless explicitly requested."""
    import os

    backend_kind = (kind or os.environ.get("AICIR_QAS_BACKEND") or "numpy").strip().lower()
    dtype_name = (dtype or os.environ.get("AICIR_QAS_DTYPE") or "").strip().lower()
    numpy_dtype = None
    torch_dtype = None
    if dtype_name:
        if dtype_name in {"complex128", "c128", "float64"}:
            numpy_dtype = np.complex128
        elif dtype_name in {"complex64", "c64", "float32"}:
            numpy_dtype = np.complex64
        else:
            raise ValueError(f"Unsupported QAS dtype: {dtype_name!r}. Use complex64 or complex128.")
        if backend_kind in {"torch", "npu"}:
            try:
                import torch
            except Exception as exc:  # pragma: no cover - depends on optional torch install.
                raise RuntimeError(f"Backend {backend_kind!r} requires torch to honor dtype={dtype_name!r}") from exc
            torch_dtype = torch.complex128 if numpy_dtype == np.complex128 else torch.complex64
    if backend_kind in {"numpy", "cpu"}:
        return NumpyBackend(dtype=numpy_dtype)
    if backend_kind == "torch":
        if TorchBackend is None:
            raise RuntimeError("TorchBackend is unavailable; install torch or use AICIR_QAS_BACKEND=numpy.")
        device = os.environ.get("AICIR_QAS_TORCH_DEVICE") or "cpu"
        return TorchBackend(dtype=torch_dtype, device=device)
    if backend_kind == "npu":
        if NPUBackend is None:
            if fallback_to_cpu:
                return NumpyBackend(dtype=numpy_dtype)
            raise RuntimeError("NPUBackend is unavailable; install torch_npu or use AICIR_QAS_BACKEND=numpy.")
        return NPUBackend.from_distributed_env(dtype=torch_dtype, fallback_to_cpu=fallback_to_cpu)
    raise ValueError(f"Unsupported QAS backend: {backend_kind!r}. Use numpy, cpu, torch, or npu.")


def backend_runtime_metadata(backend: Backend) -> Dict[str, Any]:
    """Return actual backend provenance for result manifests."""
    dtype = getattr(backend, "_dtype", None)
    device = getattr(backend, "_device", None)
    return {
        "backend_name": getattr(backend, "name", type(backend).__name__),
        "backend_class": type(backend).__name__,
        "backend_dtype": str(dtype) if dtype is not None else None,
        "backend_device": str(device) if device is not None else None,
    }


@dataclass(frozen=True)
class VQEDemoProblem:
    """Small Pauli-Hamiltonian VQE task used by the demo pipeline."""

    name: str
    n_qubits: int
    hamiltonian: Sequence[tuple[float, str]]
    reference_energy: float


@dataclass(frozen=True)
class TFIMReferenceAlignmentRow:
    n_qubits: int
    J: float
    h: float
    periodic: bool
    dense_energy: float
    free_fermion_energy: float
    abs_diff: float


@dataclass(frozen=True)
class TFIMReferenceAlignmentReport:
    rows: List[TFIMReferenceAlignmentRow]
    tolerance: float = 1e-9

    @property
    def passed(self) -> bool:
        return all(row.abs_diff < float(self.tolerance) for row in self.rows)

    def summary_lines(self) -> List[str]:
        lines = [
            "TFIM reference alignment",
            f"tolerance: {self.tolerance:.1e} | passed: {self.passed}",
            "n_qubits | J | h | boundary | dense | free_fermion | abs_diff",
        ]
        for row in self.rows:
            boundary = "PBC" if row.periodic else "OBC"
            lines.append(
                f"{row.n_qubits} | {row.J:g} | {row.h:g} | {boundary} | "
                f"{row.dense_energy:.12f} | {row.free_fermion_energy:.12f} | {row.abs_diff:.3e}"
            )
        return lines


@dataclass(frozen=True)
class HEAMask:
    """Small HEA-style architecture mask for VQE-QAS demo search."""

    n_qubits: int = 2
    layers: int = 1
    rotation_block: str = "ry_rz"
    entangler: str = "cx"
    final_rotation: str = "ry"
    entangle_pattern: str = "linear"

    def key(self) -> tuple[Any, ...]:
        return (
            self.n_qubits,
            self.layers,
            self.rotation_block,
            self.entangler,
            self.final_rotation,
            self.entangle_pattern,
        )

    def label(self) -> str:
        return (
            f"hea_mask_L{self.layers}_{self.rotation_block}_{self.entangler}_"
            f"{self.entangle_pattern}_{self.final_rotation}"
        )


@dataclass
class VQEOptimizationResult:
    architecture: ArchitectureSpec
    energy: float
    best_parameters: List[float]
    evaluations: int
    n_starts: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage1Row:
    architecture: ArchitectureSpec
    score: ArchitectureScore
    kept: bool
    reason: str


@dataclass
class SAStep:
    step: int
    temperature: float
    current_energy: float
    candidate_energy: float
    best_energy: float
    accepted: bool
    mask: HEAMask


@dataclass
class VQEHEADemoReport:
    stage1_rows: List[Stage1Row]
    sa_trace: List[SAStep]
    final_results: List[VQEOptimizationResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary_lines(self) -> List[str]:
        reference_energy = float(self.metadata.get("reference_energy", H2_REFERENCE_ENERGY))
        problem_name = str(self.metadata.get("problem_name", "h2_toy_2q"))
        kept_count = sum(1 for row in self.stage1_rows if row.kept)
        filtered_count = len(self.stage1_rows) - kept_count
        lines = [
            f"VQE-QAS HEA demo: {problem_name}",
            f"reference_energy: {reference_energy:.6f}",
            "",
            "Stage 1 zero-cost guardrail",
            f"candidates: {len(self.stage1_rows)} | kept: {kept_count} | filtered: {filtered_count}",
            "metric | min | p25 | max",
        ]
        for name, values in _stage1_metric_summary(self.stage1_rows).items():
            lines.append(f"{name} | {values['min']:.4f} | {values['p25']:.4f} | {values['max']:.4f}")
        lines.extend(
            [
                "",
                "Stage 1 rows",
            "name | kept | reason | weighted | expr | train | noise | hardware",
            ]
        )
        for row in _stage1_digest(self.stage1_rows):
            score = row.score
            lines.append(
                f"{row.architecture.name} | {row.kept} | {row.reason} | "
                f"{score.weighted_score:.4f} | {score.expressibility.score:.4f} | "
                f"{score.trainability.score:.4f} | {score.noise_robustness.score:.4f} | "
                f"{score.hardware_efficiency.score:.4f}"
            )
        lines.extend(["", "Stage 2 simulated annealing trace", "step | T | candidate | current | best | accepted | mask"])
        for item in _trace_digest(self.sa_trace):
            lines.append(
                f"{item.step} | {item.temperature:.5f} | {item.candidate_energy:.6f} | "
                f"{item.current_energy:.6f} | {item.best_energy:.6f} | {item.accepted} | {item.mask.label()}"
            )
        lines.extend(["", "Final VQE validation", "name | energy | delta_ref | evals | starts | n_params | source"])
        for result in self.final_results:
            source = result.metadata.get("source", "-")
            delta = result.energy - reference_energy
            n_params = parameter_count(result.architecture.circuit)
            lines.append(
                f"{result.architecture.name} | {result.energy:.6f} | {delta:.6f} | "
                f"{result.evaluations} | {result.n_starts} | {n_params} | {source}"
            )
        return lines


@dataclass
class SABudgetSweepRow:
    steps: int
    short_best_energy: float
    final_capped_energy: float
    final_fair_energy: float
    mask: HEAMask
    n_params: int
    capped_evaluations: int
    fair_evaluations: int


@dataclass
class VQEBudgetSweepReport:
    stage1_rows: List[Stage1Row]
    sweep_rows: List[SABudgetSweepRow]
    capped_results: List[VQEOptimizationResult]
    fair_results: List[VQEOptimizationResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary_lines(self) -> List[str]:
        problem_name = str(self.metadata.get("problem_name", "unknown"))
        reference_energy = float(self.metadata.get("reference_energy", 0.0))
        kept_count = sum(1 for row in self.stage1_rows if row.kept)
        lines = [
            f"VQE-QAS SA budget sweep: {problem_name}",
            f"reference_energy: {reference_energy:.6f}",
            f"stage1 candidates: {len(self.stage1_rows)} | kept: {kept_count} | filtered: {len(self.stage1_rows) - kept_count}",
            f"T_init: {self.metadata.get('t_init')} | T_final: {self.metadata.get('t_final')}",
            "",
            "SA budget sweep",
            "steps | short_best | final_capped | final_fair | n_params | capped_evals | fair_evals | mask",
        ]
        for row in self.sweep_rows:
            lines.append(
                f"{row.steps} | {row.short_best_energy:.6f} | {row.final_capped_energy:.6f} | "
                f"{row.final_fair_energy:.6f} | {row.n_params} | {row.capped_evaluations} | "
                f"{row.fair_evaluations} | {row.mask.label()}"
            )
        lines.extend(
            [
                "",
                "Final validation: capped budget",
                "name | energy | delta_ref | evals | n_params | source",
            ]
        )
        for result in self.capped_results:
            source = result.metadata.get("source", "-")
            n_params = parameter_count(result.architecture.circuit)
            lines.append(
                f"{result.architecture.name} | {result.energy:.6f} | {result.energy - reference_energy:.6f} | "
                f"{result.evaluations} | {n_params} | {source}"
            )
        lines.extend(
            [
                "",
                "Final validation: per-param fair budget",
                "name | energy | delta_ref | evals | n_params | source",
            ]
        )
        for result in self.fair_results:
            source = result.metadata.get("source", "-")
            n_params = parameter_count(result.architecture.circuit)
            lines.append(
                f"{result.architecture.name} | {result.energy:.6f} | {result.energy - reference_energy:.6f} | "
                f"{result.evaluations} | {n_params} | {source}"
            )
        return lines


@dataclass
class SAMultiStartRow:
    start_rank: int
    start_mask: HEAMask
    best_mask: HEAMask
    short_best_energy: float
    final_capped_energy: float
    final_fair_energy: float
    n_params: int
    capped_evaluations: int
    fair_evaluations: int


@dataclass
class VQEMultiStartSAReport:
    stage1_rows: List[Stage1Row]
    restart_rows: List[SAMultiStartRow]
    capped_results: List[VQEOptimizationResult]
    fair_results: List[VQEOptimizationResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary_lines(self) -> List[str]:
        problem_name = str(self.metadata.get("problem_name", "unknown"))
        reference_energy = float(self.metadata.get("reference_energy", 0.0))
        kept_count = sum(1 for row in self.stage1_rows if row.kept)
        lines = [
            f"VQE-QAS multi-start SA: {problem_name}",
            f"reference_energy: {reference_energy:.6f}",
            f"stage1 candidates: {len(self.stage1_rows)} | kept: {kept_count} | filtered: {len(self.stage1_rows) - kept_count}",
            f"starts: {self.metadata.get('n_starts')} | steps_per_start: {self.metadata.get('steps_per_start')}",
            f"T_init: {self.metadata.get('t_init')} | T_final: {self.metadata.get('t_final')}",
            "",
            "Multi-start SA",
            "start_rank | short_best | final_capped | final_fair | n_params | start_mask | best_mask",
        ]
        for row in self.restart_rows:
            lines.append(
                f"{row.start_rank} | {row.short_best_energy:.6f} | {row.final_capped_energy:.6f} | "
                f"{row.final_fair_energy:.6f} | {row.n_params} | {row.start_mask.label()} | {row.best_mask.label()}"
            )
        lines.extend(["", "Final validation: capped budget", "name | energy | delta_ref | evals | n_params | source"])
        for result in self.capped_results:
            source = result.metadata.get("source", "-")
            n_params = parameter_count(result.architecture.circuit)
            lines.append(
                f"{result.architecture.name} | {result.energy:.6f} | {result.energy - reference_energy:.6f} | "
                f"{result.evaluations} | {n_params} | {source}"
            )
        lines.extend(["", "Final validation: per-param fair budget", "name | energy | delta_ref | evals | n_params | source"])
        for result in self.fair_results:
            source = result.metadata.get("source", "-")
            n_params = parameter_count(result.architecture.circuit)
            lines.append(
                f"{result.architecture.name} | {result.energy:.6f} | {result.energy - reference_energy:.6f} | "
                f"{result.evaluations} | {n_params} | {source}"
            )
        return lines


@dataclass
class VQEFitnessCorrelationRow:
    rank: int
    architecture: ArchitectureSpec
    short_energy: float
    fair_energy: float
    n_params: int
    short_evaluations: int
    fair_evaluations: int


@dataclass
class VQEFitnessCorrelationReport:
    stage1_rows: List[Stage1Row]
    rows: List[VQEFitnessCorrelationRow]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def correlations(self) -> Dict[str, float]:
        short = [row.short_energy for row in self.rows]
        fair = [row.fair_energy for row in self.rows]
        overlap_k = min(int(self.metadata.get("overlap_k", 5)), len(self.rows))
        return {
            "pearson_short_vs_fair": _pearson(short, fair),
            "spearman_short_vs_fair": _spearman(short, fair),
            f"top{overlap_k}_overlap": float(_topk_overlap(self.rows, overlap_k)),
        }

    def summary_lines(self) -> List[str]:
        problem_name = str(self.metadata.get("problem_name", "unknown"))
        reference_energy = float(self.metadata.get("reference_energy", 0.0))
        correlations = self.correlations()
        lines = [
            f"VQE-QAS short/fair fitness validation: {problem_name}",
            f"reference_energy: {reference_energy:.6f}",
            f"n_candidates: {len(self.rows)}",
            "correlation | value",
        ]
        for name, value in correlations.items():
            lines.append(f"{name} | {value:.4f}")
        lines.extend(["", "rows", "rank | short | fair | delta_ref | n_params | short_evals | fair_evals | name"])
        for row in self.rows:
            lines.append(
                f"{row.rank} | {row.short_energy:.6f} | {row.fair_energy:.6f} | "
                f"{row.fair_energy - reference_energy:.6f} | {row.n_params} | "
                f"{row.short_evaluations} | {row.fair_evaluations} | {row.architecture.name}"
            )
        return lines


@dataclass
class VQEFitnessBudgetRow:
    rank: int
    architecture: ArchitectureSpec
    fair_energy: float
    n_params: int
    fair_evaluations: int
    short_by_budget: Dict[int, float] = field(default_factory=dict)
    short_evals_by_budget: Dict[int, int] = field(default_factory=dict)
    weighted_score: float = 0.0
    expressibility: float = 0.0
    trainability: float = 0.0
    noise: float = 0.0
    hardware: float = 0.0


@dataclass
class VQEFitnessBudgetSweepReport:
    stage1_rows: List[Stage1Row]
    rows: List[VQEFitnessBudgetRow]
    budgets: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def budget_correlations(self) -> Dict[int, Dict[str, float]]:
        fair = [row.fair_energy for row in self.rows]
        result: Dict[int, Dict[str, float]] = {}
        for budget in self.budgets:
            short = [row.short_by_budget[int(budget)] for row in self.rows]
            overlap_k = min(int(self.metadata.get("overlap_k", 5)), len(self.rows))
            result[int(budget)] = {
                "pearson": _pearson(short, fair),
                "spearman": _spearman(short, fair),
                "topk_overlap": float(_topk_overlap_pairs(short, fair, overlap_k)),
            }
        return result

    def zero_cost_correlations(self) -> Dict[str, Dict[str, float]]:
        fair = [row.fair_energy for row in self.rows]
        metrics = {
            "weighted": [row.weighted_score for row in self.rows],
            "expressibility": [row.expressibility for row in self.rows],
            "trainability": [row.trainability for row in self.rows],
            "noise": [row.noise for row in self.rows],
            "hardware": [row.hardware for row in self.rows],
        }
        overlap_k = min(int(self.metadata.get("overlap_k", 5)), len(self.rows))
        # zero-cost scores are larger-is-better, while energy is smaller-is-better.
        return {
            name: {
                "pearson": _pearson([-value for value in values], fair),
                "spearman": _spearman([-value for value in values], fair),
                "topk_overlap": float(_topk_overlap_pairs([-value for value in values], fair, overlap_k)),
            }
            for name, values in metrics.items()
        }

    def summary_lines(self) -> List[str]:
        problem_name = str(self.metadata.get("problem_name", "unknown"))
        reference_energy = float(self.metadata.get("reference_energy", 0.0))
        overlap_k = min(int(self.metadata.get("overlap_k", 5)), len(self.rows))
        lines = [
            f"VQE-QAS fitness budget sweep: {problem_name}",
            f"reference_energy: {reference_energy:.6f}",
            f"n_candidates: {len(self.rows)} | topk_overlap_k: {overlap_k}",
            "",
            "Improvement diagnostics",
            "rank | fair | n_params | improvement_from_short40 | name",
        ]
        first_budget = int(self.budgets[0])
        for row in self.rows:
            improvement = row.short_by_budget[first_budget] - row.fair_energy
            lines.append(
                f"{row.rank} | {row.fair_energy:.6f} | {row.n_params} | "
                f"{improvement:.6f} | {row.architecture.name}"
            )
        lines.extend(["", "Short-budget correlation", "budget | pearson | spearman | topk_overlap"])
        for budget, values in self.budget_correlations().items():
            lines.append(
                f"{budget} | {values['pearson']:.4f} | {values['spearman']:.4f} | "
                f"{int(values['topk_overlap'])}/{overlap_k}"
            )
        lines.extend(["", "Zero-cost vs fair", "metric | pearson | spearman | topk_overlap"])
        for name, values in self.zero_cost_correlations().items():
            lines.append(
                f"{name} | {values['pearson']:.4f} | {values['spearman']:.4f} | "
                f"{int(values['topk_overlap'])}/{overlap_k}"
            )
        lines.extend(["", "Rows", "rank | fair | " + " | ".join(f"short_{budget}" for budget in self.budgets) + " | weighted | expr | train | noise | hardware | name"])
        for row in self.rows:
            short_values = " | ".join(f"{row.short_by_budget[int(budget)]:.6f}" for budget in self.budgets)
            lines.append(
                f"{row.rank} | {row.fair_energy:.6f} | {short_values} | "
                f"{row.weighted_score:.4f} | {row.expressibility:.4f} | {row.trainability:.4f} | "
                f"{row.noise:.4f} | {row.hardware:.4f} | {row.architecture.name}"
            )
        return lines


@dataclass(frozen=True)
class HamiltonianProfile:
    """Coarse Pauli-term profile used for first-pass task-aware HEA mutations."""

    n_qubits: int
    total_weight: float
    zz_weight: float = 0.0
    xx_weight: float = 0.0
    yy_weight: float = 0.0
    single_z_weight: float = 0.0
    single_x_weight: float = 0.0
    single_y_weight: float = 0.0
    other_weight: float = 0.0
    max_pauli_distance: int = 0
    nearest_neighbor_weight: float = 0.0
    long_range_weight: float = 0.0

    def ratio(self, name: str) -> float:
        if self.total_weight <= 1e-12:
            return 0.0
        return float(getattr(self, name)) / float(self.total_weight)


@dataclass
class VQEB2ReliabilityRow:
    rank: int
    architecture: ArchitectureSpec
    b1_energy: float
    b2_energy: float
    fair_energy: float
    n_params: int
    b1_evaluations: int
    b2_evaluations: int
    fair_evaluations: int
    family: str
    improvement_valid: bool = True

    @property
    def improvement(self) -> float:
        return float(self.b1_energy - self.b2_energy)


@dataclass
class VQEB2ReliabilityReport:
    stage1_rows: List[Stage1Row]
    rows: List[VQEB2ReliabilityRow]
    profile: HamiltonianProfile
    b1_survivor_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def correlations(self) -> Dict[str, float]:
        b1 = [row.b1_energy for row in self.rows]
        b2 = [row.b2_energy for row in self.rows]
        valid_improvement_rows = [row for row in self.rows if row.improvement_valid]
        improvement = [row.improvement for row in valid_improvement_rows]
        fair_for_improvement = [row.fair_energy for row in valid_improvement_rows]
        fair = [row.fair_energy for row in self.rows]
        overlap_k = min(int(self.metadata.get("overlap_k", 5)), len(self.rows))
        improvement_overlap_k = min(int(self.metadata.get("overlap_k", 5)), len(valid_improvement_rows))
        return {
            "spearman_b1_vs_fair": _spearman(b1, fair),
            "spearman_b2_vs_fair": _spearman(b2, fair),
            "pearson_b2_vs_fair": _pearson(b2, fair),
            f"b2_top{overlap_k}_overlap": float(_topk_overlap_pairs(b2, fair, overlap_k)),
            "valid_improvement_rows": float(len(valid_improvement_rows)),
            "spearman_improvement_vs_fair": _spearman([-value for value in improvement], fair_for_improvement),
            f"improvement_top{improvement_overlap_k}_overlap": float(
                _topk_overlap_pairs([-value for value in improvement], fair_for_improvement, improvement_overlap_k)
            ),
        }

    def fair_energy_spread(self) -> float:
        if not self.rows:
            return 0.0
        fair = [row.fair_energy for row in self.rows]
        return float(max(fair) - min(fair))

    def summary_lines(self) -> List[str]:
        problem_name = str(self.metadata.get("problem_name", "unknown"))
        reference_energy = float(self.metadata.get("reference_energy", 0.0))
        correlations = self.correlations()
        lines = [
            f"VQE-QAS B2 reliability experiment: {problem_name}",
            f"reference_energy: {reference_energy:.6f}",
            f"stage1 candidates: {len(self.stage1_rows)} | B1 survivors: {self.b1_survivor_count} | evaluated: {len(self.rows)}",
            f"fair_energy_spread: {self.fair_energy_spread():.6f}",
            (
                "hamiltonian_profile: "
                f"ZZ={self.profile.ratio('zz_weight'):.3f}, "
                f"XX={self.profile.ratio('xx_weight'):.3f}, "
                f"YY={self.profile.ratio('yy_weight'):.3f}, "
                f"singleZ={self.profile.ratio('single_z_weight'):.3f}, "
                f"singleX={self.profile.ratio('single_x_weight'):.3f}"
            ),
            "",
            "Correlation diagnostics",
            "metric | value",
        ]
        for name, value in correlations.items():
            lines.append(f"{name} | {value:.4f}")
        lines.extend(["", "Rows ranked by B2 energy", "rank | b1 | b2 | improvement | improvement_valid | fair | n_params | b1_evals | b2_evals | fair_evals | family | name"])
        for row in self.rows:
            lines.append(
                f"{row.rank} | {row.b1_energy:.6f} | {row.b2_energy:.6f} | {row.improvement:.6f} | {row.improvement_valid} | {row.fair_energy:.6f} | "
                f"{row.n_params} | {row.b1_evaluations} | {row.b2_evaluations} | {row.fair_evaluations} | "
                f"{row.family} | {row.architecture.name}"
            )
        return lines


@dataclass
class VQEFairStabilityRow:
    rank: int
    architecture: ArchitectureSpec
    best_energy: float
    mean_energy: float
    std_energy: float
    worst_energy: float
    n_params: int
    repeat_energies: List[float]
    repeat_evaluations: List[int]
    absolute_sr_10mha: float = 0.0
    absolute_sr_20mha: float = 0.0
    posthoc_basin_sr_5mha: float = 0.0


@dataclass
class VQEFairStabilityReport:
    rows: List[VQEFairStabilityRow]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary_lines(self) -> List[str]:
        problem_name = str(self.metadata.get("problem_name", "unknown"))
        reference_energy = float(self.metadata.get("reference_energy", 0.0))
        lines = [
            f"VQE-QAS fair VQE stability: {problem_name}",
            f"reference_energy: {reference_energy:.6f}",
            f"top_k: {self.metadata.get('top_k')} | repeats: {self.metadata.get('repeats')}",
            "",
            "Rows",
            "rank | best | mean | std | worst | delta_best_ref | n_params | nfev | absolute_SR10 | absolute_SR20 | posthoc_basin_SR5 | energies | name",
        ]
        for row in self.rows:
            energies = ",".join(f"{energy:.6f}" for energy in row.repeat_energies)
            nfev = ",".join(str(value) for value in row.repeat_evaluations)
            lines.append(
                f"{row.rank} | {row.best_energy:.6f} | {row.mean_energy:.6f} | {row.std_energy:.6f} | "
                f"{row.worst_energy:.6f} | {row.best_energy - reference_energy:.6f} | {row.n_params} | {nfev} | "
                f"{row.absolute_sr_10mha:.3f} | {row.absolute_sr_20mha:.3f} | {row.posthoc_basin_sr_5mha:.3f} | "
                f"{energies} | {row.architecture.name}"
            )
        return lines


@dataclass
class VQEEnumerationReport:
    results: List[VQEOptimizationResult]
    profile: HamiltonianProfile
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary_lines(self, top_k: int = 12) -> List[str]:
        problem_name = str(self.metadata.get("problem_name", "unknown"))
        reference_energy = float(self.metadata.get("reference_energy", 0.0))
        lines = [
            f"VQE-QAS full enumeration baseline: {problem_name}",
            f"reference_energy: {reference_energy:.6f}",
            f"n_candidates: {len(self.results)}",
            (
                "hamiltonian_profile: "
                f"ZZ={self.profile.ratio('zz_weight'):.3f}, "
                f"XX={self.profile.ratio('xx_weight'):.3f}, "
                f"singleX={self.profile.ratio('single_x_weight'):.3f}"
            ),
            "",
            "Top fair VQE candidates",
            "rank | fair | delta_ref | n_params | evals | family | name",
        ]
        for rank, result in enumerate(self.results[: max(1, int(top_k))], start=1):
            n_params = parameter_count(result.architecture.circuit)
            lines.append(
                f"{rank} | {result.energy:.6f} | {result.energy - reference_energy:.6f} | "
                f"{n_params} | {result.evaluations} | {get_structure_family(result.architecture)} | {result.architecture.name}"
            )
        return lines


@dataclass
class VQETrainabilityPriorReport:
    stage1_rows: List[Stage1Row]
    trainability_top_results: List[VQEOptimizationResult]
    sa_trace: List[SAStep]
    sa_final_result: VQEOptimizationResult
    baseline_results: List[VQEOptimizationResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary_lines(self) -> List[str]:
        problem_name = str(self.metadata.get("problem_name", "unknown"))
        reference_energy = float(self.metadata.get("reference_energy", 0.0))
        kept_count = sum(1 for row in self.stage1_rows if row.kept)
        lines = [
            f"VQE-QAS trainability-prior demo: {problem_name}",
            f"reference_energy: {reference_energy:.6f}",
            f"stage1 candidates: {len(self.stage1_rows)} | kept: {kept_count} | filtered: {len(self.stage1_rows) - kept_count}",
            "",
            "Trainability-ranked Stage1 rows",
            "rank | train | weighted | expr | noise | hardware | name",
        ]
        for rank, row in enumerate(_rank_stage1_rows(self.stage1_rows, "trainability")[:10], start=1):
            score = row.score
            lines.append(
                f"{rank} | {score.trainability.score:.4f} | {score.weighted_score:.4f} | "
                f"{score.expressibility.score:.4f} | {score.noise_robustness.score:.4f} | "
                f"{score.hardware_efficiency.score:.4f} | {row.architecture.name}"
            )
        lines.extend(["", "Trainability top fair final", "name | energy | delta_ref | evals | n_params | source"])
        for result in self.trainability_top_results:
            n_params = parameter_count(result.architecture.circuit)
            lines.append(
                f"{result.architecture.name} | {result.energy:.6f} | {result.energy - reference_energy:.6f} | "
                f"{result.evaluations} | {n_params} | {result.metadata.get('source', '-')}"
            )
        lines.extend(["", "Diagnostic: trainability-seeded SA trace", "step | T | candidate | current | best | accepted | mask"])
        for item in _trace_digest(self.sa_trace):
            lines.append(
                f"{item.step} | {item.temperature:.5f} | {item.candidate_energy:.6f} | "
                f"{item.current_energy:.6f} | {item.best_energy:.6f} | {item.accepted} | {item.mask.label()}"
            )
        lines.extend(["", "Diagnostic: SA final vs baselines", "name | energy | delta_ref | evals | n_params | source"])
        combined = [self.sa_final_result] + list(self.baseline_results)
        combined.sort(key=lambda result: result.energy)
        for result in combined:
            n_params = parameter_count(result.architecture.circuit)
            lines.append(
                f"{result.architecture.name} | {result.energy:.6f} | {result.energy - reference_energy:.6f} | "
                f"{result.evaluations} | {n_params} | {result.metadata.get('source', '-')}"
            )
        return lines


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or len(right) < 2:
        return 0.0
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    cursor = 0
    while cursor < len(array):
        end = cursor + 1
        while end < len(array) and abs(array[order[end]] - array[order[cursor]]) <= 1e-12:
            end += 1
        avg_rank = 0.5 * (cursor + end - 1) + 1.0
        ranks[order[cursor:end]] = avg_rank
        cursor = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or len(right) < 2:
        return 0.0
    return _pearson(_rankdata(left), _rankdata(right))


def _topk_overlap(rows: Sequence[VQEFitnessCorrelationRow], k: int) -> int:
    if k <= 0:
        return 0
    short_top = {id(row) for row in sorted(rows, key=lambda item: item.short_energy)[:k]}
    fair_top = {id(row) for row in sorted(rows, key=lambda item: item.fair_energy)[:k]}
    return len(short_top & fair_top)


def _topk_overlap_pairs(left: Sequence[float], right: Sequence[float], k: int) -> int:
    if k <= 0:
        return 0
    left_order = np.argsort(np.asarray(left, dtype=float), kind="mergesort")[:k]
    right_order = np.argsort(np.asarray(right, dtype=float), kind="mergesort")[:k]
    return len(set(int(value) for value in left_order) & set(int(value) for value in right_order))


def _stage1_metric_summary(rows: Sequence[Stage1Row]) -> Dict[str, Dict[str, float]]:
    groups = {
        "expressibility": [row.score.expressibility.score for row in rows],
        "trainability": [row.score.trainability.score for row in rows],
        "noise": [row.score.noise_robustness.score for row in rows],
        "hardware": [row.score.hardware_efficiency.score for row in rows],
    }
    summary = {}
    for name, values in groups.items():
        array = np.asarray(values, dtype=float)
        summary[name] = {
            "min": float(np.min(array)),
            "p25": float(np.quantile(array, 0.25)),
            "max": float(np.max(array)),
        }
    return summary


def _stage1_digest(rows: Sequence[Stage1Row]) -> List[Stage1Row]:
    if len(rows) <= 14:
        return list(rows)
    kept = [row for row in rows if row.kept][:10]
    filtered = [row for row in rows if not row.kept][:4]
    return kept + filtered


def _trace_digest(trace: Sequence[SAStep]) -> List[SAStep]:
    if len(trace) <= 12:
        return list(trace)
    selected = []
    for item in trace:
        if item.step <= 3 or item.accepted or item.step == trace[-1].step or item.step % 5 == 0:
            selected.append(item)
    return selected[:18]


def enumerate_hea_masks(n_qubits: int = 2) -> List[HEAMask]:
    masks = []
    for layers in LAYER_CHOICES:
        for rotation in ROTATION_BLOCKS:
            for entangler in ENTANGLERS:
                for final_rotation in FINAL_ROTATIONS:
                    for pattern in ENTANGLE_PATTERNS:
                        masks.append(
                            HEAMask(
                                n_qubits=n_qubits,
                                layers=layers,
                                rotation_block=rotation,
                                entangler=entangler,
                                final_rotation=final_rotation,
                                entangle_pattern=pattern,
                            )
                        )
    return masks


def mutate_hea_mask(mask: HEAMask, rng: np.random.Generator) -> HEAMask:
    """Return a neighbor by changing exactly one HEAMask dimension."""
    fields = ["layers", "rotation_block", "entangler", "final_rotation", "entangle_pattern"]
    field_name = fields[int(rng.integers(0, len(fields)))]
    choices = {
        "layers": LAYER_CHOICES,
        "rotation_block": ROTATION_BLOCKS,
        "entangler": ENTANGLERS,
        "final_rotation": FINAL_ROTATIONS,
        "entangle_pattern": ENTANGLE_PATTERNS,
    }[field_name]
    old_value = getattr(mask, field_name)
    alternatives = [choice for choice in choices if choice != old_value]
    new_value = alternatives[int(rng.integers(0, len(alternatives)))]
    return replace(mask, **{field_name: new_value})


def analyze_hamiltonian(hamiltonian: Sequence[tuple[float, str]]) -> HamiltonianProfile:
    """Summarize Pauli-term weights for first-version Hamiltonian-aware rules."""
    if not hamiltonian:
        raise ValueError("Hamiltonian must contain at least one Pauli term")
    n_qubits = len(hamiltonian[0][1])
    totals: Dict[str, float] = {
        "zz_weight": 0.0,
        "xx_weight": 0.0,
        "yy_weight": 0.0,
        "single_z_weight": 0.0,
        "single_x_weight": 0.0,
        "single_y_weight": 0.0,
        "other_weight": 0.0,
        "nearest_neighbor_weight": 0.0,
        "long_range_weight": 0.0,
    }
    max_distance = 0
    total_weight = 0.0
    for coeff, pauli in hamiltonian:
        if len(pauli) != n_qubits:
            raise ValueError("All Pauli labels must have the same length")
        weight = abs(float(coeff))
        total_weight += weight
        active = [(index, label) for index, label in enumerate(pauli) if label != "I"]
        labels = [label for _, label in active]
        if len(active) == 1:
            if labels == ["Z"]:
                totals["single_z_weight"] += weight
            elif labels == ["X"]:
                totals["single_x_weight"] += weight
            elif labels == ["Y"]:
                totals["single_y_weight"] += weight
            else:
                totals["other_weight"] += weight
            continue
        if len(active) == 2:
            distance = abs(active[0][0] - active[1][0])
            max_distance = max(max_distance, distance)
            if distance == 1:
                totals["nearest_neighbor_weight"] += weight
            else:
                totals["long_range_weight"] += weight
            if labels == ["Z", "Z"]:
                totals["zz_weight"] += weight
            elif labels == ["X", "X"]:
                totals["xx_weight"] += weight
            elif labels == ["Y", "Y"]:
                totals["yy_weight"] += weight
            else:
                totals["other_weight"] += weight
            continue
        totals["other_weight"] += weight
    return HamiltonianProfile(
        n_qubits=n_qubits,
        total_weight=float(total_weight),
        max_pauli_distance=int(max_distance),
        **totals,
    )


def hamiltonian_aware_mask_preferences(profile: HamiltonianProfile) -> Dict[str, tuple[Any, ...]]:
    """Translate a Hamiltonian profile into conservative HEA mutation preferences."""
    pair_weight = profile.zz_weight + profile.xx_weight + profile.yy_weight
    rotation_preferences: List[str] = []
    entangler_preferences: List[str] = []
    final_preferences: List[str] = []
    pattern_preferences: List[str] = []

    if profile.zz_weight >= max(profile.xx_weight, profile.yy_weight, 1e-12):
        entangler_preferences.append("rzz")
    if profile.xx_weight + profile.yy_weight > 0.20 * max(profile.total_weight, 1e-12):
        rotation_preferences.append("rx_ry_rz")
    if profile.single_x_weight > profile.single_z_weight and profile.single_x_weight > 0.10 * max(profile.total_weight, 1e-12):
        rotation_preferences.append("rx_ry_rz")
    if profile.single_z_weight >= profile.single_x_weight and profile.single_z_weight > 0.10 * max(profile.total_weight, 1e-12):
        final_preferences.append("ry_rz")
        rotation_preferences.append("ry_rz")
    if pair_weight > 0 and profile.long_range_weight > profile.nearest_neighbor_weight:
        pattern_preferences.append("ring")
    elif pair_weight > 0:
        pattern_preferences.append("linear")

    return {
        "rotation_block": tuple(dict.fromkeys(rotation_preferences + list(ROTATION_BLOCKS))),
        "entangler": tuple(dict.fromkeys(entangler_preferences + list(ENTANGLERS))),
        "final_rotation": tuple(dict.fromkeys(final_preferences + list(FINAL_ROTATIONS))),
        "entangle_pattern": tuple(dict.fromkeys(pattern_preferences + list(ENTANGLE_PATTERNS))),
        "layers": LAYER_CHOICES,
    }


def derive_priority_seed_masks(
    profile: HamiltonianProfile,
    max_layers: int = 3,
    layers: Optional[Sequence[int]] = None,
) -> List[HEAMask]:
    """Derive problem-prior HEA seed masks from dominant Pauli-term weights."""
    n_qubits = int(profile.n_qubits)
    deep_layer = max(1, int(max_layers))
    seed_layers = tuple(
        dict.fromkeys(
            int(layer)
            for layer in (
                layers
                if layers is not None
                else tuple(layer for layer in LAYER_CHOICES if int(layer) <= deep_layer)
            )
            if 1 <= int(layer) <= deep_layer
        )
    )
    entanglers: List[str] = []
    rotations: List[str] = []
    patterns: List[str] = []
    final_rotations: List[str] = ["ry_rz", "ry"]

    if profile.zz_weight >= max(profile.xx_weight, profile.yy_weight, 1e-12):
        entanglers.append("rzz")
    if profile.xx_weight + profile.yy_weight > 0.10 * max(profile.total_weight, 1e-12):
        entanglers.append("cx")
        rotations.append("rx_ry_rz")
    if profile.single_x_weight > 0.10 * max(profile.total_weight, 1e-12):
        rotations.append("rx_ry_rz")
    if profile.single_z_weight > 0.10 * max(profile.total_weight, 1e-12):
        rotations.append("ry_rz")
    if not entanglers:
        entanglers.append("cx")
    if not rotations:
        rotations.append("ry_rz")

    if profile.long_range_weight > profile.nearest_neighbor_weight:
        patterns.extend(["ring", "linear"])
    else:
        patterns.extend(["linear", "ring"])

    seeds: List[HEAMask] = []
    for layer in seed_layers or (deep_layer,):
        for rotation in tuple(dict.fromkeys(rotations + ["ry_rz", "ry"])):
            for entangler in tuple(dict.fromkeys(entanglers + ["cx", "cz"])):
                for pattern in tuple(dict.fromkeys(patterns)):
                    for final_rotation in tuple(dict.fromkeys(final_rotations)):
                        seeds.append(
                            HEAMask(
                                n_qubits=n_qubits,
                                layers=int(layer),
                                rotation_block=rotation,
                                entangler=entangler,
                                final_rotation=final_rotation,
                                entangle_pattern=pattern,
                            )
                        )
    return seeds


def mutate_hea_mask_hamiltonian_aware(
    mask: HEAMask,
    profile: HamiltonianProfile,
    rng: np.random.Generator,
    bias_probability: float = 0.75,
) -> HEAMask:
    """Return a one-dimension neighbor, biased by the Hamiltonian profile."""
    fields = ["layers", "rotation_block", "entangler", "final_rotation", "entangle_pattern"]
    field_name = fields[int(rng.integers(0, len(fields)))]
    if rng.random() >= float(bias_probability):
        return mutate_hea_mask(mask, rng)
    preferences = hamiltonian_aware_mask_preferences(profile)
    choices = [choice for choice in preferences[field_name] if choice != getattr(mask, field_name)]
    if not choices:
        return mutate_hea_mask(mask, rng)
    return replace(mask, **{field_name: choices[0]})


def get_structure_family(candidate: ArchitectureSpec) -> str:
    """Return a coarse structure-family key used by diversity-aware beam updates."""
    raw_mask = candidate.metadata.get("hea_mask")
    if raw_mask:
        mask = HEAMask(*raw_mask)
        return f"{mask.rotation_block}_{mask.entangler}_{mask.entangle_pattern}_L{mask.layers}"
    family = str(candidate.metadata.get("family", candidate.name))
    layers = candidate.metadata.get("layers", "?")
    entangler = candidate.metadata.get("entangler", "?")
    topology = candidate.metadata.get("topology", "?")
    return f"{family}_{entangler}_{topology}_L{layers}"


def update_beam(
    candidates: Sequence[ArchitectureSpec],
    scores: Dict[Any, float],
    beam_width: int = 8,
) -> List[ArchitectureSpec]:
    """Select task-ranked candidates while reserving half the beam for unique families."""
    width = max(1, int(beam_width))

    def score_of(candidate: ArchitectureSpec) -> float:
        if candidate.name in scores:
            return float(scores[candidate.name])
        key = tuple(candidate.metadata.get("hea_mask", ()))
        return float(scores[key])

    ranked = sorted(candidates, key=lambda candidate: (score_of(candidate), candidate.name))
    new_beam: List[ArchitectureSpec] = []
    seen_families = set()
    diverse_slots = max(1, width // 2)
    for candidate in ranked:
        family = get_structure_family(candidate)
        if family in seen_families:
            continue
        new_beam.append(candidate)
        seen_families.add(family)
        if len(new_beam) >= diverse_slots:
            break
    for candidate in ranked:
        if candidate not in new_beam:
            new_beam.append(candidate)
        if len(new_beam) >= width:
            break
    return new_beam


def b1_bottom_filter(
    results: Sequence[VQEOptimizationResult],
    keep_fraction: float = 0.60,
) -> List[VQEOptimizationResult]:
    """Use B1 only for bottom elimination; surviving candidates keep their B2 chance."""
    if not results:
        return []
    keep_count = max(1, int(np.ceil(len(results) * float(keep_fraction))))
    ranked = sorted(results, key=lambda result: (result.energy, result.architecture.name))
    return ranked[:keep_count]


def _mask_from_architecture(architecture: ArchitectureSpec) -> Optional[HEAMask]:
    raw_mask = architecture.metadata.get("hea_mask")
    if not raw_mask:
        return None
    return HEAMask(*raw_mask)


def is_hamiltonian_favored_family(architecture: ArchitectureSpec, profile: HamiltonianProfile) -> bool:
    """Return whether a HEA-mask architecture matches first-pass Hamiltonian priors."""
    mask = _mask_from_architecture(architecture)
    if mask is None:
        return False
    if profile.zz_weight >= max(profile.xx_weight, profile.yy_weight, 1e-12) and mask.entangler == "rzz":
        return True
    if profile.single_x_weight > 0.20 * max(profile.total_weight, 1e-12) and mask.rotation_block == "rx_ry_rz":
        return True
    if profile.long_range_weight > profile.nearest_neighbor_weight and mask.entangle_pattern == "ring":
        return True
    return False


def _row_selection_key(row: Stage1Row, profile: HamiltonianProfile) -> tuple[float, float, float, str]:
    mask = _mask_from_architecture(row.architecture)
    layer_bonus = 0.05 * float(mask.layers if mask is not None else row.architecture.metadata.get("layers", 1))
    family_bonus = 0.25 if is_hamiltonian_favored_family(row.architecture, profile) else 0.0
    expressibility = float(row.score.expressibility.score)
    trainability = float(row.score.trainability.score)
    return (family_bonus + layer_bonus + expressibility, trainability, row.score.weighted_score, row.architecture.name)


def stratified_stage1_pool(
    candidates: Sequence[ArchitectureSpec],
    profile: HamiltonianProfile,
    pool_size: int = 16,
    n_samples: int = 6,
    expressibility_floor_quantile: float = 0.10,
    trainability_floor_quantile: float = 0.10,
    max_parameters: Optional[int] = None,
    layer_quota: Optional[Dict[int, int]] = None,
    family_quota: int = 2,
    backend: Optional[NumpyBackend] = None,
) -> List[Stage1Row]:
    """Build a Stage 1 pool by guardrails and quota coverage, not one scalar rank."""
    backend = backend or resolve_qas_backend()
    target_size = max(1, int(pool_size))
    scores = evaluate_architectures(
        candidates,
        backend=backend,
        n_samples=n_samples,
        active_metrics={"trainability": "gradient_norm"},
    )
    expr_bottom = _bottom_metric_keys(scores, lambda score: score.expressibility.score, expressibility_floor_quantile)
    train_bottom = _bottom_metric_keys(scores, lambda score: score.trainability.score, trainability_floor_quantile)

    rows: List[Stage1Row] = []
    eligible: List[Stage1Row] = []
    for score in scores:
        architecture = score.architecture
        key = tuple(architecture.metadata.get("hea_mask", ()))
        failures = []
        if key in expr_bottom:
            failures.append("expressibility_floor")
        if key in train_bottom:
            failures.append("trainability_floor")
        if max_parameters is not None and parameter_count(architecture.circuit) > int(max_parameters):
            failures.append("complexity_cap")
        row = Stage1Row(architecture, score, kept=False, reason="filtered:" + ",".join(failures) if failures else "eligible")
        rows.append(row)
        if not failures:
            eligible.append(row)

    selected: List[Stage1Row] = []
    selected_names: set[str] = set()

    def add_row(row: Stage1Row, reason: str) -> bool:
        if row.architecture.name in selected_names or len(selected) >= target_size:
            return False
        row.kept = True
        row.reason = reason
        selected.append(row)
        selected_names.add(row.architecture.name)
        return True

    quotas = dict(layer_quota or {1: 2, 2: 3, 3: 4})
    for layer, quota in sorted(quotas.items()):
        layer_rows = [row for row in eligible if (_mask_from_architecture(row.architecture) or HEAMask()).layers == int(layer)]
        layer_rows = sorted(layer_rows, key=lambda row: _row_selection_key(row, profile), reverse=True)
        for row in layer_rows[: max(0, int(quota))]:
            add_row(row, f"layer_quota_L{layer}")

    favored_rows = [row for row in eligible if is_hamiltonian_favored_family(row.architecture, profile)]
    favored_rows = sorted(favored_rows, key=lambda row: _row_selection_key(row, profile), reverse=True)
    favored_by_family: Dict[str, int] = {}
    for row in favored_rows:
        family = get_structure_family(row.architecture)
        if favored_by_family.get(family, 0) >= int(family_quota):
            continue
        if add_row(row, "hamiltonian_family_quota"):
            favored_by_family[family] = favored_by_family.get(family, 0) + 1
        if len(selected) >= target_size:
            break

    diversity_rows = sorted(
        eligible,
        key=lambda row: (
            get_structure_family(row.architecture) in {get_structure_family(item.architecture) for item in selected},
            _row_selection_key(row, profile),
        ),
        reverse=False,
    )
    for row in diversity_rows:
        if len(selected) >= target_size:
            break
        add_row(row, "diversity_fill")

    return sorted(rows, key=lambda row: (not row.kept, row.reason, -_row_selection_key(row, profile)[0], row.architecture.name))


def _edges(n_qubits: int, pattern: str) -> List[tuple[int, int]]:
    edges = [(i, i + 1) for i in range(n_qubits - 1)]
    if pattern == "ring":
        if n_qubits == 2:
            edges.append((1, 0))
        elif n_qubits > 2:
            edges.append((n_qubits - 1, 0))
    return edges


def _append_rotation(gates: List[Dict[str, Any]], n_qubits: int, block: str, cursor: List[int]) -> None:
    rotations = {
        "ry": ("ry",),
        "ry_rz": ("ry", "rz"),
        "rx_ry_rz": ("rx", "ry", "rz"),
    }[block]
    for qubit in range(n_qubits):
        for gate_type in rotations:
            cursor[0] += 1
            gates.append({"type": gate_type, "target_qubit": qubit, "parameter": 0.071 * cursor[0]})


def _append_entangler(gates: List[Dict[str, Any]], edges: Sequence[tuple[int, int]], entangler: str, cursor: List[int]) -> None:
    for control, target in edges:
        if entangler == "rzz":
            cursor[0] += 1
            gates.append({"type": "rzz", "qubit_1": control, "qubit_2": target, "parameter": 0.071 * cursor[0]})
        else:
            gates.append(
                {
                    "type": entangler,
                    "target_qubit": target,
                    "control_qubits": [control],
                    "control_states": [1],
                }
            )


def architecture_from_hea_mask(mask: HEAMask, backend: Optional[NumpyBackend] = None) -> ArchitectureSpec:
    gates: List[Dict[str, Any]] = []
    cursor = [0]
    edges = _edges(mask.n_qubits, mask.entangle_pattern)
    for _ in range(mask.layers):
        _append_rotation(gates, mask.n_qubits, mask.rotation_block, cursor)
        _append_entangler(gates, edges, mask.entangler, cursor)
    _append_rotation(gates, mask.n_qubits, mask.final_rotation, cursor)
    return ArchitectureSpec.from_gates(
        name=mask.label(),
        gates=gates,
        n_qubits=mask.n_qubits,
        backend=backend,
        description="HEA mask candidate for the minimal H2 VQE-QAS demo.",
        tags=["VQE", "HEA", "demo"],
        metadata={"hea_mask": mask.key(), "family": "HEA-mask"},
    )


def rotation_only_ansatz(
    n_qubits: int = 4,
    layers: int = 1,
    rotation_block: str = "ry",
    final_rotation: str = "ry",
    backend: Optional[NumpyBackend] = None,
) -> ArchitectureSpec:
    """Deliberately weak no-entanglement baseline for checking task signal size."""
    gates: List[Dict[str, Any]] = []
    cursor = [0]
    for _ in range(max(1, int(layers))):
        _append_rotation(gates, n_qubits, rotation_block, cursor)
    _append_rotation(gates, n_qubits, final_rotation, cursor)
    return ArchitectureSpec.from_gates(
        name=f"rotation_only_L{layers}_{rotation_block}_{final_rotation}",
        gates=gates,
        n_qubits=n_qubits,
        backend=backend,
        description="No-entanglement rotation-only baseline for VQE-QAS signal diagnostics.",
        tags=["VQE", "baseline", "no_entanglement"],
        metadata={
            "family": "rotation_only",
            "layers": int(layers),
            "rotation_block": rotation_block,
            "final_rotation": final_rotation,
            "entangler": "none",
            "topology": "none",
            "source": "bad_baseline",
        },
    )


def _pauli_matrix(label: str) -> np.ndarray:
    matrices = {
        "I": np.eye(2, dtype=np.complex128),
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }
    result = np.array([[1]], dtype=np.complex128)
    for char in label:
        result = np.kron(result, matrices[char])
    return result


def hamiltonian_matrix(hamiltonian: Sequence[tuple[float, str]]) -> np.ndarray:
    if not hamiltonian:
        raise ValueError("Hamiltonian must contain at least one Pauli term")
    n_qubits = len(hamiltonian[0][1])
    matrix = np.zeros((2**n_qubits, 2**n_qubits), dtype=np.complex128)
    for coeff, pauli in hamiltonian:
        if len(pauli) != n_qubits:
            raise ValueError("All Pauli labels must have the same length")
        matrix += float(coeff) * _pauli_matrix(pauli)
    return matrix


def exact_ground_energy(hamiltonian: Sequence[tuple[float, str]]) -> float:
    return float(np.min(np.linalg.eigvalsh(hamiltonian_matrix(hamiltonian))))


def tfim_open_chain_free_fermion_ground_energy(
    n_qubits: int,
    J: float = 1.0,
    h: float = 0.5,
) -> float:
    """Free-fermion ground energy for H=-J sum ZZ - h sum X with open boundaries."""
    n_qubits = int(n_qubits)
    if n_qubits < 2:
        raise ValueError("TFIM chain requires at least 2 qubits")
    A = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    B = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    np.fill_diagonal(A, -2.0 * float(h))
    for index in range(n_qubits - 1):
        A[index, index + 1] = -float(J)
        A[index + 1, index] = -float(J)
        B[index, index + 1] = -float(J)
        B[index + 1, index] = float(J)
    spectrum = np.linalg.eigvals((A - B) @ (A + B))
    eps = np.sqrt(np.maximum(np.sort(np.real(spectrum)), 0.0))
    return float(-0.5 * np.sum(eps))


def validate_tfim_reference_alignment(
    scales: Sequence[int] = (4, 6, 8),
    J: float = 1.0,
    h: float = 0.5,
    periodic: bool = False,
    tolerance: float = 1e-9,
) -> TFIMReferenceAlignmentReport:
    """Phase-0 check that dense TFIM references match the free-fermion formula."""
    if periodic:
        raise NotImplementedError("PBC free-fermion validation needs explicit parity-sector handling.")
    rows: List[TFIMReferenceAlignmentRow] = []
    for n_qubits in scales:
        hamiltonian = tfim_chain_hamiltonian(n_qubits=int(n_qubits), J=J, h=h, periodic=periodic)
        dense = exact_ground_energy(hamiltonian)
        free = tfim_open_chain_free_fermion_ground_energy(n_qubits=int(n_qubits), J=J, h=h)
        rows.append(
            TFIMReferenceAlignmentRow(
                n_qubits=int(n_qubits),
                J=float(J),
                h=float(h),
                periodic=bool(periodic),
                dense_energy=dense,
                free_fermion_energy=free,
                abs_diff=abs(dense - free),
            )
        )
    report = TFIMReferenceAlignmentReport(rows=rows, tolerance=float(tolerance))
    if not report.passed:
        worst = max(row.abs_diff for row in rows)
        raise AssertionError(f"TFIM dense/free-fermion reference mismatch: worst abs_diff={worst:.3e}")
    return report


def h2_demo_problem() -> VQEDemoProblem:
    return VQEDemoProblem(
        name="h2_toy_2q",
        n_qubits=2,
        hamiltonian=H2_HAMILTONIAN,
        reference_energy=H2_REFERENCE_ENERGY,
    )


def ising4_demo_problem() -> VQEDemoProblem:
    return VQEDemoProblem(
        name="tfim_chain_4q_J1_h0.5",
        n_qubits=4,
        hamiltonian=ISING4_HAMILTONIAN,
        reference_energy=exact_ground_energy(ISING4_HAMILTONIAN),
    )


def tfim_chain_hamiltonian(
    n_qubits: int,
    J: float = 1.0,
    h: float = 0.5,
    periodic: bool = False,
) -> tuple[tuple[float, str], ...]:
    """Open-chain transverse-field Ising Hamiltonian used by QAS scaling checks."""
    n_qubits = int(n_qubits)
    if n_qubits < 2:
        raise ValueError("TFIM chain requires at least 2 qubits")

    terms: List[tuple[float, str]] = []
    edge_count = n_qubits if periodic else n_qubits - 1
    for index in range(edge_count):
        left = index
        right = (index + 1) % n_qubits
        pauli = ["I"] * n_qubits
        pauli[left] = "Z"
        pauli[right] = "Z"
        terms.append((-float(J), "".join(pauli)))
    for index in range(n_qubits):
        pauli = ["I"] * n_qubits
        pauli[index] = "X"
        terms.append((-float(h), "".join(pauli)))
    return tuple(terms)


def tfim_chain_demo_problem(
    n_qubits: int,
    J: float = 1.0,
    h: float = 0.5,
    periodic: bool = False,
) -> VQEDemoProblem:
    """Build an exact-reference TFIM chain problem for 4q/6q/8q smoke experiments."""
    hamiltonian = tfim_chain_hamiltonian(n_qubits=n_qubits, J=J, h=h, periodic=periodic)
    boundary = "ring" if periodic else "chain"
    return VQEDemoProblem(
        name=f"tfim_{boundary}_{int(n_qubits)}q_J{J:g}_h{h:g}",
        n_qubits=int(n_qubits),
        hamiltonian=hamiltonian,
        reference_energy=exact_ground_energy(hamiltonian),
    )


def h2_hamiltonian_matrix() -> np.ndarray:
    return hamiltonian_matrix(H2_HAMILTONIAN)


def evaluate_vqe_energy(
    architecture: ArchitectureSpec,
    problem: VQEDemoProblem,
    parameters: Optional[Sequence[float]] = None,
    backend: Optional[Backend] = None,
) -> float:
    if architecture.n_qubits != problem.n_qubits:
        raise ValueError("architecture and VQE problem must use the same number of qubits")
    backend = backend or resolve_qas_backend()
    n_params = parameter_count(architecture.circuit)
    params = [0.0] * n_params if parameters is None else list(parameters)
    if len(params) != n_params:
        raise ValueError(f"Expected {n_params} parameters, got {len(params)}")
    circuit = bind_parameters(architecture.circuit, params) if n_params else architecture.circuit
    circuit.bind_backend(backend)
    result = Measure(backend).run(circuit, return_state=True)
    state = backend.cast(result.final_state)
    operator = backend.cast(hamiltonian_matrix(problem.hamiltonian))
    energy = backend.expectation_sv(state, operator)
    return float(np.real(backend.to_numpy(energy)))


def evaluate_h2_energy(
    architecture: ArchitectureSpec,
    parameters: Optional[Sequence[float]] = None,
    backend: Optional[Backend] = None,
) -> float:
    return evaluate_vqe_energy(architecture, h2_demo_problem(), parameters=parameters, backend=backend)


def _evaluation_budget(architecture: ArchitectureSpec, evals_per_param: int, max_evaluations: int) -> int:
    n_params = max(1, parameter_count(architecture.circuit))
    return max(1, min(int(n_params * evals_per_param), int(max_evaluations)))


def adaptive_fair_n_starts(architecture: ArchitectureSpec, min_starts: int = 3, params_per_start: int = 15) -> int:
    n_params = max(1, parameter_count(architecture.circuit))
    return max(int(min_starts), int(np.ceil(n_params / max(1, int(params_per_start)))))


def is_b1_improvement_valid(b1_energy: float, floor_energy: float = -3.0, tolerance: float = 0.01) -> bool:
    return abs(float(b1_energy) - float(floor_energy)) > float(tolerance)


def optimize_vqe_energy(
    architecture: ArchitectureSpec,
    problem: VQEDemoProblem,
    seed: int = 1234,
    n_starts: int = 1,
    evals_per_param: int = 10,
    max_evaluations: int = 80,
    budget_override: Optional[int] = None,
    backend: Optional[Backend] = None,
    init_mode: str = THETA_INIT_RANDOM_UNIFORM_PI,
    init_scale: float = float(np.pi),
) -> VQEOptimizationResult:
    if architecture.n_qubits != problem.n_qubits:
        raise ValueError("architecture and VQE problem must use the same number of qubits")
    backend = backend or resolve_qas_backend()
    n_params = parameter_count(architecture.circuit)
    rng = np.random.default_rng(int(seed))
    budget = int(budget_override) if budget_override is not None else _evaluation_budget(architecture, evals_per_param, max_evaluations)
    budget = max(1, budget)
    best_energy = float("inf")
    best_params = np.zeros(n_params, dtype=float)
    total_evals = 0
    per_start: List[Dict[str, Any]] = []

    def objective(params: np.ndarray) -> float:
        return evaluate_vqe_energy(architecture, problem, params, backend=backend)

    mode = str(init_mode).strip().lower()
    starts: List[np.ndarray] = []
    start_count = max(1, int(n_starts))
    if mode in {THETA_INIT_ZERO_DIAGNOSTIC, "zeros"}:
        starts = [np.zeros(n_params, dtype=float) for _ in range(start_count)]
    elif mode in {THETA_INIT_RANDOM_UNIFORM_PI, "random", "uniform"}:
        starts = [rng.uniform(-float(init_scale), float(init_scale), size=n_params) for _ in range(start_count)]
    elif mode in {"zero_then_random", "mixed"}:
        starts = [np.zeros(n_params, dtype=float)]
        for _ in range(start_count - 1):
            starts.append(rng.uniform(-float(init_scale), float(init_scale), size=n_params))
    else:
        raise ValueError("init_mode must be one of: random_uniform_pi, zero, zero_then_random")

    for start_index, start in enumerate(starts):
        start_time = perf_counter()
        if n_params == 0:
            energy = objective(start)
            nfev = 1
            params = start
        elif minimize is not None:
            result = minimize(
                objective,
                start,
                method="COBYLA",
                options={"maxiter": budget, "rhobeg": V3_COBYLA_RHOBEG, "tol": V3_COBYLA_TOL},
            )
            energy = float(result.fun)
            nfev = int(getattr(result, "nfev", budget))
            params = np.asarray(result.x, dtype=float)
        else:  # pragma: no cover
            values = [start]
            while len(values) < budget:
                values.append(rng.uniform(-np.pi, np.pi, size=n_params))
            scored = [(objective(params), params) for params in values]
            energy, params = min(scored, key=lambda item: item[0])
            nfev = len(scored)
        total_evals += nfev
        per_start.append(
            {
                "start_index": start_index,
                "seed": int(seed),
                "energy": float(energy),
                "nfev": int(nfev),
                "walltime_s": float(perf_counter() - start_time),
                "init_mode": mode,
                "init_l2": float(np.linalg.norm(start)) if n_params else 0.0,
            }
        )
        if energy < best_energy:
            best_energy = energy
            best_params = params.copy()

    return VQEOptimizationResult(
        architecture=architecture,
        energy=float(best_energy),
        best_parameters=[float(value) for value in best_params],
        evaluations=total_evals,
        n_starts=len(starts),
        metadata={
            "optimizer": "COBYLA" if minimize is not None else "budgeted_random_search",
            "budget_per_start": budget,
            "nfev": total_evals,
            "per_start": per_start,
            "theta_init_mode": mode,
            "cobyla_rhobeg": V3_COBYLA_RHOBEG,
            "cobyla_tol": V3_COBYLA_TOL,
        },
    )


def optimize_h2_energy(
    architecture: ArchitectureSpec,
    seed: int = 1234,
    n_starts: int = 1,
    evals_per_param: int = 10,
    max_evaluations: int = 80,
    backend: Optional[NumpyBackend] = None,
) -> VQEOptimizationResult:
    return optimize_vqe_energy(
        architecture,
        h2_demo_problem(),
        seed=seed,
        n_starts=n_starts,
        evals_per_param=evals_per_param,
        max_evaluations=max_evaluations,
        backend=backend,
    )


def diagnose_theta_randomness(
    architecture: ArchitectureSpec,
    problem: VQEDemoProblem,
    n_trials: int = 4,
    seed: int = 2026,
    evals_per_param: int = 20,
    maxfev: Optional[int] = None,
    init_mode: str = THETA_INIT_RANDOM_UNIFORM_PI,
    backend: Optional[Backend] = None,
) -> Dict[str, Any]:
    """Phase-0 guardrail: verify independent random theta trajectories are not identical."""
    backend = backend or resolve_qas_backend()
    n_params = parameter_count(architecture.circuit)
    budget = int(maxfev) if maxfev is not None else v3_screening_maxfev(n_params)
    results: List[VQEOptimizationResult] = []
    for trial in range(max(1, int(n_trials))):
        results.append(
            optimize_vqe_energy(
                architecture,
                problem,
                seed=int(seed) + trial,
                n_starts=1,
                evals_per_param=evals_per_param,
                max_evaluations=budget,
                budget_override=budget,
                backend=backend,
                init_mode=init_mode,
            )
        )
    energies = [float(result.energy) for result in results]
    init_l2 = [
        float(result.metadata.get("per_start", [{}])[0].get("init_l2", 0.0))
        for result in results
    ]
    nfev = [int(result.evaluations) for result in results]
    return {
        "architecture": architecture.name,
        "problem_name": problem.name,
        "theta_init_mode": init_mode,
        "n_trials": len(results),
        "n_params": n_params,
        "budget_per_trial": budget,
        "energies": energies,
        "energy_std": float(np.std(np.asarray(energies, dtype=float))),
        "init_l2": init_l2,
        "init_l2_std": float(np.std(np.asarray(init_l2, dtype=float))),
        "nfev": nfev,
        "passes_randomness_guard": bool(len(set(round(value, 12) for value in init_l2)) > 1 or n_params == 0),
    }


def _sample_masks(masks: Sequence[HEAMask], limit: int, seed: int) -> List[HEAMask]:
    masks = list(masks)
    if len(masks) <= limit:
        return masks
    rng = np.random.default_rng(int(seed))
    indices = sorted(rng.choice(np.arange(len(masks)), size=limit, replace=False).tolist())
    return [masks[index] for index in indices]


def _shard_items(items: Sequence[Any], shard_index: int = 0, num_shards: int = 1) -> List[Any]:
    shard_count = max(1, int(num_shards))
    shard = int(shard_index)
    if shard < 0 or shard >= shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count}), got {shard}")
    return [item for index, item in enumerate(items) if index % shard_count == shard]


def _append_vqe_checkpoint_row(
    checkpoint_path: Optional[str],
    result: VQEOptimizationResult,
    problem: VQEDemoProblem,
    candidate_index: int,
    shard_index: int,
    num_shards: int,
) -> None:
    if checkpoint_path is None:
        return
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_index",
        "name",
        "family",
        "energy",
        "delta_ref",
        "n_params",
        "nfev",
        "n_starts",
        "theta_init_mode",
        "budget_per_start",
        "shard_index",
        "num_shards",
    ]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "candidate_index": int(candidate_index),
                "name": result.architecture.name,
                "family": get_structure_family(result.architecture),
                "energy": f"{result.energy:.12f}",
                "delta_ref": f"{result.energy - problem.reference_energy:.12f}",
                "n_params": parameter_count(result.architecture.circuit),
                "nfev": int(result.evaluations),
                "n_starts": int(result.n_starts),
                "theta_init_mode": result.metadata.get("theta_init_mode", ""),
                "budget_per_start": result.metadata.get("budget_per_start", ""),
                "shard_index": int(shard_index),
                "num_shards": int(num_shards),
            }
        )
        handle.flush()


def zero_cost_guardrail(
    candidates: Sequence[ArchitectureSpec],
    n_samples: int = 6,
    quantile: float = 0.25,
    rescue_count: int = 3,
    backend: Optional[NumpyBackend] = None,
) -> List[Stage1Row]:
    backend = backend or resolve_qas_backend()
    scores = evaluate_architectures(
        candidates,
        backend=backend,
        n_samples=n_samples,
        active_metrics={
            "trainability": "gradient_norm",
            "hardware_efficiency": "topology_mapping_efficiency",
        },
    )
    metric_bottom_keys = {
        "trainability": _bottom_metric_keys(scores, lambda score: score.trainability.score, quantile),
        "noise": _bottom_metric_keys(scores, lambda score: score.noise_robustness.score, quantile),
        "hardware": _bottom_metric_keys(scores, lambda score: score.hardware_efficiency.score, quantile),
    }
    rows: List[Stage1Row] = []
    rejected: List[Stage1Row] = []
    kept_keys = set()
    for score in scores:
        key = tuple(score.architecture.metadata.get("hea_mask", ()))
        failures = []
        if key in metric_bottom_keys["trainability"]:
            failures.append("trainability")
        if key in metric_bottom_keys["noise"]:
            failures.append("noise")
        if key in metric_bottom_keys["hardware"]:
            failures.append("hardware")
        kept = not failures
        reason = "pass" if kept else "filtered:" + ",".join(failures)
        row = Stage1Row(score.architecture, score, kept, reason)
        rows.append(row)
        if kept:
            kept_keys.add(key)
        else:
            rejected.append(row)

    rescued = _rescue_diverse_rows(rejected, kept_keys, max(0, int(rescue_count)))
    for row in rescued:
        row.kept = True
        row.reason = "diversity_rescue"
    return sorted(rows, key=lambda row: (not row.kept, -row.score.weighted_score))


def _bottom_metric_keys(
    scores: Sequence[ArchitectureScore],
    metric: Any,
    quantile: float,
) -> set[tuple[Any, ...]]:
    values = [(tuple(score.architecture.metadata.get("hea_mask", ())), float(metric(score))) for score in scores]
    if not values:
        return set()
    raw = np.asarray([value for _, value in values], dtype=float)
    if float(np.max(raw) - np.min(raw)) <= 1e-12:
        return set()
    n_reject = max(1, int(np.floor(len(values) * float(quantile))))
    ranked = sorted(values, key=lambda item: (item[1], item[0]))
    return {key for key, _ in ranked[:n_reject]}


def _mask_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    return sum(1 for a, b in zip(left, right) if a != b)


def _rescue_diverse_rows(rejected: Sequence[Stage1Row], kept_keys: set[tuple[Any, ...]], rescue_count: int) -> List[Stage1Row]:
    rescued: List[Stage1Row] = []
    pool = list(rejected)
    while pool and len(rescued) < rescue_count:
        if not kept_keys:
            chosen = max(pool, key=lambda row: row.score.weighted_score)
        else:
            chosen = max(
                pool,
                key=lambda row: (
                    min(_mask_distance(tuple(row.architecture.metadata.get("hea_mask", ())), key) for key in kept_keys),
                    row.score.weighted_score,
                ),
            )
        pool.remove(chosen)
        key = tuple(chosen.architecture.metadata.get("hea_mask", ()))
        kept_keys.add(key)
        rescued.append(chosen)
    return rescued


def _rank_stage1_rows(rows: Sequence[Stage1Row], rank_by: str = "weighted") -> List[Stage1Row]:
    candidates = [row for row in rows if row.kept]
    if rank_by == "weighted":
        key = lambda row: row.score.weighted_score
    elif rank_by == "trainability":
        key = lambda row: row.score.trainability.score
    elif rank_by == "expressibility":
        key = lambda row: row.score.expressibility.score
    elif rank_by == "noise":
        key = lambda row: row.score.noise_robustness.score
    elif rank_by == "hardware":
        key = lambda row: row.score.hardware_efficiency.score
    else:
        raise ValueError(f"Unsupported Stage1 rank key: {rank_by}")
    return sorted(candidates, key=key, reverse=True)


def _select_diverse_stage1_seeds(rows: Sequence[Stage1Row], n_seeds: int) -> List[Stage1Row]:
    """Greedily pick high-scoring but structurally diverse Stage 1 seeds."""
    candidates = [row for row in rows if row.kept]
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda row: row.score.weighted_score, reverse=True)
    selected = [candidates[0]]
    remaining = candidates[1:]
    while remaining and len(selected) < int(n_seeds):
        chosen = max(
            remaining,
            key=lambda row: (
                min(
                    _mask_distance(
                        tuple(row.architecture.metadata.get("hea_mask", ())),
                        tuple(seed.architecture.metadata.get("hea_mask", ())),
                    )
                    for seed in selected
                ),
                row.score.weighted_score,
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def run_sa_search(
    seed_mask: HEAMask,
    n_steps: int = 24,
    seed: int = 2026,
    evals_per_param: int = 8,
    max_evaluations: int = 40,
    early_stop_delta: Optional[float] = None,
    t_init: Optional[float] = None,
    t_final: Optional[float] = None,
    problem: Optional[VQEDemoProblem] = None,
    backend: Optional[NumpyBackend] = None,
) -> tuple[VQEOptimizationResult, List[SAStep]]:
    problem = problem or h2_demo_problem()
    backend = backend or resolve_qas_backend()
    rng = np.random.default_rng(int(seed))
    current_mask = seed_mask
    current = optimize_vqe_energy(
        architecture_from_hea_mask(current_mask, backend=backend),
        problem,
        seed=seed,
        n_starts=1,
        evals_per_param=evals_per_param,
        max_evaluations=max_evaluations,
        backend=backend,
    )
    best = current
    trace: List[SAStep] = []
    start_temperature = max(abs(current.energy) * 0.1, 1e-3) if t_init is None else max(float(t_init), 1e-12)
    end_temperature = max(abs(current.energy) * 0.001, 1e-5) if t_final is None else max(float(t_final), 1e-12)
    for step in range(1, int(n_steps) + 1):
        exponent = step / max(1, int(n_steps))
        temperature = start_temperature * ((end_temperature / start_temperature) ** exponent)
        next_mask = mutate_hea_mask(current_mask, rng)
        candidate = optimize_vqe_energy(
            architecture_from_hea_mask(next_mask, backend=backend),
            problem,
            seed=seed + step,
            n_starts=1,
            evals_per_param=evals_per_param,
            max_evaluations=max_evaluations,
            backend=backend,
        )
        delta = candidate.energy - current.energy
        accepted = delta <= 0.0 or rng.random() < exp(-delta / max(temperature, 1e-12))
        if accepted:
            current_mask = next_mask
            current = candidate
        if candidate.energy < best.energy:
            best = candidate
        trace.append(
            SAStep(
                step=step,
                temperature=float(temperature),
                current_energy=float(current.energy),
                candidate_energy=float(candidate.energy),
                best_energy=float(best.energy),
                accepted=bool(accepted),
                mask=next_mask,
            )
        )
        if early_stop_delta is not None and best.energy <= problem.reference_energy + float(early_stop_delta):
            break
    best.metadata["source"] = "sa_best"
    best.architecture.metadata["source"] = "sa_best"
    return best, trace


def _dedupe_architectures(architectures: Iterable[ArchitectureSpec]) -> List[ArchitectureSpec]:
    seen = set()
    unique = []
    for architecture in architectures:
        key = tuple(architecture.circuit.gates)
        key = architecture.name if not key else repr(key)
        if key in seen:
            continue
        seen.add(key)
        unique.append(architecture)
    return unique


def _baseline_architectures(n_qubits: int, backend: NumpyBackend) -> List[ArchitectureSpec]:
    baselines = build_common_architectures(
        n_qubits=n_qubits,
        layers=2,
        backend=backend,
        names=["hea_linear", "real_amplitudes_linear", "efficient_su2_ring", "brickwork_cx"],
    )
    baselines.append(qaoa_ansatz(n_qubits, layers=2, topology="linear", backend=backend))
    for architecture in baselines:
        architecture.metadata["source"] = "baseline"
    return baselines


def _validate_candidates(
    candidates: Sequence[ArchitectureSpec],
    problem: VQEDemoProblem,
    seed: int,
    n_starts: int,
    evals_per_param: int,
    max_evaluations: int,
    backend: NumpyBackend,
) -> List[VQEOptimizationResult]:
    results = []
    for index, architecture in enumerate(_dedupe_architectures(candidates)):
        result = optimize_vqe_energy(
            architecture,
            problem,
            seed=seed + index,
            n_starts=n_starts,
            evals_per_param=evals_per_param,
            max_evaluations=max_evaluations,
            backend=backend,
        )
        result.metadata["source"] = architecture.metadata.get("source", "candidate")
        results.append(result)
    results.sort(key=lambda result: result.energy)
    return results


def run_vqe_hea_demo(
    problem: Optional[VQEDemoProblem] = None,
    seed: int = 2026,
    candidate_limit: int = 48,
    stage1_keep_top: int = 12,
    sa_steps: int = 24,
    early_stop_delta: Optional[float] = None,
    backend: Optional[NumpyBackend] = None,
) -> VQEHEADemoReport:
    problem = problem or h2_demo_problem()
    backend = backend or resolve_qas_backend()
    masks = _sample_masks(enumerate_hea_masks(problem.n_qubits), candidate_limit, seed)
    candidates = [architecture_from_hea_mask(mask, backend=backend) for mask in masks]
    stage1_rows = zero_cost_guardrail(candidates, backend=backend)
    kept = [row for row in stage1_rows if row.kept][: max(1, int(stage1_keep_top))]
    seed_row = max(kept, key=lambda row: row.score.weighted_score)
    seed_mask = HEAMask(*seed_row.architecture.metadata["hea_mask"])
    sa_best, trace = run_sa_search(
        seed_mask,
        problem=problem,
        n_steps=sa_steps,
        seed=seed,
        early_stop_delta=early_stop_delta,
        backend=backend,
    )

    final_candidates = [sa_best.architecture]
    for row in kept[:3]:
        row.architecture.metadata["source"] = "stage1_top"
        final_candidates.append(row.architecture)
    final_candidates.extend(_baseline_architectures(problem.n_qubits, backend))
    final_results = _validate_candidates(
        final_candidates,
        problem,
        seed=seed + 100,
        n_starts=3,
        evals_per_param=12,
        max_evaluations=120,
        backend=backend,
    )
    return VQEHEADemoReport(
        stage1_rows=stage1_rows,
        sa_trace=trace,
        final_results=final_results,
        metadata={
            "problem_name": problem.name,
            "hamiltonian": list(problem.hamiltonian),
            "reference_energy": problem.reference_energy,
            "candidate_limit": candidate_limit,
            "stage1_keep_top": stage1_keep_top,
            "sa_steps": sa_steps,
            "seed": seed,
        },
    )


def run_vqe_ising4_demo(
    seed: int = 2026,
    candidate_limit: int = 72,
    stage1_keep_top: int = 16,
    sa_steps: int = 36,
    early_stop_delta: Optional[float] = None,
    backend: Optional[NumpyBackend] = None,
) -> VQEHEADemoReport:
    return run_vqe_hea_demo(
        problem=ising4_demo_problem(),
        seed=seed,
        candidate_limit=candidate_limit,
        stage1_keep_top=stage1_keep_top,
        sa_steps=sa_steps,
        early_stop_delta=early_stop_delta,
        backend=backend,
    )


def run_ising4_budget_sweep(
    seed: int = 2026,
    steps: Sequence[int] = (36, 80, 120),
    candidate_limit: int = 72,
    stage1_keep_top: int = 16,
    t_init: float = 0.30,
    t_final: float = 0.001,
    search_evals_per_param: int = 8,
    search_max_evaluations: int = 40,
    final_n_starts: int = 3,
    capped_evals_per_param: int = 12,
    capped_max_evaluations: int = 120,
    fair_evals_per_param: int = 20,
    fair_min_evaluations: int = 40,
    backend: Optional[NumpyBackend] = None,
) -> VQEBudgetSweepReport:
    problem = ising4_demo_problem()
    backend = backend or resolve_qas_backend()
    masks = _sample_masks(enumerate_hea_masks(problem.n_qubits), candidate_limit, seed)
    candidates = [architecture_from_hea_mask(mask, backend=backend) for mask in masks]
    stage1_rows = zero_cost_guardrail(candidates, backend=backend)
    kept = [row for row in stage1_rows if row.kept][: max(1, int(stage1_keep_top))]
    seed_row = max(kept, key=lambda row: row.score.weighted_score)
    seed_mask = HEAMask(*seed_row.architecture.metadata["hea_mask"])

    sa_results: List[VQEOptimizationResult] = []
    sweep_rows: List[SABudgetSweepRow] = []
    for item_index, n_steps in enumerate(steps):
        sa_best, _trace = run_sa_search(
            seed_mask,
            n_steps=int(n_steps),
            seed=seed + item_index * 1000,
            evals_per_param=search_evals_per_param,
            max_evaluations=search_max_evaluations,
            t_init=t_init,
            t_final=t_final,
            problem=problem,
            backend=backend,
        )
        sa_best.architecture.metadata["source"] = f"sa_steps_{int(n_steps)}"
        sa_results.append(sa_best)

        capped = optimize_vqe_energy(
            sa_best.architecture,
            problem,
            seed=seed + 5000 + item_index,
            n_starts=final_n_starts,
            evals_per_param=capped_evals_per_param,
            max_evaluations=capped_max_evaluations,
            backend=backend,
        )
        fair_budget = max(int(fair_min_evaluations), parameter_count(sa_best.architecture.circuit) * int(fair_evals_per_param))
        fair = optimize_vqe_energy(
            sa_best.architecture,
            problem,
            seed=seed + 6000 + item_index,
            n_starts=final_n_starts,
            evals_per_param=fair_evals_per_param,
            max_evaluations=fair_budget,
            budget_override=fair_budget,
            backend=backend,
        )
        mask = HEAMask(*sa_best.architecture.metadata["hea_mask"])
        sweep_rows.append(
            SABudgetSweepRow(
                steps=int(n_steps),
                short_best_energy=sa_best.energy,
                final_capped_energy=capped.energy,
                final_fair_energy=fair.energy,
                mask=mask,
                n_params=parameter_count(sa_best.architecture.circuit),
                capped_evaluations=capped.evaluations,
                fair_evaluations=fair.evaluations,
            )
        )

    validation_candidates: List[ArchitectureSpec] = []
    validation_candidates.extend(result.architecture for result in sa_results)
    for row in kept[:3]:
        row.architecture.metadata["source"] = "stage1_top"
        validation_candidates.append(row.architecture)
    validation_candidates.extend(_baseline_architectures(problem.n_qubits, backend))

    capped_results = _validate_candidates(
        validation_candidates,
        problem,
        seed=seed + 7000,
        n_starts=final_n_starts,
        evals_per_param=capped_evals_per_param,
        max_evaluations=capped_max_evaluations,
        backend=backend,
    )
    fair_results = []
    for index, architecture in enumerate(_dedupe_architectures(validation_candidates)):
        fair_budget = max(int(fair_min_evaluations), parameter_count(architecture.circuit) * int(fair_evals_per_param))
        result = optimize_vqe_energy(
            architecture,
            problem,
            seed=seed + 8000 + index,
            n_starts=final_n_starts,
            evals_per_param=fair_evals_per_param,
            max_evaluations=fair_budget,
            budget_override=fair_budget,
            backend=backend,
        )
        result.metadata["source"] = architecture.metadata.get("source", "candidate")
        fair_results.append(result)
    fair_results.sort(key=lambda result: result.energy)

    return VQEBudgetSweepReport(
        stage1_rows=stage1_rows,
        sweep_rows=sweep_rows,
        capped_results=capped_results,
        fair_results=fair_results,
        metadata={
            "problem_name": problem.name,
            "reference_energy": problem.reference_energy,
            "seed": seed,
            "steps": list(steps),
            "t_init": t_init,
            "t_final": t_final,
            "search_evals_per_param": search_evals_per_param,
            "search_max_evaluations": search_max_evaluations,
            "final_n_starts": final_n_starts,
            "capped_evals_per_param": capped_evals_per_param,
            "capped_max_evaluations": capped_max_evaluations,
            "fair_evals_per_param": fair_evals_per_param,
            "fair_min_evaluations": fair_min_evaluations,
        },
    )


def run_ising4_multistart_sa(
    seed: int = 2026,
    n_starts: int = 5,
    steps_per_start: int = 40,
    candidate_limit: int = 72,
    stage1_keep_top: int = 16,
    t_init: float = 0.30,
    t_final: float = 0.001,
    search_evals_per_param: int = 8,
    search_max_evaluations: int = 40,
    final_n_starts: int = 3,
    capped_evals_per_param: int = 12,
    capped_max_evaluations: int = 120,
    fair_evals_per_param: int = 20,
    fair_min_evaluations: int = 40,
    backend: Optional[NumpyBackend] = None,
) -> VQEMultiStartSAReport:
    problem = ising4_demo_problem()
    backend = backend or resolve_qas_backend()
    masks = _sample_masks(enumerate_hea_masks(problem.n_qubits), candidate_limit, seed)
    candidates = [architecture_from_hea_mask(mask, backend=backend) for mask in masks]
    stage1_rows = zero_cost_guardrail(candidates, backend=backend)
    kept = [row for row in stage1_rows if row.kept][: max(1, int(stage1_keep_top))]
    start_rows = _select_diverse_stage1_seeds(kept, max(1, int(n_starts)))

    sa_results: List[VQEOptimizationResult] = []
    restart_rows: List[SAMultiStartRow] = []
    for start_index, row in enumerate(start_rows, start=1):
        start_mask = HEAMask(*row.architecture.metadata["hea_mask"])
        sa_best, _trace = run_sa_search(
            start_mask,
            n_steps=steps_per_start,
            seed=seed + start_index * 1000,
            evals_per_param=search_evals_per_param,
            max_evaluations=search_max_evaluations,
            t_init=t_init,
            t_final=t_final,
            problem=problem,
            backend=backend,
        )
        sa_best.architecture.metadata["source"] = f"multistart_{start_index}"
        sa_results.append(sa_best)

        capped = optimize_vqe_energy(
            sa_best.architecture,
            problem,
            seed=seed + 5000 + start_index,
            n_starts=final_n_starts,
            evals_per_param=capped_evals_per_param,
            max_evaluations=capped_max_evaluations,
            backend=backend,
        )
        fair_budget = max(int(fair_min_evaluations), parameter_count(sa_best.architecture.circuit) * int(fair_evals_per_param))
        fair = optimize_vqe_energy(
            sa_best.architecture,
            problem,
            seed=seed + 6000 + start_index,
            n_starts=final_n_starts,
            evals_per_param=fair_evals_per_param,
            max_evaluations=fair_budget,
            budget_override=fair_budget,
            backend=backend,
        )
        best_mask = HEAMask(*sa_best.architecture.metadata["hea_mask"])
        restart_rows.append(
            SAMultiStartRow(
                start_rank=start_index,
                start_mask=start_mask,
                best_mask=best_mask,
                short_best_energy=sa_best.energy,
                final_capped_energy=capped.energy,
                final_fair_energy=fair.energy,
                n_params=parameter_count(sa_best.architecture.circuit),
                capped_evaluations=capped.evaluations,
                fair_evaluations=fair.evaluations,
            )
        )

    validation_candidates: List[ArchitectureSpec] = []
    validation_candidates.extend(result.architecture for result in sa_results)
    for row in kept[:3]:
        row.architecture.metadata["source"] = "stage1_top"
        validation_candidates.append(row.architecture)
    validation_candidates.extend(_baseline_architectures(problem.n_qubits, backend))

    capped_results = _validate_candidates(
        validation_candidates,
        problem,
        seed=seed + 7000,
        n_starts=final_n_starts,
        evals_per_param=capped_evals_per_param,
        max_evaluations=capped_max_evaluations,
        backend=backend,
    )
    fair_results = []
    for index, architecture in enumerate(_dedupe_architectures(validation_candidates)):
        fair_budget = max(int(fair_min_evaluations), parameter_count(architecture.circuit) * int(fair_evals_per_param))
        result = optimize_vqe_energy(
            architecture,
            problem,
            seed=seed + 8000 + index,
            n_starts=final_n_starts,
            evals_per_param=fair_evals_per_param,
            max_evaluations=fair_budget,
            budget_override=fair_budget,
            backend=backend,
        )
        result.metadata["source"] = architecture.metadata.get("source", "candidate")
        fair_results.append(result)
    fair_results.sort(key=lambda result: result.energy)

    return VQEMultiStartSAReport(
        stage1_rows=stage1_rows,
        restart_rows=restart_rows,
        capped_results=capped_results,
        fair_results=fair_results,
        metadata={
            "problem_name": problem.name,
            "reference_energy": problem.reference_energy,
            "seed": seed,
            "n_starts": n_starts,
            "steps_per_start": steps_per_start,
            "t_init": t_init,
            "t_final": t_final,
            "search_evals_per_param": search_evals_per_param,
            "search_max_evaluations": search_max_evaluations,
            "final_n_starts": final_n_starts,
            "capped_evals_per_param": capped_evals_per_param,
            "capped_max_evaluations": capped_max_evaluations,
            "fair_evals_per_param": fair_evals_per_param,
            "fair_min_evaluations": fair_min_evaluations,
        },
    )


def run_ising4_fitness_correlation(
    seed: int = 2026,
    top_k: int = 10,
    candidate_limit: int = 72,
    stage1_keep_top: int = 16,
    short_n_starts: int = 1,
    short_evals_per_param: int = 8,
    short_max_evaluations: int = 40,
    fair_n_starts: int = 1,
    fair_evals_per_param: int = 20,
    fair_min_evaluations: int = 40,
    overlap_k: int = 5,
    backend: Optional[NumpyBackend] = None,
) -> VQEFitnessCorrelationReport:
    problem = ising4_demo_problem()
    backend = backend or resolve_qas_backend()
    masks = _sample_masks(enumerate_hea_masks(problem.n_qubits), candidate_limit, seed)
    candidates = [architecture_from_hea_mask(mask, backend=backend) for mask in masks]
    stage1_rows = zero_cost_guardrail(candidates, backend=backend)
    kept = [row for row in stage1_rows if row.kept][: max(1, int(stage1_keep_top))]
    selected = kept[: max(1, int(top_k))]

    rows: List[VQEFitnessCorrelationRow] = []
    for index, row in enumerate(selected, start=1):
        architecture = row.architecture
        short = optimize_vqe_energy(
            architecture,
            problem,
            seed=seed + 1000 + index,
            n_starts=short_n_starts,
            evals_per_param=short_evals_per_param,
            max_evaluations=short_max_evaluations,
            backend=backend,
        )
        fair_budget = max(int(fair_min_evaluations), parameter_count(architecture.circuit) * int(fair_evals_per_param))
        fair = optimize_vqe_energy(
            architecture,
            problem,
            seed=seed + 2000 + index,
            n_starts=fair_n_starts,
            evals_per_param=fair_evals_per_param,
            max_evaluations=fair_budget,
            budget_override=fair_budget,
            backend=backend,
        )
        rows.append(
            VQEFitnessCorrelationRow(
                rank=index,
                architecture=architecture,
                short_energy=short.energy,
                fair_energy=fair.energy,
                n_params=parameter_count(architecture.circuit),
                short_evaluations=short.evaluations,
                fair_evaluations=fair.evaluations,
            )
        )

    return VQEFitnessCorrelationReport(
        stage1_rows=stage1_rows,
        rows=rows,
        metadata={
            "problem_name": problem.name,
            "reference_energy": problem.reference_energy,
            "seed": seed,
            "top_k": top_k,
            "candidate_limit": candidate_limit,
            "stage1_keep_top": stage1_keep_top,
            "short_n_starts": short_n_starts,
            "short_evals_per_param": short_evals_per_param,
            "short_max_evaluations": short_max_evaluations,
            "fair_n_starts": fair_n_starts,
            "fair_evals_per_param": fair_evals_per_param,
            "fair_min_evaluations": fair_min_evaluations,
            "overlap_k": min(int(overlap_k), len(rows)),
        },
    )


def run_ising4_fitness_budget_sweep(
    seed: int = 2026,
    top_k: int = 10,
    budgets: Sequence[int] = (40, 100, 200, 400),
    candidate_limit: int = 72,
    stage1_keep_top: int = 16,
    short_n_starts: int = 1,
    short_evals_per_param: int = 1000,
    fair_n_starts: int = 1,
    fair_evals_per_param: int = 20,
    fair_min_evaluations: int = 40,
    overlap_k: int = 5,
    backend: Optional[NumpyBackend] = None,
) -> VQEFitnessBudgetSweepReport:
    problem = ising4_demo_problem()
    backend = backend or resolve_qas_backend()
    budgets = [int(budget) for budget in budgets]
    masks = _sample_masks(enumerate_hea_masks(problem.n_qubits), candidate_limit, seed)
    candidates = [architecture_from_hea_mask(mask, backend=backend) for mask in masks]
    stage1_rows = zero_cost_guardrail(candidates, backend=backend)
    kept = [row for row in stage1_rows if row.kept][: max(1, int(stage1_keep_top))]
    selected = kept[: max(1, int(top_k))]

    rows: List[VQEFitnessBudgetRow] = []
    for index, row in enumerate(selected, start=1):
        architecture = row.architecture
        fair_budget = max(int(fair_min_evaluations), parameter_count(architecture.circuit) * int(fair_evals_per_param))
        fair = optimize_vqe_energy(
            architecture,
            problem,
            seed=seed + 2000 + index,
            n_starts=fair_n_starts,
            evals_per_param=fair_evals_per_param,
            max_evaluations=fair_budget,
            budget_override=fair_budget,
            backend=backend,
        )
        item = VQEFitnessBudgetRow(
            rank=index,
            architecture=architecture,
            fair_energy=fair.energy,
            n_params=parameter_count(architecture.circuit),
            fair_evaluations=fair.evaluations,
            weighted_score=row.score.weighted_score,
            expressibility=row.score.expressibility.score,
            trainability=row.score.trainability.score,
            noise=row.score.noise_robustness.score,
            hardware=row.score.hardware_efficiency.score,
        )
        for budget in budgets:
            short = optimize_vqe_energy(
                architecture,
                problem,
                seed=seed + int(budget) * 100 + index,
                n_starts=short_n_starts,
                evals_per_param=short_evals_per_param,
                max_evaluations=int(budget),
                backend=backend,
            )
            item.short_by_budget[int(budget)] = short.energy
            item.short_evals_by_budget[int(budget)] = short.evaluations
        rows.append(item)

    return VQEFitnessBudgetSweepReport(
        stage1_rows=stage1_rows,
        rows=rows,
        budgets=budgets,
        metadata={
            "problem_name": problem.name,
            "reference_energy": problem.reference_energy,
            "seed": seed,
            "top_k": top_k,
            "candidate_limit": candidate_limit,
            "stage1_keep_top": stage1_keep_top,
            "short_n_starts": short_n_starts,
            "short_evals_per_param": short_evals_per_param,
            "fair_n_starts": fair_n_starts,
            "fair_evals_per_param": fair_evals_per_param,
            "fair_min_evaluations": fair_min_evaluations,
            "overlap_k": min(int(overlap_k), len(rows)),
        },
    )


def run_ising4_b2_reliability_experiment(
    seed: int = 2026,
    top_k: int = 8,
    candidate_limit: int = 72,
    stage1_keep_top: int = 16,
    b1_n_starts: int = 1,
    b1_evals_per_param: int = 1000,
    b1_max_evaluations: int = 50,
    b1_keep_fraction: float = 0.60,
    b2_n_starts: int = 1,
    b2_evals_per_param: int = 20,
    b2_max_evaluations: int = 400,
    fair_n_starts: int = 1,
    fair_evals_per_param: int = 20,
    fair_min_evaluations: int = 80,
    adaptive_fair_starts: bool = False,
    fair_params_per_start: int = 15,
    overlap_k: int = 5,
    include_aware_neighbors: bool = True,
    include_bad_baseline: bool = True,
    use_stratified_stage1: bool = True,
    include_priority_seeds: bool = True,
    improvement_floor: float = 1e-3,
    b1_floor_energy: float = -3.0,
    b1_floor_tolerance: float = 0.01,
    backend: Optional[NumpyBackend] = None,
) -> VQEB2ReliabilityReport:
    """First experiment for checking whether B2 ranking tracks fair VQE ranking."""
    problem = ising4_demo_problem()
    backend = backend or resolve_qas_backend()
    profile = analyze_hamiltonian(problem.hamiltonian)
    masks = _sample_masks(enumerate_hea_masks(problem.n_qubits), candidate_limit, seed)
    candidates = [architecture_from_hea_mask(mask, backend=backend) for mask in masks]
    if use_stratified_stage1:
        stage1_rows = stratified_stage1_pool(
            candidates,
            profile,
            pool_size=stage1_keep_top,
            backend=backend,
        )
    else:
        stage1_rows = zero_cost_guardrail(candidates, backend=backend)
    kept = [row for row in stage1_rows if row.kept][: max(1, int(stage1_keep_top))]

    selected_architectures = [row.architecture for row in kept[: max(1, int(top_k))]]
    if include_priority_seeds:
        for mask in derive_priority_seed_masks(profile, max_layers=max(LAYER_CHOICES)):
            architecture = architecture_from_hea_mask(mask, backend=backend)
            architecture.metadata["source"] = "hamiltonian_priority_seed"
            selected_architectures.append(architecture)
    if include_aware_neighbors:
        rng = np.random.default_rng(int(seed) + 500)
        for row in kept[: max(1, int(top_k))]:
            mask = HEAMask(*row.architecture.metadata["hea_mask"])
            neighbor = mutate_hea_mask_hamiltonian_aware(mask, profile, rng)
            architecture = architecture_from_hea_mask(neighbor, backend=backend)
            architecture.metadata["source"] = "hamiltonian_aware_neighbor"
            selected_architectures.append(architecture)
    if include_bad_baseline:
        selected_architectures.append(
            rotation_only_ansatz(
                n_qubits=problem.n_qubits,
                layers=1,
                rotation_block="ry",
                final_rotation="ry",
                backend=backend,
            )
        )
    selected_architectures = _dedupe_architectures(selected_architectures)

    b1_results = []
    for index, architecture in enumerate(selected_architectures, start=1):
        result = optimize_vqe_energy(
            architecture,
            problem,
            seed=seed + 1000 + index,
            n_starts=b1_n_starts,
            evals_per_param=b1_evals_per_param,
            max_evaluations=b1_max_evaluations,
            backend=backend,
        )
        b1_results.append(result)
    b1_survivors = b1_bottom_filter(b1_results, keep_fraction=b1_keep_fraction)
    survivor_names = {result.architecture.name for result in b1_survivors}
    if include_priority_seeds:
        for result in b1_results:
            if (
                result.architecture.metadata.get("source") == "hamiltonian_priority_seed"
                and result.architecture.name not in survivor_names
            ):
                b1_survivors.append(result)
                survivor_names.add(result.architecture.name)
    if include_bad_baseline:
        for result in b1_results:
            if result.architecture.metadata.get("source") == "bad_baseline" and result.architecture.name not in survivor_names:
                b1_survivors.append(result)
                survivor_names.add(result.architecture.name)
    b1_by_name = {result.architecture.name: result for result in b1_results}

    rows: List[VQEB2ReliabilityRow] = []
    for index, b1_result in enumerate(b1_survivors, start=1):
        architecture = b1_result.architecture
        b2 = optimize_vqe_energy(
            architecture,
            problem,
            seed=seed + 2000 + index,
            n_starts=b2_n_starts,
            evals_per_param=b2_evals_per_param,
            max_evaluations=b2_max_evaluations,
            backend=backend,
        )
        fair_budget = max(int(fair_min_evaluations), parameter_count(architecture.circuit) * int(fair_evals_per_param))
        starts = (
            adaptive_fair_n_starts(architecture, min_starts=fair_n_starts, params_per_start=fair_params_per_start)
            if adaptive_fair_starts
            else int(fair_n_starts)
        )
        fair = optimize_vqe_energy(
            architecture,
            problem,
            seed=seed + 3000 + index,
            n_starts=starts,
            evals_per_param=fair_evals_per_param,
            max_evaluations=fair_budget,
            budget_override=fair_budget,
            backend=backend,
        )
        b1 = b1_by_name[architecture.name]
        improvement = float(b1.energy - b2.energy)
        improvement_valid = is_b1_improvement_valid(
            b1.energy,
            floor_energy=b1_floor_energy,
            tolerance=b1_floor_tolerance,
        )
        if improvement < float(improvement_floor) and architecture.metadata.get("source") == "bad_baseline":
            continue
        rows.append(
            VQEB2ReliabilityRow(
                rank=0,
                architecture=architecture,
                b1_energy=b1.energy,
                b2_energy=b2.energy,
                fair_energy=fair.energy,
                n_params=parameter_count(architecture.circuit),
                b1_evaluations=b1.evaluations,
                b2_evaluations=b2.evaluations,
                fair_evaluations=fair.evaluations,
                family=get_structure_family(architecture),
                improvement_valid=improvement_valid,
            )
        )
    rows.sort(key=lambda row: (row.b2_energy, row.architecture.name))
    for rank, row in enumerate(rows, start=1):
        row.rank = rank

    return VQEB2ReliabilityReport(
        stage1_rows=stage1_rows,
        rows=rows,
        profile=profile,
        b1_survivor_count=len(b1_survivors),
        metadata={
            "problem_name": problem.name,
            "reference_energy": problem.reference_energy,
            "seed": seed,
            "top_k": top_k,
            "candidate_limit": candidate_limit,
            "stage1_keep_top": stage1_keep_top,
            "b1_max_evaluations": b1_max_evaluations,
            "b1_keep_fraction": b1_keep_fraction,
            "b2_evals_per_param": b2_evals_per_param,
            "b2_max_evaluations": b2_max_evaluations,
            "fair_n_starts": fair_n_starts,
            "fair_evals_per_param": fair_evals_per_param,
            "fair_min_evaluations": fair_min_evaluations,
            "adaptive_fair_starts": adaptive_fair_starts,
            "fair_params_per_start": fair_params_per_start,
            "include_aware_neighbors": include_aware_neighbors,
            "include_bad_baseline": include_bad_baseline,
            "use_stratified_stage1": use_stratified_stage1,
            "include_priority_seeds": include_priority_seeds,
            "improvement_floor": improvement_floor,
            "b1_floor_energy": b1_floor_energy,
            "b1_floor_tolerance": b1_floor_tolerance,
            "overlap_k": min(int(overlap_k), len(rows)),
        },
    )


def run_ising4_full_enumeration_baseline(
    seed: int = 2026,
    candidate_limit: Optional[int] = None,
    fair_n_starts: int = 1,
    fair_evals_per_param: int = 20,
    fair_min_evaluations: int = 80,
    adaptive_fair_starts: bool = False,
    fair_params_per_start: int = 15,
    include_bad_baseline: bool = True,
    backend: Optional[NumpyBackend] = None,
) -> VQEEnumerationReport:
    """Fair-VQE baseline over the explicit 4-qubit HEA-mask space."""
    problem = ising4_demo_problem()
    backend = backend or resolve_qas_backend()
    profile = analyze_hamiltonian(problem.hamiltonian)
    masks = enumerate_hea_masks(problem.n_qubits)
    if candidate_limit is not None:
        masks = _sample_masks(masks, int(candidate_limit), seed)
    candidates = [architecture_from_hea_mask(mask, backend=backend) for mask in masks]
    if include_bad_baseline:
        candidates.append(
            rotation_only_ansatz(
                n_qubits=problem.n_qubits,
                layers=1,
                rotation_block="ry",
                final_rotation="ry",
                backend=backend,
            )
        )
    candidates = _dedupe_architectures(candidates)

    results: List[VQEOptimizationResult] = []
    for index, architecture in enumerate(candidates, start=1):
        fair_budget = max(int(fair_min_evaluations), parameter_count(architecture.circuit) * int(fair_evals_per_param))
        starts = (
            adaptive_fair_n_starts(architecture, min_starts=fair_n_starts, params_per_start=fair_params_per_start)
            if adaptive_fair_starts
            else int(fair_n_starts)
        )
        result = optimize_vqe_energy(
            architecture,
            problem,
            seed=seed + 5000 + index,
            n_starts=starts,
            evals_per_param=fair_evals_per_param,
            max_evaluations=fair_budget,
            budget_override=fair_budget,
            backend=backend,
        )
        result.metadata["source"] = architecture.metadata.get("source", "full_enumeration")
        results.append(result)
    results.sort(key=lambda result: (result.energy, result.architecture.name))

    return VQEEnumerationReport(
        results=results,
        profile=profile,
        metadata={
            "problem_name": problem.name,
            "reference_energy": problem.reference_energy,
            "seed": seed,
            "candidate_limit": candidate_limit,
            "fair_n_starts": fair_n_starts,
            "fair_evals_per_param": fair_evals_per_param,
            "fair_min_evaluations": fair_min_evaluations,
            "adaptive_fair_starts": adaptive_fair_starts,
            "fair_params_per_start": fair_params_per_start,
            "include_bad_baseline": include_bad_baseline,
        },
    )


def run_ising4_fair_vqe_stability_experiment(
    seed: int = 2026,
    top_k: int = 5,
    repeats: int = 5,
    candidate_limit: Optional[int] = 24,
    fair_n_starts: int = 1,
    fair_evals_per_param: int = 20,
    fair_min_evaluations: int = 80,
    adaptive_fair_starts: bool = False,
    fair_params_per_start: int = 15,
    backend: Optional[NumpyBackend] = None,
) -> VQEFairStabilityReport:
    """Repeat fair VQE for top enumeration candidates to expose seed sensitivity."""
    problem = ising4_demo_problem()
    backend = backend or resolve_qas_backend()
    baseline = run_ising4_full_enumeration_baseline(
        seed=seed,
        candidate_limit=candidate_limit,
        fair_n_starts=fair_n_starts,
        fair_evals_per_param=fair_evals_per_param,
        fair_min_evaluations=fair_min_evaluations,
        adaptive_fair_starts=adaptive_fair_starts,
        fair_params_per_start=fair_params_per_start,
        include_bad_baseline=False,
        backend=backend,
    )
    return _run_fair_stability_for_architectures(
        [result.architecture for result in baseline.results[: max(1, int(top_k))]],
        problem=problem,
        seed=seed,
        repeats=repeats,
        fair_n_starts=fair_n_starts,
        fair_evals_per_param=fair_evals_per_param,
        fair_min_evaluations=fair_min_evaluations,
        adaptive_fair_starts=adaptive_fair_starts,
        fair_params_per_start=fair_params_per_start,
        backend=backend,
        metadata={
            "candidate_source": "enumeration_top",
            "candidate_limit": candidate_limit,
        },
    )


def run_tfim_priority_seed_validation(
    n_qubits: int = 6,
    J: float = 1.0,
    h: float = 0.5,
    periodic: bool = False,
    seed: int = 2026,
    repeats: int = 1,
    priority_limit: Optional[int] = 12,
    fair_n_starts: int = 1,
    fair_evals_per_param: int = 30,
    fair_min_evaluations: int = 120,
    fair_max_evaluations: int = 3000,
    adaptive_fair_starts: bool = False,
    fair_params_per_start: int = 20,
    init_mode: str = THETA_INIT_RANDOM_UNIFORM_PI,
    init_scale: float = float(np.pi),
    only_name: Optional[str] = None,
    priority_layers: Optional[Sequence[int]] = None,
    backend: Optional[Backend] = None,
) -> VQEFairStabilityReport:
    """Validate Hamiltonian-derived priority seeds on an n-qubit TFIM chain.

    This is intentionally budget-capped for first-pass 6q/8q checks. Increase
    priority_limit, repeats, and fair_evals_per_param only after the smoke run
    confirms runtime is acceptable.
    """
    backend = backend or resolve_qas_backend()
    problem = tfim_chain_demo_problem(n_qubits=n_qubits, J=J, h=h, periodic=periodic)
    profile = analyze_hamiltonian(problem.hamiltonian)
    masks = derive_priority_seed_masks(profile, max_layers=max(LAYER_CHOICES), layers=priority_layers)
    architectures = _dedupe_architectures([architecture_from_hea_mask(mask, backend=backend) for mask in masks])
    for architecture in architectures:
        architecture.metadata["source"] = "hamiltonian_priority_seed"
    if only_name:
        architectures = [architecture for architecture in architectures if str(only_name) in architecture.name]
    if priority_limit is not None:
        architectures = architectures[: max(1, int(priority_limit))]
    if not architectures:
        raise ValueError(f"No TFIM priority seed matched only_name={only_name!r}")

    rows: List[VQEFairStabilityRow] = []
    for rank, architecture in enumerate(architectures, start=1):
        fair_budget = max(int(fair_min_evaluations), parameter_count(architecture.circuit) * int(fair_evals_per_param))
        fair_budget = min(int(fair_budget), int(fair_max_evaluations))
        starts = (
            adaptive_fair_n_starts(architecture, min_starts=fair_n_starts, params_per_start=fair_params_per_start)
            if adaptive_fair_starts
            else int(fair_n_starts)
        )
        energies: List[float] = []
        evaluations: List[int] = []
        for repeat in range(max(1, int(repeats))):
            result = optimize_vqe_energy(
                architecture,
                problem,
                seed=seed + 15000 + rank * 100 + repeat,
                n_starts=starts,
                evals_per_param=fair_evals_per_param,
                max_evaluations=fair_budget,
                budget_override=fair_budget,
                backend=backend,
                init_mode=init_mode,
                init_scale=init_scale,
            )
            energies.append(float(result.energy))
            evaluations.append(int(result.evaluations))
        array = np.asarray(energies, dtype=float)
        rows.append(
            VQEFairStabilityRow(
                rank=rank,
                architecture=architecture,
                best_energy=float(np.min(array)),
                mean_energy=float(np.mean(array)),
                std_energy=float(np.std(array)),
                worst_energy=float(np.max(array)),
                n_params=parameter_count(architecture.circuit),
                repeat_energies=energies,
                repeat_evaluations=evaluations,
            )
        )
    rows.sort(key=lambda row: (row.best_energy, row.architecture.name))
    for rank, row in enumerate(rows, start=1):
        row.rank = rank
    return VQEFairStabilityReport(
        rows=rows,
        metadata={
            "problem_name": problem.name,
            "reference_energy": problem.reference_energy,
            "seed": seed,
            "top_k": len(architectures),
            "repeats": repeats,
            "candidate_source": "hamiltonian_priority_seed",
            "priority_seed_count": len(masks),
            "priority_limit": priority_limit,
            "fair_n_starts": fair_n_starts,
            "fair_evals_per_param": fair_evals_per_param,
            "fair_min_evaluations": fair_min_evaluations,
            "fair_max_evaluations": fair_max_evaluations,
            "adaptive_fair_starts": adaptive_fair_starts,
            "fair_params_per_start": fair_params_per_start,
            "init_mode": init_mode,
            "init_scale": init_scale,
            "only_name": only_name,
            "priority_layers": tuple(priority_layers) if priority_layers is not None else None,
            "hamiltonian_profile": profile,
        },
    )


def run_tfim_full_enumeration_baseline(
    n_qubits: int = 6,
    J: float = 1.0,
    h: float = 0.5,
    periodic: bool = False,
    seed: int = 2026,
    candidate_limit: Optional[int] = None,
    fair_n_starts: int = 1,
    fair_evals_per_param: int = 200,
    fair_min_evaluations: int = 1000,
    fair_max_evaluations: int = 1_000_000,
    adaptive_fair_starts: bool = False,
    fair_params_per_start: int = 20,
    init_mode: str = THETA_INIT_RANDOM_UNIFORM_PI,
    init_scale: float = float(np.pi),
    shard_index: int = 0,
    num_shards: int = 1,
    verbose: bool = False,
    checkpoint_path: Optional[str] = None,
    backend: Optional[Backend] = None,
) -> VQEEnumerationReport:
    """Budget-capped fair-VQE baseline over the explicit TFIM HEA-mask space."""
    backend = backend or resolve_qas_backend()
    problem = tfim_chain_demo_problem(n_qubits=n_qubits, J=J, h=h, periodic=periodic)
    profile = analyze_hamiltonian(problem.hamiltonian)
    masks = enumerate_hea_masks(problem.n_qubits)
    if candidate_limit is not None:
        masks = _sample_masks(masks, int(candidate_limit), seed)
    unsharded_candidate_count = len(masks)
    masks = _shard_items(masks, shard_index=shard_index, num_shards=num_shards)
    candidates = _dedupe_architectures([architecture_from_hea_mask(mask, backend=backend) for mask in masks])

    results: List[VQEOptimizationResult] = []
    for index, architecture in enumerate(candidates, start=1):
        fair_budget = max(int(fair_min_evaluations), parameter_count(architecture.circuit) * int(fair_evals_per_param))
        fair_budget = min(int(fair_budget), int(fair_max_evaluations))
        starts = (
            adaptive_fair_n_starts(architecture, min_starts=fair_n_starts, params_per_start=fair_params_per_start)
            if adaptive_fair_starts
            else int(fair_n_starts)
        )
        result = optimize_vqe_energy(
            architecture,
            problem,
            seed=seed + 17000 + int(shard_index) * 100000 + index,
            n_starts=starts,
            evals_per_param=fair_evals_per_param,
            max_evaluations=fair_budget,
            budget_override=fair_budget,
            backend=backend,
            init_mode=init_mode,
            init_scale=init_scale,
        )
        result.metadata["source"] = "tfim_full_enumeration"
        results.append(result)
        _append_vqe_checkpoint_row(
            checkpoint_path=checkpoint_path,
            result=result,
            problem=problem,
            candidate_index=index,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        if verbose:
            print(
                f"completed n={n_qubits} shard={shard_index}/{num_shards} "
                f"candidate={index}/{len(candidates)} energy={result.energy:.12f} "
                f"nfev={result.evaluations} name={architecture.name}",
                flush=True,
            )
    results.sort(key=lambda result: (result.energy, result.architecture.name))

    return VQEEnumerationReport(
        results=results,
        profile=profile,
        metadata={
            "problem_name": problem.name,
            "reference_energy": problem.reference_energy,
            "seed": seed,
            "candidate_limit": candidate_limit,
            "unsharded_candidate_count": unsharded_candidate_count,
            "candidate_count": len(candidates),
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
            "checkpoint_path": checkpoint_path,
            "fair_n_starts": fair_n_starts,
            "fair_evals_per_param": fair_evals_per_param,
            "fair_min_evaluations": fair_min_evaluations,
            "fair_max_evaluations": fair_max_evaluations,
            "adaptive_fair_starts": adaptive_fair_starts,
            "fair_params_per_start": fair_params_per_start,
            "init_mode": init_mode,
            "init_scale": init_scale,
            **backend_runtime_metadata(backend),
        },
    )


def run_tfim_stage1_stage2_search(
    n_qubits: int = 6,
    J: float = 1.0,
    h: float = 0.5,
    periodic: bool = False,
    seed: int = 2026,
    pool_count: int = 4,
    candidate_limit: int = 72,
    stage1_pool_size: int = 16,
    beam_width: int = 12,
    stage2_rounds: int = 2,
    neighbors_per_parent: int = 2,
    b2_n_starts: int = 1,
    b2_evals_per_param: int = 80,
    b2_max_evaluations: int = 1_000_000,
    fair_repeats: int = 5,
    fair_n_starts: int = 1,
    fair_evals_per_param: int = 200,
    fair_min_evaluations: int = 1000,
    fair_max_evaluations: int = 1_000_000,
    fair_params_per_start: int = 20,
    init_mode: str = THETA_INIT_RANDOM_UNIFORM_PI,
    init_scale: float = float(np.pi),
    include_priority_seeds: bool = True,
    priority_layers: Optional[Sequence[int]] = None,
    layer_quota: Optional[Dict[int, int]] = None,
    backend: Optional[Backend] = None,
) -> VQEFairStabilityReport:
    """Generic two-stage TFIM QAS search: multi-pool Stage 1, then Stage 2 beam."""
    backend = backend or resolve_qas_backend()
    problem = tfim_chain_demo_problem(n_qubits=n_qubits, J=J, h=h, periodic=periodic)
    profile = analyze_hamiltonian(problem.hamiltonian)
    all_masks = enumerate_hea_masks(problem.n_qubits)

    stage1_rows: List[Stage1Row] = []
    initial_architectures: List[ArchitectureSpec] = []
    for pool_index in range(max(1, int(pool_count))):
        masks = _sample_masks(all_masks, int(candidate_limit), seed + pool_index * 1009)
        candidates = [architecture_from_hea_mask(mask, backend=backend) for mask in masks]
        rows = stratified_stage1_pool(
            candidates,
            profile,
            pool_size=stage1_pool_size,
            layer_quota=layer_quota,
            backend=backend,
        )
        stage1_rows.extend(rows)
        for row in rows:
            if row.kept:
                row.architecture.metadata["source"] = f"stage1_pool_{pool_index}"
                initial_architectures.append(row.architecture)

    if include_priority_seeds:
        for mask in derive_priority_seed_masks(profile, max_layers=max(LAYER_CHOICES), layers=priority_layers):
            architecture = architecture_from_hea_mask(mask, backend=backend)
            architecture.metadata["source"] = "hamiltonian_priority_seed"
            initial_architectures.append(architecture)

    beam = _dedupe_architectures(initial_architectures)
    rng = np.random.default_rng(int(seed) + 31000)
    stage2_history: List[Dict[str, Any]] = []
    for round_index in range(max(1, int(stage2_rounds))):
        expanded = list(beam)
        for architecture in beam:
            mask = _mask_from_architecture(architecture)
            if mask is None:
                continue
            for _ in range(max(0, int(neighbors_per_parent))):
                neighbor_mask = mutate_hea_mask_hamiltonian_aware(mask, profile, rng)
                neighbor = architecture_from_hea_mask(neighbor_mask, backend=backend)
                neighbor.metadata["source"] = f"stage2_round_{round_index}_neighbor"
                expanded.append(neighbor)
                random_mask = mutate_hea_mask(mask, rng)
                random_neighbor = architecture_from_hea_mask(random_mask, backend=backend)
                random_neighbor.metadata["source"] = f"stage2_round_{round_index}_random_neighbor"
                expanded.append(random_neighbor)
        expanded = _dedupe_architectures(expanded)

        b2_scores: Dict[str, float] = {}
        for index, architecture in enumerate(expanded, start=1):
            b2_budget = min(
                int(b2_max_evaluations),
                max(1, parameter_count(architecture.circuit) * int(b2_evals_per_param)),
            )
            result = optimize_vqe_energy(
                architecture,
                problem,
                seed=seed + 32000 + round_index * 1000 + index,
                n_starts=b2_n_starts,
                evals_per_param=b2_evals_per_param,
                max_evaluations=b2_budget,
                budget_override=b2_budget,
                backend=backend,
                init_mode=init_mode,
                init_scale=init_scale,
            )
            b2_scores[architecture.name] = float(result.energy)
        beam = update_beam(expanded, b2_scores, beam_width=beam_width)
        stage2_history.append(
            {
                "round": round_index + 1,
                "expanded": len(expanded),
                "beam": [architecture.name for architecture in beam],
                "best_b2": min(b2_scores.values()) if b2_scores else None,
            }
        )

    final_rows: List[VQEFairStabilityRow] = []
    for rank, architecture in enumerate(beam, start=1):
        fair_budget = max(int(fair_min_evaluations), parameter_count(architecture.circuit) * int(fair_evals_per_param))
        fair_budget = min(int(fair_budget), int(fair_max_evaluations))
        starts = adaptive_fair_n_starts(
            architecture,
            min_starts=fair_n_starts,
            params_per_start=fair_params_per_start,
        )
        energies: List[float] = []
        evaluations: List[int] = []
        for repeat in range(max(1, int(fair_repeats))):
            result = optimize_vqe_energy(
                architecture,
                problem,
                seed=seed + 36000 + rank * 100 + repeat,
                n_starts=starts,
                evals_per_param=fair_evals_per_param,
                max_evaluations=fair_budget,
                budget_override=fair_budget,
                backend=backend,
                init_mode=init_mode,
                init_scale=init_scale,
            )
            energies.append(float(result.energy))
            evaluations.append(int(result.evaluations))
        array = np.asarray(energies, dtype=float)
        basin_threshold = float(np.min(array)) + 0.005
        final_rows.append(
            VQEFairStabilityRow(
                rank=rank,
                architecture=architecture,
                best_energy=float(np.min(array)),
                mean_energy=float(np.mean(array)),
                std_energy=float(np.std(array)),
                worst_energy=float(np.max(array)),
                n_params=parameter_count(architecture.circuit),
                repeat_energies=energies,
                repeat_evaluations=evaluations,
                absolute_sr_10mha=float(np.mean(array < problem.reference_energy + 0.010)),
                absolute_sr_20mha=float(np.mean(array < problem.reference_energy + 0.020)),
                posthoc_basin_sr_5mha=float(np.mean(array < basin_threshold)),
            )
        )
    final_rows.sort(key=lambda row: (row.best_energy, row.architecture.name))
    for rank, row in enumerate(final_rows, start=1):
        row.rank = rank

    return VQEFairStabilityReport(
        rows=final_rows,
        metadata={
            "problem_name": problem.name,
            "reference_energy": problem.reference_energy,
            "seed": seed,
            "candidate_source": "stage1_stage2_pipeline",
            "top_k": len(final_rows),
            "repeats": fair_repeats,
            "pool_count": pool_count,
            "candidate_limit": candidate_limit,
            "stage1_pool_size": stage1_pool_size,
            "stage1_rows": len(stage1_rows),
            "stage1_kept": len(_dedupe_architectures(initial_architectures)),
            "beam_width": beam_width,
            "stage2_rounds": stage2_rounds,
            "neighbors_per_parent": neighbors_per_parent,
            "stage2_history": stage2_history,
            "b2_n_starts": b2_n_starts,
            "b2_evals_per_param": b2_evals_per_param,
            "b2_max_evaluations": b2_max_evaluations,
            "fair_n_starts": fair_n_starts,
            "fair_evals_per_param": fair_evals_per_param,
            "fair_min_evaluations": fair_min_evaluations,
            "fair_max_evaluations": fair_max_evaluations,
            "fair_params_per_start": fair_params_per_start,
            "init_mode": init_mode,
            "init_scale": init_scale,
            "include_priority_seeds": include_priority_seeds,
            "priority_layers": tuple(priority_layers) if priority_layers is not None else None,
            "layer_quota": dict(layer_quota) if layer_quota is not None else None,
            "hamiltonian_profile": profile,
        },
    )


def _run_fair_stability_for_architectures(
    architectures: Sequence[ArchitectureSpec],
    problem: VQEDemoProblem,
    seed: int,
    repeats: int,
    fair_n_starts: int,
    fair_evals_per_param: int,
    fair_min_evaluations: int,
    adaptive_fair_starts: bool,
    fair_params_per_start: int,
    backend: Backend,
    metadata: Optional[Dict[str, Any]] = None,
    init_mode: str = THETA_INIT_RANDOM_UNIFORM_PI,
    init_scale: float = float(np.pi),
) -> VQEFairStabilityReport:
    rows: List[VQEFairStabilityRow] = []
    architectures = _dedupe_architectures(architectures)
    for rank, architecture in enumerate(architectures, start=1):
        fair_budget = max(int(fair_min_evaluations), parameter_count(architecture.circuit) * int(fair_evals_per_param))
        starts = (
            adaptive_fair_n_starts(architecture, min_starts=fair_n_starts, params_per_start=fair_params_per_start)
            if adaptive_fair_starts
            else int(fair_n_starts)
        )
        energies: List[float] = []
        evaluations: List[int] = []
        for repeat in range(max(1, int(repeats))):
            repeated = optimize_vqe_energy(
                architecture,
                problem,
                seed=seed + 9000 + rank * 100 + repeat,
                n_starts=starts,
                evals_per_param=fair_evals_per_param,
                max_evaluations=fair_budget,
                budget_override=fair_budget,
                backend=backend,
                init_mode=init_mode,
                init_scale=init_scale,
            )
            energies.append(float(repeated.energy))
            evaluations.append(int(repeated.evaluations))
        array = np.asarray(energies, dtype=float)
        basin_threshold = float(np.min(array)) + 0.005
        rows.append(
            VQEFairStabilityRow(
                rank=rank,
                architecture=architecture,
                best_energy=float(np.min(array)),
                mean_energy=float(np.mean(array)),
                std_energy=float(np.std(array)),
                worst_energy=float(np.max(array)),
                n_params=parameter_count(architecture.circuit),
                repeat_energies=energies,
                repeat_evaluations=evaluations,
                absolute_sr_10mha=float(np.mean(array < problem.reference_energy + 0.010)),
                absolute_sr_20mha=float(np.mean(array < problem.reference_energy + 0.020)),
                posthoc_basin_sr_5mha=float(np.mean(array < basin_threshold)),
            )
        )
    rows.sort(key=lambda row: (row.best_energy, row.architecture.name))
    for rank, row in enumerate(rows, start=1):
        row.rank = rank
    report_metadata = dict(metadata or {})
    report_metadata.update(
        {
            "problem_name": problem.name,
            "reference_energy": problem.reference_energy,
            "seed": seed,
            "top_k": len(architectures),
            "repeats": repeats,
            "fair_n_starts": fair_n_starts,
            "fair_evals_per_param": fair_evals_per_param,
            "fair_min_evaluations": fair_min_evaluations,
            "adaptive_fair_starts": adaptive_fair_starts,
            "fair_params_per_start": fair_params_per_start,
            "init_mode": init_mode,
            "init_scale": init_scale,
        }
    )
    return VQEFairStabilityReport(
        rows=rows,
        metadata=report_metadata,
    )


def run_ising4_final_multiseed_validation(
    seed: int = 2026,
    repeats: int = 5,
    candidate_limit: int = 72,
    stage1_keep_top: int = 16,
    b1_n_starts: int = 1,
    b1_evals_per_param: int = 1000,
    b1_max_evaluations: int = 50,
    b1_keep_fraction: float = 0.60,
    b2_n_starts: int = 1,
    b2_evals_per_param: int = 20,
    b2_max_evaluations: int = 400,
    fair_n_starts: int = 1,
    fair_evals_per_param: int = 20,
    fair_min_evaluations: int = 80,
    adaptive_fair_starts: bool = False,
    fair_params_per_start: int = 15,
    include_aware_neighbors: bool = True,
    include_priority_seeds: bool = True,
    backend: Optional[Backend] = None,
) -> VQEFairStabilityReport:
    """Final 4q pipeline: Stage 1/prior candidates, B2 diagnostics, multi-seed fair ranking."""
    backend = backend or resolve_qas_backend()
    diagnostics = run_ising4_b2_reliability_experiment(
        seed=seed,
        candidate_limit=candidate_limit,
        stage1_keep_top=stage1_keep_top,
        b1_n_starts=b1_n_starts,
        b1_evals_per_param=b1_evals_per_param,
        b1_max_evaluations=b1_max_evaluations,
        b1_keep_fraction=b1_keep_fraction,
        b2_n_starts=b2_n_starts,
        b2_evals_per_param=b2_evals_per_param,
        b2_max_evaluations=b2_max_evaluations,
        fair_n_starts=fair_n_starts,
        fair_evals_per_param=fair_evals_per_param,
        fair_min_evaluations=fair_min_evaluations,
        adaptive_fair_starts=adaptive_fair_starts,
        fair_params_per_start=fair_params_per_start,
        include_aware_neighbors=include_aware_neighbors,
        include_bad_baseline=True,
        include_priority_seeds=include_priority_seeds,
        backend=backend,
    )
    architectures = [row.architecture for row in diagnostics.rows]
    return _run_fair_stability_for_architectures(
        architectures,
        problem=ising4_demo_problem(),
        seed=seed + 12000,
        repeats=repeats,
        fair_n_starts=fair_n_starts,
        fair_evals_per_param=fair_evals_per_param,
        fair_min_evaluations=fair_min_evaluations,
        adaptive_fair_starts=adaptive_fair_starts,
        fair_params_per_start=fair_params_per_start,
        backend=backend,
        metadata={
            "candidate_source": "stage1_priority_b2_diagnostics",
            "diagnostic_candidates": len(diagnostics.rows),
            "candidate_limit": candidate_limit,
            "stage1_keep_top": stage1_keep_top,
        },
    )


def run_ising4_trainability_prior_demo(
    seed: int = 2026,
    candidate_limit: int = 72,
    stage1_keep_top: int = 16,
    trainability_top_k: int = 5,
    sa_steps: int = 40,
    t_init: float = 0.30,
    t_final: float = 0.001,
    search_evals_per_param: int = 8,
    search_max_evaluations: int = 40,
    final_n_starts: int = 3,
    fair_evals_per_param: int = 20,
    fair_min_evaluations: int = 40,
    backend: Optional[NumpyBackend] = None,
) -> VQETrainabilityPriorReport:
    problem = ising4_demo_problem()
    backend = backend or resolve_qas_backend()
    masks = _sample_masks(enumerate_hea_masks(problem.n_qubits), candidate_limit, seed)
    candidates = [architecture_from_hea_mask(mask, backend=backend) for mask in masks]
    stage1_rows = zero_cost_guardrail(candidates, backend=backend)
    ranked = _rank_stage1_rows(stage1_rows, "trainability")[: max(1, int(stage1_keep_top))]

    trainability_candidates: List[ArchitectureSpec] = []
    for index, row in enumerate(ranked[: max(1, int(trainability_top_k))], start=1):
        row.architecture.metadata["source"] = f"trainability_top_{index}"
        trainability_candidates.append(row.architecture)

    trainability_top_results = []
    for index, architecture in enumerate(_dedupe_architectures(trainability_candidates)):
        fair_budget = max(int(fair_min_evaluations), parameter_count(architecture.circuit) * int(fair_evals_per_param))
        result = optimize_vqe_energy(
            architecture,
            problem,
            seed=seed + 3000 + index,
            n_starts=final_n_starts,
            evals_per_param=fair_evals_per_param,
            max_evaluations=fair_budget,
            budget_override=fair_budget,
            backend=backend,
        )
        result.metadata["source"] = architecture.metadata.get("source", "trainability_top")
        trainability_top_results.append(result)
    trainability_top_results.sort(key=lambda result: result.energy)

    seed_mask = HEAMask(*ranked[0].architecture.metadata["hea_mask"])
    sa_best, trace = run_sa_search(
        seed_mask,
        n_steps=sa_steps,
        seed=seed,
        evals_per_param=search_evals_per_param,
        max_evaluations=search_max_evaluations,
        t_init=t_init,
        t_final=t_final,
        problem=problem,
        backend=backend,
    )
    fair_budget = max(int(fair_min_evaluations), parameter_count(sa_best.architecture.circuit) * int(fair_evals_per_param))
    sa_final = optimize_vqe_energy(
        sa_best.architecture,
        problem,
        seed=seed + 4000,
        n_starts=final_n_starts,
        evals_per_param=fair_evals_per_param,
        max_evaluations=fair_budget,
        budget_override=fair_budget,
        backend=backend,
    )
    sa_final.metadata["source"] = "trainability_seeded_sa"

    baseline_results = []
    for index, architecture in enumerate(_baseline_architectures(problem.n_qubits, backend)):
        fair_budget = max(int(fair_min_evaluations), parameter_count(architecture.circuit) * int(fair_evals_per_param))
        result = optimize_vqe_energy(
            architecture,
            problem,
            seed=seed + 5000 + index,
            n_starts=final_n_starts,
            evals_per_param=fair_evals_per_param,
            max_evaluations=fair_budget,
            budget_override=fair_budget,
            backend=backend,
        )
        result.metadata["source"] = architecture.metadata.get("source", "baseline")
        baseline_results.append(result)
    baseline_results.sort(key=lambda result: result.energy)

    return VQETrainabilityPriorReport(
        stage1_rows=stage1_rows,
        trainability_top_results=trainability_top_results,
        sa_trace=trace,
        sa_final_result=sa_final,
        baseline_results=baseline_results,
        metadata={
            "problem_name": problem.name,
            "reference_energy": problem.reference_energy,
            "seed": seed,
            "candidate_limit": candidate_limit,
            "stage1_keep_top": stage1_keep_top,
            "trainability_top_k": trainability_top_k,
            "sa_steps": sa_steps,
            "t_init": t_init,
            "t_final": t_final,
            "search_evals_per_param": search_evals_per_param,
            "search_max_evaluations": search_max_evaluations,
            "final_n_starts": final_n_starts,
            "fair_evals_per_param": fair_evals_per_param,
            "fair_min_evaluations": fair_min_evaluations,
        },
    )


__all__ = [
    "ENTANGLE_PATTERNS",
    "ENTANGLERS",
    "FINAL_ROTATIONS",
    "H2_HAMILTONIAN",
    "H2_REFERENCE_ENERGY",
    "HEAMask",
    "HamiltonianProfile",
    "ISING4_HAMILTONIAN",
    "LAYER_CHOICES",
    "ROTATION_BLOCKS",
    "SAStep",
    "Stage1Row",
    "TFIMReferenceAlignmentReport",
    "TFIMReferenceAlignmentRow",
    "THETA_INIT_RANDOM_UNIFORM_PI",
    "THETA_INIT_ZERO_DIAGNOSTIC",
    "V3_COBYLA_RHOBEG",
    "V3_COBYLA_TOL",
    "VQEB2ReliabilityReport",
    "VQEDemoProblem",
    "VQEEnumerationReport",
    "VQEFairStabilityReport",
    "VQEHEADemoReport",
    "VQETrainabilityPriorReport",
    "VQEOptimizationResult",
    "analyze_hamiltonian",
    "architecture_from_hea_mask",
    "adaptive_fair_n_starts",
    "b1_bottom_filter",
    "enumerate_hea_masks",
    "evaluate_h2_energy",
    "evaluate_vqe_energy",
    "exact_ground_energy",
    "get_structure_family",
    "derive_priority_seed_masks",
    "diagnose_theta_randomness",
    "h2_demo_problem",
    "hamiltonian_matrix",
    "hamiltonian_aware_mask_preferences",
    "h2_hamiltonian_matrix",
    "is_hamiltonian_favored_family",
    "is_b1_improvement_valid",
    "ising4_demo_problem",
    "mutate_hea_mask",
    "mutate_hea_mask_hamiltonian_aware",
    "optimize_h2_energy",
    "optimize_vqe_energy",
    "rotation_only_ansatz",
    "resolve_qas_backend",
    "run_ising4_b2_reliability_experiment",
    "run_ising4_fair_vqe_stability_experiment",
    "run_ising4_final_multiseed_validation",
    "run_ising4_full_enumeration_baseline",
    "run_ising4_trainability_prior_demo",
    "run_tfim_full_enumeration_baseline",
    "run_tfim_priority_seed_validation",
    "run_tfim_stage1_stage2_search",
    "run_sa_search",
    "stratified_stage1_pool",
    "tfim_chain_demo_problem",
    "tfim_chain_hamiltonian",
    "tfim_open_chain_free_fermion_ground_energy",
    "update_beam",
    "validate_tfim_reference_alignment",
    "v3_final_maxfev",
    "v3_screening_maxfev",
    "v3_top_k",
    "run_vqe_hea_demo",
    "run_vqe_ising4_demo",
    "zero_cost_guardrail",
]
