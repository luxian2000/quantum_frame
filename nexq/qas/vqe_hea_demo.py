"""Minimal VQE-QAS demo utilities for a 2-qubit H2 toy Hamiltonian.

The goal of this module is not chemical accuracy. It provides a small, fully
self-contained demo for the pipeline:

zero-cost guardrail -> simulated annealing with short-step VQE energy -> final VQE validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import exp
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

try:  # pragma: no cover - fallback is tested implicitly when scipy is absent.
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None

from ..channel.backends.numpy_backend import NumpyBackend
from ..measure.measure import Measure
from ._types import ArchitectureScore, ArchitectureSpec
from .architecture_candidates import build_common_architectures, qaoa_ansatz
from .evaluator import evaluate_architectures
from .task_evaluation import bind_parameters, parameter_count


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


@dataclass(frozen=True)
class VQEDemoProblem:
    """Small Pauli-Hamiltonian VQE task used by the demo pipeline."""

    name: str
    n_qubits: int
    hamiltonian: Sequence[tuple[float, str]]
    reference_energy: float


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


def h2_hamiltonian_matrix() -> np.ndarray:
    return hamiltonian_matrix(H2_HAMILTONIAN)


def evaluate_vqe_energy(
    architecture: ArchitectureSpec,
    problem: VQEDemoProblem,
    parameters: Optional[Sequence[float]] = None,
    backend: Optional[NumpyBackend] = None,
) -> float:
    if architecture.n_qubits != problem.n_qubits:
        raise ValueError("architecture and VQE problem must use the same number of qubits")
    backend = backend or NumpyBackend()
    n_params = parameter_count(architecture.circuit)
    params = [0.0] * n_params if parameters is None else list(parameters)
    if len(params) != n_params:
        raise ValueError(f"Expected {n_params} parameters, got {len(params)}")
    circuit = bind_parameters(architecture.circuit, params) if n_params else architecture.circuit
    circuit.bind_backend(backend)
    result = Measure(backend).run(circuit, return_state=True)
    state = np.asarray(result.final_state, dtype=np.complex128).reshape(-1, 1)
    energy = (np.conj(state).T @ hamiltonian_matrix(problem.hamiltonian) @ state)[0, 0]
    return float(np.real(energy))


def evaluate_h2_energy(
    architecture: ArchitectureSpec,
    parameters: Optional[Sequence[float]] = None,
    backend: Optional[NumpyBackend] = None,
) -> float:
    return evaluate_vqe_energy(architecture, h2_demo_problem(), parameters=parameters, backend=backend)


def _evaluation_budget(architecture: ArchitectureSpec, evals_per_param: int, max_evaluations: int) -> int:
    n_params = max(1, parameter_count(architecture.circuit))
    return max(1, min(int(n_params * evals_per_param), int(max_evaluations)))


def optimize_vqe_energy(
    architecture: ArchitectureSpec,
    problem: VQEDemoProblem,
    seed: int = 1234,
    n_starts: int = 1,
    evals_per_param: int = 10,
    max_evaluations: int = 80,
    backend: Optional[NumpyBackend] = None,
) -> VQEOptimizationResult:
    if architecture.n_qubits != problem.n_qubits:
        raise ValueError("architecture and VQE problem must use the same number of qubits")
    backend = backend or NumpyBackend()
    n_params = parameter_count(architecture.circuit)
    rng = np.random.default_rng(int(seed))
    budget = _evaluation_budget(architecture, evals_per_param, max_evaluations)
    best_energy = float("inf")
    best_params = np.zeros(n_params, dtype=float)
    total_evals = 0

    def objective(params: np.ndarray) -> float:
        return evaluate_vqe_energy(architecture, problem, params, backend=backend)

    starts = [np.zeros(n_params, dtype=float)]
    for _ in range(max(0, int(n_starts) - 1)):
        starts.append(rng.uniform(-np.pi, np.pi, size=n_params))

    for start in starts:
        if n_params == 0:
            energy = objective(start)
            nfev = 1
            params = start
        elif minimize is not None:
            result = minimize(objective, start, method="COBYLA", options={"maxiter": budget, "rhobeg": 1.0})
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
        if energy < best_energy:
            best_energy = energy
            best_params = params.copy()

    return VQEOptimizationResult(
        architecture=architecture,
        energy=float(best_energy),
        best_parameters=[float(value) for value in best_params],
        evaluations=total_evals,
        n_starts=len(starts),
        metadata={"optimizer": "COBYLA" if minimize is not None else "budgeted_random_search", "budget_per_start": budget},
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


def _sample_masks(masks: Sequence[HEAMask], limit: int, seed: int) -> List[HEAMask]:
    masks = list(masks)
    if len(masks) <= limit:
        return masks
    rng = np.random.default_rng(int(seed))
    indices = sorted(rng.choice(np.arange(len(masks)), size=limit, replace=False).tolist())
    return [masks[index] for index in indices]


def zero_cost_guardrail(
    candidates: Sequence[ArchitectureSpec],
    n_samples: int = 6,
    quantile: float = 0.25,
    rescue_count: int = 3,
    backend: Optional[NumpyBackend] = None,
) -> List[Stage1Row]:
    backend = backend or NumpyBackend()
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
    backend = backend or NumpyBackend()
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
    backend = backend or NumpyBackend()
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
    backend = backend or NumpyBackend()
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
    backend = backend or NumpyBackend()
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
    fair_n_starts: int = 3,
    fair_evals_per_param: int = 20,
    fair_min_evaluations: int = 40,
    overlap_k: int = 5,
    backend: Optional[NumpyBackend] = None,
) -> VQEFitnessCorrelationReport:
    problem = ising4_demo_problem()
    backend = backend or NumpyBackend()
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


__all__ = [
    "ENTANGLE_PATTERNS",
    "ENTANGLERS",
    "FINAL_ROTATIONS",
    "H2_HAMILTONIAN",
    "H2_REFERENCE_ENERGY",
    "HEAMask",
    "ISING4_HAMILTONIAN",
    "LAYER_CHOICES",
    "ROTATION_BLOCKS",
    "SABudgetSweepRow",
    "SAMultiStartRow",
    "SAStep",
    "Stage1Row",
    "VQEBudgetSweepReport",
    "VQEDemoProblem",
    "VQEFitnessCorrelationReport",
    "VQEFitnessCorrelationRow",
    "VQEHEADemoReport",
    "VQEMultiStartSAReport",
    "VQEOptimizationResult",
    "architecture_from_hea_mask",
    "enumerate_hea_masks",
    "evaluate_h2_energy",
    "evaluate_vqe_energy",
    "exact_ground_energy",
    "h2_demo_problem",
    "hamiltonian_matrix",
    "h2_hamiltonian_matrix",
    "ising4_demo_problem",
    "mutate_hea_mask",
    "optimize_h2_energy",
    "optimize_vqe_energy",
    "run_ising4_budget_sweep",
    "run_ising4_fitness_correlation",
    "run_ising4_multistart_sa",
    "run_sa_search",
    "run_vqe_hea_demo",
    "run_vqe_ising4_demo",
    "zero_cost_guardrail",
]
