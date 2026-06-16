"""Shot-based Pauli-term energy estimation.

This module estimates Hamiltonian expectation values by decomposing a
``Hamiltonian`` into Pauli strings, grouping qubit-wise commuting terms, adding
local basis-change gates, and evaluating energies from finite-shot counts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from ..backends.numpy_backend import NumpyBackend
from ..operators import Hamiltonian
from ..core.circuit import Circuit, hadamard, rz
from ..ir import circuit_gate_dicts
from .measure import Measure


@dataclass(frozen=True)
class PauliTerm:
    """A real coefficient multiplied by a Pauli string."""

    coefficient: float
    pauli: str

    @property
    def is_identity(self) -> bool:
        return all(label == "I" for label in self.pauli)


@dataclass(frozen=True)
class PauliGroup:
    """A group of qubit-wise commuting Pauli terms measured in one basis."""

    basis: str
    terms: tuple[PauliTerm, ...]

    @property
    def is_identity(self) -> bool:
        return all(label == "I" for label in self.basis)


@dataclass
class PauliTermEstimate:
    """Finite-shot estimate for one Pauli term."""

    pauli: str
    coefficient: float
    expectation: float
    variance: float
    energy: float
    energy_variance: float
    shots: int


@dataclass
class PauliGroupEstimate:
    """Finite-shot estimate for one measurement group."""

    basis: str
    shots: int
    counts: dict[str, int]
    energy: float
    variance: float
    terms: tuple[PauliTermEstimate, ...]


@dataclass
class PauliEstimateResult:
    """Shot-based Hamiltonian energy estimate."""

    energy: float
    variance: float
    shots: int
    groups: tuple[PauliGroupEstimate, ...]
    term_results: tuple[PauliTermEstimate, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def std_error(self) -> float:
        return float(math.sqrt(max(self.variance, 0.0)))


def _as_real_coefficient(value: complex, *, atol: float = 1e-12) -> float:
    coefficient = complex(value)
    if abs(coefficient.imag) > atol:
        raise ValueError("PauliEstimator only supports real Hamiltonian coefficients")
    return float(coefficient.real)


def _normalize_pauli(pauli: str) -> str:
    labels = str(pauli).strip().upper()
    if not labels:
        raise ValueError("Pauli string cannot be empty")
    invalid = sorted(set(labels) - {"I", "X", "Y", "Z"})
    if invalid:
        raise ValueError(f"Unsupported Pauli label(s): {', '.join(invalid)}")
    return labels


def hamiltonian_pauli_terms(hamiltonian: Hamiltonian | Iterable[PauliTerm | tuple[complex, str]]) -> tuple[PauliTerm, ...]:
    """Return Pauli terms from a ``Hamiltonian`` or ``(coefficient, pauli)`` iterable."""

    if isinstance(hamiltonian, Hamiltonian):
        return tuple(
            PauliTerm(
                coefficient=_as_real_coefficient(term.coefficient),
                pauli="".join(term.qubit_labels),
            )
            for term in hamiltonian.terms
        )
    if isinstance(hamiltonian, np.ndarray):
        raise TypeError("dense_matrix Hamiltonians cannot be used by PauliEstimator; pass a Hamiltonian or Pauli terms")

    terms: list[PauliTerm] = []
    expected_len: int | None = None
    for raw in hamiltonian:
        if isinstance(raw, PauliTerm):
            term = PauliTerm(raw.coefficient, _normalize_pauli(raw.pauli))
        else:
            coefficient, pauli = raw
            term = PauliTerm(_as_real_coefficient(coefficient), _normalize_pauli(pauli))

        if expected_len is None:
            expected_len = len(term.pauli)
        elif len(term.pauli) != expected_len:
            raise ValueError("All Pauli strings must have the same length")
        terms.append(term)
    return tuple(terms)


def _can_share_basis(pauli: str, basis: str) -> bool:
    for label, basis_label in zip(pauli, basis):
        if label != "I" and basis_label not in ("I", label):
            return False
    return True


def _merged_basis(paulis: Iterable[str], n_qubits: int) -> str:
    basis = ["I"] * n_qubits
    for pauli in paulis:
        for qubit, label in enumerate(pauli):
            if label != "I":
                current = basis[qubit]
                if current not in ("I", label):
                    raise ValueError("Pauli terms are not qubit-wise commuting")
                basis[qubit] = label
    return "".join(basis)


def group_pauli_terms(
    terms: Iterable[PauliTerm | tuple[complex, str]],
    *,
    strategy: str = "qwc",
) -> tuple[PauliGroup, ...]:
    """Group Pauli terms.

    ``strategy="qwc"`` greedily groups qubit-wise commuting terms. Use
    ``strategy="none"`` or ``"ungrouped"`` to measure each term separately.
    """

    normalized = hamiltonian_pauli_terms(terms)
    if not normalized:
        return ()

    n_qubits = len(normalized[0].pauli)
    method = str(strategy).strip().lower()
    if method in {"none", "ungrouped"}:
        return tuple(PauliGroup(_merged_basis([term.pauli], n_qubits), (term,)) for term in normalized)
    if method != "qwc":
        raise ValueError("grouping strategy must be 'qwc', 'none', or 'ungrouped'")

    identity_terms = tuple(term for term in normalized if term.is_identity)
    non_identity_terms = tuple(term for term in normalized if not term.is_identity)

    groups: list[PauliGroup] = []
    if identity_terms:
        groups.append(PauliGroup("I" * n_qubits, identity_terms))

    for term in non_identity_terms:
        for index, group in enumerate(groups):
            if group.is_identity:
                continue
            if _can_share_basis(term.pauli, group.basis):
                updated_terms = group.terms + (term,)
                groups[index] = PauliGroup(_merged_basis((item.pauli for item in updated_terms), n_qubits), updated_terms)
                break
        else:
            groups.append(PauliGroup(_merged_basis([term.pauli], n_qubits), (term,)))
    return tuple(groups)


def allocate_group_shots(
    groups: Sequence[PauliGroup],
    shots: int,
    *,
    strategy: str | Sequence[int] | Mapping[int, int] = "uniform",
) -> tuple[int, ...]:
    """Allocate shots across non-identity groups."""

    total_shots = int(shots)
    if total_shots < 0:
        raise ValueError("shots must be non-negative")

    if isinstance(strategy, Mapping):
        allocation = [0] * len(groups)
        for index, value in strategy.items():
            allocation[int(index)] = int(value)
    elif isinstance(strategy, Sequence) and not isinstance(strategy, (str, bytes)):
        if len(strategy) != len(groups):
            raise ValueError("shot allocation sequence length must match groups")
        allocation = [int(value) for value in strategy]
    else:
        measured = [index for index, group in enumerate(groups) if not group.is_identity]
        allocation = [0] * len(groups)
        if not measured:
            return tuple(allocation)
        if total_shots <= 0:
            raise ValueError("shots must be positive when measuring non-identity Pauli groups")

        method = str(strategy).strip().lower()
        if method == "uniform":
            base, remainder = divmod(total_shots, len(measured))
            for offset, index in enumerate(measured):
                allocation[index] = base + (1 if offset < remainder else 0)
        elif method == "coefficient":
            weights = np.array(
                [sum(abs(term.coefficient) for term in groups[index].terms) for index in measured],
                dtype=float,
            )
            if not np.any(weights > 0):
                weights = np.ones_like(weights)
            raw = total_shots * weights / float(weights.sum())
            floors = np.floor(raw).astype(int)
            remainder = int(total_shots - floors.sum())
            order = np.argsort(-(raw - floors))
            for idx in order[:remainder]:
                floors[idx] += 1
            for index, value in zip(measured, floors):
                allocation[index] = int(value)
        else:
            raise ValueError("shot allocation strategy must be 'uniform', 'coefficient', a sequence, or a mapping")

    if any(value < 0 for value in allocation):
        raise ValueError("shot allocation values must be non-negative")
    for index, group in enumerate(groups):
        if group.is_identity and allocation[index] != 0:
            raise ValueError("identity-only groups must receive zero shots")
    if sum(allocation) != total_shots and any(not group.is_identity for group in groups):
        raise ValueError("allocated shots must sum to total shots")
    return tuple(allocation)


def basis_change_gates(basis: str) -> tuple[dict[str, Any], ...]:
    """Return local basis-change gates before computational-basis measurement."""

    gates: list[dict[str, Any]] = []
    for qubit, label in enumerate(_normalize_pauli(basis)):
        if label == "X":
            gates.append(hadamard(qubit))
        elif label == "Y":
            gates.append(rz(-math.pi / 2.0, qubit))
            gates.append(hadamard(qubit))
    return tuple(gates)


def measurement_circuit(circuit: Circuit, basis: str, *, backend=None) -> Circuit:
    """Return ``circuit`` followed by local basis-change gates."""

    if len(basis) != int(circuit.n_qubits):
        raise ValueError("basis length must match circuit.n_qubits")
    circuit_backend = getattr(circuit, "backend", None)
    selected_backend = circuit_backend if circuit_backend is not None else backend
    return Circuit(*circuit_gate_dicts(circuit), *basis_change_gates(basis), n_qubits=circuit.n_qubits, backend=selected_backend)


def _bits_from_count_key(key: str, n_qubits: int) -> str:
    bits = str(key).strip()
    if bits.startswith("|") and bits.endswith(">"):
        bits = bits[1:-1]
    if len(bits) != n_qubits or any(bit not in "01" for bit in bits):
        raise ValueError(f"Invalid count key {key!r} for n_qubits={n_qubits}")
    return bits


def pauli_eigenvalue_from_bits(pauli: str, bits: str) -> int:
    """Return the +/-1 eigenvalue for a Pauli string after basis measurement."""

    labels = _normalize_pauli(pauli)
    if len(labels) != len(bits):
        raise ValueError("Pauli string and bitstring length must match")
    value = 1
    for label, bit in zip(labels, bits):
        if label != "I" and bit == "1":
            value *= -1
    return value


def pauli_expectation_from_counts(pauli: str, counts: Mapping[str, int], *, n_qubits: int | None = None) -> tuple[float, float]:
    """Estimate ``<pauli>`` and variance of the sample mean from counts."""

    labels = _normalize_pauli(pauli)
    n_qubits = len(labels) if n_qubits is None else int(n_qubits)
    shots = int(sum(counts.values()))
    if shots <= 0:
        raise ValueError("counts must contain at least one shot")

    total = 0.0
    for key, count in counts.items():
        bits = _bits_from_count_key(key, n_qubits)
        total += int(count) * pauli_eigenvalue_from_bits(labels, bits)

    mean = total / shots
    variance = max(1.0 - mean * mean, 0.0) / shots
    return float(mean), float(variance)


class PauliEstimator:
    """Shot-based Hamiltonian estimator using Pauli-term measurements."""

    def __init__(
        self,
        backend=None,
        *,
        shots: int = 1024,
        grouping: str = "qwc",
        shot_allocation: str | Sequence[int] | Mapping[int, int] = "uniform",
        noise_model=None,
        use_density_matrix: bool = False,
    ) -> None:
        self.backend = backend if backend is not None else NumpyBackend()
        self.shots = int(shots)
        if self.shots < 0:
            raise ValueError("shots must be non-negative")
        self.grouping = str(grouping)
        self.shot_allocation = shot_allocation
        self.noise_model = noise_model
        self.use_density_matrix = bool(use_density_matrix)

    def estimate(
        self,
        circuit: Circuit,
        hamiltonian: Hamiltonian | Iterable[PauliTerm | tuple[complex, str]],
        *,
        shots: int | None = None,
        grouping: str | None = None,
        shot_allocation: str | Sequence[int] | Mapping[int, int] | None = None,
        initial_state=None,
        initial_density_matrix=None,
        noise_model=None,
        use_density_matrix: bool | None = None,
    ) -> PauliEstimateResult:
        """Estimate ``<circuit|hamiltonian|circuit>`` from finite-shot counts."""

        if not isinstance(circuit, Circuit):
            raise TypeError("circuit must be a Circuit")

        terms = hamiltonian_pauli_terms(hamiltonian)
        if not terms:
            raise ValueError("hamiltonian must contain at least one Pauli term")
        n_qubits = int(circuit.n_qubits)
        if any(len(term.pauli) != n_qubits for term in terms):
            raise ValueError("Hamiltonian Pauli strings must match circuit.n_qubits")

        groups = group_pauli_terms(terms, strategy=self.grouping if grouping is None else grouping)
        total_shots = self.shots if shots is None else int(shots)
        allocations = allocate_group_shots(
            groups,
            total_shots,
            strategy=self.shot_allocation if shot_allocation is None else shot_allocation,
        )

        backend = circuit.backend if circuit.backend is not None else self.backend
        measure = Measure(backend)
        density_mode = self.use_density_matrix if use_density_matrix is None else bool(use_density_matrix)
        active_noise = self.noise_model if noise_model is None else noise_model

        all_term_results: list[PauliTermEstimate] = []
        group_results: list[PauliGroupEstimate] = []
        total_energy = 0.0
        total_variance = 0.0

        for group, group_shots in zip(groups, allocations):
            if group.is_identity:
                term_estimates = tuple(
                    PauliTermEstimate(
                        pauli=term.pauli,
                        coefficient=term.coefficient,
                        expectation=1.0,
                        variance=0.0,
                        energy=term.coefficient,
                        energy_variance=0.0,
                        shots=0,
                    )
                    for term in group.terms
                )
                energy = sum(term.energy for term in term_estimates)
                group_result = PauliGroupEstimate(group.basis, 0, {}, float(energy), 0.0, term_estimates)
                group_results.append(group_result)
                all_term_results.extend(term_estimates)
                total_energy += float(energy)
                continue

            if group_shots <= 0:
                raise ValueError("non-identity Pauli groups require positive shots")

            measured_circuit = measurement_circuit(circuit, group.basis, backend=backend)
            # 统一走 run()：run_density_matrix 已移除。
            # 若有噪声模型，动态附加到临时电路（Measure.run 通过 getattr 读取）。
            if active_noise is not None:
                measured_circuit.noise_model = active_noise

            # 初始态：density_mode 下使用 initial_density_matrix，否则 initial_state
            init = initial_density_matrix if (density_mode and initial_density_matrix is not None) else initial_state
            result = measure.run(
                measured_circuit,
                shots=group_shots,
                initial_state=init,
                return_state=False,
            )

            # counts(-1) 返回裸比特串（如 "00"），包装为 |..> 格式供测试断言及
            # pauli_expectation_from_counts（_bits_from_count_key 两者均接受）
            raw_counts = result.counts(-1)
            counts = {f"|{k}>": v for k, v in raw_counts.items()}
            term_estimates = []
            for term in group.terms:
                expectation, variance = pauli_expectation_from_counts(term.pauli, counts, n_qubits=n_qubits)
                energy = term.coefficient * expectation
                energy_variance = term.coefficient * term.coefficient * variance
                term_estimates.append(
                    PauliTermEstimate(
                        pauli=term.pauli,
                        coefficient=term.coefficient,
                        expectation=expectation,
                        variance=variance,
                        energy=float(energy),
                        energy_variance=float(energy_variance),
                        shots=group_shots,
                    )
                )

            group_energy, group_variance = self._group_energy_stats(group, counts, n_qubits=n_qubits)
            group_result = PauliGroupEstimate(
                basis=group.basis,
                shots=group_shots,
                counts=counts,
                energy=group_energy,
                variance=group_variance,
                terms=tuple(term_estimates),
            )
            group_results.append(group_result)
            all_term_results.extend(term_estimates)
            total_energy += group_energy
            total_variance += group_variance

        return PauliEstimateResult(
            energy=float(total_energy),
            variance=float(max(total_variance, 0.0)),
            shots=int(sum(allocations)),
            groups=tuple(group_results),
            term_results=tuple(all_term_results),
            metadata={
                "estimator": type(self).__name__,
                "grouping": self.grouping if grouping is None else grouping,
                "shot_allocation": self.shot_allocation if shot_allocation is None else shot_allocation,
                "n_terms": len(terms),
                "n_groups": len(groups),
                "state_mode": "density_matrix" if density_mode or active_noise is not None else "state_vector",
            },
        )

    @staticmethod
    def _group_energy_stats(group: PauliGroup, counts: Mapping[str, int], *, n_qubits: int) -> tuple[float, float]:
        shots = int(sum(counts.values()))
        if shots <= 0:
            raise ValueError("counts must contain at least one shot")

        mean = 0.0
        second = 0.0
        for key, count in counts.items():
            bits = _bits_from_count_key(key, n_qubits)
            sample_energy = 0.0
            for term in group.terms:
                sample_energy += term.coefficient * pauli_eigenvalue_from_bits(term.pauli, bits)
            weight = int(count) / shots
            mean += weight * sample_energy
            second += weight * sample_energy * sample_energy

        variance = max(second - mean * mean, 0.0) / shots
        return float(mean), float(variance)


__all__ = [
    "PauliTerm",
    "PauliGroup",
    "PauliTermEstimate",
    "PauliGroupEstimate",
    "PauliEstimateResult",
    "PauliEstimator",
    "allocate_group_shots",
    "basis_change_gates",
    "group_pauli_terms",
    "hamiltonian_pauli_terms",
    "measurement_circuit",
    "pauli_eigenvalue_from_bits",
    "pauli_expectation_from_counts",
]
