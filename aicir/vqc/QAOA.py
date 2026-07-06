"""Basic QAOA implementation.

QAOA (Quantum Approximate Optimization Algorithm) is a hybrid variational method
for combinatorial optimization with alternating problem/mixer Hamiltonian layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..backends.numpy_backend import NumpyBackend
from ..core.circuit import Circuit, cnot, hadamard, pauli_x, pauli_y, pauli_z, rx, ry, rz, rzz
from ..core.operators import Hamiltonian
from ..measure import Measure


def _infer_n_qubits(dim: int) -> int:
    if dim <= 0:
        raise ValueError("Hamiltonian dimension must be positive")
    n_qubits = int(round(np.log2(dim)))
    if (1 << n_qubits) != dim:
        raise ValueError("Hamiltonian dimension must be a power of 2")
    return n_qubits


def _normalize_statevector(state: np.ndarray, dim: int) -> np.ndarray:
    vec = np.asarray(state, dtype=np.complex128).reshape(-1)
    if vec.size != dim:
        raise ValueError(f"Statevector size {vec.size} does not match expected dimension {dim}")
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        raise ValueError("Statevector norm must be positive")
    return vec / norm


def _pauli_x() -> np.ndarray:
    return np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex128)


def _identity_2() -> np.ndarray:
    return np.eye(2, dtype=np.complex128)


def _build_mixer_hamiltonian(n_qubits: int) -> np.ndarray:
    dim = 1 << n_qubits
    mixer = np.zeros((dim, dim), dtype=np.complex128)
    x = _pauli_x()
    identity = _identity_2()
    for target in range(n_qubits):
        term = np.array([[1.0 + 0.0j]], dtype=np.complex128)
        for qubit in range(n_qubits):
            term = np.kron(term, x if qubit == target else identity)
        mixer += term
    return mixer


def _plus_state(n_qubits: int) -> np.ndarray:
    dim = 1 << n_qubits
    return np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)


def _exp_hermitian(generator: np.ndarray, angle: float) -> np.ndarray:
    # exp(-i * angle * generator) for Hermitian generator.
    eigvals, eigvecs = np.linalg.eigh(generator)
    phases = np.exp(-1j * float(angle) * eigvals)
    return eigvecs @ np.diag(phases) @ eigvecs.conj().T


@dataclass(frozen=True)
class _PauliCostTerm:
    paulis: tuple[str, ...]
    qubits: tuple[int, ...]
    coefficient: float


def _real_coefficient(value: complex, *, label: str) -> float:
    coeff = complex(value)
    if abs(coeff.imag) > 1e-12:
        raise ValueError(f"{label} coefficient must be real for canonical QAOA")
    return float(coeff.real)


def _pauli_cost_terms(hamiltonian: Hamiltonian) -> tuple[tuple[_PauliCostTerm, ...], float, bool]:
    terms: list[_PauliCostTerm] = []
    offset = 0.0
    is_diagonal = True
    for term in hamiltonian.terms:
        labels = tuple(term.qubit_labels)
        active = tuple((index, label) for index, label in enumerate(labels) if label != "I")
        coeff = _real_coefficient(term.coefficient, label="problem_hamiltonian")
        if abs(coeff) <= 1e-12:
            continue
        if not active:
            offset += coeff
            continue
        qubits = tuple(index for index, _ in active)
        paulis = tuple(label for _, label in active)
        if any(label != "Z" for label in paulis):
            is_diagonal = False
        terms.append(_PauliCostTerm(paulis, qubits, coeff))
    return tuple(terms), offset, is_diagonal


def _append_basis_change(circuit: Circuit, pauli: str, qubit: int) -> None:
    if pauli == "X":
        circuit.append(hadamard(qubit))
    elif pauli == "Y":
        circuit.append(rz(-np.pi / 2.0, qubit))
        circuit.append(hadamard(qubit))


def _append_basis_uncompute(circuit: Circuit, pauli: str, qubit: int) -> None:
    if pauli == "X":
        circuit.append(hadamard(qubit))
    elif pauli == "Y":
        circuit.append(hadamard(qubit))
        circuit.append(rz(np.pi / 2.0, qubit))


def _append_pauli_evolution(circuit: Circuit, term: _PauliCostTerm, angle: float) -> None:
    """Append a gate sequence for exp(-i * angle * P)."""

    if abs(angle) <= 1e-15:
        return
    rotation = 2.0 * float(angle)
    if len(term.qubits) == 1:
        pauli = term.paulis[0]
        qubit = term.qubits[0]
        if pauli == "X":
            circuit.append(rx(rotation, qubit))
        elif pauli == "Y":
            circuit.append(ry(rotation, qubit))
        else:
            circuit.append(rz(rotation, qubit))
        return
    if len(term.qubits) == 2 and term.paulis == ("Z", "Z"):
        circuit.append(rzz(rotation, term.qubits[0], term.qubits[1]))
        return

    for pauli, qubit in zip(term.paulis, term.qubits):
        _append_basis_change(circuit, pauli, qubit)

    pivot = term.qubits[-1]
    for control in term.qubits[:-1]:
        circuit.append(cnot(pivot, [control]))
    circuit.append(rz(rotation, pivot))
    for control in reversed(term.qubits[:-1]):
        circuit.append(cnot(pivot, [control]))

    for pauli, qubit in reversed(tuple(zip(term.paulis, term.qubits))):
        _append_basis_uncompute(circuit, pauli, qubit)


def _append_trotter_slice(
    circuit: Circuit,
    terms: tuple[_PauliCostTerm, ...],
    gamma_step: float,
    trotter_order: int,
) -> None:
    if not terms:
        return

    if trotter_order == 1:
        for term in terms:
            _append_pauli_evolution(circuit, term, gamma_step * term.coefficient)
        return

    if len(terms) == 1:
        term = terms[0]
        _append_pauli_evolution(circuit, term, gamma_step * term.coefficient)
        return

    for term in terms[:-1]:
        _append_pauli_evolution(circuit, term, 0.5 * gamma_step * term.coefficient)
    last = terms[-1]
    _append_pauli_evolution(circuit, last, gamma_step * last.coefficient)
    for term in reversed(terms[:-1]):
        _append_pauli_evolution(circuit, term, 0.5 * gamma_step * term.coefficient)


@dataclass(frozen=True)
class _GateRecord:
    """QAOA 线路磁带的单门记录。

    name:   门类型（"h"/"rx"/"ry"/"rz"/"rzz"/"cx"）。
    qubits: 作用比特（cx 为 (pivot, control)，rzz 为 (q0, q1)）。
    arg:    旋转门参数；h/cx 为 None。
    owner:  变分参数的扁平索引（γ 层为 0..p-1，β 层为 p..2p-1）；固定门为 None。
    dcoeff: d(arg)/d(theta_owner)；固定门为 None。
    """

    name: str
    qubits: tuple[int, ...]
    arg: float | None
    owner: int | None
    dcoeff: float | None


def _basis_change_records(records: list[_GateRecord], pauli: str, qubit: int) -> None:
    if pauli == "X":
        records.append(_GateRecord("h", (qubit,), None, None, None))
    elif pauli == "Y":
        records.append(_GateRecord("rz", (qubit,), -np.pi / 2.0, None, None))
        records.append(_GateRecord("h", (qubit,), None, None, None))


def _basis_uncompute_records(records: list[_GateRecord], pauli: str, qubit: int) -> None:
    if pauli == "X":
        records.append(_GateRecord("h", (qubit,), None, None, None))
    elif pauli == "Y":
        records.append(_GateRecord("h", (qubit,), None, None, None))
        records.append(_GateRecord("rz", (qubit,), np.pi / 2.0, None, None))


def _pauli_evolution_records(
    records: list[_GateRecord], term: _PauliCostTerm, rotation: float, drotation: float, owner: int
) -> None:
    """记录 exp(-i angle P) 的门序列，rotation = 2*angle，drotation = d(rotation)/d(theta_owner)。"""

    if len(term.qubits) == 1:
        pauli = term.paulis[0]
        qubit = term.qubits[0]
        name = {"X": "rx", "Y": "ry"}.get(pauli, "rz")
        records.append(_GateRecord(name, (qubit,), float(rotation), owner, float(drotation)))
        return
    if len(term.qubits) == 2 and term.paulis == ("Z", "Z"):
        records.append(
            _GateRecord("rzz", (term.qubits[0], term.qubits[1]), float(rotation), owner, float(drotation))
        )
        return

    for pauli, qubit in zip(term.paulis, term.qubits):
        _basis_change_records(records, pauli, qubit)

    pivot = term.qubits[-1]
    for control in term.qubits[:-1]:
        records.append(_GateRecord("cx", (pivot, control), None, None, None))
    records.append(_GateRecord("rz", (pivot,), float(rotation), owner, float(drotation)))
    for control in reversed(term.qubits[:-1]):
        records.append(_GateRecord("cx", (pivot, control), None, None, None))

    for pauli, qubit in reversed(tuple(zip(term.paulis, term.qubits))):
        _basis_uncompute_records(records, pauli, qubit)


def _trotter_slice_records(
    records: list[_GateRecord],
    terms: tuple[_PauliCostTerm, ...],
    gamma: float,
    owner: int,
    trotter_steps: int,
    trotter_order: int,
) -> None:
    if not terms:
        return
    gamma_step = gamma / trotter_steps

    def emit(term: _PauliCostTerm, mult: float) -> None:
        rotation = 2.0 * mult * gamma_step * term.coefficient
        drotation = 2.0 * mult * term.coefficient / trotter_steps
        _pauli_evolution_records(records, term, rotation, drotation, owner)

    if trotter_order == 1 or len(terms) == 1:
        for term in terms:
            emit(term, 1.0)
        return

    for term in terms[:-1]:
        emit(term, 0.5)
    emit(terms[-1], 1.0)
    for term in reversed(terms[:-1]):
        emit(term, 0.5)


def _circuit_from_tape(records: list[_GateRecord], n_qubits: int, backend: Any = None) -> Circuit:
    circuit = Circuit(n_qubits=n_qubits, backend=backend)
    for rec in records:
        if rec.name == "h":
            circuit.append(hadamard(rec.qubits[0]))
        elif rec.name == "cx":
            circuit.append(cnot(rec.qubits[0], [rec.qubits[1]]))
        elif rec.name == "rx":
            circuit.append(rx(rec.arg, rec.qubits[0]))
        elif rec.name == "ry":
            circuit.append(ry(rec.arg, rec.qubits[0]))
        elif rec.name == "rz":
            circuit.append(rz(rec.arg, rec.qubits[0]))
        elif rec.name == "rzz":
            circuit.append(rzz(rec.arg, rec.qubits[0], rec.qubits[1]))
        else:
            raise ValueError(f"未知磁带门类型 {rec.name!r}")
    return circuit


@dataclass
class QAOAResult:
    """Container for a QAOA run."""

    energy: float
    gammas: np.ndarray
    betas: np.ndarray
    statevector: np.ndarray | None
    energy_history: list[float]
    parameters: np.ndarray | None = None
    counts: dict[str, int] | None = None
    optimizer_result: Any = None
    measurement_result: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BasicQAOA:
    """Canonical gate-level QAOA solver with a dense-matrix compatibility path.

    Layered ansatz:
        |psi(theta)> = prod_{l=1..p} exp(-i beta_l H_M) exp(-i gamma_l H_C) |+>^n

    The canonical path accepts an :class:`aicir.core.operators.Hamiltonian`
    with real Pauli terms and builds an executable Circuit with H initial
    state preparation, first- or second-order Trotterized cost evolution,
    and RX mixer rotations. Dense matrix inputs are still accepted as a legacy
    exact-simulator fallback.
    """

    def __init__(
        self,
        problem_hamiltonian: np.ndarray | None = None,
        p: int = 1,
        n_qubits: int | None = None,
        mixer_hamiltonian: np.ndarray | None = None,
        seed: int | None = None,
        trotter_steps: int = 1,
        trotter_order: int = 1,
        *,
        cost: Any = None,
    ) -> None:
        self.cost = cost
        self._gate_level = False
        self._cost_terms: tuple[_PauliCostTerm, ...] = ()
        self._cost_offset = 0.0
        if isinstance(trotter_steps, bool) or not isinstance(trotter_steps, (int, np.integer)):
            raise ValueError("trotter_steps must be a positive integer")
        self.trotter_steps = int(trotter_steps)
        if self.trotter_steps <= 0:
            raise ValueError("trotter_steps must be a positive integer")
        if isinstance(trotter_order, bool) or not isinstance(trotter_order, (int, np.integer)):
            raise ValueError("trotter_order must be 1 or 2")
        self.trotter_order = int(trotter_order)
        if self.trotter_order not in {1, 2}:
            raise ValueError("trotter_order must be 1 or 2")
        self._diagonal_cost = True
        if cost is not None:
            if getattr(cost, "_multi", False):
                raise ValueError("BasicQAOA 的 cost 必须是单观测量 qfun（多观测量无标量能量）")
            if not callable(cost) or not hasattr(cost, "grad"):
                raise TypeError("cost 必须是可调用且带 .grad 的对象（如 qfun）")
            self.p = int(p)
            if self.p <= 0:
                raise ValueError("p must be a positive integer")
            self.n_qubits = None
            self.dim = None
            self.problem_hamiltonian = None
            self.mixer_hamiltonian = None
            self._rng = np.random.default_rng(seed)
            return

        if problem_hamiltonian is None:
            raise ValueError("BasicQAOA 需要 problem_hamiltonian 或 cost")

        self.p = int(p)
        if self.p <= 0:
            raise ValueError("p must be a positive integer")

        if isinstance(problem_hamiltonian, Hamiltonian):
            if mixer_hamiltonian is not None:
                raise ValueError(
                    "Hamiltonian-based BasicQAOA uses the standard X mixer; "
                    "dense custom mixers require matrix input"
                )
            self.n_qubits = problem_hamiltonian.n_qubits if n_qubits is None else int(n_qubits)
            if self.n_qubits != problem_hamiltonian.n_qubits:
                raise ValueError(
                    f"n_qubits={self.n_qubits} does not match Hamiltonian width {problem_hamiltonian.n_qubits}"
                )
            self.dim = 1 << self.n_qubits
            self.problem_hamiltonian = problem_hamiltonian
            self.mixer_hamiltonian = None
            self._cost_terms, self._cost_offset, self._diagonal_cost = _pauli_cost_terms(problem_hamiltonian)
            self._gate_level = True
            self._rng = np.random.default_rng(seed)
            return

        ham_c = np.asarray(problem_hamiltonian, dtype=np.complex128)
        if ham_c.ndim != 2 or ham_c.shape[0] != ham_c.shape[1]:
            raise ValueError("problem_hamiltonian must be a square matrix")

        inferred = _infer_n_qubits(ham_c.shape[0])
        self.n_qubits = inferred if n_qubits is None else int(n_qubits)
        if self.n_qubits != inferred:
            raise ValueError(
                f"n_qubits={self.n_qubits} does not match Hamiltonian dimension {ham_c.shape[0]}"
            )

        self.dim = 1 << self.n_qubits
        self.problem_hamiltonian = ham_c

        if mixer_hamiltonian is None:
            self.mixer_hamiltonian = _build_mixer_hamiltonian(self.n_qubits)
        else:
            ham_m = np.asarray(mixer_hamiltonian, dtype=np.complex128)
            if ham_m.shape != (self.dim, self.dim):
                raise ValueError(
                    f"mixer_hamiltonian shape {ham_m.shape} does not match expected {(self.dim, self.dim)}"
                )
            self.mixer_hamiltonian = ham_m

        self._rng = np.random.default_rng(seed)

    @property
    def n_params(self) -> int:
        return 2 * self.p

    def initial_params(self, gamma_scale: float = np.pi, beta_scale: float = np.pi / 2.0) -> np.ndarray:
        gammas = self._rng.uniform(-gamma_scale, gamma_scale, size=self.p)
        betas = self._rng.uniform(-beta_scale, beta_scale, size=self.p)
        return np.concatenate([gammas, betas])

    def split_params(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flat = np.asarray(params, dtype=float).reshape(-1)
        if flat.size != self.n_params:
            raise ValueError(f"params size {flat.size} does not match expected {self.n_params}")
        return flat[: self.p].copy(), flat[self.p :].copy()

    def _qaoa_tape(self, params: np.ndarray) -> list[_GateRecord]:
        """构造 QAOA 线路的门磁带（单一事实来源，build_circuit 与解析梯度共用）。"""

        gammas, betas = self.split_params(params)
        records: list[_GateRecord] = []
        for qubit in range(self.n_qubits):
            records.append(_GateRecord("h", (qubit,), None, None, None))

        for layer in range(self.p):
            gamma = float(gammas[layer])
            for _ in range(self.trotter_steps):
                _trotter_slice_records(
                    records, self._cost_terms, gamma, layer, self.trotter_steps, self.trotter_order
                )
            beta = float(betas[layer])
            for qubit in range(self.n_qubits):
                records.append(_GateRecord("rx", (qubit,), 2.0 * beta, self.p + layer, 2.0))
        return records

    def build_circuit(self, params: np.ndarray, *, backend: Any = None) -> Circuit:
        """Build the canonical gate-level QAOA circuit for ``params``."""

        if self.cost is not None:
            raise ValueError("build_circuit is unavailable when BasicQAOA delegates to an external cost")
        if not self._gate_level:
            raise ValueError("build_circuit requires an aicir Hamiltonian problem_hamiltonian")
        return _circuit_from_tape(self._qaoa_tape(params), self.n_qubits, backend)

    def measure(
        self,
        params: np.ndarray,
        *,
        shots: int | None = None,
        backend: Any = None,
        seed: int | None = None,
        method: str = "statevector",
        return_state: bool = True,
    ):
        """Execute the gate-level QAOA circuit through the measurement runner."""

        active_backend = NumpyBackend() if backend is None else backend
        circuit = self.build_circuit(params, backend=active_backend)
        return Measure(active_backend).run(
            circuit,
            shots=shots,
            measure_qubits=(),
            seed=seed,
            return_state=return_state,
            method=method,
        )

    def probabilities(self, params: np.ndarray, *, backend: Any = None, method: str = "statevector") -> np.ndarray:
        """Return exact computational-basis probabilities from the gate-level circuit."""

        result = self.measure(params, shots=None, backend=backend, method=method, return_state=False)
        return np.asarray(result.probabilities, dtype=float)

    def sample(
        self,
        params: np.ndarray,
        *,
        shots: int = 1024,
        backend: Any = None,
        seed: int | None = None,
        method: str = "statevector",
    ) -> dict[str, int]:
        """Sample the gate-level QAOA circuit in the computational basis."""

        if int(shots) <= 0:
            raise ValueError("shots must be positive")
        result = self.measure(
            params,
            shots=int(shots),
            backend=backend,
            seed=seed,
            method=method,
            return_state=False,
        )
        return result.counts(-1)

    def bitstring_energy(self, bitstring: str) -> float:
        """Evaluate the diagonal cost Hamiltonian on a computational-basis bitstring."""

        if not self._gate_level:
            raise ValueError("bitstring_energy requires an aicir Hamiltonian problem_hamiltonian")
        if not self._diagonal_cost:
            raise ValueError("bitstring_energy is only available for diagonal I/Z-only Hamiltonians")
        bits = str(bitstring).strip()
        if bits.startswith("|") and bits.endswith(">"):
            bits = bits[1:-1]
        if len(bits) != self.n_qubits or any(bit not in {"0", "1"} for bit in bits):
            raise ValueError(f"bitstring must be a {self.n_qubits}-bit computational-basis string")

        value = self._cost_offset
        for term in self._cost_terms:
            eigenvalue = 1
            for qubit in term.qubits:
                eigenvalue *= 1 if bits[int(qubit)] == "0" else -1
            value += term.coefficient * eigenvalue
        return float(value)

    def _sparse_cost_expectation(self, state, backend) -> float:
        """稀疏计算 ⟨ψ|H_C|ψ⟩ = Σ_j c_j ⟨ψ|P_j|ψ⟩，逐 Pauli 串作用到态矢量，避免构造 2^n×2^n 稠密矩阵。

        参数:
            state:   后端原生态矢量张量（如 ``State.data``）。
            backend: 计算后端。
        返回:
            实数期望值（含常数偏移 ``self._cost_offset``）。
        """
        from ..core.gates import apply_gate_to_state

        factory = {"X": pauli_x, "Y": pauli_y, "Z": pauli_z}
        psi = np.asarray(backend.to_numpy(state), dtype=np.complex128).reshape(-1)
        value = float(self._cost_offset)
        for term in self._cost_terms:
            transformed = state
            for pauli, qubit in zip(term.paulis, term.qubits):
                transformed = apply_gate_to_state(
                    factory[pauli](int(qubit)), transformed, self.n_qubits, backend
                )
            t_np = np.asarray(backend.to_numpy(transformed), dtype=np.complex128).reshape(-1)
            value += term.coefficient * float(np.vdot(psi, t_np).real)
        return float(value)

    def ansatz_state(self, params: np.ndarray, init_state: np.ndarray | None = None) -> np.ndarray:
        if self._gate_level:
            if init_state is not None:
                raise ValueError("gate-level BasicQAOA uses the canonical |+> initial state")
            result = self.measure(params, shots=None, return_state=True)
            return np.asarray(result.final_state.array, dtype=np.complex128)

        gammas, betas = self.split_params(params)
        state = _plus_state(self.n_qubits) if init_state is None else _normalize_statevector(init_state, self.dim)

        for layer in range(self.p):
            u_cost = _exp_hermitian(self.problem_hamiltonian, gammas[layer])
            u_mixer = _exp_hermitian(self.mixer_hamiltonian, betas[layer])
            state = u_cost @ state
            state = u_mixer @ state

        return state

    def energy(
        self,
        params: np.ndarray,
        *,
        shots: int | None = None,
        backend: Any = None,
        seed: int | None = None,
        method: str = "statevector",
    ) -> float:
        if self.cost is not None:
            return float(self.cost(np.asarray(params, dtype=float).reshape(-1)))
        if self._gate_level:
            if shots is not None:
                if not self._diagonal_cost:
                    raise ValueError(
                        "shots-based energy for non-diagonal Hamiltonians requires a Pauli estimator; "
                        "use shots=None for exact energy"
                    )
                counts = self.sample(params, shots=int(shots), backend=backend, seed=seed, method=method)
                total = sum(counts.values()) or 1
                return float(sum(self.bitstring_energy(bitstring) * count for bitstring, count in counts.items()) / total)
            active_backend = NumpyBackend() if backend is None else backend
            result = self.measure(params, backend=active_backend, method=method, return_state=True)
            # 精确能量（shots=None）：稀疏逐项期望，避免稠密矩阵与 2^n Python 循环
            return self._sparse_cost_expectation(result.final_state.data, active_backend)
        if shots is not None:
            raise ValueError("shots-based QAOA energy requires an aicir Hamiltonian problem_hamiltonian")
        state = self.ansatz_state(params)
        value = np.vdot(state, self.problem_hamiltonian @ state)
        return float(np.real(value))

    def _gradient(
        self, params: np.ndarray, *, grad_method: str = "fd", **energy_kwargs: Any
    ) -> np.ndarray:
        if grad_method not in ("fd", "analytic", "psr"):
            raise ValueError(f"grad_method 必须是 fd/analytic/psr，收到 {grad_method!r}")
        if self.cost is not None:
            flat = np.asarray(params, dtype=float).reshape(-1)
            return np.asarray(self.cost.grad(flat), dtype=float).reshape(flat.shape)
        if grad_method in ("analytic", "psr"):
            if energy_kwargs.get("shots") is not None:
                raise ValueError("analytic/psr gradient requires shots=None for exact energy")
            return self.analytic_gradient(
                params,
                backend=energy_kwargs.get("backend"),
                method=energy_kwargs.get("method", "statevector"),
            )
        return self.finite_difference_gradient(params, **energy_kwargs)

    def finite_difference_gradient(self, params: np.ndarray, eps: float = 1e-4, **energy_kwargs: Any) -> np.ndarray:
        flat = np.asarray(params, dtype=float).reshape(-1)
        grad = np.zeros_like(flat)
        for index in range(flat.size):
            plus = flat.copy()
            minus = flat.copy()
            plus[index] += eps
            minus[index] -= eps
            grad[index] = (
                self.energy(plus, **energy_kwargs) - self.energy(minus, **energy_kwargs)
            ) / (2.0 * eps)
        return grad

    def _tape_energy(self, records, gate_index: int, delta: float, backend, method: str) -> float:
        """将 records[gate_index] 的参数平移 delta 后重建线路并返回精确能量。"""

        rec = records[gate_index]
        shifted = list(records)
        shifted[gate_index] = _GateRecord(rec.name, rec.qubits, rec.arg + delta, rec.owner, rec.dcoeff)
        circuit = _circuit_from_tape(shifted, self.n_qubits, backend)
        result = Measure(backend).run(
            circuit, shots=None, measure_qubits=(), return_state=True, method=method
        )
        return self._sparse_cost_expectation(result.final_state.data, backend)

    def analytic_gradient(
        self, params: np.ndarray, *, backend: Any = None, method: str = "statevector"
    ) -> np.ndarray:
        """精确能量的解析梯度：对磁带中每个变分门做逐门参数移位（生成元谱 +-1/2，移位 pi/2），
        再按链式法则 dE/dtheta_k = sum_g (d arg_g/dtheta_k)*1/2[E(arg_g+pi/2)-E(arg_g-pi/2)] 聚合。

        对被评估的（Trotter 化）线路是解析精确的。仅支持门级 Hamiltonian 路径。
        """

        if not self._gate_level:
            raise ValueError("analytic_gradient requires an aicir Hamiltonian problem_hamiltonian")

        active_backend = NumpyBackend() if backend is None else backend
        theta = np.asarray(params, dtype=float).reshape(-1)
        if theta.size != self.n_params:
            raise ValueError(f"params size {theta.size} does not match expected {self.n_params}")

        records = self._qaoa_tape(theta)
        grad = np.zeros(self.n_params, dtype=float)
        shift = np.pi / 2.0
        for gate_index, rec in enumerate(records):
            if rec.owner is None:
                continue
            e_plus = self._tape_energy(records, gate_index, shift, active_backend, method)
            e_minus = self._tape_energy(records, gate_index, -shift, active_backend, method)
            grad[rec.owner] += rec.dcoeff * 0.5 * (e_plus - e_minus)
        return grad

    def run(
        self,
        max_iters: int = 200,
        lr: float = 0.05,
        init_params: np.ndarray | None = None,
        callback: Callable[[int, float, np.ndarray], None] | None = None,
        *,
        optimizer: Any = None,
        shots: int | None = None,
        backend: Any = None,
        seed: int | None = None,
        method: str = "statevector",
        grad_method: str = "fd",
    ) -> QAOAResult:
        if max_iters <= 0:
            raise ValueError("max_iters must be a positive integer")
        if optimizer is None and lr <= 0:
            raise ValueError("lr must be positive")

        params = self.initial_params() if init_params is None else np.asarray(init_params, dtype=float)
        params = params.reshape(-1)
        if params.size != self.n_params:
            raise ValueError(f"init_params size {params.size} does not match expected {self.n_params}")

        history: list[float] = []
        best_energy = np.inf
        best_params = params.copy()
        energy_kwargs = {"shots": shots, "backend": backend, "seed": seed, "method": method}

        if optimizer is not None:
            if not hasattr(optimizer, "minimize"):
                raise TypeError("optimizer must expose a minimize(fn, init_params, ...) method")

            def objective(theta):
                return self.energy(theta, **energy_kwargs)

            def opt_callback(step, value, theta):
                history.append(float(value))
                if callback is not None:
                    callback(step, value, theta)

            opt_result = optimizer.minimize(objective, params, callback=opt_callback)
            best_params = np.asarray(getattr(opt_result, "best_x", opt_result.x), dtype=float).reshape(-1)
            best_energy = float(getattr(opt_result, "best_fun", opt_result.fun))
            if not history:
                history.append(best_energy)
            gammas, betas = self.split_params(best_params)
            counts = None
            measurement_result = None
            final_state = None
            if self._gate_level:
                if shots is None:
                    measurement_result = self.measure(best_params, backend=backend, method=method, return_state=True)
                    final_state = np.asarray(measurement_result.final_state.array, dtype=np.complex128)
                else:
                    counts = self.sample(best_params, shots=int(shots), backend=backend, seed=seed, method=method)
            elif self.cost is None:
                final_state = self.ansatz_state(best_params)
            return QAOAResult(
                energy=best_energy,
                gammas=gammas,
                betas=betas,
                statevector=final_state,
                energy_history=history,
                parameters=best_params,
                counts=counts,
                optimizer_result=opt_result,
                measurement_result=measurement_result,
            )

        for step in range(max_iters):
            current_energy = self.energy(params, **energy_kwargs)
            history.append(current_energy)
            if current_energy < best_energy:
                best_energy = current_energy
                best_params = params.copy()

            if callback is not None:
                callback(step, current_energy, params)

            grad = self._gradient(params, grad_method=grad_method, **energy_kwargs)
            params = params - lr * grad

        final_state = None
        counts = None
        if self.cost is None:
            if self._gate_level and shots is not None:
                counts = self.sample(best_params, shots=int(shots), backend=backend, seed=seed, method=method)
            else:
                final_state = self.ansatz_state(best_params)
        gammas, betas = self.split_params(best_params)
        return QAOAResult(
            energy=float(best_energy),
            gammas=gammas,
            betas=betas,
            statevector=final_state,
            energy_history=history,
            parameters=best_params,
            counts=counts,
        )


def run_qaoa(
    problem_hamiltonian: Any,
    p: int,
    n_qubits: int | None = None,
    max_iters: int = 200,
    lr: float = 0.05,
    mixer_hamiltonian: np.ndarray | None = None,
    seed: int | None = None,
) -> QAOAResult:
    """Convenience function to run the basic QAOA solver."""
    solver = BasicQAOA(
        problem_hamiltonian=problem_hamiltonian,
        p=p,
        n_qubits=n_qubits,
        mixer_hamiltonian=mixer_hamiltonian,
        seed=seed,
    )
    return solver.run(max_iters=max_iters, lr=lr)
