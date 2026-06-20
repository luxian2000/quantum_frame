"""Variational Quantum Eigensolver implementation.

``BasicVQE`` keeps the original minimal dense_matrix RY/CNOT path for backward
compatibility, and also supports a general orchestration path built around
``Circuit`` templates, ``Hamiltonian`` objects, ``Measure`` backends, optional
shot sampling/noise, and pluggable classical optimizers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..backends.numpy_backend import NumpyBackend
from ..core.operators import Hamiltonian
from ..core.circuit import Circuit
from ..ir import circuit_gate_dicts
from ..measure import Measure
from ..qml.deriv import psr


def _infer_n_qubits(dim: int) -> int:
    if dim <= 0:
        raise ValueError("Hamiltonian dimension must be positive")
    n_qubits = int(round(np.log2(dim)))
    if (1 << n_qubits) != dim:
        raise ValueError("Hamiltonian dimension must be a power of 2")
    return n_qubits


def _bit_mask(n_qubits: int, qubit: int) -> int:
    if qubit < 0 or qubit >= n_qubits:
        raise ValueError(f"Invalid qubit index {qubit} for n_qubits={n_qubits}")
    # Keep qubit-0 as MSB to match most of this repository's matrix conventions.
    return 1 << (n_qubits - qubit - 1)


def _apply_single_qubit_gate(state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    psi = np.asarray(state, dtype=np.complex128).reshape([2] * n_qubits)
    psi = np.moveaxis(psi, qubit, 0)
    psi = np.tensordot(gate, psi, axes=([1], [0]))
    psi = np.moveaxis(psi, 0, qubit)
    return psi.reshape(-1)


def _apply_cnot(state: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    if control == target:
        raise ValueError("control and target must be different")

    out = np.asarray(state, dtype=np.complex128).copy().reshape(-1)
    ctrl_mask = _bit_mask(n_qubits, control)
    tgt_mask = _bit_mask(n_qubits, target)

    # Swap amplitudes where control=1 and target flips 0 -> 1.
    for index in range(out.size):
        if (index & ctrl_mask) and not (index & tgt_mask):
            pair = index | tgt_mask
            out[index], out[pair] = out[pair], out[index]
    return out


def _ry(theta: float) -> np.ndarray:
    t = float(theta)
    c = np.cos(t / 2.0)
    s = np.sin(t / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


@dataclass
class VQEResult:
    """Container for a VQE run."""

    energy: float
    parameters: np.ndarray
    statevector: np.ndarray | None
    energy_history: list[float]
    circuit: Circuit | None = None
    measurement_result: Any = None
    estimator_result: Any = None
    optimizer_result: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_state(self) -> np.ndarray | None:
        """Alias for the final pure state or flattened density matrix."""

        return self.statevector


class BasicVQE:
    """VQE solver with legacy and circuit-orchestration execution modes.

    Without an explicit ``ansatz`` this class preserves the original minimal
    dense_matrix solver with an RY + nearest-neighbor CNOT ansatz:
    1) Apply RY(theta[layer, q]) on each qubit q.
    2) Apply a nearest-neighbor CNOT chain q -> q+1.

    With ``ansatz`` supplied, it becomes a general VQE orchestration layer:
    ``ansatz`` may be a parameterized ``Circuit`` or a callable
    ``ansatz(params) -> Circuit``; ``hamiltonian`` may be a dense_matrix or
    ``Hamiltonian`` object; measurement runs through ``Measure`` with the chosen
    backend, optional shots, density-matrix noise, and optimizer.
    """

    def __init__(
        self,
        hamiltonian: np.ndarray | Hamiltonian | None = None,
        n_qubits: int | None = None,
        depth: int = 1,
        seed: int | None = None,
        *,
        cost: Any = None,
        ansatz: Circuit | Callable[[np.ndarray], Circuit] | None = None,
        n_params: int | None = None,
        parameter_shape: tuple[int, ...] | None = None,
        backend: Any = None,
        optimizer: Any = None,
        shots: int | None = None,
        noise_model: Any = None,
        use_density_matrix: bool = False,
        initial_state: Any = None,
        initial_density_matrix: Any = None,
        observable_name: str = "energy",
        energy_estimator: str | Any = "exact",
    ) -> None:
        self.cost = cost
        if cost is not None:
            self._init_cost_mode(
                cost,
                seed=seed,
                n_params=n_params,
                parameter_shape=parameter_shape,
                backend=backend,
                optimizer=optimizer,
                shots=shots,
                observable_name=observable_name,
            )
            return
        if hamiltonian is None:
            raise ValueError("BasicVQE 需要 hamiltonian 或 cost")
        self.backend = backend if backend is not None else NumpyBackend()
        self.observable = hamiltonian
        ham, inferred = self._coerce_hamiltonian(hamiltonian, self.backend)
        self.n_qubits = self._resolve_n_qubits(n_qubits, inferred, ansatz)
        self.hamiltonian = ham

        self.depth = int(depth)
        if self.depth <= 0:
            raise ValueError("depth must be a positive integer")

        self.ansatz = ansatz
        self.optimizer = optimizer
        self.shots = shots
        if shots is not None and int(shots) < 0:
            raise ValueError("shots must be non-negative")
        self.noise_model = noise_model
        self.use_density_matrix = bool(use_density_matrix)
        self.initial_state = initial_state
        self.initial_density_matrix = initial_density_matrix
        self.observable_name = str(observable_name)
        self.energy_estimator = self._resolve_energy_estimator(energy_estimator)
        if ansatz is None and not self._uses_exact_energy_estimator():
            raise ValueError("non-exact energy_estimator requires a Circuit or callable ansatz")

        self._parameter_shape = self._resolve_parameter_shape(ansatz, n_params, parameter_shape)
        self._last_circuit: Circuit | None = None
        self._last_measurement: Any = None
        self._last_estimator_result: Any = None
        self._rng = np.random.default_rng(seed)

    def _init_cost_mode(
        self,
        cost: Any,
        *,
        seed: int | None,
        n_params: int | None,
        parameter_shape: tuple[int, ...] | None,
        backend: Any,
        optimizer: Any,
        shots: int | None,
        observable_name: str,
    ) -> None:
        """cost 模式：直接用外部 qfun 作代价函数，旁路 ansatz/hamiltonian 编排。"""

        if getattr(cost, "_multi", False):
            raise ValueError("BasicVQE 的 cost 必须是单观测量 qfun（多观测量无标量能量）")
        if not callable(cost) or not hasattr(cost, "grad"):
            raise TypeError("cost 必须是可调用且带 .grad 的对象（如 qfun）")
        if n_params is None and parameter_shape is None:
            raise ValueError("cost 模式需要 n_params 或 parameter_shape")

        self.backend = backend if backend is not None else getattr(cost, "_backend", NumpyBackend())
        self.observable = getattr(cost, "observable", None)
        self.hamiltonian = None
        self.n_qubits = None
        self.depth = 1
        self.ansatz = None
        self.optimizer = optimizer
        self.shots = shots
        self.noise_model = None
        self.use_density_matrix = False
        self.initial_state = None
        self.initial_density_matrix = None
        self.observable_name = str(observable_name)
        self.energy_estimator = "exact"
        self._parameter_shape = (
            tuple(int(d) for d in parameter_shape) if parameter_shape is not None else (int(n_params),)
        )
        self._last_circuit = None
        self._last_measurement = None
        self._last_estimator_result = None
        self._rng = np.random.default_rng(seed)

    def _resolve_energy_estimator(self, energy_estimator: str | Any) -> str | Any:
        if energy_estimator is None:
            return "exact"
        if isinstance(energy_estimator, str):
            name = energy_estimator.strip().lower()
            if name != "exact":
                raise ValueError("energy_estimator string must be 'exact'")
            return "exact"
        if not hasattr(energy_estimator, "estimate"):
            raise TypeError("energy_estimator must be 'exact' or expose estimate(circuit, hamiltonian, ...)")
        return energy_estimator

    def _energy_estimator_name(self) -> str:
        if self._uses_exact_energy_estimator():
            return "exact"
        return type(self.energy_estimator).__name__

    def _uses_exact_energy_estimator(self) -> bool:
        return isinstance(self.energy_estimator, str) and self.energy_estimator == "exact"

    def _coerce_hamiltonian(self, hamiltonian: np.ndarray | Hamiltonian, backend) -> tuple[np.ndarray, int]:
        if hasattr(hamiltonian, "to_matrix"):
            raw = hamiltonian.to_matrix(backend)
            ham = np.asarray(backend.to_numpy(raw), dtype=np.complex128)
            inferred = int(getattr(hamiltonian, "n_qubits", _infer_n_qubits(ham.shape[0])))
        else:
            ham = np.asarray(hamiltonian, dtype=np.complex128)
            inferred = _infer_n_qubits(ham.shape[0]) if ham.ndim == 2 and ham.shape[0] == ham.shape[1] else -1

        if ham.ndim != 2 or ham.shape[0] != ham.shape[1]:
            raise ValueError("hamiltonian must be a square matrix or expose to_matrix(backend)")
        matrix_inferred = _infer_n_qubits(ham.shape[0])
        if inferred != matrix_inferred:
            raise ValueError(
                f"Hamiltonian n_qubits={inferred} does not match matrix dimension {ham.shape[0]}"
            )
        return ham, matrix_inferred

    def _resolve_n_qubits(
        self,
        n_qubits: int | None,
        inferred: int,
        ansatz: Circuit | Callable[[np.ndarray], Circuit] | None,
    ) -> int:
        value = inferred if n_qubits is None else int(n_qubits)
        if value != inferred:
            raise ValueError(f"n_qubits={value} does not match Hamiltonian dimension {1 << inferred}")
        if isinstance(ansatz, Circuit) and int(ansatz.n_qubits) != value:
            raise ValueError(f"ansatz.n_qubits={ansatz.n_qubits} does not match n_qubits={value}")
        return value

    def _resolve_parameter_shape(
        self,
        ansatz: Circuit | Callable[[np.ndarray], Circuit] | None,
        n_params: int | None,
        parameter_shape: tuple[int, ...] | None,
    ) -> tuple[int, ...] | None:
        if ansatz is None:
            return (self.depth, self.n_qubits)

        if isinstance(ansatz, Circuit):
            inferred_count = len(ansatz.parameters)
            if n_params is not None and int(n_params) != inferred_count:
                raise ValueError(f"n_params={n_params} does not match ansatz parameter count {inferred_count}")
            if parameter_shape is None:
                return (inferred_count,)
        elif parameter_shape is None and n_params is not None:
            return (int(n_params),)

        if parameter_shape is not None:
            shape = tuple(int(dim) for dim in parameter_shape)
            if any(dim < 0 for dim in shape):
                raise ValueError("parameter_shape dimensions must be non-negative")
            if n_params is not None and int(np.prod(shape, dtype=int)) != int(n_params):
                raise ValueError("parameter_shape size does not match n_params")
            if isinstance(ansatz, Circuit) and int(np.prod(shape, dtype=int)) != len(ansatz.parameters):
                raise ValueError("parameter_shape size does not match ansatz parameter count")
            return shape

        return None

    @property
    def n_params(self) -> int:
        if self._parameter_shape is None:
            raise ValueError("n_params is unknown for callable ansatz; pass n_params or parameter_shape")
        return int(np.prod(self._parameter_shape, dtype=int))

    def initial_params(self, scale: float = 0.1) -> np.ndarray:
        if self._parameter_shape is None:
            raise ValueError("Cannot initialize params for callable ansatz without n_params or parameter_shape")
        return self._rng.uniform(-scale, scale, size=self._parameter_shape)

    def _normalize_params(self, params: np.ndarray) -> np.ndarray:
        theta = np.asarray(params, dtype=float)
        if self._parameter_shape is None:
            return theta
        expected = int(np.prod(self._parameter_shape, dtype=int))
        if theta.size != expected:
            raise ValueError(f"Expected {expected} parameter value(s), got {theta.size}")
        return theta.reshape(self._parameter_shape)

    def ansatz_state(self, params: np.ndarray) -> np.ndarray | None:
        if self.cost is not None:
            return None
        if self.ansatz is not None:
            _, measurement = self._measure_circuit_exact(params, return_state=True)
            # 取测量前的完整末态；shots>0 时 final_state 已是坍缩/约化后的态
            state = None if measurement is None else getattr(measurement, "state", None)
            if state is None and measurement is not None:
                state = measurement.final_state
            return None if state is None else np.asarray(state)

        theta = self._normalize_params(params)
        state = np.zeros(1 << self.n_qubits, dtype=np.complex128)
        state[0] = 1.0 + 0.0j

        for layer in range(self.depth):
            for qubit in range(self.n_qubits):
                state = _apply_single_qubit_gate(state, _ry(theta[layer, qubit]), qubit, self.n_qubits)

            for qubit in range(self.n_qubits - 1):
                state = _apply_cnot(state, control=qubit, target=qubit + 1, n_qubits=self.n_qubits)

        return state

    def _observable_matrix(self, backend) -> Any:
        if hasattr(self.observable, "to_matrix"):
            return self.observable.to_matrix(backend)
        return backend.cast(self.hamiltonian)

    def bind_ansatz(self, params: np.ndarray) -> Circuit:
        """Return a fully-bound ansatz circuit for ``params``."""

        if self.ansatz is None:
            raise ValueError("No Circuit ansatz was supplied")

        theta = self._normalize_params(params)
        if isinstance(self.ansatz, Circuit):
            circuit = self.ansatz.bind_parameters(theta.reshape(-1))
        elif callable(self.ansatz):
            circuit = self.ansatz(theta)
            if not isinstance(circuit, Circuit):
                raise TypeError("ansatz(params) must return a Circuit")
            if circuit.parameters:
                circuit = circuit.bind_parameters(theta.reshape(-1))
        else:
            raise TypeError("ansatz must be a Circuit, callable, or None")

        if int(circuit.n_qubits) != self.n_qubits:
            raise ValueError(f"ansatz circuit n_qubits={circuit.n_qubits} does not match {self.n_qubits}")
        if circuit.parameters:
            names = ", ".join(parameter.name for parameter in circuit.parameters)
            raise ValueError(f"ansatz circuit has unbound parameter(s): {names}")
        if circuit.backend is None and self.backend is not None:
            circuit = Circuit(*circuit_gate_dicts(circuit), n_qubits=circuit.n_qubits, backend=self.backend)
        return circuit

    def _measure_circuit_exact(self, params: np.ndarray, *, return_state: bool) -> tuple[float, Any]:
        circuit = self.bind_ansatz(params)
        backend = circuit.backend if circuit.backend is not None else self.backend
        measure = Measure(backend)
        observable = self._observable_matrix(backend)
        observables = {self.observable_name: observable}

        if self.noise_model is not None or self.use_density_matrix:
            # 噪声路径：将 noise_model 附加到线路，由 run() 内部读取
            if self.noise_model is not None:
                circuit.noise_model = self.noise_model
            measurement = measure.run(
                circuit,
                shots=self.shots,
                initial_density_matrix=self.initial_density_matrix,
                observables=observables,
                return_state=return_state,
            )
        else:
            measurement = measure.run(
                circuit,
                shots=self.shots,
                initial_state=self.initial_state,
                observables=observables,
                return_state=return_state,
            )

        self._last_circuit = circuit
        self._last_measurement = measurement
        return float(measurement.expectation_values[self.observable_name]), measurement

    def _evaluate_circuit(self, params: np.ndarray, *, return_state: bool) -> tuple[float, Any]:
        if self._uses_exact_energy_estimator():
            return self._measure_circuit_exact(params, return_state=return_state)

        circuit = self.bind_ansatz(params)
        estimate_kwargs: dict[str, Any] = {
            "initial_state": self.initial_state,
            "initial_density_matrix": self.initial_density_matrix,
            "noise_model": self.noise_model,
            "use_density_matrix": self.use_density_matrix,
        }
        if self.shots is not None:
            estimate_kwargs["shots"] = self.shots

        estimator_result = self.energy_estimator.estimate(circuit, self.observable, **estimate_kwargs)
        self._last_circuit = circuit
        self._last_estimator_result = estimator_result
        if return_state:
            _, measurement = self._measure_circuit_exact(params, return_state=True)
            return float(estimator_result.energy), measurement
        self._last_measurement = None
        return float(estimator_result.energy), estimator_result

    def energy(self, params: np.ndarray) -> float:
        if self.cost is not None:
            return float(self.cost(self._normalize_params(params)))
        if self.ansatz is not None:
            value, _ = self._evaluate_circuit(params, return_state=False)
            return value

        params = self._normalize_params(params)
        state = self.ansatz_state(params)
        value = np.vdot(state, self.hamiltonian @ state)
        return float(np.real(value))

    def parameter_shift_gradient(self, params: np.ndarray) -> np.ndarray:
        params = self._normalize_params(params)
        if self.cost is not None:
            return np.asarray(self.cost.grad(params), dtype=float).reshape(params.shape)
        return psr(self.energy, params)

    def run(
        self,
        max_iters: int = 200,
        lr: float = 0.1,
        init_params: np.ndarray | None = None,
        optimizer: Any = None,
        callback: Callable[[int, float, np.ndarray], None] | None = None,
    ) -> VQEResult:
        selected_optimizer = optimizer if optimizer is not None else self.optimizer
        if selected_optimizer is not None:
            return self._run_with_optimizer(selected_optimizer, init_params=init_params, callback=callback)

        params = self.initial_params() if init_params is None else np.asarray(init_params, dtype=float)
        params = self._normalize_params(params)
        if max_iters <= 0:
            raise ValueError("max_iters must be a positive integer")
        if lr <= 0:
            raise ValueError("lr must be positive")

        history: list[float] = []
        best_energy = np.inf
        best_params = params.copy()

        for step in range(max_iters):
            current_energy = self.energy(params)
            history.append(current_energy)
            if current_energy < best_energy:
                best_energy = current_energy
                best_params = params.copy()

            if callback is not None:
                callback(step, current_energy, params)

            grad = self.parameter_shift_gradient(params)
            params = params - lr * grad

        final_state = self.ansatz_state(best_params)
        final_circuit = self._last_circuit
        final_measurement = self._last_measurement
        return VQEResult(
            energy=float(best_energy),
            parameters=best_params,
            statevector=final_state,
            energy_history=history,
            circuit=final_circuit,
            measurement_result=final_measurement,
            estimator_result=self._last_estimator_result,
            metadata={
                "mode": "qfun" if self.cost is not None else ("circuit" if self.ansatz is not None else "legacy_dense"),
                "backend": getattr(self.backend, "name", None),
                "shots": self.shots,
                "noise_model": type(self.noise_model).__name__ if self.noise_model is not None else None,
                "energy_estimator": self._energy_estimator_name(),
            },
        )

    def _run_with_optimizer(
        self,
        optimizer: Any,
        *,
        init_params: np.ndarray | None,
        callback: Callable[[int, float, np.ndarray], None] | None,
    ) -> VQEResult:
        if not hasattr(optimizer, "minimize"):
            raise TypeError("optimizer must expose minimize(fn, init_params, ...)")

        params = self.initial_params() if init_params is None else self._normalize_params(init_params)
        opt_result = optimizer.minimize(self.energy, params, callback=callback)
        raw_best_params = getattr(opt_result, "best_x", None)
        if raw_best_params is None:
            raw_best_params = getattr(opt_result, "x")
        raw_best_energy = getattr(opt_result, "best_fun", None)
        if raw_best_energy is None:
            raw_best_energy = getattr(opt_result, "fun")
        best_params = np.asarray(raw_best_params, dtype=float).reshape(params.shape)
        best_energy = float(raw_best_energy)
        final_state = self.ansatz_state(best_params)
        history = [
            float(entry["fun"])
            for entry in getattr(opt_result, "history", [])
            if isinstance(entry, dict) and "fun" in entry
        ]
        return VQEResult(
            energy=best_energy,
            parameters=best_params,
            statevector=final_state,
            energy_history=history,
            circuit=self._last_circuit,
            measurement_result=self._last_measurement,
            estimator_result=self._last_estimator_result,
            optimizer_result=opt_result,
            metadata={
                "mode": "qfun" if self.cost is not None else ("circuit" if self.ansatz is not None else "legacy_dense"),
                "backend": getattr(self.backend, "name", None),
                "optimizer": type(optimizer).__name__,
                "shots": self.shots,
                "noise_model": type(self.noise_model).__name__ if self.noise_model is not None else None,
                "energy_estimator": self._energy_estimator_name(),
            },
        )


def run_vqe(
    hamiltonian: np.ndarray | Hamiltonian,
    n_qubits: int | None = None,
    depth: int = 1,
    max_iters: int = 200,
    lr: float = 0.1,
    seed: int | None = None,
    *,
    ansatz: Circuit | Callable[[np.ndarray], Circuit] | None = None,
    optimizer: Any = None,
    backend: Any = None,
    shots: int | None = None,
    noise_model: Any = None,
    use_density_matrix: bool = False,
    initial_state: Any = None,
    initial_density_matrix: Any = None,
    observable_name: str = "energy",
    energy_estimator: str | Any = "exact",
    n_params: int | None = None,
    parameter_shape: tuple[int, ...] | None = None,
    init_params: np.ndarray | None = None,
) -> VQEResult:
    """Convenience function to run VQE."""
    solver = BasicVQE(
        hamiltonian=hamiltonian,
        n_qubits=n_qubits,
        depth=depth,
        seed=seed,
        ansatz=ansatz,
        optimizer=optimizer,
        backend=backend,
        shots=shots,
        noise_model=noise_model,
        use_density_matrix=use_density_matrix,
        initial_state=initial_state,
        initial_density_matrix=initial_density_matrix,
        observable_name=observable_name,
        energy_estimator=energy_estimator,
        n_params=n_params,
        parameter_shape=parameter_shape,
    )
    return solver.run(max_iters=max_iters, lr=lr, init_params=init_params)
