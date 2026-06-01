"""Basic VQD implementation.

VQD (Variational Quantum Deflation) finds multiple eigenstates sequentially by
adding overlap-penalty terms against previously optimized states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..qml.grad import psr
from .VQE import _apply_cnot, _apply_single_qubit_gate, _infer_n_qubits, _ry


def _normalize_statevector(state: np.ndarray, dim: int) -> np.ndarray:
    vec = np.asarray(state, dtype=np.complex128).reshape(-1)
    if vec.size != dim:
        raise ValueError(f"Statevector size {vec.size} does not match expected dimension {dim}")
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        raise ValueError("Statevector norm must be positive")
    return vec / norm


def _default_penalties(n_states: int, hamiltonian: np.ndarray) -> np.ndarray:
    if n_states <= 1:
        return np.zeros(0, dtype=float)
    spectral_hint = float(np.linalg.norm(hamiltonian, ord=2))
    beta = max(1.0, spectral_hint)
    return np.full(n_states - 1, beta, dtype=float)


@dataclass
class VQDResult:
    """Container for a VQD run."""

    energies: np.ndarray
    parameters: np.ndarray
    statevectors: np.ndarray
    objective_histories: list[list[float]]


class BasicVQD:
    """A minimal VQD solver with RY + nearest-neighbor CNOT ansatz.

    For target state k, the objective is:
    E_k(theta) + sum_{j < k} beta_j * |<psi(theta)|phi_j>|^2
    where phi_j are previously found states.
    """

    def __init__(
        self,
        hamiltonian: np.ndarray,
        n_states: int,
        n_qubits: int | None = None,
        depth: int = 1,
        penalties: np.ndarray | None = None,
        reference_state: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        ham = np.asarray(hamiltonian, dtype=np.complex128)
        if ham.ndim != 2 or ham.shape[0] != ham.shape[1]:
            raise ValueError("hamiltonian must be a square matrix")

        inferred = _infer_n_qubits(ham.shape[0])
        self.n_qubits = inferred if n_qubits is None else int(n_qubits)
        if self.n_qubits != inferred:
            raise ValueError(
                f"n_qubits={self.n_qubits} does not match Hamiltonian dimension {ham.shape[0]}"
            )

        self.hamiltonian = ham
        self.dim = 1 << self.n_qubits

        self.n_states = int(n_states)
        if self.n_states <= 0:
            raise ValueError("n_states must be a positive integer")
        if self.n_states > self.dim:
            raise ValueError(f"n_states={self.n_states} cannot exceed Hilbert space dimension {self.dim}")

        self.depth = int(depth)
        if self.depth <= 0:
            raise ValueError("depth must be a positive integer")

        if penalties is None:
            self.penalties = _default_penalties(self.n_states, self.hamiltonian)
        else:
            p = np.asarray(penalties, dtype=float).reshape(-1)
            if p.size != max(0, self.n_states - 1):
                raise ValueError(
                    f"penalties size {p.size} does not match required size {max(0, self.n_states - 1)}"
                )
            if np.any(p <= 0):
                raise ValueError("penalties must be positive")
            self.penalties = p

        if reference_state is None:
            ref = np.zeros(self.dim, dtype=np.complex128)
            ref[0] = 1.0 + 0.0j
            self.reference_state = ref
        else:
            self.reference_state = _normalize_statevector(reference_state, self.dim)

        self._rng = np.random.default_rng(seed)

    @property
    def n_params(self) -> int:
        return self.depth * self.n_qubits

    def initial_params(self, scale: float = 0.1) -> np.ndarray:
        return self._rng.uniform(-scale, scale, size=(self.depth, self.n_qubits))

    def ansatz_state(self, params: np.ndarray) -> np.ndarray:
        theta = np.asarray(params, dtype=float).reshape(self.depth, self.n_qubits)
        state = self.reference_state.copy()

        for layer in range(self.depth):
            for qubit in range(self.n_qubits):
                state = _apply_single_qubit_gate(state, _ry(theta[layer, qubit]), qubit, self.n_qubits)

            for qubit in range(self.n_qubits - 1):
                state = _apply_cnot(state, control=qubit, target=qubit + 1, n_qubits=self.n_qubits)

        return state

    def energy(self, params: np.ndarray) -> float:
        state = self.ansatz_state(params)
        value = np.vdot(state, self.hamiltonian @ state)
        return float(np.real(value))

    def objective(self, params: np.ndarray, prev_states: list[np.ndarray], level: int) -> float:
        value = self.energy(params)
        if level <= 0:
            return value

        state = self.ansatz_state(params)
        for idx in range(level):
            overlap = np.vdot(prev_states[idx], state)
            value += float(self.penalties[idx] * (np.abs(overlap) ** 2))
        return float(value)

    def parameter_shift_gradient(self, params: np.ndarray, prev_states: list[np.ndarray], level: int) -> np.ndarray:
        params = np.asarray(params, dtype=float).reshape(self.depth, self.n_qubits)
        return psr(lambda theta: self.objective(theta, prev_states, level), params)

    def _optimize_single_state(
        self,
        level: int,
        prev_states: list[np.ndarray],
        max_iters: int,
        lr: float,
        init_params: np.ndarray | None,
        callback: Callable[[int, int, float, np.ndarray], None] | None,
    ) -> tuple[float, np.ndarray, np.ndarray, list[float]]:
        params = self.initial_params() if init_params is None else np.asarray(init_params, dtype=float)
        params = params.reshape(self.depth, self.n_qubits)

        history: list[float] = []
        best_objective = np.inf
        best_params = params.copy()

        for step in range(max_iters):
            current_objective = self.objective(params, prev_states, level)
            history.append(current_objective)

            if current_objective < best_objective:
                best_objective = current_objective
                best_params = params.copy()

            if callback is not None:
                callback(level, step, current_objective, params)

            grad = self.parameter_shift_gradient(params, prev_states, level)
            params = params - lr * grad

        best_state = self.ansatz_state(best_params)
        best_energy = float(np.real(np.vdot(best_state, self.hamiltonian @ best_state)))
        return best_energy, best_params, best_state, history

    def run(
        self,
        max_iters: int = 250,
        lr: float = 0.1,
        init_params_list: list[np.ndarray] | None = None,
        callback: Callable[[int, int, float, np.ndarray], None] | None = None,
    ) -> VQDResult:
        if max_iters <= 0:
            raise ValueError("max_iters must be a positive integer")
        if lr <= 0:
            raise ValueError("lr must be positive")

        prev_states: list[np.ndarray] = []
        energies: list[float] = []
        params_list: list[np.ndarray] = []
        statevectors: list[np.ndarray] = []
        histories: list[list[float]] = []

        for level in range(self.n_states):
            level_init = None
            if init_params_list is not None:
                if level >= len(init_params_list):
                    raise ValueError("init_params_list does not provide enough entries for n_states")
                level_init = init_params_list[level]

            energy, parameters, state, history = self._optimize_single_state(
                level=level,
                prev_states=prev_states,
                max_iters=max_iters,
                lr=lr,
                init_params=level_init,
                callback=callback,
            )

            prev_states.append(state)
            energies.append(energy)
            params_list.append(parameters)
            statevectors.append(state)
            histories.append(history)

        return VQDResult(
            energies=np.asarray(energies, dtype=float),
            parameters=np.stack(params_list, axis=0),
            statevectors=np.stack(statevectors, axis=0),
            objective_histories=histories,
        )


def run_vqd(
    hamiltonian: np.ndarray,
    n_states: int,
    n_qubits: int | None = None,
    depth: int = 1,
    max_iters: int = 250,
    lr: float = 0.1,
    penalties: np.ndarray | None = None,
    reference_state: np.ndarray | None = None,
    seed: int | None = None,
) -> VQDResult:
    """Convenience function to run the basic VQD solver."""
    solver = BasicVQD(
        hamiltonian=hamiltonian,
        n_states=n_states,
        n_qubits=n_qubits,
        depth=depth,
        penalties=penalties,
        reference_state=reference_state,
        seed=seed,
    )
    return solver.run(max_iters=max_iters, lr=lr)
