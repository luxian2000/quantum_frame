"""Basic SSVQE implementation.

SSVQE (Subspace-Search Variational Quantum Eigensolver) extends VQE to
simultaneously optimize multiple eigenstates with a shared parameterized unitary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..qml.gradient import psr
from .VQE import _apply_cnot, _apply_single_qubit_gate, _infer_n_qubits, _ry


def _normalize_statevector(state: np.ndarray, dim: int) -> np.ndarray:
    vec = np.asarray(state, dtype=np.complex128).reshape(-1)
    if vec.size != dim:
        raise ValueError(f"Statevector size {vec.size} does not match expected dimension {dim}")
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        raise ValueError("Statevector norm must be positive")
    return vec / norm


def _validate_reference_states(reference_states: np.ndarray, dim: int) -> np.ndarray:
    refs = np.asarray(reference_states, dtype=np.complex128)
    if refs.ndim != 2:
        raise ValueError("reference_states must be a 2D array of shape (n_states, dim)")
    if refs.shape[1] != dim:
        raise ValueError(f"reference_states second dimension must be {dim}, got {refs.shape[1]}")

    normalized = np.zeros_like(refs, dtype=np.complex128)
    for idx in range(refs.shape[0]):
        normalized[idx] = _normalize_statevector(refs[idx], dim)

    gram = normalized @ normalized.conj().T
    if not np.allclose(gram, np.eye(normalized.shape[0], dtype=np.complex128), atol=1e-8):
        raise ValueError("reference_states must be orthonormal")

    return normalized


def _default_reference_states(n_states: int, dim: int) -> np.ndarray:
    if n_states > dim:
        raise ValueError(f"n_states={n_states} cannot exceed Hilbert space dimension {dim}")

    refs = np.zeros((n_states, dim), dtype=np.complex128)
    for index in range(n_states):
        refs[index, index] = 1.0 + 0.0j
    return refs


@dataclass
class SSVQEResult:
    """Container for an SSVQE run."""

    weighted_cost: float
    energies: np.ndarray
    parameters: np.ndarray
    statevectors: np.ndarray
    cost_history: list[float]


class BasicSSVQE:
    """A minimal SSVQE solver based on RY + nearest-neighbor CNOT ansatz.

    The ansatz reuses a single parameter tensor for all target states:
    1) For each layer, apply RY(theta[layer, q]) to each qubit.
    2) Apply CNOT chain q -> q+1.
    """

    def __init__(
        self,
        hamiltonian: np.ndarray,
        n_states: int,
        n_qubits: int | None = None,
        depth: int = 1,
        reference_states: np.ndarray | None = None,
        weights: np.ndarray | None = None,
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

        self.depth = int(depth)
        if self.depth <= 0:
            raise ValueError("depth must be a positive integer")

        if reference_states is None:
            self.reference_states = _default_reference_states(self.n_states, self.dim)
        else:
            self.reference_states = _validate_reference_states(reference_states, self.dim)
            if self.reference_states.shape[0] != self.n_states:
                raise ValueError(
                    f"reference_states has {self.reference_states.shape[0]} states, expected {self.n_states}"
                )

        if weights is None:
            # Descending positive weights prioritize lower states in the weighted objective.
            self.weights = np.arange(self.n_states, 0, -1, dtype=float)
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.size != self.n_states:
                raise ValueError(f"weights size {w.size} does not match n_states={self.n_states}")
            if np.any(w <= 0):
                raise ValueError("weights must be positive")
            self.weights = w

        self._rng = np.random.default_rng(seed)

    @property
    def n_params(self) -> int:
        return self.depth * self.n_qubits

    def initial_params(self, scale: float = 0.1) -> np.ndarray:
        return self._rng.uniform(-scale, scale, size=(self.depth, self.n_qubits))

    def ansatz_state(self, params: np.ndarray, init_state: np.ndarray) -> np.ndarray:
        theta = np.asarray(params, dtype=float).reshape(self.depth, self.n_qubits)
        state = _normalize_statevector(init_state, self.dim)

        for layer in range(self.depth):
            for qubit in range(self.n_qubits):
                state = _apply_single_qubit_gate(state, _ry(theta[layer, qubit]), qubit, self.n_qubits)

            for qubit in range(self.n_qubits - 1):
                state = _apply_cnot(state, control=qubit, target=qubit + 1, n_qubits=self.n_qubits)

        return state

    def energies(self, params: np.ndarray) -> np.ndarray:
        theta = np.asarray(params, dtype=float).reshape(self.depth, self.n_qubits)
        values = np.zeros(self.n_states, dtype=float)

        for idx in range(self.n_states):
            state = self.ansatz_state(theta, self.reference_states[idx])
            energy = np.vdot(state, self.hamiltonian @ state)
            values[idx] = float(np.real(energy))

        return values

    def cost(self, params: np.ndarray) -> float:
        return float(np.dot(self.weights, self.energies(params)))

    def parameter_shift_gradient(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=float).reshape(self.depth, self.n_qubits)
        return psr(self.cost, params)

    def run(
        self,
        max_iters: int = 300,
        lr: float = 0.08,
        init_params: np.ndarray | None = None,
        callback: Callable[[int, float, np.ndarray], None] | None = None,
    ) -> SSVQEResult:
        if max_iters <= 0:
            raise ValueError("max_iters must be a positive integer")
        if lr <= 0:
            raise ValueError("lr must be positive")

        params = self.initial_params() if init_params is None else np.asarray(init_params, dtype=float)
        params = params.reshape(self.depth, self.n_qubits)

        history: list[float] = []
        best_cost = np.inf
        best_params = params.copy()

        for step in range(max_iters):
            current_cost = self.cost(params)
            history.append(current_cost)

            if current_cost < best_cost:
                best_cost = current_cost
                best_params = params.copy()

            if callback is not None:
                callback(step, current_cost, params)

            grad = self.parameter_shift_gradient(params)
            params = params - lr * grad

        best_energies = self.energies(best_params)
        best_states = np.stack(
            [self.ansatz_state(best_params, self.reference_states[idx]) for idx in range(self.n_states)],
            axis=0,
        )

        # Keep energies sorted as approximation to low-lying spectrum.
        order = np.argsort(best_energies)
        return SSVQEResult(
            weighted_cost=float(best_cost),
            energies=best_energies[order],
            parameters=best_params,
            statevectors=best_states[order],
            cost_history=history,
        )


def run_ssvqe(
    hamiltonian: np.ndarray,
    n_states: int,
    n_qubits: int | None = None,
    depth: int = 1,
    max_iters: int = 300,
    lr: float = 0.08,
    reference_states: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    seed: int | None = None,
) -> SSVQEResult:
    """Convenience function to run the basic SSVQE solver."""
    solver = BasicSSVQE(
        hamiltonian=hamiltonian,
        n_states=n_states,
        n_qubits=n_qubits,
        depth=depth,
        reference_states=reference_states,
        weights=weights,
        seed=seed,
    )
    return solver.run(max_iters=max_iters, lr=lr)
