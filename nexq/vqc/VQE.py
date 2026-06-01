"""Basic VQE implementation.

This module provides a minimal, self-contained Variational Quantum Eigensolver
implementation based on state-vector simulation with NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..qml.gradient import psr


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
    statevector: np.ndarray
    energy_history: list[float]


class BasicVQE:
    """A minimal VQE solver with RY + nearest-neighbor CNOT ansatz.

    The ansatz per layer is:
    1) Apply RY(theta[layer, q]) on each qubit q.
    2) Apply a nearest-neighbor CNOT chain q -> q+1.
    """

    def __init__(
        self,
        hamiltonian: np.ndarray,
        n_qubits: int | None = None,
        depth: int = 1,
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
        self.depth = int(depth)
        if self.depth <= 0:
            raise ValueError("depth must be a positive integer")

        self._rng = np.random.default_rng(seed)

    @property
    def n_params(self) -> int:
        return self.depth * self.n_qubits

    def initial_params(self, scale: float = 0.1) -> np.ndarray:
        return self._rng.uniform(-scale, scale, size=(self.depth, self.n_qubits))

    def ansatz_state(self, params: np.ndarray) -> np.ndarray:
        theta = np.asarray(params, dtype=float).reshape(self.depth, self.n_qubits)
        state = np.zeros(1 << self.n_qubits, dtype=np.complex128)
        state[0] = 1.0 + 0.0j

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

    def parameter_shift_gradient(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=float).reshape(self.depth, self.n_qubits)
        return psr(self.energy, params)

    def run(
        self,
        max_iters: int = 200,
        lr: float = 0.1,
        init_params: np.ndarray | None = None,
        callback: Callable[[int, float, np.ndarray], None] | None = None,
    ) -> VQEResult:
        if max_iters <= 0:
            raise ValueError("max_iters must be a positive integer")
        if lr <= 0:
            raise ValueError("lr must be positive")

        params = self.initial_params() if init_params is None else np.asarray(init_params, dtype=float)
        params = params.reshape(self.depth, self.n_qubits)

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
        return VQEResult(
            energy=float(best_energy),
            parameters=best_params,
            statevector=final_state,
            energy_history=history,
        )


def run_vqe(
    hamiltonian: np.ndarray,
    n_qubits: int | None = None,
    depth: int = 1,
    max_iters: int = 200,
    lr: float = 0.1,
    seed: int | None = None,
) -> VQEResult:
    """Convenience function to run the basic VQE solver."""
    solver = BasicVQE(hamiltonian=hamiltonian, n_qubits=n_qubits, depth=depth, seed=seed)
    return solver.run(max_iters=max_iters, lr=lr)
