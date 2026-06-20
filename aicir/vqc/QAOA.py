"""Basic QAOA implementation.

QAOA (Quantum Approximate Optimization Algorithm) is a hybrid variational method
for combinatorial optimization with alternating problem/mixer Hamiltonian layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


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


@dataclass
class QAOAResult:
    """Container for a QAOA run."""

    energy: float
    gammas: np.ndarray
    betas: np.ndarray
    statevector: np.ndarray | None
    energy_history: list[float]


class BasicQAOA:
    """A minimal QAOA solver for matrix-form Hamiltonians.

    Layered ansatz:
        |psi(theta)> = prod_{l=1..p} exp(-i beta_l H_M) exp(-i gamma_l H_C) |+>^n

    where H_C is the problem Hamiltonian and H_M is the mixer Hamiltonian.
    """

    def __init__(
        self,
        problem_hamiltonian: np.ndarray | None = None,
        p: int = 1,
        n_qubits: int | None = None,
        mixer_hamiltonian: np.ndarray | None = None,
        seed: int | None = None,
        *,
        cost: Any = None,
    ) -> None:
        self.cost = cost
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

        self.p = int(p)
        if self.p <= 0:
            raise ValueError("p must be a positive integer")

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

    def ansatz_state(self, params: np.ndarray, init_state: np.ndarray | None = None) -> np.ndarray:
        gammas, betas = self.split_params(params)
        state = _plus_state(self.n_qubits) if init_state is None else _normalize_statevector(init_state, self.dim)

        for layer in range(self.p):
            u_cost = _exp_hermitian(self.problem_hamiltonian, gammas[layer])
            u_mixer = _exp_hermitian(self.mixer_hamiltonian, betas[layer])
            state = u_cost @ state
            state = u_mixer @ state

        return state

    def energy(self, params: np.ndarray) -> float:
        if self.cost is not None:
            return float(self.cost(np.asarray(params, dtype=float).reshape(-1)))
        state = self.ansatz_state(params)
        value = np.vdot(state, self.problem_hamiltonian @ state)
        return float(np.real(value))

    def _gradient(self, params: np.ndarray) -> np.ndarray:
        if self.cost is not None:
            flat = np.asarray(params, dtype=float).reshape(-1)
            return np.asarray(self.cost.grad(flat), dtype=float).reshape(flat.shape)
        return self.finite_difference_gradient(params)

    def finite_difference_gradient(self, params: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        flat = np.asarray(params, dtype=float).reshape(-1)
        grad = np.zeros_like(flat)
        for index in range(flat.size):
            plus = flat.copy()
            minus = flat.copy()
            plus[index] += eps
            minus[index] -= eps
            grad[index] = (self.energy(plus) - self.energy(minus)) / (2.0 * eps)
        return grad

    def run(
        self,
        max_iters: int = 200,
        lr: float = 0.05,
        init_params: np.ndarray | None = None,
        callback: Callable[[int, float, np.ndarray], None] | None = None,
    ) -> QAOAResult:
        if max_iters <= 0:
            raise ValueError("max_iters must be a positive integer")
        if lr <= 0:
            raise ValueError("lr must be positive")

        params = self.initial_params() if init_params is None else np.asarray(init_params, dtype=float)
        params = params.reshape(-1)
        if params.size != self.n_params:
            raise ValueError(f"init_params size {params.size} does not match expected {self.n_params}")

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

            grad = self._gradient(params)
            params = params - lr * grad

        final_state = None if self.cost is not None else self.ansatz_state(best_params)
        gammas, betas = self.split_params(best_params)
        return QAOAResult(
            energy=float(best_energy),
            gammas=gammas,
            betas=betas,
            statevector=final_state,
            energy_history=history,
        )


def run_qaoa(
    problem_hamiltonian: np.ndarray,
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
