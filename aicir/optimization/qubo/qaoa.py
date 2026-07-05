"""Convenience helpers connecting QUBO models to BasicQAOA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ...backends.numpy_backend import NumpyBackend
from ...vqc.QAOA import BasicQAOA, QAOAResult
from .adapters import builder_to_hamiltonian, model_to_hamiltonian


@dataclass(frozen=True)
class QAOAAssignment:
    """Decoded most-likely computational basis state from a QAOA statevector."""

    bitstring: str
    probability: float
    assignment: dict[str, int]


def model_to_qaoa_matrix(
    model,
    *,
    compact: bool = True,
    include_offset: bool = True,
    backend: NumpyBackend | None = None,
) -> tuple[np.ndarray, int]:
    """Convert a QUBO ``Model`` into the legacy dense matrix accepted by ``BasicQAOA``."""

    active_backend = NumpyBackend() if backend is None else backend
    hamiltonian = model_to_hamiltonian(
        model,
        compact=compact,
        include_offset=include_offset,
    )
    matrix = active_backend.to_numpy(hamiltonian.to_matrix(active_backend))
    return np.asarray(matrix, dtype=np.complex128), hamiltonian.n_qubits


def builder_to_qaoa_matrix(
    builder,
    *,
    compact: bool = True,
    include_offset: bool = True,
    backend: NumpyBackend | None = None,
) -> tuple[np.ndarray, int]:
    """Convert a ``QuboBuilder`` into the legacy dense matrix accepted by ``BasicQAOA``."""

    active_backend = NumpyBackend() if backend is None else backend
    hamiltonian = builder_to_hamiltonian(
        builder,
        compact=compact,
        include_offset=include_offset,
    )
    matrix = active_backend.to_numpy(hamiltonian.to_matrix(active_backend))
    return np.asarray(matrix, dtype=np.complex128), hamiltonian.n_qubits


def model_to_basic_qaoa(
    model,
    p: int,
    *,
    compact: bool = True,
    include_offset: bool = True,
    mixer_hamiltonian: np.ndarray | None = None,
    seed: int | None = None,
) -> BasicQAOA:
    """Build a ``BasicQAOA`` solver from a QUBO ``Model``."""

    if mixer_hamiltonian is not None:
        matrix, n_qubits = model_to_qaoa_matrix(
            model,
            compact=compact,
            include_offset=include_offset,
        )
        return BasicQAOA(
            problem_hamiltonian=matrix,
            p=p,
            n_qubits=n_qubits,
            mixer_hamiltonian=mixer_hamiltonian,
            seed=seed,
        )

    hamiltonian = model_to_hamiltonian(
        model,
        compact=compact,
        include_offset=include_offset,
    )
    return BasicQAOA(
        problem_hamiltonian=hamiltonian,
        p=p,
        seed=seed,
    )


def builder_to_basic_qaoa(
    builder,
    p: int,
    *,
    compact: bool = True,
    include_offset: bool = True,
    mixer_hamiltonian: np.ndarray | None = None,
    seed: int | None = None,
) -> BasicQAOA:
    """Build a ``BasicQAOA`` solver from a ``QuboBuilder``."""

    if mixer_hamiltonian is not None:
        matrix, n_qubits = builder_to_qaoa_matrix(
            builder,
            compact=compact,
            include_offset=include_offset,
        )
        return BasicQAOA(
            problem_hamiltonian=matrix,
            p=p,
            n_qubits=n_qubits,
            mixer_hamiltonian=mixer_hamiltonian,
            seed=seed,
        )

    hamiltonian = builder_to_hamiltonian(
        builder,
        compact=compact,
        include_offset=include_offset,
    )
    return BasicQAOA(
        problem_hamiltonian=hamiltonian,
        p=p,
        seed=seed,
    )


def run_model_qaoa(
    model,
    p: int,
    *,
    max_iters: int = 200,
    lr: float = 0.05,
    init_params: np.ndarray | None = None,
    callback: Callable[[int, float, np.ndarray], None] | None = None,
    compact: bool = True,
    include_offset: bool = True,
    mixer_hamiltonian: np.ndarray | None = None,
    seed: int | None = None,
) -> QAOAResult:
    """Run ``BasicQAOA`` on a QUBO ``Model``."""

    solver = model_to_basic_qaoa(
        model,
        p,
        compact=compact,
        include_offset=include_offset,
        mixer_hamiltonian=mixer_hamiltonian,
        seed=seed,
    )
    return solver.run(
        max_iters=max_iters,
        lr=lr,
        init_params=init_params,
        callback=callback,
    )


run_qubo_qaoa = run_model_qaoa


def most_likely_bitstring(statevector: np.ndarray, n_qubits: int | None = None) -> tuple[str, float]:
    """Return the most likely computational-basis bitstring and probability."""

    state = np.asarray(statevector, dtype=np.complex128).reshape(-1)
    if state.size == 0:
        raise ValueError("statevector must not be empty.")
    if n_qubits is None:
        inferred = int(round(np.log2(state.size)))
        if (1 << inferred) != state.size:
            raise ValueError("statevector size must be a power of 2.")
        n_qubits = inferred
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")
    if state.size != (1 << n_qubits):
        raise ValueError("statevector size does not match n_qubits.")

    probabilities = np.abs(state) ** 2
    index = int(np.argmax(probabilities))
    bitstring = format(index, f"0{n_qubits}b")
    return bitstring, float(probabilities[index])


def bitstring_to_qubo_assignment(bitstring: str, variable_names: list[str]) -> dict[str, int]:
    """Decode a computational-basis bitstring using ``x=(1+Z)/2``.

    With this convention, computational bit ``0`` maps to QUBO value ``1`` and
    computational bit ``1`` maps to QUBO value ``0``.
    """

    if len(bitstring) != len(variable_names):
        raise ValueError("bitstring length must match variable_names length.")
    if any(bit not in {"0", "1"} for bit in bitstring):
        raise ValueError("bitstring must contain only '0' and '1'.")
    return {name: 1 - int(bit) for name, bit in zip(variable_names, bitstring)}


def most_likely_qaoa_assignment(
    statevector: np.ndarray,
    variable_names: list[str],
) -> QAOAAssignment:
    """Decode the highest-probability statevector entry as QUBO variables."""

    bitstring, probability = most_likely_bitstring(statevector, n_qubits=len(variable_names))
    return QAOAAssignment(
        bitstring=bitstring,
        probability=probability,
        assignment=bitstring_to_qubo_assignment(bitstring, variable_names),
    )


__all__ = [
    "builder_to_basic_qaoa",
    "builder_to_qaoa_matrix",
    "bitstring_to_qubo_assignment",
    "model_to_basic_qaoa",
    "model_to_qaoa_matrix",
    "most_likely_bitstring",
    "most_likely_qaoa_assignment",
    "QAOAAssignment",
    "run_model_qaoa",
    "run_qubo_qaoa",
]
