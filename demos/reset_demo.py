"""Demonstrate reset execution using result.snap intermediate states."""

from __future__ import annotations

import numpy as np

from aicir import Circuit, Measure, NumpyBackend, cnot, hadamard, measure, reset
from aicir.core import State


def _flat_state(state) -> np.ndarray:
    """Return a snapshot as a one-dimensional state vector."""
    return np.asarray(state).reshape(-1)


def _state_ket(state, backend: NumpyBackend) -> str:
    """Format a state vector as Dirac notation for readable demo output."""
    return State.from_array(_flat_state(state), backend=backend).ket


def build_circuit() -> Circuit:
    """Build the requested three-qubit reset verification circuit."""
    return Circuit(
        hadamard(0),  # gate 0: H(0)
        cnot(1, [0]), # gate 1: cnot(1, 0)
        cnot(2, [1]), # gate 2: cnot(2, 1)
        measure(1),   # gate 3: measure(1), state is unchanged
        reset(1),     # gate 4: reset(1), q1 is set to |0>
        cnot(1, [2]), # gate 5: cnot(1, 2), proves evolution continues after reset
        n_qubits=3,
    )


def run_demo(*, verbose: bool = True) -> dict[str, object]:
    """Run the demo and verify reset by comparing recorded snapshots."""
    backend = NumpyBackend()
    measurement = Measure(backend)
    circuit = build_circuit()
    result = measurement.run(circuit, shots=None, snap=[0, 1, 2, 3, 4, 5])

    after_hadamard = _flat_state(result.snap(0))
    after_cnot_10 = _flat_state(result.snap(1))
    after_cnot_21 = _flat_state(result.snap(2))
    after_measure = _flat_state(result.snap(3))
    after_reset = _flat_state(result.snap(4))
    after_cnot_12 = _flat_state(result.snap(5))

    expected_after_hadamard = _basis_state([0, 4], [1 / np.sqrt(2), 1 / np.sqrt(2)])
    expected_after_cnot_10 = _basis_state([0, 6], [1 / np.sqrt(2), 1 / np.sqrt(2)])
    expected_after_cnot_21 = _basis_state([0, 7], [1 / np.sqrt(2), 1 / np.sqrt(2)])
    expected_after_reset = _basis_state([0, 5], [1 / np.sqrt(2), 1 / np.sqrt(2)])
    expected_after_cnot_12 = expected_after_cnot_21

    reset_verified = (
        np.allclose(after_hadamard, expected_after_hadamard, atol=1e-6)
        and np.allclose(after_cnot_10, expected_after_cnot_10, atol=1e-6)
        and np.allclose(after_cnot_21, expected_after_cnot_21, atol=1e-6)
        and np.allclose(after_measure, expected_after_cnot_21, atol=1e-6)
        and np.allclose(after_reset, expected_after_reset, atol=1e-6)
        and np.allclose(after_cnot_12, expected_after_cnot_12, atol=1e-6)
    )

    if verbose:
        print("Use result.snap to verify reset execution in the requested 3-qubit circuit")
        print("snap(0), after H(0):", _state_ket(after_hadamard, backend))
        print("snap(1), after cnot(1, 0):", _state_ket(after_cnot_10, backend))
        print("snap(2), after cnot(2, 1):", _state_ket(after_cnot_21, backend))
        print("snap(3), after measure(1):", _state_ket(after_measure, backend))
        print("snap(4), after reset(1):", _state_ket(after_reset, backend))
        print("snap(5), after cnot(1, 2):", _state_ket(after_cnot_12, backend))
        print("reset verified:", reset_verified)

    if not reset_verified:
        raise AssertionError("reset snapshot verification failed")

    return {
        "circuit": circuit,
        "result": result,
        "after_hadamard": after_hadamard,
        "after_cnot_10": after_cnot_10,
        "after_cnot_21": after_cnot_21,
        "after_measure": after_measure,
        "after_reset": after_reset,
        "after_cnot_12": after_cnot_12,
        "reset_verified": reset_verified,
    }


def _basis_state(indices, values) -> np.ndarray:
    state = np.zeros(8, dtype=np.complex64)
    for index, value in zip(indices, values):
        state[index] = value
    return state


def main() -> None:
    run_demo(verbose=True)


if __name__ == "__main__":
    main()
