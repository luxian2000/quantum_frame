"""Demonstrate Measure.run(..., snap=[...]) intermediate state snapshots."""

from __future__ import annotations

import numpy as np

from aicir import Circuit, Measure, NumpyBackend, cnot, hadamard, rx, measure
from aicir.core import State


def describe_state(label: str, state_vector, backend: NumpyBackend) -> None:
    state = State.from_array(state_vector, backend=backend)
    print(f"{label}:")
    print("  ket:", state.ket)
    print("  vector:", np.array2string(state.array, precision=3, suppress_small=True))


def main() -> None:
    backend = NumpyBackend()
    measurement = Measure(backend)

    circuit = Circuit(
        hadamard(0),
        rx(0.5, 0),
        cnot(1, [0]),
        n_qubits=2,
    )

    result = measurement.run(circuit, shots=None, snap=[0, 1, 2])

    print("Circuit snapshots recorded by snap=[0, 1, 2]")
    # snap 操作下标集合已内嵌于 snapshot_states，metadata 不再重复存储
    print("snap op indices:", sorted(result.snapshot_states.keys()))
    print()

    describe_state("After gate 0: hadamard(0)", result.snap(0), backend)
    print()
    describe_state("After gate 1: rx(0.5, 0)", result.snap(1), backend)
    print()
    describe_state("After gate 2: cnot(1, [0])", result.snap(2), backend)
    print()
    describe_state("Final state", result.final_state, backend)

    print()
    print("Unrecorded gate snapshot:", result.snap(2))
    print(
        "snap(2) equals final_state:",
        np.allclose(result.snap(2).array, result.final_state.array),
    )


if __name__ == "__main__":
    main()
