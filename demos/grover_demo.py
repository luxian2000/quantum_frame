"""Two-qubit Grover demo focused on State/Circuit interaction.

This example keeps the algorithm intentionally small:

1. Prepare |00> as an aicir ``State``.
2. Build the Grover iteration as an aicir ``Circuit``.
3. Apply the circuit unitary to the state with ``State.evolve(...)``.
4. Show that the marked state |11> is amplified to probability 1.
"""

from __future__ import annotations

import numpy as np

from aicir import Circuit, NumpyBackend, cz, hadamard, pauli_x
from aicir.core import State


def hadamard_layer(n_qubits: int) -> Circuit:
    """Return H on every qubit."""

    return Circuit(*(hadamard(qubit) for qubit in range(n_qubits)), n_qubits=n_qubits)


def phase_oracle_for_11() -> Circuit:
    """Mark |11> by a phase flip."""

    return Circuit(cz(1, [0]), n_qubits=2)


def diffusion_operator() -> Circuit:
    """Standard 2-qubit inversion-about-the-mean operator."""

    return Circuit(
        hadamard(0),
        hadamard(1),
        pauli_x(0),
        pauli_x(1),
        cz(1, [0]),
        pauli_x(0),
        pauli_x(1),
        hadamard(0),
        hadamard(1),
        n_qubits=2,
    )


def sorted_probabilities(state: State) -> list[tuple[str, float]]:
    """Return basis-state probabilities in descending order."""

    probs = np.asarray(state.probabilities(), dtype=float).reshape(-1)
    labels = [f"|{idx:0{state.n_qubits}b}>" for idx in range(len(probs))]
    entries = list(zip(labels, probs.tolist()))
    return sorted(entries, key=lambda item: item[1], reverse=True)


def main() -> None:
    backend = NumpyBackend()
    initial_state = State.zero_state(n_qubits=2, backend=backend)

    prepare_uniform = hadamard_layer(2)
    oracle = phase_oracle_for_11()
    diffusion = diffusion_operator()
    grover = (prepare_uniform + oracle + diffusion).bind_backend(backend)

    final_state = initial_state.evolve(grover.unitary())
    probabilities = sorted_probabilities(final_state)

    print("=== aicir Grover demo ===")
    print("Marked state: |11>")
    print()

    print("Initial State:")
    print(f"  {initial_state.ket}")
    print()

    print("Grover Circuit:")
    grover.show()
    print()

    print("State after applying Circuit.unitary() to State.evolve(...):")
    print(f"  {final_state.ket}")
    print()

    print("Basis-state probabilities:")
    for label, prob in probabilities:
        print(f"  {label}: {prob:.3f}")
    print()

    print("Example direct State measurement after Grover:")
    print(f"  {final_state.measure(shots=32)}")
    print()

    best_label, best_prob = probabilities[0]
    print(f"Most probable state: {best_label}")
    print(f"Amplified probability: {best_prob:.3f}")


if __name__ == "__main__":
    main()
