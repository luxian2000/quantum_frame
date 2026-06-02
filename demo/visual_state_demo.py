"""Demo: visualize a quantum state vector.

Run from the repository root:
    python -m demo.visual_state_demo
"""

from __future__ import annotations

import argparse
import numpy as np

from aicir import Circuit, NumpyBackend, StateVector, cnot, hadamard, rz
from aicir.visual import plot_state_amplitudes, plot_state_phase, plot_state_probs

from ._visual_demo_utils import add_common_visual_args, configure_matplotlib, save_figure


def build_bell_phase_state() -> StateVector:
    backend = NumpyBackend()
    circuit = Circuit(
        hadamard(0),
        cnot(1, [0]),
        rz(np.pi / 3, 1),
        n_qubits=2,
        backend=backend,
    )
    return StateVector.zero_state(2, backend).evolve(circuit.unitary())


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize state probabilities, amplitudes, and phases.")
    add_common_visual_args(parser)
    args = parser.parse_args()

    plt = configure_matplotlib(args.show)
    state = build_bell_phase_state()

    print("=== State ket form ===")
    print(state.format())
    print("\n=== Probabilities ===")
    print(state.probabilities())

    fig, _ = plot_state_probs(state, title="Bell-phase state probabilities")
    print(f"Saved probabilities figure: {save_figure(fig, args.output_dir, 'visual_state_probs.png')}")

    fig, _ = plot_state_amplitudes(state, title="Bell-phase state amplitudes")
    print(f"Saved amplitudes figure: {save_figure(fig, args.output_dir, 'visual_state_amplitudes.png')}")

    fig, _ = plot_state_phase(state, title="Bell-phase state phases")
    print(f"Saved phases figure: {save_figure(fig, args.output_dir, 'visual_state_phase.png')}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
