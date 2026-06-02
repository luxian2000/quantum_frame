"""Demo: visualize a aicir quantum circuit.

Run from the repository root:
    python -m aicir.demos.visual_circuit_demo
"""

from __future__ import annotations

import argparse
import numpy as np

from aicir import Circuit, cnot, crx, hadamard, rzz, rz, swap
from aicir.visual import circuit_to_text, draw_circuit, gate_histogram

from ._visual_demo_utils import add_common_visual_args, configure_matplotlib, save_figure


def build_demo_circuit() -> Circuit:
    return Circuit(
        hadamard(0),
        cnot(1, [0]),
        rz(np.pi / 4, 1),
        crx(np.pi / 3, 2, [1]),
        rzz(np.pi / 2, 0, 2),
        swap(1, 2),
        n_qubits=3,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a quantum circuit with aicir.visual.")
    add_common_visual_args(parser)
    args = parser.parse_args()

    plt = configure_matplotlib(args.show)
    circuit = build_demo_circuit()

    print("=== Circuit ASCII diagram ===")
    print(circuit_to_text(circuit))
    print("\n=== Gate histogram ===")
    print(gate_histogram(circuit))

    fig, _ = draw_circuit(circuit, output="mpl", title="aicir circuit visualization")
    saved = save_figure(fig, args.output_dir, "visual_circuit_demo.png")
    print(f"\nSaved circuit figure: {saved}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
