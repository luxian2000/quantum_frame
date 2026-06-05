"""Demo: render a quantum circuit as a coloured PNG figure.

Run from the repository root:
    python -m demos.visual_circuit_png_demo
"""

from __future__ import annotations

import argparse
import math

from aicir import Circuit, cnot, crz, cz, hadamard, pauli_x, s_gate, swap, t_gate
from aicir.visual import plot

from ._visual_demo_utils import add_common_visual_args, configure_matplotlib


def build_mini_qft() -> Circuit:
    return Circuit(
        pauli_x(0),
        cnot(1, [0]),
        s_gate(0),
        t_gate(1),
        hadamard(0),
        crz(math.pi / 2, 0, [1]),
        hadamard(1),
        swap(0, 1),
        n_qubits=2,
    )


def build_cz_chain() -> Circuit:
    return Circuit(
        hadamard(0),
        hadamard(1),
        hadamard(2),
        hadamard(3),
        cz(1, [0]),
        cz(3, [0]),
        cz(2, [1]),
        cz(3, [1]),
        cz(3, [2]),
        n_qubits=4,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a circuit PNG with aicir.visual.")
    add_common_visual_args(parser)
    args = parser.parse_args()

    plt = configure_matplotlib(args.show)

    # By default `plot` writes <this-file>_<variable>.png next to the script.
    # Here we redirect into the demo output directory; the file name still
    # derives from the variable each circuit is bound to.
    cz_chain = build_cz_chain()
    mini_qft = build_mini_qft()

    plot(cz_chain, path=args.output_dir, title="Hadamards + CZ chain")
    print(f"Saved: {args.output_dir / 'visual_circuit_png_demo_cz_chain.png'}")

    plot(mini_qft, path=args.output_dir, title="X, CNOT, phases, mini-QFT, swap")
    print(f"Saved: {args.output_dir / 'visual_circuit_png_demo_mini_qft.png'}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
