"""Demo: visualize a density matrix.

Run from the repository root:
    python -m demo.visual_density_demo
"""

from __future__ import annotations

import argparse
import numpy as np

from aicir import Circuit, NumpyBackend, StateVector, cnot, hadamard, rz
from aicir.visual import plot_density_matrix, plot_density_real_imag

from ._visual_demo_utils import add_common_visual_args, configure_matplotlib, save_figure


def build_density_matrix():
    backend = NumpyBackend()
    circuit = Circuit(
        hadamard(0),
        cnot(1, [0]),
        rz(np.pi / 5, 0),
        rz(-np.pi / 7, 1),
        n_qubits=2,
        backend=backend,
    )
    state = StateVector.zero_state(2, backend).evolve(circuit.unitary())
    return state.to_density_matrix()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize density-matrix heatmaps.")
    add_common_visual_args(parser)
    args = parser.parse_args()

    plt = configure_matplotlib(args.show)
    rho = build_density_matrix()

    print("=== Density matrix summary ===")
    print(rho)
    print(f"purity: {rho.purity():.6f}")
    print(f"von_neumann_entropy: {rho.von_neumann_entropy():.6f}")

    fig, _ = plot_density_matrix(rho, part="abs", title="Density matrix magnitude")
    print(f"Saved magnitude figure: {save_figure(fig, args.output_dir, 'visual_density_abs.png')}")

    fig, _ = plot_density_matrix(rho, part="phase", cmap="twilight", title="Density matrix phase")
    print(f"Saved phase figure: {save_figure(fig, args.output_dir, 'visual_density_phase.png')}")

    fig, _ = plot_density_real_imag(rho, title="Density matrix real/imag")
    print(f"Saved real/imag figure: {save_figure(fig, args.output_dir, 'visual_density_real_imag.png')}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
