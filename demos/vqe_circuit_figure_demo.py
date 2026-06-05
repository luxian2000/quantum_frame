"""Demo: a richer VQE ansatz + circuit figure on the 4-qubit H2 Hamiltonian.

``aicir.chemistry`` deliberately ships only *confirmed-source* H2 presets (it is
not an electronic-structure pipeline, and there is no PySCF / OpenFermion
backend bundled), so the **4-qubit Jordan-Wigner H2** preset (``h2_jw``, 15
Pauli terms including the ``XXYY`` / ``YYXX`` exchange couplings) is the largest
molecule available here. To exercise a genuinely more complex VQE *circuit*,
this demo pairs that Hamiltonian with a deeper, all-to-all-entangled
hardware-efficient ansatz. It

1. loads the 4-qubit Hamiltonian,
2. builds a multi-layer HEA (``Ry``/``Rz`` rotations with a configurable
   entangler topology, ``full`` / all-to-all by default),
3. trains it with ``BasicVQE`` and compares against exact diagonalization, and
4. renders the trained ansatz with :func:`aicir.visual.plot`.

Run from the repository root::

    python -m demos.vqe_circuit_figure_demo
    python -m demos.vqe_circuit_figure_demo --layers 2 --topology linear
"""

from __future__ import annotations

import argparse

import numpy as np

from aicir import NumpyBackend
from aicir.chemistry import get_molecule, molecule_hamiltonian, molecule_matrix
from aicir.optimizer import COBYLA
from aicir.vqc import BasicVQE
from aicir.vqc.ansatz import hea
from aicir.visual import plot

from ._visual_demo_utils import add_common_visual_args, configure_matplotlib

MOLECULE = "h2_jw"  # 4-qubit Jordan-Wigner H2 (largest aicir.chemistry preset).


def build_ansatz(n_qubits: int, layers: int, topology: str, backend: NumpyBackend):
    """Hardware-efficient ansatz: Ry+Rz rotations with a CX entangler."""
    return hea(
        n_qubits,
        layers=layers,
        rotation_gates=("ry", "rz"),
        entangler="cx",
        topology=topology,
        final_rotation_layer=True,
        backend=backend,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="VQE + circuit figure on a 4-qubit Hamiltonian.")
    add_common_visual_args(parser)
    parser.add_argument("--layers", type=int, default=3, help="HEA layers.")
    parser.add_argument(
        "--topology",
        choices=("full", "linear", "ring"),
        default="full",
        help="Entangler topology (full = all-to-all).",
    )
    parser.add_argument("--maxiter", type=int, default=600, help="Optimizer iterations.")
    parser.add_argument("--seed", type=int, default=11, help="Seed for initial parameters.")
    args = parser.parse_args()

    plt = configure_matplotlib(args.show)

    backend = NumpyBackend()
    preset = get_molecule(MOLECULE)
    hamiltonian = molecule_hamiltonian(MOLECULE)
    ansatz = build_ansatz(hamiltonian.n_qubits, args.layers, args.topology, backend)

    print("=" * 72)
    print(f"VQE demo: {preset.formula} ({preset.mapping}, {hamiltonian.n_qubits} qubits)")
    print("=" * 72)
    print(f"  ansatz : HEA Ry+Rz + {args.topology} CX, layers={args.layers}")
    print(f"  gates  : {len(ansatz.gates)} | parameters: {len(ansatz.parameters)}")

    rng = np.random.default_rng(args.seed)
    init_params = rng.uniform(-0.1, 0.1, size=len(ansatz.parameters))

    solver = BasicVQE(
        hamiltonian,
        ansatz=ansatz,
        backend=backend,
        optimizer=COBYLA(options={"maxiter": args.maxiter, "rhobeg": 0.3, "tol": 1e-9}),
        energy_estimator="exact",
    )
    # COBYLA probes wide parameter ranges; ignore transient overflow warnings.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        result = solver.run(init_params=init_params)

    exact_energy = float(np.linalg.eigvalsh(molecule_matrix(MOLECULE, backend=backend)).min())
    print(f"  VQE energy   : {result.energy:.8f} Ha")
    print(f"  exact ground : {exact_energy:.8f} Ha")
    print(f"  VQE - exact  : {result.energy - exact_energy:+.3e} Ha")

    # `plot` accepts the trained ansatz directly; here we redirect the figure
    # into the demo output directory.
    plot(
        ansatz,
        path=args.output_dir,
        name="vqe_h2_jw_ansatz",
        title=f"{preset.formula} VQE ansatz (4-qubit JW, HEA Ry+Rz + {args.topology} CX, {args.layers} layers)",
    )
    print(f"\nSaved circuit figure: {args.output_dir / 'vqe_h2_jw_ansatz.png'}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
