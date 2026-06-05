"""Construct a PySCF/Qiskit Nature Hamiltonian for lithium hydride.

Run from the repository root:

    python -m demos.LiH

The coefficients below were generated with PySCF/Qiskit Nature for neutral
singlet LiH at 1.595 angstrom, STO-3G basis, a 2-electron/2-spatial-orbital
active space, and Jordan-Wigner mapping. The constant term includes the nuclear
repulsion energy and active-space inactive-energy offset reported by Qiskit
Nature.
"""

from __future__ import annotations

import numpy as np

from aicir import Hamiltonian, NumpyBackend
from aicir.measure import hamiltonian_pauli_terms


LIH_GEOMETRY_ANGSTROM = (
    ("Li", (0.000000, 0.000000, 0.000000)),
    ("H", (0.000000, 0.000000, 1.595000)),
)

LIH_BASIS = "sto3g"
LIH_CHARGE = 0
LIH_SPIN = 0
LIH_ACTIVE_ELECTRONS = 2
LIH_ACTIVE_SPATIAL_ORBITALS = 2
LIH_QUBIT_MAPPER = "JordanWignerMapper"

LIH_GENERATION = (
    "PySCFDriver(atom='Li 0 0 0; H 0 0 1.595', basis='sto3g', "
    "charge=0, spin=0) -> "
    "ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=2) -> "
    "JordanWignerMapper"
)

LIH_NUCLEAR_REPULSION_ENERGY = 0.995317638094044
LIH_ACTIVE_SPACE_OFFSET = -7.798291188105942


def build_lih_hamiltonian() -> Hamiltonian:
    """Return the 4-qubit LiH active-space Hamiltonian.

    The term format mirrors Qiskit's sparse-list ordering:
    ``("ZZ", [0, 3], coeff)`` means ``coeff * Z_0 Z_3``.
    """

    return Hamiltonian(n_qubits=4, terms=[
        ("IIII", -0.7059409881285760),
        ("IIIZ", 0.1561395312006330),
        ("IIZI", -0.0149911864321253),
        ("IIZZ", 0.0526847769863971),
        ("IZII", 0.1561395312006330),
        ("IZIZ", 0.1219144565378520),
        ("YYII", 0.0139782944733026),
        ("YYIZ", 0.0121237381452783),
        ("XXII", 0.0139782944733026),
        ("XXIZ", 0.0121237381452783),
        ("ZIII", -0.0149911864321253),
        ("ZIIZ", 0.0559382693417738),
        ("IIYY", 0.0139782944733026),
        ("IZYY", 0.0121237381452783),
        ("IIXX", 0.0139782944733026),
        ("IZXX", 0.0121237381452783),
        ("YYYY", 0.0032534923553767),
        ("XXYY", 0.0032534923553767),
        ("YYXX", 0.0032534923553767),
        ("XXXX", 0.0032534923553767),
        ("ZIYY", -0.0018545501857935),
        ("ZIXX", -0.0018545501857935),
        ("IZZI", 0.0559382693417738),
        ("YYZI", -0.0018545501857935),
        ("XXZI", -0.0018545501857935),
        ("ZIZI", 0.0844837493973667),
        ("ZZII", 0.0526847769863971),
    ])


def exact_ground_energy(hamiltonian: Hamiltonian) -> float:
    """Return the minimum eigenvalue of the dense Hamiltonian matrix."""

    backend = NumpyBackend()
    matrix = hamiltonian.to_matrix(backend)
    matrix_np = np.asarray(backend.to_numpy(matrix), dtype=np.complex64)
    return float(np.linalg.eigvalsh(matrix_np).min())


def main() -> None:
    """Print the LiH geometry, Pauli terms, and exact dense-matrix energy."""

    hamiltonian = build_lih_hamiltonian()
    terms = hamiltonian_pauli_terms(hamiltonian)

    print("LiH PySCF/Qiskit Nature active-space Hamiltonian")
    print("Geometry (angstrom):")
    for atom, coords in LIH_GEOMETRY_ANGSTROM:
        x, y, z = coords
        print(f"  {atom:>2s}: ({x:+.6f}, {y:+.6f}, {z:+.6f})")

    print(f"\nBasis : {LIH_BASIS}")
    print(f"Charge: {LIH_CHARGE}")
    print(f"Spin  : {LIH_SPIN}")
    print(
        "Active space: "
        f"{LIH_ACTIVE_ELECTRONS} electrons, "
        f"{LIH_ACTIVE_SPATIAL_ORBITALS} spatial orbitals"
    )
    print(f"Mapper: {LIH_QUBIT_MAPPER}")
    print(f"Nuclear repulsion energy: {LIH_NUCLEAR_REPULSION_ENERGY:+.10f}")
    print(f"Active-space offset     : {LIH_ACTIVE_SPACE_OFFSET:+.10f}")
    print(f"\nQubits: {hamiltonian.n_qubits}")
    print(f"Terms : {len(terms)}")
    for term in terms:
        print(f"  {term.coefficient:+.10f} * {term.pauli}")

    print(f"\nDense-matrix exact ground energy: {exact_ground_energy(hamiltonian):+.10f}")


if __name__ == "__main__":
    main()
