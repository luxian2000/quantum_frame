"""Construct a PySCF/Qiskit Nature Hamiltonian for a water molecule.

Run from the repository root:

    python -m demos.H2O

The coefficients below were generated with PySCF/Qiskit Nature for neutral
singlet H2O, STO-3G basis, a 4-electron/3-spatial-orbital active space, and
Jordan-Wigner mapping. The constant term includes the nuclear repulsion energy
and active-space inactive-energy offset reported by Qiskit Nature.
"""

from __future__ import annotations

import numpy as np

from aicir import Hamiltonian, NumpyBackend
from aicir.measure import hamiltonian_pauli_terms


H2O_GEOMETRY_ANGSTROM = (
    ("O", (0.000000, 0.000000, 0.000000)),
    ("H", (0.757160, 0.586260, 0.000000)),
    ("H", (-0.757160, 0.586260, 0.000000)),
)

H2O_BASIS = "sto3g"
H2O_CHARGE = 0
H2O_SPIN = 0
H2O_ACTIVE_ELECTRONS = 4
H2O_ACTIVE_SPATIAL_ORBITALS = 3
H2O_QUBIT_MAPPER = "JordanWignerMapper"

H2O_GENERATION = (
    "PySCFDriver("
    "atom='O 0 0 0; H 0.757160 0.586260 0; H -0.757160 0.586260 0', "
    "basis='sto3g', charge=0, spin=0) -> "
    "ActiveSpaceTransformer(num_electrons=4, num_spatial_orbitals=3) -> "
    "JordanWignerMapper"
)

H2O_NUCLEAR_REPULSION_ENERGY = 9.191200742618042
H2O_ACTIVE_SPACE_OFFSET = -77.99892062901534


def build_h2o_hamiltonian() -> Hamiltonian:
    """Return the 6-qubit H2O active-space Hamiltonian."""

    return Hamiltonian(n_qubits=6, terms=[
        ("IIIIII", -4.5241061234101245),
        ("IIIIIZ", 0.5153159797373158),
        ("IIIYZY", -0.0777577585041152),
        ("IIIXZX", -0.0777577585041152),
        ("IIIIZI", 0.4813205218163351),
        ("IIIZII", 0.0902426376429984),
        ("IIZIII", 0.5153159797373158),
        ("YZYIII", -0.0777577585041153),
        ("XZXIII", -0.0777577585041153),
        ("IZIIII", 0.4813205218163352),
        ("ZIIIII", 0.0902426376429983),
        ("IIIIZZ", 0.1682539303981457),
        ("IIIZIZ", 0.1200923081829575),
        ("IIZIIZ", 0.1956895651366700),
        ("YZYIIZ", -0.0303380764625682),
        ("XZXIIZ", -0.0303380764625682),
        ("IZIIIZ", 0.1822330598869542),
        ("ZIIIIZ", 0.1372661146010317),
        ("IIIYIY", -0.0295021027117906),
        ("IIIXIX", -0.0295021027117906),
        ("IYYIYY", 0.0139791294888085),
        ("IXXIYY", 0.0139791294888085),
        ("IYYIXX", 0.0139791294888085),
        ("IXXIXX", 0.0139791294888085),
        ("YYIIYY", -0.0004375325735518),
        ("XXIIYY", -0.0004375325735518),
        ("YYIIXX", -0.0004375325735518),
        ("XXIIXX", -0.0004375325735518),
        ("IIZYZY", -0.0303380764625682),
        ("IIZXZX", -0.0303380764625682),
        ("YZYYZY", 0.0171738064180742),
        ("XZXYZY", 0.0171738064180742),
        ("YZYXZX", 0.0171738064180742),
        ("XZXXZX", 0.0171738064180742),
        ("IZIYZY", -0.0290645701382388),
        ("IZIXZX", -0.0290645701382388),
        ("ZIIYZY", -0.0111466920449807),
        ("ZIIXZX", -0.0111466920449807),
        ("IIIZZI", 0.1375870149850004),
        ("IIZIZI", 0.1822330598869542),
        ("YZYIZI", -0.0290645701382388),
        ("XZXIZI", -0.0290645701382388),
        ("IZIIZI", 0.2200397733437616),
        ("ZIIIZI", 0.1472359927649237),
        ("IYYYYI", -0.0004375325735518),
        ("IXXYYI", -0.0004375325735518),
        ("IYYXXI", -0.0004375325735518),
        ("IXXXXI", -0.0004375325735518),
        ("YYIYYI", 0.0096489777799233),
        ("XXIYYI", 0.0096489777799233),
        ("YYIXXI", 0.0096489777799233),
        ("XXIXXI", 0.0096489777799233),
        ("IIZZII", 0.1372661146010317),
        ("YZYZII", -0.0111466920449807),
        ("XZXZII", -0.0111466920449807),
        ("IZIZII", 0.1472359927649237),
        ("ZIIZII", 0.1492816648983666),
        ("IZZIII", 0.1682539303981457),
        ("ZIZIII", 0.1200923081829575),
        ("YIYIII", -0.0295021027117906),
        ("XIXIII", -0.0295021027117906),
        ("ZZIIII", 0.1375870149850004),
    ])


def exact_ground_energy(hamiltonian: Hamiltonian) -> float:
    """Return the minimum eigenvalue of the dense Hamiltonian matrix."""

    backend = NumpyBackend()
    matrix = hamiltonian.to_matrix(backend)
    matrix_np = np.asarray(backend.to_numpy(matrix), dtype=np.complex64)
    return float(np.linalg.eigvalsh(matrix_np).min())


def main() -> None:
    """Print the H2O geometry, Pauli terms, and exact dense-matrix energy."""

    hamiltonian = build_h2o_hamiltonian()
    terms = hamiltonian_pauli_terms(hamiltonian)

    print("H2O PySCF/Qiskit Nature active-space Hamiltonian")
    print("Geometry (angstrom):")
    for atom, coords in H2O_GEOMETRY_ANGSTROM:
        x, y, z = coords
        print(f"  {atom:>2s}: ({x:+.6f}, {y:+.6f}, {z:+.6f})")

    print(f"\nBasis : {H2O_BASIS}")
    print(f"Charge: {H2O_CHARGE}")
    print(f"Spin  : {H2O_SPIN}")
    print(
        "Active space: "
        f"{H2O_ACTIVE_ELECTRONS} electrons, "
        f"{H2O_ACTIVE_SPATIAL_ORBITALS} spatial orbitals"
    )
    print(f"Mapper: {H2O_QUBIT_MAPPER}")
    print(f"Nuclear repulsion energy: {H2O_NUCLEAR_REPULSION_ENERGY:+.10f}")
    print(f"Active-space offset     : {H2O_ACTIVE_SPACE_OFFSET:+.10f}")
    print(f"\nQubits: {hamiltonian.n_qubits}")
    print(f"Terms : {len(terms)}")
    for term in terms:
        print(f"  {term.coefficient:+.10f} * {term.pauli}")

    print(f"\nDense-matrix exact ground energy: {exact_ground_energy(hamiltonian):+.10f}")


if __name__ == "__main__":
    main()
