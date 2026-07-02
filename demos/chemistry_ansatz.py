"""Closed-shell chemistry ansatz helpers for molecular VQE demos."""

from __future__ import annotations

from itertools import product
from pathlib import Path

from aicir.core.io.qasm import save_circuit_qasm3


def closed_shell_hf_occupied_qubits(
    active_electrons: int,
    active_spatial_orbitals: int,
) -> tuple[int, ...]:
    """Return q0-leftmost Jordan-Wigner qubits occupied in the HF determinant."""

    if active_electrons % 2 != 0:
        raise ValueError("closed-shell HF helpers require an even electron count")
    n_spin_orbitals = 2 * int(active_spatial_orbitals)
    n_occ_spatial = int(active_electrons) // 2
    occupied_spin_orbitals = (
        *range(n_occ_spatial),
        *range(int(active_spatial_orbitals), int(active_spatial_orbitals) + n_occ_spatial),
    )
    return tuple(sorted(n_spin_orbitals - 1 - orbital for orbital in occupied_spin_orbitals))


def closed_shell_excitation_pools(
    active_electrons: int,
    active_spatial_orbitals: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...], tuple[tuple[int, int, int, int], ...]]:
    """Return HF qubits, spin-preserving singles, and paired doubles.

    The double-excitation pool is intentionally the closed-shell paired-UCCD
    subset, not the full O(n_occ^2 n_virt^2) UCCSD pool. That keeps the QAS
    search space usable for the BeH2/CH4 demos while still searching the
    chemically important electron-pair excitations.
    """

    n_spatial = int(active_spatial_orbitals)
    n_occ = int(active_electrons) // 2
    if n_occ > n_spatial:
        raise ValueError("active_electrons exceeds active spin-orbital capacity")

    def alpha(spatial_orbital: int) -> int:
        return 2 * n_spatial - 1 - int(spatial_orbital)

    def beta(spatial_orbital: int) -> int:
        return 2 * n_spatial - 1 - (n_spatial + int(spatial_orbital))

    occupied = tuple(range(n_occ))
    virtual = tuple(range(n_occ, n_spatial))
    hf = closed_shell_hf_occupied_qubits(active_electrons, n_spatial)

    singles = tuple(
        (virt_q, occ_q)
        for occ, virt in product(occupied, virtual)
        for virt_q, occ_q in ((alpha(virt), alpha(occ)), (beta(virt), beta(occ)))
    )
    paired_doubles = tuple(
        (*sorted((alpha(virt), beta(virt))), *sorted((alpha(occ), beta(occ))))
        for occ, virt in product(occupied, virtual)
    )
    return hf, singles, paired_doubles


def save_qasm3_if_supported(circuit, file_path: str | Path) -> tuple[bool, str]:
    """Save QASM 3 when the circuit uses only currently exportable gates."""

    path = Path(file_path)
    try:
        save_circuit_qasm3(circuit, path)
    except ValueError as exc:
        return False, str(exc)
    return True, str(path)
