"""Placeholder for the N2 spin-preserving supernet-searched circuit.

This file is overwritten by ``demos/N2/N2_npu.py`` after a real supernet QAS
run (``torchrun --nproc_per_node=4 demos/N2/N2_npu.py``). Until then it only
prepares the closed-shell Hartree-Fock reference state (occupied qubits 2, 3, 4, 5, 6, 9, 10, 11, 12, 13), so
that ``demos/N2/N2_result_spin.py`` and the inline analysis can import cleanly.

Regenerate with: ``torchrun --nproc_per_node=4 demos/N2/N2_npu.py``.
Plot with:       ``python -m demos.N2.N2_cir_spin``.
"""

from __future__ import annotations

from pathlib import Path

from aicir.core.circuit import Circuit, double_excitation, pauli_x, single_excitation  # noqa: F401
from aicir.visual import plot


def build_n2_npu_qas_circuit():
    """Return the N2 ground-state circuit (placeholder: HF reference only)."""
    gates = [
        pauli_x(2),
        pauli_x(3),
        pauli_x(4),
        pauli_x(5),
        pauli_x(6),
        pauli_x(9),
        pauli_x(10),
        pauli_x(11),
        pauli_x(12),
        pauli_x(13),
    ]
    return Circuit(*gates, n_qubits=14)


circuit = build_n2_npu_qas_circuit()


if __name__ == "__main__":
    figure_path = Path(__file__).with_name("N2_cir_spin.png")
    plot(circuit, figure_path, title="N2 supernet ground-state ansatz (spin-preserving)")
    print(f"Saved circuit figure to {figure_path}")
