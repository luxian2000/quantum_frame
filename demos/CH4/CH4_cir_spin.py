"""Placeholder for the CH4 spin-preserving supernet-searched circuit.

This file is overwritten by ``demos/CH4/CH4_npu.py`` after a real supernet QAS run
(``torchrun --nproc_per_node=4 demos/CH4/CH4_npu.py``). Until then it only prepares
the closed-shell Hartree-Fock reference state (occupied qubits 4,5,6,7,8,13,14,15,16,17),
so that ``demos/CH4/CH4_result_spin.py`` and the inline analysis can import cleanly.

Regenerate with: ``torchrun --nproc_per_node=4 demos/CH4/CH4_npu.py``.
Plot with:       ``python -m demos.CH4.CH4_cir_spin``.
"""

from __future__ import annotations

from pathlib import Path

from aicir.core.circuit import Circuit, double_excitation, pauli_x, single_excitation  # noqa: F401
from aicir.visual import plot


def build_ch4_npu_qas_circuit():
    """Return the CH4 ground-state circuit (placeholder: HF reference only)."""
    gates = [
        pauli_x(4),
        pauli_x(5),
        pauli_x(6),
        pauli_x(7),
        pauli_x(8),
        pauli_x(13),
        pauli_x(14),
        pauli_x(15),
        pauli_x(16),
        pauli_x(17),
    ]
    return Circuit(*gates, n_qubits=18)


circuit = build_ch4_npu_qas_circuit()


if __name__ == "__main__":
    figure_path = Path(__file__).with_name("CH4_cir_spin.png")
    plot(circuit, figure_path, title="CH4 supernet ground-state ansatz (spin-preserving)")
    print(f"Saved circuit figure to {figure_path}")
