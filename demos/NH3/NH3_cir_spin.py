"""Placeholder for the NH3 spin-preserving supernet-searched circuit.

This file is overwritten by ``demos/NH3/NH3_npu.py`` after a real supernet QAS
run (``torchrun --nproc_per_node=4 demos/NH3/NH3_npu.py``). Until then it only
prepares the closed-shell Hartree-Fock reference state (occupied qubits 3, 4, 5, 9, 10, 11), so
that ``demos/NH3/NH3_result_spin.py`` and the inline analysis can import cleanly.

Regenerate with: ``torchrun --nproc_per_node=4 demos/NH3/NH3_npu.py``.
Plot with:       ``python -m demos.NH3.NH3_cir_spin``.
"""

from __future__ import annotations

from pathlib import Path

from aicir.core.circuit import Circuit, double_excitation, pauli_x, single_excitation  # noqa: F401
from aicir.visual import plot


def build_nh3_npu_qas_circuit():
    """Return the NH3 ground-state circuit (placeholder: HF reference only)."""
    gates = [
        pauli_x(3),
        pauli_x(4),
        pauli_x(5),
        pauli_x(9),
        pauli_x(10),
        pauli_x(11),
    ]
    return Circuit(*gates, n_qubits=12)


circuit = build_nh3_npu_qas_circuit()


if __name__ == "__main__":
    figure_path = Path(__file__).with_name("NH3_cir_spin.png")
    plot(circuit, figure_path, title="NH3 supernet ground-state ansatz (spin-preserving)")
    print(f"Saved circuit figure to {figure_path}")
