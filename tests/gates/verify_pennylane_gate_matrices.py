"""Validate aicir basic gate matrices against PennyLane.

Run from the repository root:

    PYTHONPATH=. python tests/gates/verify_pennylane_gate_matrices.py

PennyLane matrices are requested with ``wire_order=[0, 1, ...]``.  That wire
order matches aicir's q0-as-left-most tensor convention, so no qubit-order
remapping is needed.

The in-circuit measure marker is intentionally excluded because it is not a
unitary gate and has no matrix representation in aicir.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pennylane as qml

from aicir import (
    ccnot,
    cnot,
    crx,
    cry,
    crz,
    cx,
    cy,
    cz,
    hadamard,
    molmer_sorensen,
    ms_gate,
    pauli_x,
    pauli_y,
    pauli_z,
    rx,
    rxx,
    ry,
    rz,
    rzz,
    s_gate,
    swap,
    t_gate,
    toffoli,
    u2,
    u3,
)
from aicir.core.gates import gate_to_matrix


DEFAULT_OUTPUT_PATH = Path(__file__).with_suffix(".txt")


@dataclass(frozen=True)
class GateCase:
    name: str
    n_qubits: int
    aicir_gate: object
    pennylane_op: object


def _pennylane_matrix(case: GateCase) -> np.ndarray:
    return np.asarray(
        qml.matrix(case.pennylane_op, wire_order=list(range(case.n_qubits))),
        dtype=np.complex128,
    )


def _aicir_matrix(case: GateCase) -> np.ndarray:
    return np.asarray(gate_to_matrix(case.aicir_gate, cir_qubits=case.n_qubits), dtype=np.complex128)


def gate_cases() -> list[GateCase]:
    a = math.pi / 7
    b = math.pi / 5
    c = math.pi / 3
    return [
        GateCase("identity", 3, {"type": "identity", "n_qubits": 3}, qml.Identity(wires=[0, 1, 2])),
        GateCase("pauli_x", 3, pauli_x(0), qml.PauliX(wires=0)),
        GateCase("pauli_y", 3, pauli_y(1), qml.PauliY(wires=1)),
        GateCase("pauli_z", 3, pauli_z(2), qml.PauliZ(wires=2)),
        GateCase("hadamard", 3, hadamard(0), qml.Hadamard(wires=0)),
        GateCase("s_gate", 3, s_gate(1), qml.S(wires=1)),
        GateCase("t_gate", 3, t_gate(2), qml.T(wires=2)),
        GateCase("rx", 3, rx(a, 0), qml.RX(a, wires=0)),
        GateCase("ry", 3, ry(b, 1), qml.RY(b, wires=1)),
        GateCase("rz", 3, rz(c, 2), qml.RZ(c, wires=2)),
        GateCase("u2", 3, u2(a, b, 1), qml.U2(a, b, wires=1)),
        GateCase("u3", 3, u3(a, b, c, 2), qml.U3(a, b, c, wires=2)),
        GateCase("cx", 3, cx(1, [0]), qml.CNOT(wires=[0, 1])),
        GateCase("cnot", 3, cnot(2, [0]), qml.CNOT(wires=[0, 2])),
        GateCase("cy", 3, cy(0, [2]), qml.CY(wires=[2, 0])),
        GateCase("cz", 3, cz(2, [1]), qml.CZ(wires=[1, 2])),
        GateCase("crx", 3, crx(a, 1, [0]), qml.CRX(a, wires=[0, 1])),
        GateCase("cry", 3, cry(b, 0, [2]), qml.CRY(b, wires=[2, 0])),
        GateCase("crz", 3, crz(c, 2, [1]), qml.CRZ(c, wires=[1, 2])),
        GateCase("swap", 3, swap(0, 2), qml.SWAP(wires=[0, 2])),
        GateCase("rzz", 3, rzz(a, 0, 2), qml.IsingZZ(a, wires=[0, 2])),
        GateCase("rxx", 3, rxx(b, 1, 2), qml.IsingXX(b, wires=[1, 2])),
        GateCase("ms_gate", 3, ms_gate(c, 0, 1), qml.IsingXX(c, wires=[0, 1])),
        GateCase("molmer_sorensen", 3, molmer_sorensen(a, 0, 2), qml.IsingXX(a, wires=[0, 2])),
        GateCase("toffoli", 3, toffoli(2, [0, 1]), qml.Toffoli(wires=[0, 1, 2])),
        GateCase("ccnot", 3, ccnot(0, [1, 2]), qml.Toffoli(wires=[1, 2, 0])),
    ]


def _format_matrix(matrix: np.ndarray) -> str:
    return np.array2string(
        matrix,
        precision=8,
        suppress_small=False,
        max_line_width=160,
    )


def _report_block(case: GateCase, aicir_matrix: np.ndarray, pennylane_matrix: np.ndarray, max_diff: float) -> str:
    return "\n".join(
        [
            f"Gate: {case.name}",
            f"n_qubits: {case.n_qubits}",
            f"max|diff|: {max_diff:.12e}",
            "PennyLane matrix:",
            _format_matrix(pennylane_matrix),
            "aicir matrix:",
            _format_matrix(aicir_matrix),
            "",
        ]
    )


def validate_cases(atol: float, rtol: float, output_path: str | Path | None = DEFAULT_OUTPUT_PATH) -> list[tuple[GateCase, float]]:
    failures: list[tuple[GateCase, float]] = []
    report_blocks = [
        "aicir basic gate matrix verification against PennyLane",
        "PennyLane matrices use wire_order=[0, 1, ...] to match aicir basis order.",
        "",
    ]
    for case in gate_cases():
        aicir_matrix = _aicir_matrix(case)
        pennylane_matrix = _pennylane_matrix(case)
        max_diff = float(np.max(np.abs(aicir_matrix - pennylane_matrix)))
        if not np.allclose(aicir_matrix, pennylane_matrix, atol=atol, rtol=rtol):
            failures.append((case, max_diff))
        print(f"{case.name:<18} dim={aicir_matrix.shape[0]:>2} max|diff|={max_diff:.3e}")
        report_blocks.append(_report_block(case, aicir_matrix, pennylane_matrix, max_diff))
    summary = (
        f"All {len(gate_cases())} aicir basic gate matrix checks match PennyLane."
        if not failures
        else f"{len(failures)} aicir basic gate matrix check(s) failed."
    )
    report_blocks.append(summary)
    report_blocks.append("")

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(report_blocks), encoding="utf-8")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="matrix report output path (default: tests/gates/verify_pennylane_gate_matrices.txt)",
    )
    args = parser.parse_args(argv)

    failures = validate_cases(atol=args.atol, rtol=args.rtol, output_path=args.output)
    if failures:
        print("\nFAILED gate matrix checks:", file=sys.stderr)
        for case, max_diff in failures:
            print(f"- {case.name}: max|diff|={max_diff:.6e}", file=sys.stderr)
        return 1

    print(f"\nAll {len(gate_cases())} aicir basic gate matrix checks match PennyLane.")
    print(f"Matrix report written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
