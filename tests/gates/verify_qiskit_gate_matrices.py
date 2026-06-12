"""Validate aicir basic gate matrices against Qiskit.

Run from the repository root:

    PYTHONPATH=. python tests/gates/verify_qiskit_gate_matrices.py

Qiskit displays matrices in little-endian qubit order, while aicir uses q0 as
the left-most tensor factor.  Each aicir qubit q is therefore emitted to Qiskit
qubit n_qubits - 1 - q before comparing matrices.

The in-circuit measure marker is intentionally excluded because it is not a
unitary gate and has no matrix representation in aicir.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

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


QiskitEmitter = Callable[[QuantumCircuit, Callable[[int], int]], None]
DEFAULT_OUTPUT_PATH = Path(__file__).with_suffix(".txt")


@dataclass(frozen=True)
class GateCase:
    name: str
    n_qubits: int
    aicir_gate: object
    emit_qiskit: QiskitEmitter


def _qmap(n_qubits: int) -> Callable[[int], int]:
    return lambda qubit: n_qubits - 1 - int(qubit)


def _qiskit_matrix(case: GateCase) -> np.ndarray:
    qc = QuantumCircuit(case.n_qubits)
    case.emit_qiskit(qc, _qmap(case.n_qubits))
    return np.asarray(Operator(qc).data, dtype=np.complex128)


def _aicir_matrix(case: GateCase) -> np.ndarray:
    return np.asarray(gate_to_matrix(case.aicir_gate, cir_qubits=case.n_qubits), dtype=np.complex128)


def _single(method_name: str, target: int) -> QiskitEmitter:
    def emit(qc: QuantumCircuit, q):
        getattr(qc, method_name)(q(target))

    return emit


def _single_param(method_name: str, theta: float, target: int) -> QiskitEmitter:
    def emit(qc: QuantumCircuit, q):
        getattr(qc, method_name)(theta, q(target))

    return emit


def _controlled(method_name: str, control: int, target: int) -> QiskitEmitter:
    def emit(qc: QuantumCircuit, q):
        getattr(qc, method_name)(q(control), q(target))

    return emit


def _controlled_param(method_name: str, theta: float, control: int, target: int) -> QiskitEmitter:
    def emit(qc: QuantumCircuit, q):
        getattr(qc, method_name)(theta, q(control), q(target))

    return emit


def _pair(method_name: str, qubit_1: int, qubit_2: int) -> QiskitEmitter:
    def emit(qc: QuantumCircuit, q):
        getattr(qc, method_name)(q(qubit_1), q(qubit_2))

    return emit


def _pair_param(method_name: str, theta: float, qubit_1: int, qubit_2: int) -> QiskitEmitter:
    def emit(qc: QuantumCircuit, q):
        getattr(qc, method_name)(theta, q(qubit_1), q(qubit_2))

    return emit


def _u(theta: float, phi: float, lam: float, target: int) -> QiskitEmitter:
    def emit(qc: QuantumCircuit, q):
        qc.u(theta, phi, lam, q(target))

    return emit


def _ccx(control_1: int, control_2: int, target: int) -> QiskitEmitter:
    def emit(qc: QuantumCircuit, q):
        qc.ccx(q(control_1), q(control_2), q(target))

    return emit


def gate_cases() -> list[GateCase]:
    a = math.pi / 7
    b = math.pi / 5
    c = math.pi / 3
    return [
        GateCase("identity", 3, {"type": "identity", "n_qubits": 3}, lambda qc, q: None),
        GateCase("pauli_x", 3, pauli_x(0), _single("x", 0)),
        GateCase("pauli_y", 3, pauli_y(1), _single("y", 1)),
        GateCase("pauli_z", 3, pauli_z(2), _single("z", 2)),
        GateCase("hadamard", 3, hadamard(0), _single("h", 0)),
        GateCase("s_gate", 3, s_gate(1), _single("s", 1)),
        GateCase("t_gate", 3, t_gate(2), _single("t", 2)),
        GateCase("rx", 3, rx(a, 0), _single_param("rx", a, 0)),
        GateCase("ry", 3, ry(b, 1), _single_param("ry", b, 1)),
        GateCase("rz", 3, rz(c, 2), _single_param("rz", c, 2)),
        GateCase("u2", 3, u2(a, b, 1), _u(math.pi / 2.0, a, b, 1)),
        GateCase("u3", 3, u3(a, b, c, 2), _u(a, b, c, 2)),
        GateCase("cx", 3, cx(1, [0]), _controlled("cx", 0, 1)),
        GateCase("cnot", 3, cnot(2, [0]), _controlled("cx", 0, 2)),
        GateCase("cy", 3, cy(0, [2]), _controlled("cy", 2, 0)),
        GateCase("cz", 3, cz(2, [1]), _controlled("cz", 1, 2)),
        GateCase("crx", 3, crx(a, 1, [0]), _controlled_param("crx", a, 0, 1)),
        GateCase("cry", 3, cry(b, 0, [2]), _controlled_param("cry", b, 2, 0)),
        GateCase("crz", 3, crz(c, 2, [1]), _controlled_param("crz", c, 1, 2)),
        GateCase("swap", 3, swap(0, 2), _pair("swap", 0, 2)),
        GateCase("rzz", 3, rzz(a, 0, 2), _pair_param("rzz", a, 0, 2)),
        GateCase("rxx", 3, rxx(b, 1, 2), _pair_param("rxx", b, 1, 2)),
        GateCase("ms_gate", 3, ms_gate(c, 0, 1), _pair_param("rxx", c, 0, 1)),
        GateCase("molmer_sorensen", 3, molmer_sorensen(a, 0, 2), _pair_param("rxx", a, 0, 2)),
        GateCase("toffoli", 3, toffoli(2, [0, 1]), _ccx(0, 1, 2)),
        GateCase("ccnot", 3, ccnot(0, [1, 2]), _ccx(1, 2, 0)),
    ]


def _format_matrix(matrix: np.ndarray) -> str:
    return np.array2string(
        matrix,
        precision=8,
        suppress_small=False,
        max_line_width=160,
    )


def _report_block(case: GateCase, aicir_matrix: np.ndarray, qiskit_matrix: np.ndarray, max_diff: float) -> str:
    return "\n".join(
        [
            f"Gate: {case.name}",
            f"n_qubits: {case.n_qubits}",
            f"max|diff|: {max_diff:.12e}",
            "Qiskit matrix:",
            _format_matrix(qiskit_matrix),
            "aicir matrix:",
            _format_matrix(aicir_matrix),
            "",
        ]
    )


def validate_cases(atol: float, rtol: float, output_path: str | Path | None = DEFAULT_OUTPUT_PATH) -> list[tuple[GateCase, float]]:
    failures: list[tuple[GateCase, float]] = []
    report_blocks = [
        "aicir basic gate matrix verification against Qiskit",
        "Qiskit qubits are emitted as n_qubits - 1 - aicir_qubit to align basis order.",
        "",
    ]
    for case in gate_cases():
        aicir_matrix = _aicir_matrix(case)
        qiskit_matrix = _qiskit_matrix(case)
        max_diff = float(np.max(np.abs(aicir_matrix - qiskit_matrix)))
        if not np.allclose(aicir_matrix, qiskit_matrix, atol=atol, rtol=rtol):
            failures.append((case, max_diff))
        print(f"{case.name:<18} dim={aicir_matrix.shape[0]:>2} max|diff|={max_diff:.3e}")
        report_blocks.append(_report_block(case, aicir_matrix, qiskit_matrix, max_diff))
    summary = (
        f"All {len(gate_cases())} aicir basic gate matrix checks match Qiskit."
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
        help="matrix report output path (default: tests/gates/verify_qiskit_gate_matrices.txt)",
    )
    args = parser.parse_args(argv)

    failures = validate_cases(atol=args.atol, rtol=args.rtol, output_path=args.output)
    if failures:
        print("\nFAILED gate matrix checks:", file=sys.stderr)
        for case, max_diff in failures:
            print(f"- {case.name}: max|diff|={max_diff:.6e}", file=sys.stderr)
        return 1

    print(f"\nAll {len(gate_cases())} aicir basic gate matrix checks match Qiskit.")
    print(f"Matrix report written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
