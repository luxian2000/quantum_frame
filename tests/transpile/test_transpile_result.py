"""PassManager.run_with_result → TranspileResult（NEXT.md 第 9 节）。"""

import numpy as np

from aicir import Circuit, cx, hadamard, pauli_x
from aicir.transpile import (
    CancelInversePass,
    LayoutPass,
    PassManager,
    TranspileResult,
)


def test_run_with_result_reports_depth_and_passes():
    # h·h 在 qubit 0 上自逆抵消：depth 3 → 1。
    circuit = Circuit(hadamard(0), hadamard(0), pauli_x(0), n_qubits=1)
    pm = PassManager([CancelInversePass()], fixed_point=True)

    result = pm.run_with_result(circuit)

    assert isinstance(result, TranspileResult)
    assert result.passes == ("CancelInversePass",)
    assert result.depth_before == 3
    assert result.depth_after == 1
    assert result.layout is None
    # circuit 与 run() 一致
    assert result.circuit.gates == pm.run(circuit).gates


def test_run_with_result_captures_layout():
    circuit = Circuit(cx(target_qubit=1, control_qubits=[0]), n_qubits=2)
    pm = PassManager([LayoutPass(initial_layout={0: 1, 1: 0})])

    result = pm.run_with_result(circuit)

    assert result.layout == {0: 1, 1: 0}
    assert "LayoutPass" in result.passes
    assert result.circuit.n_qubits == 2


def test_run_with_result_default_metadata_empty():
    circuit = Circuit(pauli_x(0), n_qubits=1)
    result = PassManager([CancelInversePass()]).run_with_result(circuit)
    assert dict(result.metadata) == {}
    assert result.depth_before == result.depth_after == 1
