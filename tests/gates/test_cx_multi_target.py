"""多目标 cnot/cx：cx([t1, t2, ...], [controls]) 与单目标模式完全兼容。

多目标受控 X 在语义上等价于对每个目标分别施加同一组控制位的单目标 CX
（它们彼此对易），因此存储为单个多目标 Operation，模拟/导出时按目标展开。
"""

import numpy as np
import pytest

from aicir import Circuit, cnot, cx
from aicir.core.io import circuit_to_qasm
from aicir.ir import Operation


def test_single_target_factory_is_unchanged():
    op = cx(0, [1])
    assert isinstance(op, Operation)
    assert op.qubits == (0,)
    assert op.controls == (1,)
    assert op.control_states == (1,)


def test_multi_target_factory_returns_single_multi_qubit_op():
    op = cnot([0, 1], [2])
    assert isinstance(op, Operation)
    assert op.qubits == (0, 1)
    assert op.controls == (2,)
    assert op.control_states == (1,)


def test_multi_target_passes_control_states_through():
    op = cnot([0, 2], [3], [0])
    assert op.qubits == (0, 2)
    assert op.controls == (3,)
    assert op.control_states == (0,)


def test_multi_target_unitary_matches_separate_cx():
    multi = Circuit(cnot([0, 1], [2]), n_qubits=3)
    separate = Circuit(cnot(0, [2]), cnot(1, [2]), n_qubits=3)
    np.testing.assert_allclose(multi.unitary(), separate.unitary(), atol=1e-12)


def test_multi_target_state_matches_separate_cx():
    from aicir import State, hadamard

    multi = Circuit(hadamard(2), cnot([0, 1], [2]), n_qubits=3)
    separate = Circuit(hadamard(2), cnot(0, [2]), cnot(1, [2]), n_qubits=3)
    sv_multi = State.zero_state(3).evolve(multi.unitary())
    sv_separate = State.zero_state(3).evolve(separate.unitary())
    np.testing.assert_allclose(sv_multi.array, sv_separate.array, atol=1e-12)


def test_multi_target_qasm_export_decomposes():
    qasm = circuit_to_qasm(Circuit(cnot([0, 1], [2]), n_qubits=3))
    assert "cx q[2],q[0];" in qasm
    assert "cx q[2],q[1];" in qasm
