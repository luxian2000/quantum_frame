"""Operation 构造期接入 GateSpec 校验：已注册门检查目标比特数/参数个数/控制位，未知门保持宽松。"""

import pytest

from aicir.core.circuit import cx, rz, swap, u3
from aicir.ir import Operation


def test_registered_gate_with_wrong_param_count_raises():
    with pytest.raises(ValueError, match="rz"):
        Operation("rz", qubits=(0,))  # 缺参数

    with pytest.raises(ValueError, match="u3"):
        Operation("u3", qubits=(0,), params=(0.1, 0.2))  # 少一个参数

    with pytest.raises(ValueError, match="hadamard"):
        Operation("hadamard", qubits=(0,), params=(0.5,))  # 多余参数


def test_registered_gate_with_wrong_qubit_count_raises():
    with pytest.raises(ValueError, match="rz"):
        Operation("rz", qubits=(0, 1), params=(0.5,))

    with pytest.raises(ValueError, match="swap"):
        Operation("swap", qubits=(0,))


def test_controlled_gate_without_controls_raises():
    with pytest.raises(ValueError, match="cx"):
        Operation("cx", qubits=(0,))


def test_from_dict_path_is_validated_too():
    with pytest.raises(ValueError, match="rx"):
        Operation.from_dict({"type": "rx", "target_qubit": 0})  # 缺 parameter


def test_unknown_gate_names_stay_permissive():
    op = Operation("my_custom_block", qubits=(0, 1, 2), params=(1.0, 2.0))
    assert op.qubits == (0, 1, 2)
    assert op.params == (1.0, 2.0)


def test_factories_still_construct_valid_operations():
    assert rz(0.5, 1).params == (0.5,)
    assert cx(1, [0]).controls == (0,)
    assert u3(0.1, 0.2, 0.3, 2).params == (0.1, 0.2, 0.3)
    assert swap(0, 3).qubits == (0, 3)


def test_symbolic_parameter_counts_as_one_param():
    from aicir.core.circuit import Parameter

    op = rz(Parameter("theta"), 0)
    assert len(op.params) == 1
