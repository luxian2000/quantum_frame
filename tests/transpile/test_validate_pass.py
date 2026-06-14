"""ValidatePass 实质校验测试：qubit 越界、目标/控制冲突、重复比特。"""

import pytest

from aicir.core.circuit import Circuit, cx, hadamard, measure, rz
from aicir.ir import Operation
from aicir.transpile import PassManager, ValidatePass


def _circuit(*gates, n_qubits):
    return Circuit(*gates, n_qubits=n_qubits)


def test_valid_circuit_passes_and_is_equivalent():
    cir = _circuit(hadamard(0), rz(0.5, 1), cx(1, [0]), measure([0, 1]), n_qubits=2)
    out = ValidatePass().run(cir)
    assert out.n_qubits == cir.n_qubits
    assert out.gates == cir.gates


def test_target_qubit_out_of_range_raises():
    cir = _circuit(hadamard(3), n_qubits=2)
    with pytest.raises(ValueError, match="hadamard"):
        ValidatePass().run(cir)


def test_control_qubit_out_of_range_raises():
    cir = _circuit(cx(0, [5]), n_qubits=2)
    with pytest.raises(ValueError, match="cx"):
        ValidatePass().run(cir)


def test_measure_qubit_out_of_range_raises():
    cir = _circuit(hadamard(0), measure([0, 4]), n_qubits=2)
    with pytest.raises(ValueError):
        ValidatePass().run(cir)


def test_target_overlapping_control_raises():
    cir = _circuit(Operation("cx", qubits=(1,), controls=(1,)), n_qubits=2)
    with pytest.raises(ValueError, match="cx"):
        ValidatePass().run(cir)


def test_duplicate_target_qubits_raise():
    cir = _circuit(Operation("swap", qubits=(1, 1)), n_qubits=2)
    with pytest.raises(ValueError, match="swap"):
        ValidatePass().run(cir)


def test_validate_runs_inside_passmanager_by_name():
    good = _circuit(hadamard(0), cx(1, [0]), n_qubits=2)
    assert PassManager(["validate"]).run(good).gates == good.gates

    bad = _circuit(hadamard(9), n_qubits=2)
    with pytest.raises(ValueError):
        PassManager(["validate"]).run(bad)
