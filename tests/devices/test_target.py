"""Target 硬件能力描述测试：门集查询与连接拓扑。"""

import pytest

from aicir.devices import Target


def test_basis_gates_are_canonicalized_and_supports_uses_aliases():
    target = Target(n_qubits=2, basis_gates=("X", "cnot", "rz"))
    # 别名归一为规范名
    assert set(target.basis_gates) == {"pauli_x", "cx", "rz"}
    # 查询同样按规范名匹配，可用别名
    assert target.supports("X") and target.supports("pauli_x")
    assert target.supports("cnot") and target.supports("cx")
    assert not target.supports("hadamard")


def test_empty_basis_supports_any_gate():
    target = Target(n_qubits=2)
    assert target.supports("anything")


def test_coupling_map_is_undirected_and_neighbors_sorted():
    target = Target(n_qubits=4, coupling_map=[(0, 1), (1, 2), (2, 3)])
    assert target.coupled(0, 1) and target.coupled(1, 0)
    assert not target.coupled(0, 2)
    assert target.neighbors(1) == (0, 2)
    assert target.neighbors(0) == (1,)
    assert not target.fully_connected


def test_fully_connected_when_no_coupling_map():
    target = Target(n_qubits=3)
    assert target.fully_connected
    assert target.coupled(0, 2)
    assert not target.coupled(1, 1)
    assert target.neighbors(0) == (1, 2)


def test_invalid_target_rejected():
    with pytest.raises(ValueError):
        Target(n_qubits=0)
    with pytest.raises(ValueError):
        Target(n_qubits=2, coupling_map=[(0, 5)])
    with pytest.raises(ValueError):
        Target(n_qubits=2, coupling_map=[(1, 1)])
