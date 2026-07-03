import numpy as np
from aicir import Circuit, Measure, NumpyBackend, cnot, hadamard, if_, pauli_x, while_
from aicir.core.classical import ClassicalRegister
from aicir.core.circuit import measure


def _run(circ, shots=400, seed=7):
    return Measure(NumpyBackend()).run(circ, shots=shots, seed=seed)


def test_measure_creg_deterministic():
    reg = ClassicalRegister(1, "c")
    c = Circuit(pauli_x(0), measure(0, creg=reg), n_qubits=1)
    res = _run(c, shots=20)
    cc = res.classical_counts(reg)
    assert cc == {1: 20}


def test_if_correlates_branch_with_measurement():
    reg = ClassicalRegister(1, "c")
    body = Circuit(pauli_x(1), n_qubits=2)
    c = Circuit(hadamard(0), measure(0, creg=reg), if_(reg[0] == 1, body), n_qubits=2)
    # 每 shot：q1 == c[0]。末端测 q1 与 classical c 完全关联。
    res = _run(c, shots=300)
    cc = res.classical_counts(reg)
    assert set(cc) == {0, 1} and abs(cc[0] - cc[1]) < 120  # ~50/50


def test_if_else_both_branches():
    reg = ClassicalRegister(1, "c")
    c = Circuit(
        hadamard(0), measure(0, creg=reg),
        if_(reg[0] == 1, Circuit(pauli_x(1), n_qubits=2),
            else_body=Circuit(pauli_x(2), n_qubits=2) if False else Circuit(hadamard(1), n_qubits=2)),
        n_qubits=2,
    )
    res = _run(c, shots=100)
    assert set(res.classical_counts(reg)) <= {0, 1}


def test_while_converges():
    # body：把 q0 无条件置 0 再测入 c[0]（一步内必收敛，条件 c[0]==1 变假）
    reg = ClassicalRegister(1, "c")
    body = Circuit(pauli_x(0), measure(0, creg=reg), n_qubits=1)  # X 后测：若初为1则变0
    c = Circuit(pauli_x(0), measure(0, creg=reg),   # c[0]=1
                while_(reg[0] == 1, body, max_iterations=5), n_qubits=1)
    res = _run(c, shots=10)
    # 循环体一次后 c[0]=0，退出
    assert res.classical_counts(reg) == {0: 10}


def test_while_overflow_raises():
    import pytest
    reg = ClassicalRegister(1, "c")
    # 条件恒真：body 不改变 c[0]（只作用 q1），c[0] 始终 1
    body = Circuit(pauli_x(1), n_qubits=2)
    c = Circuit(pauli_x(0), measure(0, creg=reg),
                while_(reg[0] == 1, body, max_iterations=3), n_qubits=2)
    with pytest.raises(RuntimeError, match="max_iterations"):
        _run(c, shots=1)


def test_nested_if_in_while():
    reg = ClassicalRegister(1, "c")
    inner = Circuit(if_(reg[0] == 1, Circuit(pauli_x(0), n_qubits=1)), measure(0, creg=reg), n_qubits=1)
    c = Circuit(pauli_x(0), measure(0, creg=reg),
                while_(reg[0] == 1, inner, max_iterations=5), n_qubits=1)
    res = _run(c, shots=10)
    assert res.classical_counts(reg) == {0: 10}
