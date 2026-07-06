import numpy as np
import pytest
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


# ---- 回归测试：op_index/spec 记账修复（见 sdd/task-4-report.md 第二轮独立评审） ----


def test_snap_after_top_level_op_following_if():
    """Critical 1 回归：if_ 之后的顶层门必须保留其在 circuit_instructions 中的
    扁平下标，snap 到该下标不应 KeyError，且快照态须与手算结果一致。

    电路（n_qubits=2, op 下标）：
      0: X(0)                     -> q0=1
      1: measure(0, creg=c)       -> c[0]=1（确定性）
      2: if_(c[0]==1, X(1))       -> body 执行 X(1)，q1=1
      3: X(1)                     -> 顶层门，紧跟在 if 之后；翻回 q1=0
    修复前：body 内的 X(1) 会消耗共享 op_index_ref，顶层 X(1) 实际拿到的
    op_index 不是 3，导致 snap=3 要么 KeyError、要么记录了错误时刻的态。
    """
    reg = ClassicalRegister(1, "c")
    c = Circuit(
        pauli_x(0), measure(0, creg=reg),
        if_(reg[0] == 1, Circuit(pauli_x(1), n_qubits=2)),
        pauli_x(1),
        n_qubits=2,
    )
    res = Measure(NumpyBackend()).run(c, shots=None, snap=3)
    assert 3 in res.snapshot_states

    ref = Measure(NumpyBackend()).run(Circuit(pauli_x(0), n_qubits=2), shots=None)
    snap_arr = np.asarray(res.snap(3)).reshape(-1)
    ref_arr = np.asarray(ref.state).reshape(-1)
    assert np.allclose(snap_arr, ref_arr)


def test_trailing_measure_after_control_flow_counts():
    """Critical 1 回归：控制流块之后的顶层 measure(id=...) 必须拿到正确的
    扁平 op_index，否则 res.counts("m1") 会因 spec.op_index 与
    tr.incircuit 的键不一致而 KeyError。

    电路：X(0) -> measure(0,creg=c) -> if_(c[0]==1, X(1)) -> measure([1], id="m1")
    q1 在 if 分支后确定性地翻到 1，Z 基本征值约定 |1> -> -1。
    """
    reg = ClassicalRegister(1, "c")
    c = Circuit(
        pauli_x(0), measure(0, creg=reg),
        if_(reg[0] == 1, Circuit(pauli_x(1), n_qubits=2)),
        measure([1], id="m1"),
        n_qubits=2,
    )
    res = _run(c, shots=20)
    assert res.counts("m1") == {-1: 20}


def test_creg_measure_not_registered_as_joint_pauli_spec():
    """Critical 2 回归：creg 目标的 measure 门不应注册为联合 Pauli
    MeasureSpec，否则 shots=1 / shots=None(exact) 路径在 _build_result 中
    按 op_index 读取 tr.incircuit 时会 KeyError（_exec_ops 从不为 creg
    measure 写 incircuit，只写 classical store）。
    """
    reg = ClassicalRegister(1, "c")
    c = Circuit(pauli_x(0), measure(0, creg=reg), n_qubits=1)

    res_exact = Measure(NumpyBackend()).run(c, shots=None)
    assert res_exact.classical_counts(reg) == {1: 1}

    res_one = Measure(NumpyBackend()).run(c, shots=1, seed=0)
    assert res_one.classical_counts(reg) == {1: 1}


def test_snap_after_top_level_op_following_while():
    """Critical 1 回归（while 变体）：while_ 收敛后紧跟的顶层门必须保留
    扁平 op_index。虽然本用例 while 恒定迭代 1 次（跨 shot 稳定），但修复前
    共享计数器仍会把 body 内的操作计入顶层编号，使顶层 X(0) 的真实
    op_index 偏离 snap 请求的下标 3，导致 KeyError 或记录错误时刻的态。
    """
    reg = ClassicalRegister(1, "c")
    body = Circuit(pauli_x(0), measure(0, creg=reg), n_qubits=1)
    c = Circuit(
        pauli_x(0), measure(0, creg=reg),
        while_(reg[0] == 1, body, max_iterations=5),
        pauli_x(0),
        n_qubits=1,
    )
    res = Measure(NumpyBackend()).run(c, shots=None, snap=3)
    assert 3 in res.snapshot_states

    ref = Measure(NumpyBackend()).run(Circuit(pauli_x(0), n_qubits=1), shots=None)
    snap_arr = np.asarray(res.snap(3)).reshape(-1)
    ref_arr = np.asarray(ref.state).reshape(-1)
    assert np.allclose(snap_arr, ref_arr)
