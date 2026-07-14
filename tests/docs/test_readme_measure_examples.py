"""README §4 示例测试 — 确保文档示例与真实 API 行为一致。

每个测试对应 README §4 的一个关键示例，断言输出精确匹配，
防止文档与实现静默偏离。
"""

import numpy as np
import pytest
from aicir import (
    Circuit, Measure, NumpyBackend,
    hadamard, cnot, pauli_x, measure, reset,
)
from aicir.core import State


def run(cir, **kw):
    return Measure(NumpyBackend()).run(cir, **kw)


# ---------- §4.3 线路内 measure：Bell ZZ 确定性 +1 且保持相干 ----------

def test_bell_zz_incircuit_deterministic_plus1_shots_none():
    """Bell 态 ZZ 联合投影：shots=None 时 output 为标量 +1（确定性），
    态矢仍是 Bell 态（未坍缩到计算基）。"""
    cir = Circuit(hadamard(0), cnot(1, [0]), measure([0, 1], id="zz"), n_qubits=2)
    r = run(cir, shots=None, measure_qubits=None)

    # output("zz") 和 output(2) 均应给出标量 +1
    assert r.output("zz") == 1
    assert r.output(2) == 1

    # Bell 态仍相干：振幅约为 [1/√2, 0, 0, 1/√2]
    sv = np.asarray(r.state).reshape(-1)
    expected = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=complex)
    assert np.allclose(np.abs(sv), np.abs(expected), atol=1e-5)


def test_bell_zz_incircuit_shots_m_output_shape_and_values():
    """Bell ZZ，shots=8：output(2) 形状 (8,1)，全为 +1；counts(2) 仅含 {1}。"""
    cir = Circuit(hadamard(0), cnot(1, [0]), measure([0, 1]), n_qubits=2)
    r = run(cir, shots=8, measure_qubits=None)

    out = r.output(2)
    assert out.shape == (8, 1)
    assert np.all(out == 1)

    cts = r.counts(2)
    # Bell 态是 ZZ 的 +1 本征态，-1 不应出现
    assert set(cts.keys()) <= {1}
    assert cts.get(1, 0) == 8


# ---------- §4.6 shots 语义：exact 模式 ----------

def test_shots_none_exact_no_terminal_final_equals_state():
    """shots=None：不做末端测量，final_state == state，output(-1) 报 ValueError。"""
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
    r = run(cir, shots=None)

    assert np.allclose(r.final_state, r.state, atol=1e-6)
    with pytest.raises(ValueError):
        r.output(-1)


def test_shots_none_counts_raises():
    """shots=None 时调用 counts 应报 RuntimeError（单轨迹模式）。"""
    cir = Circuit(hadamard(0), measure(0), n_qubits=1)
    r = run(cir, shots=None, measure_qubits=None)
    with pytest.raises(RuntimeError):
        r.counts(1)


# ---------- §4.6 shots=M：末端输出形状与 DM state ----------

def test_shots_m_terminal_output_shape_and_state_is_dm():
    """shots=16，Bell 态全比特末端测量：output(-1) 形状 (16,2)，
    counts(-1) ⊆ {"00","11"}；无噪声共享纯态前态下 state 保持向量形态，
    末端测量后的 final_state 为 (4,4) 密度矩阵（真混合态）。"""
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
    r = run(cir, shots=16)

    out = r.output(-1)
    assert out.shape == (16, 2)

    cts = r.counts(-1)
    assert set(cts.keys()) <= {"00", "11"}
    assert sum(cts.values()) == 16

    assert np.asarray(r.state).shape == (4,)
    assert np.asarray(r.final_state).shape == (4, 4)


# ---------- §4.5 末端输出顺序：保留 measure_qubits 输入顺序 ----------

def test_terminal_output_order_preserved():
    """measure_qubits=[1,0]：qubit1=|1>→-1 在前，qubit0=|0>→+1 在后。"""
    cir = Circuit(pauli_x(1), n_qubits=2)
    r = run(cir, shots=1, measure_qubits=[1, 0])

    assert r.output(-1).tolist() == [[-1, 1]]


# ---------- §4.4 reset 信道：纠缠比特 reset → DM ----------

def test_reset_entangled_qubit_yields_density_matrix():
    """Bell 态 qubit0 施加 reset：snap(2) 应升级为 (4,4) 密度矩阵。"""
    cir = Circuit(hadamard(0), cnot(1, [0]), reset(0), n_qubits=2)
    r = run(cir, shots=None, snap=[2])

    snap = r.snap(2)
    arr = np.asarray(snap)
    assert arr.ndim == 2 and arr.shape == (4, 4), f"期望 (4,4) DM，实际形状 {arr.shape}"

    # 对角线：|00> 和 |01> 各 0.5（reset(0) 后 q0=|0>，q1 随机 |0>/|1>）
    diag = np.real(np.diag(arr))
    # 各计算基 |00>=idx0, |01>=idx1, |10>=idx2, |11>=idx3
    assert abs(diag[0] + diag[1] - 1.0) < 1e-5, f"对角线之和异常：{diag}"


# ---------- §4.7 snap + reduce 偏迹 ----------

def test_snap_bell_state_building():
    """snap([0,1]) 记录 H(0) 后与 Bell 态；snap(0) 叠加态，snap(1) 为 Bell 态。"""
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
    r = run(cir, shots=None, snap=[0, 1])

    s0 = np.asarray(r.snap(0)).reshape(-1)
    # H(0)|00> = (|00>+|10>)/√2，下标 0 和 2
    assert abs(abs(s0[0]) - 1 / np.sqrt(2)) < 1e-5
    assert abs(abs(s0[2]) - 1 / np.sqrt(2)) < 1e-5

    s1 = np.asarray(r.snap(1)).reshape(-1)
    # Bell 态 (|00>+|11>)/√2，下标 0 和 3
    assert abs(abs(s1[0]) - 1 / np.sqrt(2)) < 1e-5
    assert abs(abs(s1[3]) - 1 / np.sqrt(2)) < 1e-5


def test_reduce_partial_trace_bell():
    """shots=16 Bell 态，reduce([0], pos='state') 应得约化 DM ≈ I/2。"""
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
    r = run(cir, shots=16)

    red = r.reduce([0], pos="state")
    assert red.shape == (2, 2)
    expected = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
    assert np.allclose(red, expected, atol=0.15), f"偏迹异常：{red}"


# ---------- §4.8 期望值 ----------

def test_observables_expectation_x1_state():
    """|1> 态的 Z 期望值精确为 -1.0（exact 模式）。"""
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    cir = Circuit(pauli_x(0), n_qubits=1)
    r = run(cir, shots=None, observables={"Z0": Z})
    assert abs(r.expectation_values["Z0"] - (-1.0)) < 1e-6


# ---------- §4.9 State 直接测量 ----------

def test_state_measure_zero_state():
    """|00> 态的 measure 所有 shots 均应返回 '|00>'。"""
    sv = State.zero_state(2, NumpyBackend())
    counts = sv.measure(shots=4)
    assert counts == {"|00>": 4}
    probs = sv.probabilities()
    assert abs(probs[0] - 1.0) < 1e-6
