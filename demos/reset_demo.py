"""演示 reset 作为信道的执行效果（统一测量模型）。

新模型要点：
- reset(*qubits) 是直接的重置信道，无需事先执行 measure。
- in-circuit measure() 现在是投影（坍缩）测量，与 reset 语义解耦。
- 对纠缠态的某比特施加 reset 会产生混合态（密度矩阵），可用 is_density 验证。

电路结构（3 量子比特，无 measure 门）：
    op 0: H(0)
    op 1: CNOT(1, [0])
    op 2: CNOT(2, [1])
    op 3: reset(1)      ← 直接信道，无需先测量；纠缠使结果升级为密度矩阵
    op 4: CNOT(1, [2])

期望状态（不含随机性，数学上精确）：
    snap(0): (|000> + |100>) / sqrt(2)           纯态
    snap(1): (|000> + |110>) / sqrt(2)           纯态
    snap(2): (|000> + |111>) / sqrt(2)           纯态（GHZ 态）
    snap(3): ρ = (|000><000| + |101><101|) / 2  混合态（密度矩阵）
    snap(4): ρ = (|000><000| + |111><111|) / 2  混合态（密度矩阵）
"""

from __future__ import annotations

import numpy as np

from aicir import Circuit, Measure, NumpyBackend, cnot, hadamard, reset
from aicir.core import State


def _flat_sv(state) -> np.ndarray:
    """把纯态快照展平为一维状态向量。"""
    return np.asarray(state).reshape(-1)


def _diag_dm(state) -> np.ndarray:
    """把密度矩阵快照展平为对角线（计算基概率）。"""
    arr = np.asarray(state)
    return np.real(np.diag(arr.reshape(8, 8)))


def _sv_ket(state, backend: NumpyBackend) -> str:
    """把纯态向量格式化为 Dirac 符号。"""
    return State.from_array(_flat_sv(state), backend=backend).ket


def build_circuit() -> Circuit:
    """构造 3 量子比特 reset 演示电路（无 measure 门，直接使用 reset 信道）。"""
    return Circuit(
        hadamard(0),   # op 0: 使 q0 进入叠加态
        cnot(1, [0]),  # op 1: 纠缠 q0-q1
        cnot(2, [1]),  # op 2: 纠缠 q1-q2，生成 GHZ 态
        reset(1),      # op 3: 对纠缠比特 q1 施加重置信道 → 混合态
        cnot(1, [2]),  # op 4: 继续演化，证明混合态可继续参与门操作
        n_qubits=3,
    )


def _basis_state(indices, values) -> np.ndarray:
    """构造指定计算基叠加的纯态向量（用于断言比较）。"""
    state = np.zeros(8, dtype=np.complex128)
    for index, value in zip(indices, values):
        state[index] = value
    return state


def _dm_from_states(vecs) -> np.ndarray:
    """构造等权混合密度矩阵 ρ = Σ_k |ψ_k><ψ_k| / K。"""
    rho = np.zeros((8, 8), dtype=np.complex128)
    for v in vecs:
        v = v.reshape(-1, 1)
        rho += v @ v.conj().T
    return rho / len(vecs)


def run_demo(*, verbose: bool = True) -> dict[str, object]:
    """运行演示并用 result.snap 验证 reset 的信道语义。

    返回包含各阶段快照和验证结果的字典，供测试调用。
    """
    backend = NumpyBackend()
    measurement = Measure(backend)
    circuit = build_circuit()
    # shots=None 为 exact 模式（单条确定性轨迹）；reset 无随机性，结果完全确定
    result = measurement.run(circuit, shots=None, snap=[0, 1, 2, 3, 4])

    # 读取各操作后的完整态快照
    snap_after_hadamard = result.snap(0)  # 纯态
    snap_after_cnot_10  = result.snap(1)  # 纯态
    snap_after_cnot_21  = result.snap(2)  # 纯态（GHZ 态）
    snap_after_reset    = result.snap(3)  # 混合态（密度矩阵）
    snap_after_cnot_12  = result.snap(4)  # 混合态（密度矩阵）

    # --- 验证纯态阶段 ---
    # snap(0): (|000> + |100>) / sqrt(2)，flat index 0=000, 4=100
    sv2 = 1.0 / np.sqrt(2.0)
    expected_sv0 = _basis_state([0, 4], [sv2, sv2])
    # snap(1): (|000> + |110>) / sqrt(2)，flat index 0=000, 6=110
    expected_sv1 = _basis_state([0, 6], [sv2, sv2])
    # snap(2): (|000> + |111>) / sqrt(2)，flat index 0=000, 7=111（GHZ 态）
    expected_sv2 = _basis_state([0, 7], [sv2, sv2])

    sv_ok = (
        np.allclose(_flat_sv(snap_after_hadamard), expected_sv0, atol=1e-6)
        and np.allclose(_flat_sv(snap_after_cnot_10),  expected_sv1, atol=1e-6)
        and np.allclose(_flat_sv(snap_after_cnot_21),  expected_sv2, atol=1e-6)
    )

    # --- 验证 reset 产生密度矩阵 ---
    # snap(3) 必须是密度矩阵（shape (8,8)）
    arr_reset = np.asarray(snap_after_reset)
    is_dm_after_reset = (arr_reset.ndim == 2 and arr_reset.shape == (8, 8))

    # snap(3): ρ = (|000><000| + |101><101|) / 2
    # |000>=index 0，|101>=index 5（101 binary = 5）
    expected_rho_reset = _dm_from_states([
        _basis_state([0], [1.0]),   # |000>
        _basis_state([5], [1.0]),   # |101>
    ])

    # snap(4) 也必须是密度矩阵
    arr_cnot12 = np.asarray(snap_after_cnot_12)
    is_dm_after_cnot12 = (arr_cnot12.ndim == 2 and arr_cnot12.shape == (8, 8))

    # snap(4): ρ = (|000><000| + |111><111|) / 2
    # CNOT(1,[2])：控制=q2，目标=q1；|101> → q2=1 触发翻转 q1 → |111>
    expected_rho_cnot12 = _dm_from_states([
        _basis_state([0], [1.0]),   # |000>
        _basis_state([7], [1.0]),   # |111>
    ])

    dm_ok = (
        is_dm_after_reset
        and is_dm_after_cnot12
        and np.allclose(arr_reset.reshape(8, 8),   expected_rho_reset,   atol=1e-6)
        and np.allclose(arr_cnot12.reshape(8, 8),  expected_rho_cnot12,  atol=1e-6)
    )

    reset_verified = sv_ok and dm_ok

    if verbose:
        print("=== reset 信道演示（无前置 measure，纠缠 → 混合态） ===")
        print()
        print("snap(0), H(0) 后（纯态）：", _sv_ket(snap_after_hadamard, backend))
        print("snap(1), CNOT(1,[0]) 后（纯态）：", _sv_ket(snap_after_cnot_10,  backend))
        print("snap(2), CNOT(2,[1]) 后（纯态，GHZ）：", _sv_ket(snap_after_cnot_21, backend))
        print()
        print("snap(3), reset(1) 后 —— 升级为密度矩阵：", is_dm_after_reset)
        print("  对角线（计算基概率）：", _diag_dm(snap_after_reset))
        print("  期望：|000> 和 |101> 各以概率 0.5 出现")
        print()
        print("snap(4), CNOT(1,[2]) 后 —— 仍为密度矩阵：", is_dm_after_cnot12)
        print("  对角线（计算基概率）：", _diag_dm(snap_after_cnot_12))
        print("  期望：|000> 和 |111> 各以概率 0.5 出现")
        print()
        print("纯态阶段验证（sv_ok）：", sv_ok)
        print("密度矩阵阶段验证（dm_ok）：", dm_ok)
        print("reset 验证总体结果：", reset_verified)

    if not reset_verified:
        raise AssertionError("reset snapshot verification failed")

    return {
        "circuit": circuit,
        "result": result,
        "snap_after_hadamard": snap_after_hadamard,
        "snap_after_cnot_10":  snap_after_cnot_10,
        "snap_after_cnot_21":  snap_after_cnot_21,
        "snap_after_reset":    snap_after_reset,
        "snap_after_cnot_12":  snap_after_cnot_12,
        "is_dm_after_reset":   is_dm_after_reset,
        "is_dm_after_cnot12":  is_dm_after_cnot12,
        "reset_verified":      reset_verified,
    }


def main() -> None:
    run_demo(verbose=True)


if __name__ == "__main__":
    main()
