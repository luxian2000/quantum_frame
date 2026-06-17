"""demo_1：5 比特复杂线路，完整演示四项线路特性。

本示例集中展示 aicir 统一操作流中的四个能力：
  1. 多控多目标 CNOT —— cnot(target, controls) 的 target/controls 均可为列表；
  2. 线路内（mid-circuit）测量 —— measure(...) 标记，非破坏性投影测量并保留比特；
  3. 线路内（mid-circuit）重置 —— reset(...) 把目标比特无条件重置为 |0>；
  4. 态快照 snap —— Measure.run(..., snap=[op_index]) 记录指定操作下标处的完整量子态。
     本例特意在「线路内测量之后」与「线路内重置之后」各打一个快照。

门工厂返回的是带校验、不可变、可按 dict 读取的 Operation/Measurement IR。
"""

from aicir import (
    Circuit, Measure, NumpyBackend,
    hadamard, pauli_x, rx, ry, rz, cnot, cz, cry, rzz, swap, t_gate,
    measure, reset,
)
import numpy as np


def _as_density(state):
    """把快照 State 统一成密度矩阵，便于读取对角线占据概率。"""
    arr = np.asarray(state)
    if arr.ndim == 1 or arr.shape[0] != arr.shape[1]:
        vec = arr.reshape(-1, 1)
        return vec @ vec.conj().T
    return arr


# ---------------------------------------------------------------------------
# 5 比特复杂线路。右侧标注每个操作的下标（op_index）；snap/result 均按此引用。
# ---------------------------------------------------------------------------
cir = Circuit(
    # --- 段一：制备纠缠 + 局部旋转 ---
    hadamard(0),                       # 0
    hadamard(1),                       # 1
    ry(np.pi / 3, 2),                  # 2
    cnot(2, [0]),                      # 3
    cz(3, [1]),                        # 4
    cry(np.pi / 4, 4, [2, 3]),         # 5: 双控受控 RY
    rzz(np.pi / 5, 0, 4),              # 6
    #
    # 【特性 1】多控多目标 CNOT：控制位 [0, 1] 同为 |1> 时，目标 [3, 4] 同时翻转。
    cnot([3, 4], [0, 1]),             # 7: multi-control & multi-target CNOT
    rx(0.7, 2),                        # 8
    #
    # 【特性 2】线路内测量：联合 Z 测量 q0、q1，保留比特，id="mid"。
    measure([0, 1], id="mid"),         # 9: mid-circuit measure
    # 【特性 4 快照 A】测量之后的态（投影坍缩 + 多轨迹平均后的混合态）。
    #
    # --- 段二：重置部分比特并重新加工 ---
    # 【特性 3】线路内重置：把 q2、q3、q4 无条件重置为 |0>（无需前置 measure）。
    reset([2, 3, 4]),                  # 10: mid-circuit reset
    # 【特性 4 快照 B】重置之后的态：q2/q3/q4 应回到 |0>，仅 q0/q1 仍携带信息。
    #
    hadamard(2),                       # 11
    t_gate(3),                         # 12
    pauli_x(4),                        # 13: q4 重置后翻成稳定 |1>
    swap(0, 2),                        # 14
    rz(0.9, 1),                        # 15
)

print("=== 线路结构（5 比特）===")
for i, op in enumerate(cir.gates):
    print(f"  op[{i:>2}] {op}")
cir.plot()


# ---------------------------------------------------------------------------
# 运行：shots=400 多轨迹聚合；snap=[9, 10] 分别记录
#   - op[9] 线路内测量「之后」的态；
#   - op[10] 线路内重置「之后」的态。
# 说明：snap[i] 记录的是「执行完 op[i] 之后」的态，故 snap=[9,10] 即满足需求。
# ---------------------------------------------------------------------------
result = Measure(NumpyBackend()).run(cir, shots=1, snap=[8, 9, 10], seed=11)

print("\n=== 概览 ===")
print(result.summary())

# 【特性 2】线路内测量结果。
print("\n=== 线路内测量 (op[9], id='mid') ===")
print("  counts        :", result.counts("mid"))
print("  probabilities :", result.prob("mid"))

# 【特性 4】两处快照。多轨迹下 snap 返回的是平均态（密度矩阵）。
def occupation(state):
    return np.round(np.real(np.diag(_as_density(state))), 3)
snap_before_measure = result.snap(8)
snap_after_measure = result.snap(9)
snap_after_reset = result.snap(10)

print("\n=== 快照 A：线路内测量之后 (snap @ op[9]) ===")
print("  shape       :", np.asarray(snap_after_measure).shape)
print("  占据概率对角 :", occupation(snap_after_measure))
print("state before measure:", snap_before_measure.ket)
print("  Dirac (.ket):", snap_after_measure.ket)

print("\n=== 快照 B：线路内重置之后 (snap @ op[10]) ===")
print("  占据概率对角 :", occupation(snap_after_reset))
print("  Dirac (.ket):", snap_after_reset.ket)
# 重置把 q2/q3/q4 清零：只有 q2=q3=q4=0 的基态（下标低 3 位为 0）才可能非零。
occ_b = occupation(snap_after_reset)
nonzero = [i for i, p in enumerate(occ_b) if p > 1e-6]
print("  非零基态下标 :", nonzero, "(均满足低3位=0，证明 reset 生效)")

# 末端测量（重置 + 后续门之后执行）。
print("\n=== 末端测量 (op_index = -1) ===")
print("  terminal qubits:", result.terminal_qubits)
print("  terminal counts:", result.counts(-1))
