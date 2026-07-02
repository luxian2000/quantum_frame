"""demo_1：4 比特复杂线路，完整演示四项线路特性。

本示例集中展示 aicir 统一操作流中的四个能力：
  1. 多控多目标 CNOT —— cnot(target, controls) 的 target/controls 均可为列表；
  2. 线路内（mid-circuit）测量 —— measure(...) 标记，非破坏性投影测量并保留比特；
  3. 线路内（mid-circuit）重置 —— reset(...) 把目标比特无条件重置为 |0>；
  4. 态快照 snap —— Measure.run(..., snap=[op_index]) 记录指定操作下标处的完整量子态。
     本例特意在「测量和重置之前与之后」各打一个快照。

门工厂返回的是带校验、不可变、可按 dict 读取的 Operation/Measurement IR。
"""

from aicir import (
    Circuit, Measure, NumpyBackend,
    hadamard, pauli_x, rx, ry, rz, cnot, cz, cry, rzz, swap, t_gate,
    measure, reset,
)
import numpy as np


# ---------------------------------------------------------------------------
# 4 比特复杂线路。右侧标注每个操作的下标（op_index）；snap/result 均按此引用。
# ---------------------------------------------------------------------------
cir = Circuit(
    hadamard(0),
    hadamard(1),                      # 0
    cnot(2, [0, 1]),                 # 1
    measure([0, 1], id="mid"),            # 2: mid-circuit measure
    cnot(0, [2]),                      # 3
)

cir.plot()


# ---------------------------------------------------------------------------
# 运行：shots=1 多轨迹聚合；snap=[3, 4, 5] 分别记录
#   - op[3] 线路内测量「之前」的态；
#   - op[4] 线路内测量「之后」的态（也是重置之前）；
#   - op[5] 线路内重置「之后」的态。
# 说明：snap[i] 记录的是「执行完 op[i] 之后」的态，故 snap=[8, 9, 10] 即满足需求。
# ---------------------------------------------------------------------------
result = Measure(NumpyBackend()).run(cir, shots=1000, snap=[2, 3, 4])


# 【特性 4】快照。多轨迹下 snap 返回的是平均态（密度矩阵）。
snap_before_measure = result.snap(2)
snap_after_measure = result.snap(3)
snap_after_cnot = result.snap(4)

print("  Dirac (.ket):", snap_before_measure.ket)

print("  Dirac (.ket):", snap_after_measure.ket)

print("  Dirac (.ket):", snap_after_cnot.ket)
