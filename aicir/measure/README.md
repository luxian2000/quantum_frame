# aicir.measure — 测量模型 + 经典控制流

本模块提供统一测量入口 `Measure`（`measure.py`）、结果对象 `Result`（`result.py`）、单轨迹执行器（`trajectory.py`）、shots 聚合（`aggregate.py`）与 shot-based Pauli 期望估计（`estimator.py`）。经典寄存器与 `if_`/`while_` 控制流（跨 `aicir.core.classical` + `aicir.core.circuit` + `aicir.ir.control_flow`）依赖本模块的测量轨迹执行器，因此一并在此说明。

---

## 1. 两种测量方式

- 显式指定读出比特：`Measure(backend).run(circuit, shots=..., measure_qubits=[...])`。
- 在 `Circuit` 内嵌入 `measure(*qubits)` 标记，`Measure.run()` 自动识别。

两者**互斥**：线路内已有 `measure()` 标记时再传 `measure_qubits` 会抛 `ValueError`。

---

## 2. 经典控制流（`ClassicalRegister` + `if_`/`while_`）

用测量结果驱动后续线路分支/循环，模型分三层：

- **经典寄存器**（`aicir.core.classical`）：`ClassicalRegister(size, name)` 声明一组经典位；`reg[i]` 取到 `Bit`；位比较用 `reg[i] == 0/1`，整寄存器比较用 `reg == N`（LSB=`reg[0]`），两者都返回 `Condition`，可直接传给 `if_`/`while_`。
- **写入**：`measure(qubits, creg=reg)` 按序把每个比特的 Z 基投影结果写入 `reg` 的前几位；也可用 `measure(qubits, cbits=[reg[1], reg[0]])` 显式指定每比特写到哪一位（`creg`/`cbits` 互斥，且有经典目标时只支持 Z 基）。
- **控制流指令**：`if_(condition, body, else_body=None)` / `while_(condition, body, *, max_iterations)`（`max_iterations` 必填；循环到达上限仍满足条件会抛 `RuntimeError`）。`body`/`else_body` 的 `n_qubits` 必须与外层线路一致。

**执行边界**：控制流指令只能在 `Measure.run()` 的测量轨迹路径上求值/递归执行（逐条轨迹读取当时的经典寄存器状态）。`Circuit.unitary()` 与张量网络引擎（`aicir.simulator`）遇到控制流指令一律抛 `ValueError` —— 它们没有"测量-反馈"语义，无法表示成酉矩阵或缩并成单个振幅。

读出用 `Result.classical_counts(reg)`：统计各轨迹末尾该寄存器整数取值的分布（从未写入过该寄存器的轨迹直接跳过，不计入分布）。

### 完整示例：H → measure → if（测量后条件翻转）

```python
from aicir import Circuit, Measure, NumpyBackend, hadamard, if_, ClassicalRegister
from aicir.core.circuit import measure, pauli_x

c = ClassicalRegister(1, "c")

qc = Circuit(n_qubits=1)
qc.append(hadamard(0))          # 制备叠加态
qc.append(measure(0, creg=c))   # 测量结果写入 c[0]

flip = Circuit(n_qubits=1)
flip.append(pauli_x(0))
qc.append(if_(c[0] == 1, flip))  # 测到 1 就翻转回 |0>

result = Measure(NumpyBackend()).run(qc, shots=200, seed=0)
print(result.classical_counts(c))  # 例如 {0: 87, 1: 113}（测量结果本身仍是随机的）
print(result.counts(-1))           # 例如 {'0': 200}（if_ 把 |1> 分支纠正回 |0>，末端读出恒为 0）
```

`while_` 用法类似，条件为真则重复执行 body（例如反复重测直到落在目标子空间），超过 `max_iterations` 仍未满足终止条件会报错，避免死循环：

```python
from aicir import while_

body = Circuit(n_qubits=1)
body.append(hadamard(0))
body.append(measure(0, creg=c))

qc2 = Circuit(n_qubits=1)
qc2.append(hadamard(0))
qc2.append(measure(0, creg=c))
qc2.append(while_(c[0] == 1, body, max_iterations=50))  # 反复重掷，直到测到 0

result2 = Measure(NumpyBackend()).run(qc2, shots=100, seed=1)
print(result2.classical_counts(c))  # {0: 100}
```

含控制流指令的电路也支持 `aicir.core.io` 的 JSON 往返（`circuit_to_json`/`circuit_from_json`），序列化后重新执行结果与原电路一致；QASM 导出暂不支持控制流指令。

---

## 3. Pauli 期望估计（`estimator.py`）

`PauliEstimator` 把 `Hamiltonian` 拆分成 qubit-wise commuting 分组（`group_pauli_terms`）、按组做基变换测量（`basis_change_gates`/`measurement_circuit`）、并按方差/项数分配 shots（`allocate_group_shots`），供 `aicir.vqc.BasicVQE(energy_estimator=...)` 等上层调用做有限 shots 能量估计。
