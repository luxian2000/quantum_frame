# my_frame

一个用于量子门矩阵构造与简单量子线路计算的 Python 仓库，提供两套实现：

- PyTorch 实现
- MindSpore 实现

代码风格已按 PEP 8 统一（函数与变量采用 snake_case，常量采用 UPPER_SNAKE_CASE）。

## 项目结构

- [basic_torch.py](basic_torch.py) — PyTorch 的基础量子门与矩阵工具（新增 `partial_trace`）。
- [define_torch.py](define_torch.py) — PyTorch 的线路层封装（以 `Circuit` 为主API，保留 `circuit()` 兼容别名）。
- [basic_mind.py](basic_mind.py) — MindSpore 的基础量子门与矩阵工具（新增 `partial_trace`）。
- [define_mind.py](define_mind.py) — MindSpore 的线路层封装（与 Torch 对齐，提供 `Circuit`）。
- [run_tests.py](run_tests.py) — 统一测试入口脚本。

## 新增与改动要点

- `Circuit` 类（在 `define_torch.py` 与 `define_mind.py` 中）：
  - 面向对象的线路表示，构造时**必须显式传入** `n_qubits`。
  - 支持 `+` 运算符表示“门序拼接”（`c1 + c2`），拼接前会检查 `n_qubits` 相等，否则抛出 `ValueError`。
  - 提供 `append()` / `extend()` 增量构建方法，以及 `unitary()` / `matrix()` 生成整体矩阵。
  - 为了兼容旧代码，函数式接口 `circuit(..., n_qubits=...)` 仍然可用，内部委托给 `Circuit(...).unitary()`。

- `partial_trace`（在 `basic_torch.py` 与 `basic_mind.py` 中）：
  - 对方阵密度矩阵执行偏迹（reduced density matrix）。
  - 接口示例（PyTorch）：
    - `partial_trace(rho, keep, n_qubits=None)`
      - `rho`: 形状为 `(2^N, 2^N)` 的密度矩阵
      - `keep`: 要保留的量子比特索引（int 或 list）
      - `n_qubits`: 可选，总量子比特数（不传时从 `rho` 维度推断）

## 快速示例（PyTorch）

Circuit 用法：

```python
from define_torch import Circuit, pauli_x, pauli_y, cnot

c1 = Circuit(pauli_x(0), pauli_y(1), n_qubits=2)
c2 = Circuit(cnot(1, [0]), n_qubits=2)
c_big = c1 + c2          # 门序拼接（先 c1 再 c2）
u_big = c_big.unitary()  # 获取整体矩阵
```

兼容旧接口：

```python
from define_torch import circuit, pauli_x, pauli_y, cnot
u = circuit(pauli_x(0), pauli_y(1), cnot(1, [0]), n_qubits=2)
```

partial_trace 用法：

```python
from basic_torch import KET_0, KET_1, tensor_product, dagger, partial_trace

psi01 = tensor_product(KET_0, KET_1)          # |0>|1>
rho01 = psi01 @ dagger(psi01)                 # 密度矩阵 |01><01|
rho_keep0 = partial_trace(rho01, keep=[0], n_qubits=2)  # 保留第0比特 -> 得到 |0><0|
```

MindSpore 使用方法与 PyTorch 对齐（参见 `basic_mind.py` / `define_mind.py`）。

## 运行测试

在仓库根目录运行：

```bash
python run_tests.py
```

- 脚本会优先执行 Torch 路径的测试。若系统未安装 `mindspore`，MindSpore 相关测试会被跳过（这是预期行为）。

## 迁移说明

- 推荐新代码使用 `Circuit` 类作为主入口，能更方便做增量构建与组合。
- 为了最小化侵入，`circuit(...)` 保持兼容，但 `Circuit` 提供更丰富的方法与类型检查（例如强制 `n_qubits`）。

## 贡献与开发

- 如需新增门或优化底层矩阵构造，优先在 `basic_torch.py` / `basic_mind.py` 添加门矩阵实现，再在 `define_*` 层补充封装与测试。

欢迎反馈改进建议。
