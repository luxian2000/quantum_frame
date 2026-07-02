# 设计：完成 §5 qfun 测量返回构造器与 §7 GateSpec.matrix 字段

日期：2026-06-29
关联：`NEXT.md` §5（qfun）、§7（GateSpec）、第三阶段第 2 项（`expval`/`probs`/`sample`）

## 背景

`NEXT.md` 中 §5 与 §7 各剩一处**有意保留**的尾项：

- **§5**：qfun 当前在装饰器上用 `observable=` 声明观测量、函数体显式 `return Circuit`，
  因此未提供 §5 原草图中 `return expval(H)` 风格的测量返回构造器（设计取舍，规避全局 tape）。
  剩 `expval`（及 `probs`/`sample`）测量返回帮助器一项。
- **§7**：`GateSpec` 已落地 `name`/`num_*`/`aliases`/`qasm_name`/`symbol`/`generator`/
  `shift_rule`/`decomposition` 字段，但 `matrix` 字段未迁移——矩阵构造仍由
  `aicir.core.gates.gate_to_matrix` 负责。

本设计完成这两处尾项，遵循仓库"渐进迁移：先新增薄抽象，再逐步把算法层切过去"的原则。

## 目标与非目标

**目标**
- §5：新增 `expval`/`probs`/`sample` 三个测量返回构造器，函数体可返回测量对象表达测量意图。
- §7：新增 `GateSpec.matrix` 字段（Approach C），使**自定义不受控门**注册 `matrix=` 后即可
  经 `gate_to_matrix` 模拟，无需改动 core。

**非目标**
- §7 不重写内置门 `gate_to_matrix`/`apply_gate_to_state` 的分发链（那是 Approach A/B）。
  内置门的 numpy/torch 自动微分热路径与内存敏感的 local-apply 路径**保持不变**。
- §7 不为受控门（`cx`/`cy`/`cz`/`cr*`/`toffoli`）提供 `matrix` 局部矩阵（受控结构留在现有代码）。
- §5 `.grad` 仍仅对 `expval` 有效；`probs`/`sample` 为前向测量返回，不提供梯度。

## Part A — §5：qfun 测量返回构造器

### 新增测量返回对象

在 `aicir.qml`（qfun 同模块或邻近小模块）新增三个轻量 frozen dataclass，并从 `aicir.qml` 导出。
由于不依赖全局 tape，构造器**显式携带 circuit**：

- `expval(circuit, observable)` → 期望值（可微）
- `probs(circuit, wires=None)` → 概率向量
- `sample(circuit, wires=None)` → counts 字典（shots 取自装饰器 `shots=`）

### QFun 解析与分发

`QFun` 调用用户函数后按返回值类型解析：

- 返回裸 `Circuit` → 用**装饰器** `observable=`（现有行为不变），返回类型 = expval。
- 返回测量对象 → 取其 circuit + observable/wires + 返回类型；装饰器 `observable=` 变为可选。

`__call__` 按返回类型分发：

- `expval` → 现有期望值路径（单观测量返回 float，多观测量返回 `(n_obs,)` 数组）。
- `probs` → `Measure.run(...).probabilities`；给定 `wires` 时限制到对应比特（经 `measure_qubits`）。
- `sample` → `Result.counts`；`shots is None` 时报明确错误。

`.grad` 仅对 expval 返回类型有效；解析为 probs/sample 时抛出清晰错误。

### 行为变更（需记 CHANGELOG）

`observable=` 不再于**装饰期**强制要求（函数体可提供）。"缺少 observable" 的错误下移到
**调用期**，且仅在返回裸 `Circuit` 且无装饰器 observable 时触发。现有
`test_observable_required` 更新为断言调用期错误。其余保持向后兼容。

## Part B — §7：GateSpec.matrix 字段（Approach C）

### GateSpec.matrix 字段

`GateSpec` 新增可选字段 `matrix`：后端感知可调用对象，签名
`(params, backend) -> 局部稠密幺正矩阵`，作用于该门 `num_qubits` 个目标比特（比特顺序 =
门的比特列表顺序）；无参门的 builder 忽略 `params`。

为全部**不受控**标准门填充 builder（复用现有矩阵构造原语）：
`pauli_x/y/z`、`hadamard`、`s_gate`、`t_gate`、`identity`、`rx/ry/rz`、`u2/u3`、
`swap`、`rzz`、`rxx`、`single_excitation`、`double_excitation`。
受控门与 `unitary`/`measure`/`reset` 保持 `matrix=None`。

### 注册表访问器

`registry.py` 新增 `gate_matrix(name, params=(), backend=None)`：返回该门的**局部**矩阵
（纯元信息读取）；门未注册或 `matrix is None` 时返回 `None`。从 `aicir.gates` 导出。

### gate_to_matrix 回退

`gate_to_matrix` 的未知门 `else` 分支：查 `get_gate_spec(name).matrix`，若存在则构造局部
矩阵并经现有 `_expand_local_matrix_to_full` 按门的目标比特嵌入到整线路空间。
**内置门分发完全不变**——回退仅对自定义不受控门触发，使其无需改 core 即可模拟。

## 测试

### §5（扩展 `tests/qfun/test_qfun.py`）
- `expval` 函数体返回与装饰器 `observable=` 路径数值一致。
- `probs` 返回合法分布（和为 1，非负）；`wires` 限制生效。
- `sample` 返回 counts；`shots=None` 时报错。
- `.grad` 经 `expval` 函数体返回可用；对 `probs`/`sample` 返回类型报错。
- `observable_required` 更新为调用期语义。

### §7（新增 `tests/gates/` 测试）
- **一致性测试**（漂移护栏）：参数化遍历每个带 `matrix` builder 的标准门，断言嵌入后的
  `gate_matrix(...)` 与 `gate_to_matrix(...)` 在 **numpy 与 torch** 后端上一致。
- 自定义不受控门注册 `matrix=` 后端到端经 `gate_to_matrix` 模拟。
- `gate_matrix` 对 `matrix=None` 门（如受控门、`measure`）返回 `None`，调用方宽松处理。

## 文档

- 实现后更新 `aicir/qml/README.md` §16 qfun 用法：补充 `expval`/`probs`/`sample` 函数体返回风格，
  并修订"暂不提供 `expval` 帮助器"一句。
- `CHANGELOG.md` 记 §5 observable 检查时机变更、新增测量返回构造器、§7 `matrix` 字段。
- 实现后在 `NEXT.md` §5/§7/第三阶段相应位置标注落地状态。

## 设计取舍

- §5 测量构造器显式携带 circuit（而非 PennyLane 隐式 tape），与 qfun "返回 Circuit" 契约一致，
  规避门工厂队列化的侵入式改动。
- §7 选 Approach C 而非 B/A：C 已完整交付"自定义门注册一次即可在矩阵路径复用"的核心价值，
  内置门 autograd/local-apply 热路径零改动；一致性测试提供与 B 等价的防漂移保证，风险远低。
- §7 受控门不纳入局部矩阵抽象：受控嵌入非简单张量因子，留在现有专用代码。
