# WuYueSDK 与 quantum_frame/aicir 功能对比（基于当前源码）

> 对比范围：
>
> - WuYueSDK：`/Users/luxian/GitSpace/WuYueSDK/wuyue`
> - quantum_frame：`/Users/luxian/GitSpace/quantum_frame/aicir`
>
> 统计口径：仅按当前仓库已实现代码统计，不按规划或 README 愿景统计。

## 1. 总览结论

- 两者都具备：基础门电路构建、态矢量与密度矩阵测量路径、QASM 互转（子集）、噪声注入能力。
- WuYueSDK 的特点：
  - 电路/程序层功能更“工程化”，含 `QuantumProg`、`qif/qwhile`、受控子电路、读出噪声、批处理参数绑定、本地后端 + 作业接口。
  - 门与噪声种类更偏“类定义式”（每个门/噪声是类）。
- quantum_frame/aicir 的特点：
  - 计算后端抽象更清晰（`Numpy/Torch/NPU` 统一接口），测量与结果对象更现代化（`Result`、`run_batch`）。
  - 明确提供数据编码器（Amplitude/Angle/Basis）与 JSON/DAG/QASM 多种电路表示。

## 2. 能力矩阵（核心维度）

| 维度                      | WuYueSDK                                                                                                                                                                                                 | aicir                                                                                                                                                         |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 量子门（公开构建能力）    | 丰富：`H/X/Y/Z/T/S/RX/RY/RZ/P/U1/U2/U3/CNOT/CX/CZ/CCX/TOFFOLI/SWAP/IsingZZ/IZZ/MEASURE`，含 `BARRIER/RESET`                                                                                          | 丰富：`pauli_x/y/z, hadamard, rx/ry/rz, s/t, cx/cnot/cy/cz, crx/cry/crz, swap, toffoli/ccnot, u2/u3`；底层还支持 `unitary`、`identity`、`rzz` 字典门 |
| 门表示方式                | 面向对象门类（类 + matrix 属性）                                                                                                                                                                         | 门字典（`{"type": ...}`）+ `gate_to_matrix`                                                                                                              |
| 后端                      | `CPU`                                                                                                                                                                                                  | 统一抽象 `Backend`，实现 `NumpyBackend`、`TorchBackend`、`NPUBackend`                                                                                |
| 噪声通道                  | 单比特：`BitFlip/PhaseFlip/PauliChannel/Depolarizing/AmplitudeDamping/GeneralizedAmplitudeDamping/PhaseDamping`；双比特：`TwoQubitDepolarizing/TwoQubitDephasing`；另有 `ResetNoise`、读出噪声封装 | 当前内置：`DepolarizingChannel/BitFlipChannel/PhaseFlipChannel/AmplitudeDampingChannel`（单比特）                                                          |
| 噪声模型                  | `NoiseModel` 支持按门、按比特、全比特添加噪声；`full_noise_model` 便捷构造                                                                                                                           | `NoiseModel + NoiseRule`，支持“在指定门后施加通道”                                                                                                       |
| 测量                      | 后端 `run(shots)`；可取 `get_probs/get_states/get_density_matrix/expval_pauli`                                                                                                                       | `Measure.run`（态矢量）与 `run_density_matrix`（密度矩阵），返回统一 `Result`；支持 `run_batch`                                                      |
| 电路导入导出              | `from_qasm/QASM`；还支持伪代码 `from_compiler`                                                                                                                                                       | `circuit_to_qasm(2.0/3.0)`、`circuit_from_qasm`、JSON 序列化、DAG 转换                                                                                   |
| 编码器                    | 未见独立 `encoder` 模块                                                                                                                                                                                | 已实现 `AmplitudeEncoder`、`AngleEncoder`、`BasisEncoder`                                                                                              |
| 控制流/程序结构           | `QuantumProg` 提供 `qif/trueif/falseif/qwhile/addwhile`，支持受控子电路 `control(...)`                                                                                                             | 以电路与测量为主，未见 `qif/qwhile` 这类程序控制流对象                                                                                                     |
| 批处理能力                | 参数/电路存在 batch 概念（如门参数 batch、`set_batch`）                                                                                                                                                | `Measure.run_batch` 显式批量运行                                                                                                                           |
| 量子架构搜索（QAS）       | 未见独立 QAS 模块                                                                                                                                                                                        | 已实现 `qas`，含 `CRLQAS`、`PPR_DQL`、`PPO_RB` 等强化学习式架构搜索实现                                                                   |
| 量子线路优化（Optimizer） | 未见独立优化器模块，主要是手工电路构建与转换接口                                                                                                                                                         | 已实现 `optimizer/basic.py`，支持局部重写优化（门对消、旋转门合并、QASM 文本级优化）                                                                       |
| 作业系统                  | `job/job.py` 对接外部 QCOS Client（提交、查询、取消、删除任务）                                                                                                                                        | 当前仓库未见等价远程作业客户端                                                                                                                               |

## 3. 详细对比

## 3.1 量子门能力

### WuYueSDK

- 门类集中在 `wuyue/element/gate.py`，以类形式实现，含单比特、双比特、三比特、参数化门、测量门。
- `QuantumCircuit.gate_list` 与 `QuantumProg.gate_list` 都纳入了常用门集合，并支持 `BARRIER/RESET`。
- `IsingZZ` 与 `IZZ` 同时出现：前者在电路接口中使用，后者在 `gate.py` 中也有门类定义。

### aicir

- `aicir/core/circuit.py` 对外暴露门构造函数（返回 gate dict）。
- `aicir/core/gates.py` 的 `gate_to_matrix` 支持别名（如 `pauli_x` 与 `X`），也支持高级字典门：`unitary`、`identity`、`rzz`。
- 顶层 `aicir/__init__.py` 当前未把 `rzz` 作为便捷函数导出（但底层矩阵映射已支持）。

## 3.2 后端与执行模型

### WuYueSDK

- 后端入口 `Backend.get_device("Full amplitude"|"Density matrix")`。
- 全振幅后端提供：`run/get_states/get_density_matrix/get_probs/get_amps/get_phase/expval_pauli`。
- 密度矩阵后端提供：`apply_noise` 与 `run/get_density_matrix/get_probs/expval_pauli`。
- 量子比特规模限制：文档与代码中对不同路径有上限检查（如全振幅应用时 30 比特、密度矩阵后端 15 比特）。

### aicir

- `Backend` 抽象定义统一的张量、线代、测量、采样、期望值接口。
- 实现有 `NumpyBackend`、`TorchBackend`、`NPUBackend`，上层 `StateVector/DensityMatrix/Measure` 不感知底层框架。
- 更适合多算力后端切换与统一 API 复用。

## 3.3 噪声模型

### WuYueSDK

- 噪声类多、覆盖广，含单比特 + 双比特噪声。
- `NoiseModel` 可按：
  - 指定门 + 指定位（`add_one_qubit_error`）
  - 多比特门 + 指定位组（`add_mul_qubit_error`）
  - 指定门 + 全比特（`add_all_qubit_error`）
- 支持 `Readout_Noise`（测量读出噪声）插入。

### aicir

- 当前实现的是单比特噪声通道集合（4 种）。
- `NoiseModel` 通过规则匹配 `after_gates` 在门后逐步作用到密度矩阵。
- 噪声框架扩展性较好（`NoiseChannel` 抽象），但当前内置通道种类少于 WuYueSDK。

## 3.4 编码（Encoding）

### WuYueSDK

- 未发现独立编码器目录或统一 `encode/decode` 抽象。
- 主要通过电路构建、参数绑定、状态设定（`set_state`）完成输入表达。

### aicir

- 明确提供编码器体系：
  - `AmplitudeEncoder`
  - `AngleEncoder`
  - `BasisEncoder`
- 编码输出可直接转 `dict/QASM/DAG`，并给出对应量子态对象。

## 3.5 测量与结果对象

### WuYueSDK

- 测量主要由后端 `run(shots)` 完成，返回计数字典。
- 分析类接口散落在后端方法中（概率、相位、振幅、期望等）。

### aicir

- `Measure` 统一测量入口，支持：
  - 态矢量路径 `run`
  - 密度矩阵路径 `run_density_matrix`（可接 `noise_model`）
  - 批量路径 `run_batch`
- `Result` 统一承载概率、计数、期望值、方差、末态、元数据。

## 3.6 编译/表示/互操作

### WuYueSDK

- `QuantumCircuit` 与 `QuantumProg` 均支持 `from_qasm` / `QASM`（OpenQASM 2.0 子集）。
- 还支持自定义伪代码导入 `from_compiler`。
- `QuantumProg` 额外支持控制流结构（if/while）并能体现在程序绘图中。

### aicir

- I/O 侧更模块化：
  - QASM 2.0/3.0 导入导出
  - JSON 序列化
  - DAG 转换（用于图特征）
- QASM 说明明确标注了不支持项（如 `if/reset/opaque/自定义 gate` 在导入侧多为跳过或不支持）。

## 3.7 QAS 与线路优化模块专项检查

### quantum_frame/aicir

- `qas` 为实装模块，不是纯占位：
  - `CRLQAS.py`：DDQN + Adam-SPSA + curriculum 机制做架构搜索。
  - `PPR_DQL.py`：DQN + 经验回放 + policy reuse（持续学习）做态制备架构搜索。
  - `PPO_RB.py`：TR-PPO rollback 风格的架构搜索实现。
- `optimizer/basic.py` 为实装优化器：
  - 针对 gate dict：做相邻门对消（如 `X/X`、`H/H`、`CNOT/CNOT`）与 `rx/ry/rz` 参数合并。
  - 针对 QASM 文本：支持可解析语句的局部重写与简化。

### WuYueSDK/wuyue

- 未发现与 `qas` 对等的“量子架构搜索”框架模块：
  - `wuyue` 目录下无 `algorithms` 或 `qas` 子包。
  - `wuyue/example/algorithm.py` 是算法示例（QFT/Grover/Shor 等），非通用搜索训练框架。
- 未发现与 `optimizer/basic.py` 对等的“线路优化器”模块：
  - `wuyue` 目录下无独立 `optimizer` 目录/文件。
  - 电路相关代码以构建、导入导出、绘制、参数绑定为主。

结论：在“QAS（量子架构搜索）”与“线路自动优化器”这两个点上，当前 `aicir` 明显领先于 `wuyue` 的模块化实现深度。

## 4. “已实现”与“占位”观察

- WuYueSDK：模块相对集中在模拟器、电路/程序、噪声、可视化、作业接口，偏“功能完整型 SDK”。
- aicir：核心模拟/测量/编码较完善；同时在 `qas` 与 `optimizer/basic` 上已有实装。提升后的算法子包中除 `qas` 外，多数子目录当前仍以 `__init__.py` 占位为主。

## 5. 结论建议（按选型场景）

- 如果你要：
  - 直接做量子程序控制流（if/while）、读出噪声、QCOS 作业流集成：WuYueSDK 更合适。
- 如果你要：
  - 统一多后端（NumPy/Torch/NPU）、结构化测量结果、编码器与多表示互转（QASM/JSON/DAG）：aicir 更合适。
- 如果你要统一两者：
  - 可考虑以 aicir 的后端与测量抽象为主干，逐步吸收 WuYueSDK 的 `QuantumProg` 控制流与更丰富噪声库。

## 6. 关键证据文件

- WuYueSDK
  - `wuyue/element/gate.py`
  - `wuyue/element/noise.py`
  - `wuyue/circuit/noise_model.py`
  - `wuyue/circuit/circuit.py`
  - `wuyue/programe/prog.py`
  - `wuyue/backend/backend.py`
  - `wuyue/backend/backend_amplitude.py`
  - `wuyue/backend/backend_density_matrix.py`
  - `wuyue/job/job.py`
  - `wuyue/example/algorithm.py`
- quantum_frame/aicir
  - `aicir/core/circuit.py`
  - `aicir/core/gates.py`
  - `aicir/channel/backends/base.py`
  - `aicir/channel/backends/numpy_backend.py`
  - `aicir/channel/backends/torch_backend.py`
  - `aicir/channel/backends/npu_backend.py`
  - `aicir/channel/noise/base.py`
  - `aicir/channel/noise/channels.py`
  - `aicir/channel/noise/model.py`
  - `aicir/measure/measure.py`
  - `aicir/encoder/amplitude.py`
  - `aicir/encoder/angle.py`
  - `aicir/encoder/basis.py`
  - `aicir/qas/CRLQAS.py`
  - `aicir/qas/PPR_DQL.py`
  - `aicir/qas/PPO_RB.py`
  - `aicir/optimizer/basic.py`
  - `aicir/core/io/qasm.py`
  - `aicir/core/io/json_io.py`
  - `aicir/core/io/dag.py`
