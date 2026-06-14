# aicir 量子计算框架架构总结

## 📋 项目概览

**aicir** 是一个从零实现的量子线路模拟器与量子算法框架，提供态矢量/密度矩阵仿真、噪声模型、变分量子算法（VQE/QAOA/VQD/SSVQE）、量子架构搜索（QAS）、量子机器学习（QML）梯度计算以及 OpenQASM I/O 等功能。

- **版本**: 0.1.0
- **Python 要求**: >= 3.11
- **核心依赖**: NumPy（必需），PyTorch（可选，用于 GPU/NPU 后端）

---

## 🏗️ 整体架构设计

### 架构分层

```
┌─────────────────────────────────────────────────┐
│           应用层 (Applications)                  │
│  VQC (VQE/QAOA/VQD/SSVQE) | QAS | QML          │
├─────────────────────────────────────────────────┤
│           原语层 (Primitives)                    │
│  Estimator | Sampler | Measure                 │
├─────────────────────────────────────────────────┤
│           核心层 (Core)                          │
│  Circuit | State | Backend | Gates             │
├─────────────────────────────────────────────────┤
│         中间表示层 (IR)                          │
│  Operation | Measurement | Observable          │
├─────────────────────────────────────────────────┤
│          后端抽象层 (Backends)                   │
│  NumpyBackend | GPUBackend | NPUBackend        │
└─────────────────────────────────────────────────┘
```

---

## 📦 核心模块详解

### 1. **backends** - 计算后端抽象

**职责**: 提供统一的量子计算后端接口，支持多种硬件平台

#### 架构特点
- **抽象基类**: [`Backend`](file:///Users/luxian/GitSpace/quantum_frame/aicir/backends/base.py) 定义统一接口
- **多后端支持**:
  - [`NumpyBackend`](file:///Users/luxian/GitSpace/quantum_frame/aicir/backends/numpy_backend.py) - CPU 后端（基于 NumPy）
  - [`GPUBackend`](file:///Users/luxian/GitSpace/quantum_frame/aicir/backends/gpu_backend.py) - GPU 后端（基于 PyTorch）
  - [`NPUBackend`](file:///Users/luxian/GitSpace/quantum_frame/aicir/backends/npu_backend.py) - Ascend NPU 后端（基于 torch_npu）

#### 关键接口
```python
class Backend(ABC):
    # 张量工厂
    zeros(shape, dtype)      # 创建零张量
    eye(dim)                 # 创建单位矩阵
    cast(array, dtype)       # 类型转换
    to_numpy(tensor)         # 转为 NumPy
    
    # 量子态操作
    zeros_state(n_qubits)    # 初始化 |0...0⟩
    kron(a, b)              # Kronecker 积
    matmul(a, b)            # 矩阵乘法
    dagger(a)               # 共轭转置
    
    # 期望值计算
    expectation_sv(state, operator)
    expectation_dm(density_matrix, operator)
```

#### 设计模式
- **策略模式**: 不同后端实现相同接口
- **适配器模式**: `to_numpy()` 统一输出格式
- **装饰器模式**: NPU 后端的 fallback 机制

---

### 2. **core** - 核心数据结构

#### 2.1 [`State`](file:///Users/luxian/GitSpace/quantum_frame/aicir/core/state.py) - 量子态统一封装

**功能**:
- 纯态（态矢量）: shape `(2^n, 1)`
- 混合态（密度矩阵）: shape `(2^n, 2^n)`
- 不可变设计，返回新对象

**关键方法**:
```python
class State:
    @classmethod
    def from_array(array, backend)     # 从数组创建
    @classmethod
    def zero_state(n_qubits, backend)  # |0...0⟩ 态
    
    evolve(U)              # U|ψ⟩ 或 UρU†
    to_density_matrix()    # 转为密度矩阵
    probabilities()        # 测量概率
    partial_trace(keep)    # 偏迹（约化密度矩阵）
    
    @property
    def array   # 访问底层数据
    def matrix  # 密度矩阵形态
    def ket     # Dirac 符号表示
```

#### 2.2 [`Circuit`](file:///Users/luxian/GitSpace/quantum_frame/aicir/core/circuit.py) - 量子线路

**功能**:
- 存储门序列（内部为字典列表）
- 参数化支持（`Parameter` 占位符）
- 酉矩阵计算
- 后端绑定

**关键特性**:
```python
class Circuit:
    # 构造
    Circuit(*gates, n_qubits=N, backend=None)
    
    # 参数绑定
    bind_parameters(bindings, inplace=False)
    
    # 矩阵计算
    unitary(backend=None)  # 获取全局酉矩阵
    matrix()               # unitary() 的别名
    
    # 组合
    __add__(other)         # 线路拼接
    
    # 属性
    gates                  # 门字典列表（旧版兼容）
    operations             # typed IR 视图
    ir                     # CircuitIR 对象
    parameters             # 未绑定参数列表
    n_qubits              # 量子比特数
```

**门构造函数** (返回 [`Operation`](file:///Users/luxian/GitSpace/quantum_frame/aicir/ir/operation.py)):
- 单比特门: `hadamard`, `pauli_x/y/z`, `rx/ry/rz`, `s_gate`, `t_gate`, `u2`, `u3`
- 双比特门: `cx/cnot`, `cy`, `cz`, `swap`, `rzz`, `rxx`
- 受控门: `crx/cry/crz`
- 多控门: `toffoli/ccnot`
- 测量: `measure(qubits)` → [`Measurement`](file:///Users/luxian/GitSpace/quantum_frame/aicir/ir/measurement.py)

#### 2.3 [`BatchSV`](file:///Users/luxian/GitSpace/quantum_frame/aicir/core/batch.py) - 批量态矢量

**用途**: 深度学习场景的批量量子态演化

**特点**:
- 批量处理: `batch_size` 个态矢量并行演化
- 逐样本参数: 旋转门角度可以是 `(batch,)` 张量
- 端到端可微: 支持 PyTorch autograd
- NPU 安全: 使用实部/虚部分离避免 complex64 内核问题

```python
bsv = BatchSV(n_qubits=3, batch_size=8, backend=backend)
bsv.apply_gate(hadamard(0))
bsv.apply_gate(ry(enc_angles, 1))  # enc_angles: (8,) tensor
z = bsv.z_expectations()  # (8, 3)
```

#### 2.4 **io** - 输入输出子系统

**支持的格式**:
- **OpenQASM 2.0/3.0**: [`qasm.py`](file:///Users/luxian/GitSpace/quantum_frame/aicir/core/io/qasm.py)
  - `circuit_to_qasm()`, `circuit_from_qasm()`
  - `circuit_to_qasm3()`, `save/load_circuit_qasm()`
  
- **JSON**: [`json_io.py`](file:///Users/luxian/GitSpace/quantum_frame/aicir/core/io/json_io.py)
  - `circuit_to_json()`, `circuit_from_json()`
  
- **Qiskit**: [`qiskit_io.py`](file:///Users/luxian/GitSpace/quantum_frame/aicir/core/io/qiskit_io.py) (可选依赖)
  - `circuit_to_qiskit()`, `circuit_from_qiskit()`
  
- **PennyLane**: [`pennylane_io.py`](file:///Users/luxian/GitSpace/quantum_frame/aicir/core/io/pennylane_io.py) (可选依赖)
  - `circuit_to_pennylane()`, `circuit_from_pennylane()`
  
- **WuYue**: [`wuyue_io.py`](file:///Users/luxian/GitSpace/quantum_frame/aicir/core/io/wuyue_io.py) (可选依赖)
  - `circuit_to_wuyue()`, `circuit_from_wuyue()`

---

### 3. **ir** - 中间表示层 (Intermediate Representation)

**职责**: 提供类型化的电路指令表示

#### 核心类

##### [`Operation`](file:///Users/luxian/GitSpace/quantum_frame/aicir/ir/operation.py) - 量子门操作
```python
@dataclass(frozen=True)
class Operation(LegacyGateView):
    name: str              # 门名称
    qubits: tuple[int]     # 目标量子比特
    params: tuple          # 参数元组
    controls: tuple[int]   # 控制比特（可选）
    control_states: tuple[int]  # 控制态（可选）
    
    # 工厂方法
    @classmethod
    def from_dict(gate_dict)
    
    # 兼容旧版字典（只读）
    def to_dict()
    def __getitem__(key)
    def get(key, default)
```

##### [`Measurement`](file:///Users/luxian/GitSpace/quantum_frame/aicir/ir/measurement.py) - 测量指令
```python
@dataclass(frozen=True)
class Measurement:
    qubits: tuple[int]
    id: str | None
```

##### [`Observable`](file:///Users/luxian/GitSpace/quantum_frame/aicir/ir/observable.py) - 可观测量
```python
@dataclass
class Observable:
    pauli_strings: list[PauliString]
    coeffs: list[float]
    n_qubits: int | None
```

##### [`CircuitIR`](file:///Users/luxian/GitSpace/quantum_frame/aicir/ir/circuit_ir.py) - 线路级 IR
```python
@dataclass
class CircuitIR:
    instructions: list[CircuitInstruction]
    n_qubits: int
```

#### 设计优势
- **类型安全**: 构造期校验（目标比特数、参数个数等）
- **向后兼容**: `LegacyGateView`  mixin 提供旧字典读取接口
- **单一事实来源**: 通过 [`GateSpec`](file:///Users/luxian/GitSpace/quantum_frame/aicir/gates/spec.py) 注册表统一管理门元信息

---

### 4. **gates** - 门元信息注册表

**职责**: 门的元数据管理中心

#### [`GateSpec`](file:///Users/luxian/GitSpace/quantum_frame/aicir/gates/spec.py)
```python
@dataclass(frozen=True)
class GateSpec:
    name: str              # 规范名称
    num_qubits: int | None # 目标比特数
    num_params: int | None # 参数个数
    aliases: list[str]     # 别名
    qasm_name: str | None  # QASM 导出名
    symbol: str | None     # 绘图符号
```

#### 关键函数
```python
register_gate(spec)        # 注册新门
get_gate_spec(name)        # 查询门规格
canonical_gate_name(name)  # 规范化门名
```

**已注册门示例**:
- `hadamard`: 1 qubit, 0 params, symbol "H"
- `rz`: 1 qubit, 1 param, symbol "Rz"
- `cx`: 2 qubits (1 target + 1 control), 0 params
- `rzz`: 2 qubits, 1 param

---

### 5. **measure** - 测量子系统

**职责**: 量子测量和结果分析

#### 核心类

##### [`Measure`](file:///Users/luxian/GitSpace/quantum_frame/aicir/measure/measure.py) - 测量执行器
```python
class Measure:
    def __init__(self, backend=None)
    
    def run(circuit, shots=None, snap=None, bit_order="msb")
    # 返回 Result 对象
    
    def run_density_matrix(circuit, shots, ...)
    # 密度矩阵路径
```

**测量策略**:
- `shots=None`: 精确态矢量演化（无采样噪声）
- `shots=N`: 采样 N 次，返回统计结果
- `snap=[idx]`: 记录中间态（快照）

##### [`Result`](file:///Users/luxian/GitSpace/quantum_frame/aicir/measure/result.py) - 测量结果
```python
class Result:
    state              # 最终态（向量或密度矩阵）
    counts             # 计数字典 {"00": 512, "11": 512}
    probabilities      # 概率分布
    shots              # 采样次数
    
    snap(index)        # 获取快照
    reduce(qubits)     # 偏迹约化
```

##### [`PauliEstimator`](file:///Users/luxian/GitSpace/quantum_frame/aicir/measure/estimator.py) - Pauli 期望值估计
```python
class PauliEstimator:
    def estimate(circuit, observable, shots)
    # 返回 PauliEstimateResult
```

---

### 6. **operators** - 算符与哈密顿量

**职责**: 构建和操作量子算符

#### 核心类

##### [`PauliOp`](file:///Users/luxian/GitSpace/quantum_frame/aicir/operators.py) - Pauli 算符
```python
class PauliOp:
    label: str  # "I", "X", "Y", "Z"
```

##### [`PauliString`](file:///Users/luxian/GitSpace/quantum_frame/aicir/operators.py) - Pauli 串
```python
class PauliString:
    ops: list[PauliOp]
    coefficient: complex
    
    # 例如: 2.5 * X₀ ⊗ Z₁ ⊗ I₂
```

##### [`Hamiltonian`](file:///Users/luxian/GitSpace/quantum_frame/aicir/operators.py) - 哈密顿量
```python
class Hamiltonian:
    pauli_strings: list[PauliString]
    
    def to_matrix(n_qubits, backend)  # 转为矩阵
    def __add__(other)                 # 哈密顿量相加
    def __mul__(scalar)                # 标量乘法
```

**示例**:
```python
from aicir import Hamiltonian, PauliString, PauliOp

# H = 0.5 * Z₀Z₁ + 0.3 * X₀
H = Hamiltonian([
    PauliString([PauliOp("Z"), PauliOp("Z")], 0.5),
    PauliString([PauliOp("X"), PauliOp("I")], 0.3),
])
```

---

### 7. **vqc** - 变分量子电路算法

**包含算法**:

#### [`VQE`](file:///Users/luxian/GitSpace/quantum_frame/aicir/vqc/VQE.py) - 变分量子本征求解器
- 用于求解分子基态能量
- 支持 UCCSD、HEA 等 ansatz
- 经典优化器集成（SciPy）

#### [`QAOA`](file:///Users/luxian/GitSpace/quantum_frame/aicir/vqc/QAOA.py) - 量子近似优化算法
- 组合优化问题求解
- MaxCut 等图问题应用

#### [`VQD`](file:///Users/luxian/GitSpace/quantum_frame/aicir/vqc/VQD.py) - 变分量子对角化
- 求解激发态能量
- 正交性约束

#### [`SSVQE`](file:///Users/luxian/GitSpace/quantum_frame/aicir/vqc/SSVQE.py) - 子空间搜索 VQE
- 同时求解多个本征态

#### **ansatz** 子模块
- [`hea_ti`](file:///Users/luxian/GitSpace/quantum_frame/aicir/vqc/ansatz/hea_ti.py) - HEA-TI（硬件高效 ansatz + trapped-ion 演化）
- 全局演化酉矩阵生成
- 幂律耦合系数

---

### 8. **qas** - 量子架构搜索 (Quantum Architecture Search)

**职责**: 自动搜索最优量子电路结构

#### 核心组件

##### 搜索算法
- [`PPO_RB`](file:///Users/luxian/GitSpace/quantum_frame/aicir/qas/PPO_RB.py) - Proximal Policy Optimization with Reward Shaping
- [`PPR_DQL`](file:///Users/luxian/GitSpace/quantum_frame/aicir/qas/PPR_DQL.py) - Policy-Value Dual Q-Learning
- [`CRLQAS`](file:///Users/luxian/GitSpace/quantum_frame/aicir/qas/CRLQAS.py) - Curriculum Reinforcement Learning QAS

##### 关键模块
- [`supernet.py`](file:///Users/luxian/GitSpace/quantum_frame/aicir/qas/supernet.py) - 超网络定义（64KB，最大模块）
- [`search_env.py`](file:///Users/luxian/GitSpace/quantum_frame/aicir/qas/search_env.py) - 强化学习环境
- [`reward.py`](file:///Users/luxian/GitSpace/quantum_frame/aicir/qas/reward.py) - 奖励函数
- [`multi_objective_reward.py`](file:///Users/luxian/GitSpace/quantum_frame/aicir/qas/multi_objective_reward.py) - 多目标奖励
- [`evaluator.py`](file:///Users/luxian/GitSpace/quantum_frame/aicir/qas/evaluator.py) - 电路评估器
- [`architecture_candidates.py`](file:///Users/luxian/GitSpace/quantum_frame/aicir/qas/architecture_candidates.py) - 候选架构库

##### 工作流程
```
Agent (PPO/DQL) 
    ↓ 选择动作（添加门）
Environment (SearchEnv)
    ↓ 构建电路
Evaluator
    ↓ 计算保真度、深度、门数等指标
Reward Function
    ↓ 计算奖励
Agent 更新策略
```

---

### 9. **qml** - 量子机器学习梯度工具

**功能**: 参数移位法（Parameter-Shift）计算量子电路梯度

#### 核心函数
```python
from aicir.qml import parameter_shift, natural_gradient, dqng, rotosolve

# Parameter-Shift Rule
grad = parameter_shift(fn, params, shift=np.pi/2)

# Natural Gradient (使用 QFIM)
direction = natural_gradient(grad, qfim, damping=1e-5)

# Diagonal QFIM Natural Gradient
direction = dqng(grad, qfim_diag=diag_values)

# ROTOSOLVE (gradient-free)
params = rotosolve(objective_fn, params)
```

#### 特点
- **NPU 兼容**: 所有计算保持在设备端，避免 CPU 往返
- **自动微分集成**: 可与 PyTorch autograd 结合使用

---

### 10. **noise** - 噪声模型

**职责**: 模拟真实量子设备的噪声

#### 噪声通道
- [`DepolarizingChannel`](file:///Users/luxian/GitSpace/quantum_frame/aicir/noise/channels.py) - 去极化噪声
- [`BitFlipChannel`](file:///Users/luxian/GitSpace/quantum_frame/aicir/noise/channels.py) - 比特翻转
- [`PhaseFlipChannel`](file:///Users/luxian/GitSpace/quantum_frame/aicir/noise/channels.py) - 相位翻转
- [`AmplitudeDampingChannel`](file:///Users/luxian/GitSpace/quantum_frame/aicir/noise/channels.py) - 振幅阻尼

#### [`NoiseModel`](file:///Users/luxian/GitSpace/quantum_frame/aicir/noise/model.py)
```python
model = NoiseModel()
model.add_channel(DepolarizingChannel(prob=0.01), gate_types=["cx"])
model.add_channel(AmplitudeDampingChannel(gamma=0.005), gate_types=["all"])

# 应用到电路
noisy_result = Measure(backend).run(circuit, noise_model=model)
```

#### 离子阱噪声
- [`ion_trap.py`](file:///Users/luxian/GitSpace/quantum_frame/aicir/noise/ion_trap.py) - 离子阱特定噪声配置

---

### 11. **transpile** - 编译优化

**职责**: 量子电路编译和优化

#### 架构
- [`PassManager`](file:///Users/luxian/GitSpace/quantum_frame/aicir/transpile/passmanager.py) - Pass 管理器
- [`BasePass`](file:///Users/luxian/GitSpace/quantum_frame/aicir/transpile/base.py) - Pass 基类

#### 内置 Pass
- **ValidatePass**: 验证电路合法性
- **CanonicalizePass**: 规范化门名称
- **旋转合并**: 合并相邻同类旋转门（rx/ry/rz）
- **恒等消除**: 删除角度为 0 的门
- **安全重排**: 有限范围内的门交换优化

**示例**:
```python
from aicir.transpile import PassManager, ValidatePass, CanonicalizePass

pm = PassManager([
    ValidatePass(),
    CanonicalizePass(),
])
optimized_circuit = pm.run(circuit)
```

---

### 12. **encoder** - 数据编码

**功能**: 经典数据编码到量子态

#### 编码策略
- [`AngleEncoder`](file:///Users/luxian/GitSpace/quantum_frame/aicir/encoder/angle.py) - 角度编码
- [`AmplitudeEncoder`](file:///Users/luxian/GitSpace/quantum_frame/aicir/encoder/amplitude.py) - 振幅编码
- [`BasisEncoder`](file:///Users/luxian/GitSpace/quantum_frame/aicir/encoder/basis.py) - 基态编码

**示例**:
```python
from aicir.encoder import AngleEncoder

encoder = AngleEncoder(n_qubits=3)
circuit = encoder.encode([0.1, 0.2, 0.3])  # 返回 Circuit
```

---

### 13. **chemistry** - 量子化学

**功能**: 分子哈密顿量构建

#### [`Molecule`](file:///Users/luxian/GitSpace/quantum_frame/aicir/chemistry/molecule.py)
```python
from aicir.chemistry import Molecule

mol = Molecule.from_geometry("H2", geometry=[...], basis="sto-3g")
hamiltonian = mol.get_hamiltonian()  # 返回 Hamiltonian 对象
```

**支持**:
- PySCF 集成（可选依赖）
- Jordan-Wigner 变换
- Bravyi-Kitaev 变换

---

### 14. **visual** - 可视化

**功能**: 量子电路和状态可视化

#### 模块
- [`circuit.py`](file:///Users/luxian/GitSpace/quantum_frame/aicir/visual/circuit.py) - 电路绘制
- [`state.py`](file:///Users/luxian/GitSpace/quantum_frame/aicir/visual/state.py) - 态矢量可视化
- [`density.py`](file:///Users/luxian/GitSpace/quantum_frame/aicir/visual/density.py) - 密度矩阵热力图
- [`qas.py`](file:///Users/luxian/GitSpace/quantum_frame/aicir/visual/qas.py) - QAS 搜索结果可视化

**示例**:
```python
circuit.show()  # ASCII 艺术电路显示
circuit.plot()  # matplotlib 图形（需安装 matplotlib）
```

---

### 15. **primitives** - 量子原语

**职责**: 提供标准化的量子原语接口（类似 Qiskit Primitives）

#### 原语类型
- [`Estimator`](file:///Users/luxian/GitSpace/quantum_frame/aicir/primitives/estimator.py) - 期望值估计
- [`Sampler`](file:///Users/luxian/GitSpace/quantum_frame/aicir/primitives/sampler.py) - 采样

**接口**:
```python
from aicir.primitives import Estimator, Sampler

estimator = Estimator(backend)
result = estimator.run([(circuit, observable)]).result()

sampler = Sampler(backend)
result = sampler.run([circuit], shots=1024).result()
```

---

### 16. **metrics** - 电路度量

**功能**: 计算量子电路的各种度量指标

#### 度量类型
- **Expressibility**: 电路表达能力
- **Trainability**: 可训练性（barren plateau 检测）
- **Hardware Metrics**: 硬件相关指标（深度、门数等）
- **Noisy Expressibility**: 噪声下的表达能力

**示例**:
```python
from aicir.metrics import expressibility, trainability

expr = expressibility(circuit, num_samples=100)
train_score = trainability(circuit, backend)
```

---

### 17. **optimization** - 经典优化

**包含**:
- **QUBO** (Quadratic Unconstrained Binary Optimization)
  - 建模工具：Binary, Integer, Linear, Polynomial
  - 求解器：brute_force, simulated_annealing
  - 约束处理：equality, inequality, one-hot
  
- **SB** (Simulated Bifurcation)
  - 基于物理的优化算法

---

### 18. **optimizer** - 量子电路优化器

**功能**: 变分量子算法的参数优化

#### 组件
- [`circuit.py`](file:///Users/luxian/GitSpace/quantum_frame/aicir/optimizer/circuit.py) - 电路优化器
- [`params.py`](file:///Users/luxian/GitSpace/quantum_frame/aicir/optimizer/params.py) - 参数管理

---

### 19. **universal** - 通用量子算法

**包含**:
- [`qft.py`](file:///Users/luxian/GitSpace/quantum_frame/aicir/universal/qft.py) - 量子傅里叶变换

---

### 20. **devices** & **wireless** - 设备与无线通信

**状态**: 预留模块（当前为空或最小实现）

---

## 🎯 设计模式与架构原则

### 1. **分层架构**
```
用户接口层 → 原语层 → 核心层 → IR 层 → 后端层
```
每层职责清晰，依赖单向流动

### 2. **后端抽象**
- 所有数值计算委托给 Backend
- 上层代码不感知具体后端实现
- 易于扩展新硬件平台

### 3. **不可变性**
- `State`, `Operation`, `Measurement` 等核心类采用 frozen dataclass
- 函数式风格，返回新对象而非原地修改
- 便于推理和调试

### 4. **类型安全**
- PEP 604 联合类型 (`str | None`)
- 构造期校验（GateSpec）
- Typed IR 替代裸字典

### 5. **向后兼容**
- `LegacyGateView` mixin 提供旧字典接口
- 门构造函数签名保持不变
- 渐进式迁移路径

### 6. **可选依赖**
- Torch 仅在需要时导入
- Qiskit/PennyLane/WuYue 互转为可选功能
- 优雅降级机制

### 7. **单一事实来源**
- GateSpec 注册表统一管理门元信息
- 避免多处定义导致的不一致

---

## 🔄 典型工作流程

### 场景 1: 基础量子线路仿真

```python
from aicir import Circuit, NumpyBackend, State, hadamard, cnot, Measure

# 1. 构建电路
cir = Circuit(
    hadamard(0),
    cnot(1, [0]),
    n_qubits=2,
)

# 2. 选择后端
backend = NumpyBackend()

# 3. 执行测量
m = Measure(backend)
result = m.run(cir, shots=1024)

# 4. 分析结果
print(result.counts)       # {"00": 512, "11": 511}
print(result.probabilities) # {0: 0.5, 3: 0.5}
```

### 场景 2: 参数化电路与优化

```python
from aicir import Circuit, Parameter, rx, ry, GPUBackend
import torch

# 1. 创建参数化电路
theta = Parameter("θ")
phi = Parameter("φ")

cir = Circuit(
    rx(theta, 0),
    ry(phi, 1),
    n_qubits=2,
)

# 2. 绑定参数并获取酉矩阵
backend = GPUBackend(device="cpu")
bound_cir = cir.bind_parameters({"θ": 0.5, "φ": 0.3})
U = bound_cir.unitary(backend=backend)

# 3. 自动微分（直接使用 Torch 张量）
theta_t = torch.tensor(0.5, requires_grad=True)
cir_torch = Circuit(rx(theta_t, 0), n_qubits=1)
U_torch = cir_torch.unitary(backend=backend)
loss = torch.real(U_torch[0, 0])
loss.backward()
print(theta_t.grad)
```

### 场景 3: VQE 算法

```python
from aicir.vqc import VQE
from aicir.chemistry import Molecule
from aicir import NumpyBackend

# 1. 构建分子哈密顿量
mol = Molecule.from_geometry("H2", geometry=[...], basis="sto-3g")
hamiltonian = mol.get_hamiltonian()

# 2. 运行 VQE
backend = NumpyBackend()
vqe = VQE(hamiltonian, backend=backend)
energy, circuit, params = vqe.run()

print(f"基态能量: {energy}")
```

### 场景 4: 量子架构搜索

```python
from aicir.qas import config, run
import numpy as np

# 1. 定义目标态（GHZ 态）
target = np.zeros((8, 1), dtype=np.complex64)
target[0, 0] = 1 / np.sqrt(2)
target[7, 0] = 1 / np.sqrt(2)
rho_target = target @ target.conj().T

# 2. 配置搜索
cfg = config.ppo_rb(episode_num=800, max_steps_per_episode=8)

# 3. 执行搜索
theta, circuit = run("ppo_rb", target_density_matrix=rho_target, config=cfg)

print(f"找到的电路: {len(circuit.gates)} 个门")
print(circuit.show())
```

---

## 📊 技术栈总结

| 层级 | 技术 | 用途 |
|------|------|------|
| **计算后端** | NumPy | CPU 串行计算 |
| | PyTorch | GPU 并行计算 + 自动微分 |
| | torch_npu | Ascend NPU 加速 |
| **核心语言** | Python 3.11+ | 主要开发语言 |
| **类型系统** | typing, dataclass | 类型安全和结构化数据 |
| **可选依赖** | Qiskit | 量子生态互操作 |
| | PennyLane | 量子机器学习互操作 |
| | WuYue SDK | 国产量子平台支持 |
| | PySCF | 量子化学计算 |
| | matplotlib | 可视化 |
| | scipy | 经典优化算法 |

---

## 🚀 性能考虑

### 内存优化
- **大比特数电路**: 建议使用逐门演化而非组装全局矩阵
- **BatchSV**: 批量处理减少重复计算

### NPU 特殊处理
- **complex64 内核缺失**: BatchSV 使用实部/虚部分离
- **Fallback 机制**: NPUBackend 支持回退到 CPU

### 并行化
- **GPUBackend**: 利用 CUDA 并行加速
- **批量演化**: BatchSV 支持 per-sample 参数

---

## 🔮 未来发展方向

根据 [`NEXT.md`](file:///Users/luxian/GitSpace/quantum_frame/NEXT.md)，可能的演进方向：

1. **更多量子算法**: Grover, Shor, QFT 等
2. **更强的编译器**: 更激进的优化 pass
3. **分布式仿真**: 多节点大规模仿真
4. **错误校正**: 量子纠错码支持
5. **更多硬件后端**: 其他加速器支持

---

## 📚 学习路径建议

### 初学者
1. 阅读 README.md 第 1-5 节
2. 运行 `demos/demo_1.py`
3. 尝试构建简单电路（Bell 态、GHZ 态）

### 进阶用户
1. 学习参数化电路和自动微分
2. 探索 VQE/QAOA 算法
3. 使用 QASM/Qiskit 互操作

### 高级用户
1. 研究 QAS 量子架构搜索
2. 自定义噪声模型
3. 开发新的 Backend 或 Pass

---

## 🎓 关键概念映射

| aicir 概念 | Qiskit 对应 | PennyLane 对应 |
|-----------|------------|---------------|
| `Backend` | `BackendV2` | `Device` |
| `State` | `Statevector` / `DensityMatrix` | - |
| `Circuit` | `QuantumCircuit` | `QuantumTape` |
| `Operation` | `Instruction` | `Operator` |
| `Measure` | `Sampler` / `Estimator` | `QNode` |
| `Hamiltonian` | `SparsePauliOp` | `Hamiltonian` |
| `Parameter` | `Parameter` | - |

---

**文档版本**: 2026-06-14  
**aicir 版本**: 0.1.0
