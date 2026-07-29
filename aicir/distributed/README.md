# `aicir.distributed` 使用手册

`aicir.distributed` 用多张 Ascend NPU 协同保存和演化**同一个**量子态。
它是一套独立的显式 API，不会改变 `State`、`Circuit`、`Measure` 或
`NPUBackend` 的既有语义，也不会把普通模拟调用自动转换为分布式执行。

## 目录

- [1. 功能模型与前提条件](#1-功能模型与前提条件)
- [2. 启动分布式进程](#2-启动分布式进程)
- [3. 构建分布式后端](#3-构建分布式后端)
- [4. 构建分布式量子态](#4-构建分布式量子态)
- [5. 构建分布式量子线路](#5-构建分布式量子线路)
- [6. 构建并调用分布式模拟器](#6-构建并调用分布式模拟器)
- [7. 状态向量模拟](#7-状态向量模拟)
- [8. 密度矩阵模拟](#8-密度矩阵模拟)
- [9. 确定性 Kraus 噪声](#9-确定性-kraus-噪声)
- [10. 观测量与期望值](#10-观测量与期望值)
- [11. 末端 Z 基采样与坍缩](#11-末端-z-基采样与坍缩)
- [12. 读取 DistResult 与显式聚合](#12-读取-distresult-与显式聚合)
- [13. 复用 DistState 继续演化](#13-复用-diststate-继续演化)
- [14. 自动与显式 layout](#14-自动与显式-layout)
- [15. 存储、通信与内存公式](#15-存储通信与内存公式)
- [16. 支持边界和错误条件](#16-支持边界和错误条件)
- [17. 真机验证与故障定位](#17-真机验证与故障定位)
- [18. 公共 API 参考](#18-公共-api-参考)

---

## 1. 功能模型与前提条件

分布式模式采用“一进程一 NPU”：

```text
torchrun
├── rank 0 / local_rank 0 / npu:0 ── 状态分片 0
├── rank 1 / local_rank 1 / npu:1 ── 状态分片 1
├── ...
└── rank W-1                       ── 状态分片 W-1
```

所有进程执行相同的 Python 程序、构建相同的线路，并共同完成每个
collective。`rank` 只标识某段存储属于哪个进程，不是量子寄存器，也不
消耗逻辑量子比特。

首期前提：

- `world_size = 2^p`，例如 1、2、4、8；
- 线路量子比特数满足 `n_qubits >= p`；
- 状态和门矩阵固定使用 `torch.complex64`；
- 仅支持前向模拟，不支持 autograd 或 `requires_grad=True`；
- Ascend 真机使用 HCCL；CPU/Gloo 只用于本地契约测试；
- 正常模拟不会隐式聚合完整状态、密度矩阵或概率向量。

分布式状态是一个逻辑整体，但每个进程只持有自己的 `DistState` 分片。
因此所有参与 collective 的方法都必须由每个 rank 按相同顺序调用。

---

## 2. 启动分布式进程

假设程序保存为 `dist_example.py`，两张或四张 NPU 的启动方式为：

```bash
source /usr/local/Ascend/cann/set_env.sh
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=2 dist_example.py
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=4 dist_example.py
```

`set_env.sh` 的实际位置取决于 CANN 安装方式；旧版组合包常见路径为
`/usr/local/Ascend/ascend-toolkit/set_env.sh`。必须在启动
`torchrun` 的同一个 shell 中加载它。

不要写成 `PYTHONPATH=. torchrun ...`。这种写法会覆盖
`set_env.sh` 注入的 CANN Python 路径，可能使 `torch_npu` 在
`set_device()` 阶段报 `ModuleNotFoundError: No module named 'tbe'`。
`PYTHONPATH=.:${PYTHONPATH}` 才是在加入当前仓库的同时保留 TBE 等
CANN 模块。

`torchrun` 为每个进程设置：

| 环境变量                          | 含义                                           |
| --------------------------------- | ---------------------------------------------- |
| `WORLD_SIZE`                    | 全局进程数，也是本功能的状态分片数             |
| `RANK`                          | 全局进程编号，用于通信和分片编号               |
| `LOCAL_RANK`                    | 当前节点内进程编号，用于绑定`npu:LOCAL_RANK` |
| `MASTER_ADDR` / `MASTER_PORT` | process group rendezvous 地址                  |

单节点上 `RANK` 通常等于 `LOCAL_RANK`。多节点上两者不一定相同：
设备绑定必须使用 `LOCAL_RANK`，全局通信必须使用 `RANK`。不要额外
设置 `ASCEND_RT_VISIBLE_DEVICES` 来重映射这些编号。

程序结束时，如调用方还需要显式释放 process group，可执行：

```python
import torch

if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
```

---

## 3. 构建分布式后端

### 3.1 通过模拟器直接构建

最短且推荐的入口是：

```python
from aicir.distributed import DistSimulator

simulator = DistSimulator.from_env(fallback_to_cpu=False)
backend = simulator.backend
```

`from_env()` 读取 `torchrun` 环境、绑定当前 NPU，并初始化 process
group。

### 3.2 显式构建后端

需要先保存后端对象时：

```python
from aicir.distributed import DistNPUBackend, DistSimulator

backend = DistNPUBackend.from_env(fallback_to_cpu=False)
simulator = DistSimulator(backend)
```

`DistNPUBackend.from_env()` 参数：

| 参数                      | 默认值    | 说明                                                |
| ------------------------- | --------- | --------------------------------------------------- |
| `fallback_to_cpu`       | `False` | 真机应保持`False`；NPU 不可用时直接失败           |
| `init_process_group`    | `True`  | 调用方已自行初始化 process group 时可设为`False`  |
| `process_group_backend` | `None`  | 真机自动选择 HCCL；本地多进程测试可显式传`"gloo"` |

常用只读属性：

```python
backend.world_size
backend.rank
backend.local_rank
```

不要使用普通构造函数 `DistNPUBackend()` 代替 `from_env()` 启动多卡
模拟；普通构造函数没有 `torchrun` 的 rank 上下文。也不要把
`DistNPUBackend` 传给现有批次任务并行接口：它表示单状态分片，不是
“每卡独立运行一个任务”的后端。

---

## 4. 构建分布式量子态

用户通常不单独实例化 `DistState`，而是通过 `DistSimulator.run()`
建立正确的 layout 和分片元数据。首期支持三种构建路径。

### 4.1 自动构建零态

不传 `initial_state` 和 `initial_density_matrix` 时，模拟器建立
\(\lvert 0\cdots0\rangle\)：

```python
from aicir import Circuit

circuit = Circuit(n_qubits=4)
result = simulator.run(circuit)
state = result.state
```

只有 rank 0 的首个分片包含振幅 1，其余元素和其余 rank 的分片均为
0。这个差异只是存储位置不同，逻辑上仍是一个完整零态。

### 4.2 由 rank 0 提供完整状态向量

完整状态向量只在 rank 0 创建；其他 rank 必须传 `None`：

```python
import numpy as np

from aicir import Circuit

circuit = Circuit(n_qubits=2)
initial_state = (
    np.array([0, 0, 1, 0], dtype=np.complex64)
    if simulator.backend.rank == 0
    else None
)
result = simulator.run(circuit, initial_state=initial_state)
```

rank 0 可以提供：

- 一维 NumPy 数组，元素数为 `2**n_qubits`；
- 形状为 `(2**n_qubits, 1)` 的数组；
- 可转换为 NumPy 的 PyTorch tensor；
- 现有非分布式 `State`。

模拟器将逻辑顺序转换为选定的存储顺序，再把连续分片 scatter 到各
rank。输入由调用方负责满足归一化条件。

### 4.3 由 rank 0 提供完整密度矩阵

密度矩阵必须具有精确形状
`(2**n_qubits, 2**n_qubits)`：

```python
import numpy as np

from aicir import Circuit

circuit = Circuit(n_qubits=1)
initial_density_matrix = (
    np.array(
        [[0.25, 0.0], [0.0, 0.75]],
        dtype=np.complex64,
    )
    if simulator.backend.rank == 0
    else None
)
result = simulator.run(
    circuit,
    initial_density_matrix=initial_density_matrix,
)
assert result.state.is_density
```

模拟器按行分片密度矩阵。调用方负责输入的 Hermitian、半正定和
trace-one 物理条件；首期入口只校验形状和数据类型转换。

### 4.4 使用已有 `DistState`

上一轮 `run()` 返回的 `DistState` 可以作为下一轮的分布式初态。
此时每个 rank 都传自己的分片，而不是仅 rank 0 传值。完整示例见
[第 13 节](#13-复用-diststate-继续演化)。

### 4.5 `DistState` 中保存什么

```python
state.local_data       # 当前 rank 的 torch.complex64 张量
state.local_shape      # 当前 rank 的分片形状
state.global_shape     # 逻辑整体的形状
state.n_qubits
state.kind             # "vector" 或 "matrix"
state.is_density
state.rank
state.world_size
state.layout
state.backend
```

`DistState.from_local()` 和 `DistState.zero()` 依赖内部的分片/layout
元数据。首期没有面向用户的“任意 local shard 直接构造”接口，因此
不应通过内部 `_ShardSpec`、`_Layout` 或 `_Communicator` 拼装状态。

---

## 5. 构建分布式量子线路

分布式功能**没有** `DistCircuit`。线路继续使用现有 `Circuit` 和门
构造函数：

```python
from aicir import Circuit, cx, hadamard, rz

circuit = Circuit(
    hadamard(0),
    cx(target_qubit=1, control_qubits=(0,)),
    rz(0.3, 1),
    n_qubits=2,
)
```

每个 rank 必须构建相同线路。`DistSimulator` 会比较线路、layout 和
关键运行选项的摘要；不一致时在执行分布式门之前失败。

“分布式线路”指由 `DistSimulator` 在分片状态上执行的普通线路：

- 目标存储轴全部位于 rank 内部的门是**局部门**，无需交换状态分片；
- 涉及 rank 分布式存储轴的门是**通信门**，按确定的 XOR partner
  交换所需分片；
- 同一个逻辑门是否通信取决于 layout，不需要另一套门构造 API。

首期接受能够解析出有限局部门矩阵的酉门。线路内 `measure`、reset、
`if` 和 `while` 会在预检阶段被拒绝；末端采样通过 `run()` 参数指定。

---

## 6. 构建并调用分布式模拟器

### 6.1 `run()` 签名

```python
def run(
    self,
    circuit,
    *,
    initial_state=None,
    initial_density_matrix=None,
    observables=None,
    shots=None,
    measure_qubits=(),
    collapse=False,
    seed=None,
    layout=None,
    return_state=True,
    return_probabilities=True,
) -> DistResult:
    ...
```

### 6.2 参数说明

| 参数                       | 说明                                                        |
| -------------------------- | ----------------------------------------------------------- |
| `circuit`                | 每个 rank 上内容相同的现有`Circuit`                       |
| `initial_state`          | rank 0 完整状态向量，或每个 rank 的匹配 vector`DistState` |
| `initial_density_matrix` | rank 0 完整密度矩阵，或每个 rank 的匹配 matrix`DistState` |
| `observables`            | 名称到结构化 observable 的映射                              |
| `shots`                  | `None` 表示不采样；正整数表示末端 Z 基采样                |
| `measure_qubits`         | 空序列表示全部逻辑比特；非空序列按给定顺序读出子集          |
| `collapse`               | 仅在`shots == 1` 时允许，返回采样后的坍缩态               |
| `seed`                   | 末端采样随机种子                                            |
| `layout`                 | `None` 自动选择；或传逻辑比特到存储轴的完整排列           |
| `return_state`           | 是否通过`DistResult.state` 暴露最终分片态                 |
| `return_probabilities`   | 是否计算并保存每个 rank 的局部概率分片                      |

`initial_state` 与 `initial_density_matrix` 不能同时提供。除“rank 0
完整初态”这种约定外，每个 rank 的线路、参数和执行顺序必须一致。

### 6.3 一次完整调用

```python
import numpy as np

from aicir import Circuit, PauliString, cx, hadamard
from aicir.distributed import DistSimulator

simulator = DistSimulator.from_env(fallback_to_cpu=False)
circuit = Circuit(
    hadamard(0),
    cx(target_qubit=1, control_qubits=(0,)),
    n_qubits=2,
)

result = simulator.run(
    circuit,
    observables={"zz": PauliString("ZZ", n_qubits=2)},
    shots=1024,
    seed=7,
)

print(result.expectations["zz"])  # 所有 rank 均可读取
if result.is_root:
    print(dict(result.counts))     # counts 只在 rank 0 存在

# 这是 collective：每个 rank 都必须调用。
full_state = result.state.to_numpy(root=0)
probabilities = result.gather_probabilities(root=0)
if result.is_root:
    print(np.asarray(full_state).reshape(-1))
    print(probabilities)
```

---

## 7. 状态向量模拟

无密度矩阵初态且没有匹配噪声规则时，状态保持为 vector
`DistState`：

```python
from aicir import Circuit, cx, hadamard

bell = Circuit(
    hadamard(0),
    cx(target_qubit=1, control_qubits=(0,)),
    n_qubits=2,
)
result = simulator.run(bell)

state = result.state
assert state.kind == "vector"
assert not state.is_density
print(state.local_shape)
```

设 `W = world_size`，每个 rank 的持久状态形状为：

```text
(2^n / W, 1)
```

状态向量门内核只读取当前分片和本门需要的 partner 分片，不构建完整
`2^n × 2^n` 全局酉矩阵。局部门完全在当前 rank 内计算。

`result.local_probabilities` 是当前 rank 的归一化概率分片。它不是完整
概率向量；完整逻辑顺序概率只能通过
`result.gather_probabilities(root=...)` 显式聚合。

---

## 8. 密度矩阵模拟

传入 `initial_density_matrix` 后，酉门按
\(\rho \mapsto U\rho U^\dagger\) 演化行分片：

```python
import numpy as np

from aicir import Circuit, pauli_x

rho0 = (
    np.array([[1, 0], [0, 0]], dtype=np.complex64)
    if simulator.backend.rank == 0
    else None
)
circuit = Circuit(pauli_x(0), n_qubits=1)
result = simulator.run(
    circuit,
    initial_density_matrix=rho0,
)

assert result.state.kind == "matrix"
assert result.state.is_density
```

每个 rank 保存连续的行块：

```text
(2^n / W, 2^n)
```

密度矩阵不会在正常执行路径中完整聚合。`local_probabilities` 只提取
当前行块拥有的全局对角元素，并通过标量 all-reduce 归一化。

---

## 9. 确定性 Kraus 噪声

分布式噪声沿用现有 `NoiseModel` 的“门后触发”语义：

```python
from aicir import (
    AmplitudeDampingChannel,
    Circuit,
    NoiseModel,
    hadamard,
)

noisy = Circuit(hadamard(0), n_qubits=1)
noisy.noise_model = NoiseModel().add_channel(
    AmplitudeDampingChannel(target_qubit=0, gamma=0.05)
)

result = simulator.run(noisy)
assert result.state.is_density
```

执行过程：

1. 先执行当前酉门；
2. 检查 `NoiseModel` 中匹配该门的规则；
3. 如果当前状态是状态向量，先提升为行分片密度矩阵；
4. 对各局部 Kraus 项计算 \(K_i\rho K_i^\dagger\)；
5. 在每个 rank 的密度行块上确定性累加。

没有门的空线路不会触发“门后噪声”。`after_gates` 和
`exclude_gate_qubits` 的匹配方式与现有 `NoiseModel` 一致。

首期支持提供局部 Kraus 表示的内置信道，不使用随机纯态轨迹近似，
也不接受只能生成全系统稠密 Kraus 矩阵的自定义信道。

状态向量提升为密度矩阵时，每个 rank 会临时 all-gather 完整
`2^n` 状态向量，用于构造自己的密度行块；不会 all-gather 完整
`4^n` 密度矩阵。

---

## 10. 观测量与期望值

`observables` 必须是“名称 → observable”的映射。首期支持三类结构化
输入。

### 10.1 `PauliString`

```python
from aicir import PauliString

observables = {
    "zz": PauliString("ZZ", n_qubits=2),
}
```

### 10.2 Pauli Hamiltonian

```python
from aicir import Hamiltonian

observables = {
    "energy": Hamiltonian(
        n_qubits=2,
        terms=[("ZI", 0.5), ("XX", -0.25)],
    ),
}
```

Hamiltonian 逐 Pauli 项规约，不会先构造完整全系统 Hamiltonian
矩阵。

### 10.3 显式目标比特的局部稠密 observable

```python
import numpy as np

from aicir import Observable

observables = {
    "x0": Observable.matrix(
        np.array([[0, 1], [1, 0]], dtype=np.complex64),
        metadata={"qubits": [0]},
    ),
}
```

`metadata["qubits"]` 必须显式给出矩阵作用的逻辑比特，且矩阵维度必须
等于 `2**len(qubits)`。无目标信息的全系统稠密矩阵会被拒绝。

综合调用：

```python
result = simulator.run(circuit, observables=observables)
for name, value in result.expectations.items():
    print(name, value)
```

每个 rank 都参与局部贡献计算和 all-reduce，因此
`result.expectations` 中的标量在所有 rank 上相同。

---

## 11. 末端 Z 基采样与坍缩

### 11.1 全部逻辑比特

```python
result = simulator.run(
    circuit,
    shots=1024,
    seed=7,
)
```

`measure_qubits=()` 是默认值，表示按逻辑比特 `0..n-1` 的顺序读取
全部比特。

### 11.2 逻辑比特子集

```python
result = simulator.run(
    circuit,
    shots=1024,
    measure_qubits=(1, 0),
    seed=7,
)
```

计数字符串遵循 `measure_qubits` 的输入顺序；上例的第一个字符对应
逻辑比特 1，第二个字符对应逻辑比特 0。

### 11.3 counts 的 rank 语义

```python
if result.is_root:
    print(dict(result.counts))
else:
    assert result.counts is None
```

采样采用分层流程：rank 0 先根据各 rank 的概率质量分配 shots，各
rank 再从自己的条件分布采样，最后只把采样索引送到 rank 0。不会为了
采样聚合完整概率向量。

### 11.4 单次采样并坍缩

```python
collapsed = simulator.run(
    circuit,
    shots=1,
    collapse=True,
    seed=7,
)
collapsed_state = collapsed.state
```

`collapse=True` 仅支持 `shots == 1`。多 shot 坍缩会抛出
`ValueError`。首期只支持末端计算基（Z 基）采样，不支持线路中途测量
或非 Z 末端采样。

---

## 12. 读取 `DistResult` 与显式聚合

### 12.1 结果字段

| 字段/属性               | 所有 rank 是否存在                     | 说明                                         |
| ----------------------- | -------------------------------------- | -------------------------------------------- |
| `state`               | 是，除非`return_state=False`         | 当前 rank 的最终`DistState`                |
| `local_probabilities` | 是，除非`return_probabilities=False` | 当前 rank 的概率分片                         |
| `expectations`        | 是                                     | 不可变名称到标量映射，各 rank 相同           |
| `counts`              | 仅 rank 0                              | 末端采样计数；未采样时所有 rank 均为`None` |
| `rank`                | 是                                     | 当前全局 rank                                |
| `world_size`          | 是                                     | 状态分片数                                   |
| `is_root`             | 是                                     | `rank == 0` 的便捷判断                     |

### 12.2 聚合完整状态

```python
full_state_object = result.state.gather(root=0)
full_array = result.state.to_numpy(root=0)
```

rank 0 分别得到普通 CPU `State` 和 NumPy 数组；非 root rank 得到
`None`。

### 12.3 聚合完整概率

```python
full_probabilities = result.gather_probabilities(root=0)
```

rank 0 得到逻辑比特顺序的 NumPy 概率数组；非 root rank 得到
`None`。

`gather()`、`to_numpy()` 和 `gather_probabilities()` 都是
collective。即使只有 rank 0 使用返回值，每个 rank 也必须执行同一
调用，否则会等待其他进程。

### 12.4 独立关闭状态或概率返回

```python
probability_only = simulator.run(
    circuit,
    return_state=False,
    return_probabilities=True,
)
assert probability_only.state is None
full_probabilities = probability_only.gather_probabilities(root=0)

state_only = simulator.run(
    circuit,
    return_state=True,
    return_probabilities=False,
)
assert state_only.local_probabilities is None
```

两个开关彼此独立。`return_state=False` 不妨碍显式聚合已请求的概率。

当 `collapse=True` 时，`result.state` 是采样后的坍缩态，而
`result.local_probabilities` 是采样前用于产生该结果的概率分布。

---

## 13. 复用 `DistState` 继续演化

### 13.1 继续演化状态向量

第一轮和第二轮必须使用同一 `DistNPUBackend`、相同量子比特数和相同
layout。每个 rank 都传自己的 `first.state`：

```python
from aicir import Circuit, hadamard, pauli_x

first_circuit = Circuit(hadamard(0), n_qubits=2)
first = simulator.run(first_circuit)

layout = first.state.layout.logical_to_storage
next_circuit = Circuit(pauli_x(1), n_qubits=2)
second = simulator.run(
    next_circuit,
    initial_state=first.state,
    layout=layout,
)
```

不要只在 rank 0 传 `first.state`；`DistState` 模式要求所有 rank 都
提供与当前 backend 匹配的本地分片。

### 13.2 继续演化密度矩阵

如果第一轮经过噪声或本来就是密度矩阵，须使用
`initial_density_matrix`：

```python
layout = first.state.layout.logical_to_storage
second = simulator.run(
    next_circuit,
    initial_density_matrix=first.state,
    layout=layout,
)
```

把 matrix `DistState` 传给 `initial_state`，或把 vector `DistState`
传给 `initial_density_matrix`，都会因状态种类不匹配而失败。

---

## 14. 自动与显式 layout

### 14.1 layout 表示什么

当 `world_size = 2^p` 时，前 `p` 个**存储轴**决定分片所属 rank。
layout 是全部逻辑量子比特到全部存储轴的双射：

```text
layout[q] = 逻辑量子比特 q 所在的存储轴
```

例如两卡运行 `n_qubits=2` 时，`p=1`：

```python
layout = (1, 0)
```

表示：

- 逻辑比特 0 → 存储轴 1，位于 rank 内部；
- 逻辑比特 1 → 存储轴 0，是 rank 分布式轴。

逻辑比特仍然有两个。rank 前缀复用了状态张量的存储索引，不增加也不
占用任何逻辑 qubit。

### 14.2 自动 layout

```python
result = simulator.run(circuit, layout=None)
```

模拟器在执行前分析整条静态线路，选择预计通信代价较低的逻辑到存储
映射。同一次 `run()` 中 layout 不再变化。

### 14.3 显式 layout

```python
result = simulator.run(circuit, layout=(1, 0))
```

显式 layout 必须：

- 长度等于 `n_qubits`；
- 是 `range(n_qubits)` 的完整排列；
- 在所有 rank 上完全相同。

显式 layout 适合复现实验、与外部存储约定对齐，或复用已有
`DistState`。它改变存储和通信路径，不改变逻辑线路或显式聚合后的
逻辑顺序。

---

## 15. 存储、通信与内存公式

设：

- `n` 为逻辑量子比特数；
- `W = world_size`；
- `complex64` 每个元素占 8 字节。

### 15.1 持久状态存储

| 状态     | 全局元素数 | 每 rank 元素数 | 每 rank 持久字节数 |
| -------- | ---------: | -------------: | -----------------: |
| 状态向量 |    `2^n` |    `2^n / W` |    `8 * 2^n / W` |
| 密度矩阵 |    `4^n` |    `4^n / W` |    `8 * 4^n / W` |

密度矩阵的分片方式是连续行块，因此每 rank 形状为
`(2^n / W, 2^n)`。

### 15.2 门通信

- 局部门：不交换状态分片；
- 涉及一个或多个分布式轴的门：按门计划与 XOR partner 交换分片；
- 规约：范数、trace 和期望值只 all-reduce 标量；
- 采样：通信量主要与 `world_size + shots` 成正比；
- 显式 gather：在 root 物化完整数组，是潜在的大内存边界。

### 15.3 密度提升临时缓冲

状态向量首次遇到匹配噪声时，每 rank 除自己的密度行块外，还会临时
持有一个完整 `2^n` 状态向量。容量评估必须把这个缓冲计入峰值。

以上公式描述存储缩放，不等价于性能加速。实际性能还受门通信比例、
HCCL 带宽、kernel 粒度和临时缓冲影响。

---

## 16. 支持边界和错误条件

### 16.1 功能边界

| 支持                                                    | 不支持或拒绝                     |
| ------------------------------------------------------- | -------------------------------- |
| `world_size=2^p` 且 `n_qubits >= p`                 | 非 2 的幂 world size             |
| 前向`complex64` 状态向量                              | autograd、`requires_grad=True` |
| 行分片密度矩阵                                          | 任意二维分块或列分块             |
| 能解析局部门矩阵的酉门                                  | 隐式完整全局酉矩阵               |
| 门后触发的确定性局部 Kraus 噪声                         | 随机纯态轨迹、全系统自定义 Kraus |
| Pauli、Pauli Hamiltonian、显式目标的局部稠密 observable | 无结构的全系统稠密 observable    |
| 末端 Z 基 shots 和逻辑比特子集                          | 中途测量、reset、非 Z 末端采样   |
| `shots=1, collapse=True`                              | 多 shot collapse                 |
| rank 0 完整初态或所有 rank 的匹配`DistState`          | 混合初态提供模式                 |
| 显式 root gather                                        | 隐式完整状态/概率 gather         |

### 16.2 常见预检错误

| 错误条件                                                 | 处理                     |
| -------------------------------------------------------- | ------------------------ |
| `initial_state` 和 `initial_density_matrix` 同时非空 | 抛出`ValueError`       |
| `shots <= 0`                                           | 抛出`ValueError`       |
| `collapse=True` 且 `shots != 1`                      | 抛出`ValueError`       |
| 线路包含 measure/reset/控制流                            | 在状态分配和门执行前拒绝 |
| 门参数`requires_grad=True`                             | 在门规划时拒绝           |
| layout 不是完整排列                                      | 抛出`ValueError`       |
| 不同 rank 的线路或关键选项不一致                         | 摘要一致性检查失败       |
| 复用的`DistState` 属于另一 backend/layout              | 抛出`ValueError`       |

首期没有容错恢复或弹性 rank 变更；任一进程异常退出后，应由外部启动
和作业系统终止整个任务。

---

## 17. 真机验证与故障定位

### 17.1 严格 Ascend 探针

从仓库根目录运行：

```bash
source /usr/local/Ascend/cann/set_env.sh
python -c "import tbe; print(tbe.__file__)"
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=2 scripts/npu/distributed_state_probe.py
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=4 scripts/npu/distributed_state_probe.py
```

探针强制 `fallback_to_cpu=False`，并检查：

- 每个分片位于 `npu:LOCAL_RANK`；
- rank、local rank 和设备对应关系；
- 局部门和通信门；
- 状态向量数值、范数和概率；
- 密度矩阵、Kraus 噪声和 trace；
- 结构化期望值；
- 末端采样；
- 每 rank 本地张量元素数。

只有 rank 0 输出最终 JSON。任一不变量失败时，所有 rank 应以非零状态
退出。探针通过证明该软硬件组合完成了本组正确性检查，但不证明获得
多 NPU 加速。

### 17.2 本地 Gloo 回归

没有 Ascend NPU 时可以验证通信契约，但不能代替 HCCL 真机结果：

```bash
PYTHONPATH=. pytest tests/distributed -q
```

### 17.3 常见故障

| 现象                            | 检查                                                     |
| ------------------------------- | -------------------------------------------------------- |
| `world_size` 不是 2 的幂      | 调整`--nproc-per-node` 为 1、2、4、8 等                |
| `n_qubits < log2(world_size)` | 减少进程数或增加线路量子比特数                           |
| rank 卡在 collective            | 检查所有 rank 是否调用了同一`run()`/gather，且线路一致 |
| 分片不在预期 NPU                | 检查`LOCAL_RANK` 和启动器配置，不要自行重映射可见设备  |
| gather 后 root 内存不足         | 避免完整 gather，只消费分片概率、标量期望值或 counts     |
| 噪声后状态变成 matrix           | 这是确定性 Kraus 演化的预期行为                          |
| CPU/Gloo 通过但 HCCL 失败       | 以目标 CANN、PyTorch、torch-npu 组合的真机探针为准       |

---

## 18. 公共 API 参考

公共类型只从 `aicir.distributed` 导出：

```python
from aicir.distributed import (
    DistNPUBackend,
    DistResult,
    DistSimulator,
    DistState,
)
```

这些名称不从顶层 `aicir` 重导出。

### 18.1 `DistNPUBackend`

推荐工厂：

```python
def from_env(
    *,
    fallback_to_cpu=False,
    init_process_group=True,
    process_group_backend=None,
) -> DistNPUBackend:
    ...
```

用户属性：

- `world_size`
- `rank`
- `local_rank`

`communicator` 是运行时内部协作对象，不是稳定的用户通信 API。

### 18.2 `DistState`

用户主要从 `DistResult.state` 获得 `DistState`。

只读属性：

- `local_data`
- `local_shape`
- `global_shape`
- `n_qubits`
- `kind`
- `is_density`
- `bit_order`
- `rank`
- `world_size`
- `layout`
- `backend`

显式物化方法：

```python
state.gather(root=0)
state.to_numpy(root=0)
```

`from_local()` 和 `zero()` 是当前实现所需的低层入口，但依赖内部
metadata，不是首期推荐的用户构建方法。

### 18.3 `DistSimulator`

构造：

```python
DistSimulator(backend)
DistSimulator.from_env(**backend_options)
```

属性：

- `backend`

执行：

```python
simulator.run(...)
```

所有 rank 必须调用相同的执行入口；方法返回当前 rank 的
`DistResult`。

### 18.4 `DistResult`

只读字段/属性：

- `state`
- `local_probabilities`
- `expectations`
- `counts`
- `rank`
- `world_size`
- `is_root`

显式概率聚合：

```python
result.gather_probabilities(root=0)
```

`DistResult` 是不可变数据类；`expectations` 和非空 `counts` 以只读
映射暴露。
