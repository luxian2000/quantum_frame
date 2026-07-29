# `aicir.distributed` 使用指南

`aicir.distributed` 提供一套独立、显式的 API，把同一个量子态分片到
多个 Ascend NPU。它不会改变 `State`、`Measure`、`NPUBackend` 等现有
API 的语义，也不会自动把普通模拟切换成分布式执行。

## 快速使用

每个进程必须执行同一段代码，由 `torchrun` 提供 `RANK`、
`LOCAL_RANK` 和 `WORLD_SIZE`：

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

# 完整初态只能由 rank 0 提供；其他 rank 传 None。
initial = (
    np.array([1, 0, 0, 0], dtype=np.complex64)
    if simulator.backend.rank == 0
    else None
)
result = simulator.run(
    circuit,
    initial_state=initial,
    observables={"zz": PauliString("ZZ", n_qubits=2)},
    shots=1024,
    seed=7,
)

# 标量期望值在所有 rank 相同；counts 只在 rank 0 存在。
print(result.expectations["zz"])
if result.is_root:
    print(dict(result.counts))

# 只有显式调用才会在 root 物化完整数组。
full_state = result.state.to_numpy(root=0)
full_probabilities = result.gather_probabilities(root=0)
```

真机严格探针：

```bash
PYTHONPATH=. torchrun --nproc-per-node=2 scripts/npu/distributed_state_probe.py
PYTHONPATH=. torchrun --nproc-per-node=4 scripts/npu/distributed_state_probe.py
```

探针禁用 CPU 回退，检查每个分片位于 `npu:LOCAL_RANK`，并覆盖局部门、
跨 rank 门、状态向量、密度矩阵、Kraus 噪声、期望值、概率和采样。它
只验证正确性和分片存储，不证明多 NPU 加速。

## API

公共类型只从 `aicir.distributed` 导出：

- `DistNPUBackend`：一进程一 NPU 的分布式后端；
- `DistState`：当前 rank 持有的状态向量或密度矩阵行分片；
- `DistSimulator`：线路预检、布局、执行、规约和采样入口；
- `DistResult`：不可变结果；不隐式聚合完整状态。

推荐用 `DistSimulator.from_env()`。也可以先构造
`DistNPUBackend.from_env()`，再显式传给 `DistSimulator(backend)`。
不要把 `DistNPUBackend` 用于现有批次任务并行接口。

`DistSimulator.run` 的分布式新增参数包括根进程完整初态、
已分片 `DistState`、结构化 observables、末端 Z 采样、显式 layout
以及独立的 `return_state` / `return_probabilities` 开关。

## rank、逻辑量子比特和存储轴

`rank` 是全局进程编号；`local_rank` 是当前节点内的设备编号。单节点
启动时两者通常相同，但多节点时不能互换。后端将当前进程绑定到
`npu:LOCAL_RANK`，通信使用全局 `rank`。

当 `world_size = 2^p` 时，状态张量的前 `p` 个存储轴决定分片所属
rank。这些轴只是存储布局，不是额外量子比特，也不会占用或删除线路
中的任何逻辑量子比特。静态 layout 只是把全部 `n` 个逻辑量子比特
双射到 `n` 个存储轴，以减少预计的跨 rank 门通信。用户提供 layout
时，序列第 `q` 项表示逻辑量子比特 `q` 所在的存储轴。

## 存储和通信

设 `W = world_size`，首期数据类型固定为 `complex64`：

| 状态 | 全局元素数 | 每 rank 持久元素数 | 每 rank 持久字节数 |
| --- | ---: | ---: | ---: |
| 状态向量 | `2^n` | `2^n / W` | `8 * 2^n / W` |
| 密度矩阵 | `4^n` | `4^n / W` 行分片 | `8 * 4^n / W` |

跨分片门按确定的 XOR partner 交换分片块。状态向量提升为密度矩阵时，
每个 rank 会临时收集完整的 `2^n` 状态向量以构造自己的密度行块；这
是声明过的有界工作缓冲，不会收集完整 `4^n` 密度矩阵。Kraus 各项按
本地密度行块累加。

常规 `run()` 不会聚合完整状态、密度矩阵或概率向量。以下操作是显式
且可能造成根进程大内存占用的边界：

- `DistState.gather(root=0)`；
- `DistState.to_numpy(root=0)`；
- `DistResult.gather_probabilities(root=0)`。

非 root 进程在这些调用中返回 `None`，但仍须参与相应 collective。

## 首期支持边界

| 支持 | 不支持或拒绝 |
| --- | --- |
| `world_size=2^p` 且 `n_qubits >= p` | 非 2 的幂的 world size |
| 前向 `complex64` 状态向量和行分片密度矩阵 | autograd / `requires_grad=True` |
| 有局部门矩阵的酉门 | 中途测量、reset、`if` / `while` |
| 内置局部 Kraus 信道的确定性密度演化 | 随机纯态轨迹近似、全系统自定义 Kraus |
| `PauliString`、Pauli Hamiltonian、显式目标比特的局部稠密 observable | 无结构的全系统稠密 observable |
| 末端 Z 基 shots、子集测量和单 shot collapse | 多 shot collapse、非 Z 末端采样 |
| 根进程完整初态或所有 rank 的匹配 `DistState` | 隐式 CPU 回退、隐式完整 gather |

当前实现使用一进程一 NPU 和 HCCL。多节点部署需要外部启动器正确设置
rendezvous 与 rank 环境；本接口不负责调度或容错恢复。

## 验证范围

CPU/Gloo 测试用于验证 1、2、4 rank 的数值与通信契约。发布或声称
Ascend 可用前，还必须在目标 CANN、PyTorch 和 torch-npu 组合上运行
上面的 2/4 NPU 严格探针。探针通过不等价于获得线性加速；性能结论
需要独立的同问题、同精度、同硬件基线测量。
