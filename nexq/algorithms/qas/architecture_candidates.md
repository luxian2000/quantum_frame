# QAS Candidate Architecture Library

本文档说明当前 `architecture_candidates.py` 中预设候选架构库的定义、线路组成和适用场景。候选库的目标不是只做目标态制备，而是给噪声自适应 QAS 提供一批可比较、可排名的 ansatz / architecture candidates。

## 1. 公共约定

每个候选架构都会构造成 `ArchitectureSpec`，包含：

```text
name: 架构名称
circuit: nexq.core.Circuit
description: 架构说明
tags: 架构标签
metadata: family / layers / topology / entangler 等结构信息
```

默认入口：

```python
from nexq.algorithms.qas import build_common_architectures

architectures = build_common_architectures(n_qubits=3, layers=2, backend=backend)
```

当前默认候选库混合了浅层/深层、稀疏/稠密、局域/全连接、固定骨架/参数化骨架，便于评分器体现表达能力、可训练性、噪声鲁棒性和硬件效率之间的权衡。

拓扑约定：

```text
linear:      (0,1), (1,2), ..., (n-2,n-1)
ring:        linear + (n-1,0)，n > 2 时启用
all_to_all:  所有 i < j 的 qubit pair
brickwork:   偶边层 (0,1),(2,3),... 与奇边层 (1,2),(3,4),... 交替
hierarchical: MERA-like 中 stride = 1,2,4,... 的层级连接
```

参数约定：候选库使用确定性的占位参数 `0.071, 0.142, ...`。评分时 `kl_haar` 会重新采样参数，因此这些占位值主要用于构造合法参数化线路和统计参数数量。

## 2. 当前预设架构总览

| 预设名 | 生成函数 | family | 主要组成 |
|---|---|---|---|
| `hea_linear` | `hardware_efficient_ansatz(..., topology="linear", entangler="cx")` | HEA | RY/RZ 旋转层 + 线性 CX 层 + 末端 RY 层 |
| `hea_ring` | `hardware_efficient_ansatz(..., topology="ring", entangler="cx")` | HEA | RY/RZ 旋转层 + 环形 CX 层 + 末端 RY 层 |
| `real_amplitudes_linear` | `real_amplitudes_ansatz(..., topology="linear")` | RealAmplitudes | RY 旋转层 + 线性 CX 层 + 末端 RY 层 |
| `efficient_su2_ring` | `efficient_su2_ansatz(..., topology="ring", entangler="cx")` | EfficientSU2 | RX/RY/RZ 旋转层 + 环形 CX 层 + 末端 RY/RZ 层 |
| `two_local_all_to_all` | `two_local_ansatz(..., topology="all_to_all", entangler="cz")` | TwoLocal | RX/RY/RZ 旋转层 + 全连接 CZ 层 |
| `qaoa_chain` | `qaoa_ansatz(..., topology="linear")` | QAOA | 初始 H 层 + 线性 RZZ cost 层 + RX mixer 层 |
| `qaoa_complete` | `qaoa_ansatz(..., topology="all_to_all")` | QAOA | 初始 H 层 + 全连接 RZZ cost 层 + RX mixer 层 |
| `brickwork_cx` | `brickwork_ansatz(..., entangler="cx")` | Brickwork | RY/RZ 层 + 偶边 CX + RX 层 + 奇边 CX |
| `cascade_cx` | `cascade_entangler_ansatz(...)` | Cascade | RY/RZ 层 + 正向 CX ladder + RX 层 + 反向 CX ladder |
| `strongly_entangling_crx` | `strongly_entangling_layers_ansatz(...)` | StronglyEntanglingLayers | U3 层 + shifted-ring CRX 层 |
| `ghz_ladder` | `ghz_ladder_ansatz(...)` | GHZ | H(0) + 线性 CX ladder + RZ/RY dressing 层 |
| `mera_like` | `mera_like_ansatz(...)` | MERA-like | 层级 RY/RZ 层 + stride 递增的 CX 层 |

## 3. 各候选架构定义

### 3.1 `hea_linear`

定义：hardware-efficient ansatz 的线性连接版本。

每层组成：

```text
for each layer:
    for q in qubits: RY(q), RZ(q)
    for edge in linear_edges: CX(edge)
final:
    for q in qubits: RY(q)
```

特点：结构简单，双比特门数量随 `n` 线性增长，适合离子阱以外的 nearest-neighbor 硬件，也适合做 QAS baseline。

### 3.2 `hea_ring`

定义：hardware-efficient ansatz 的环形连接版本。

每层组成：

```text
for each layer:
    for q in qubits: RY(q), RZ(q)
    for edge in ring_edges: CX(edge)
final:
    for q in qubits: RY(q)
```

特点：比 `hea_linear` 多一条闭环 entangler，表达能力通常更强，但双比特门、error budget 和硬件代价也更高。

### 3.3 `real_amplitudes_linear`

定义：RealAmplitudes 风格 ansatz，只使用实振幅旋转 `RY` 和线性 `CX`。

每层组成：

```text
for each layer:
    for q in qubits: RY(q)
    for edge in linear_edges: CX(edge)
final:
    for q in qubits: RY(q)
```

特点：参数量较少，可训练性和噪声鲁棒性通常较好；表达能力比含 `RZ/RX/U3` 的模板保守。

### 3.4 `efficient_su2_ring`

定义：EfficientSU2 风格 ansatz，使用完整局域 SU(2) 旋转块和环形 entangler。

每层组成：

```text
for each layer:
    for q in qubits: RX(q), RY(q), RZ(q)
    for edge in ring_edges: CX(edge)
final:
    for q in qubits: RY(q), RZ(q)
```

特点：局域旋转自由度更高，表达能力更强；同时参数数量、深度和训练风险也更高。

### 3.5 `two_local_all_to_all`

定义：TwoLocal 风格全连接模板，局域旋转层后接全连接 `CZ` entangler。

每层组成：

```text
for each layer:
    for q in qubits: RX(q), RY(q), RZ(q)
    for edge in all_to_all_edges: CZ(edge)
```

特点：全连接 entangler 提高表达能力，但双比特门数量为 `O(n^2)`，在噪声鲁棒性和硬件效率上通常会被惩罚。

### 3.6 `qaoa_chain`

定义：QAOA-like 线性 Ising ansatz，使用 `RZZ` cost layer 和 `RX` mixer layer。

组成：

```text
initial:
    for q in qubits: H(q)
for each layer p:
    for edge in linear_edges: RZZ(edge)
    for q in qubits: RX(q)
```

特点：适合组合优化/Ising 问题。线性连接版本门数较低，通常在噪声鲁棒性和表达能力之间有较好折中。

### 3.7 `qaoa_complete`

定义：QAOA-like 全连接 Ising ansatz。

组成：

```text
initial:
    for q in qubits: H(q)
for each layer p:
    for edge in all_to_all_edges: RZZ(edge)
    for q in qubits: RX(q)
```

特点：比 `qaoa_chain` 更接近 dense Ising cost Hamiltonian，表达能力更强；双比特门和 error budget 随 `O(n^2)` 增长。

### 3.8 `brickwork_cx`

定义：brickwork nearest-neighbor ansatz，交替施加偶边和奇边 entangler。

每层组成：

```text
for each layer:
    for q in qubits: RY(q), RZ(q)
    for edge in even_edges: CX(edge)
    for q in qubits: RX(q)
    for edge in odd_edges: CX(edge)
```

特点：结构接近很多硬件友好的浅层 VQA 模板，层内部分 entangler 可并行；在当前串行 error budget 代理下仍按门数累计噪声。

### 3.9 `cascade_cx`

定义：正向和反向 CX ladder 组成的 cascade entangler 模板。

每层组成：

```text
for each layer:
    for q in qubits: RY(q), RZ(q)
    for edge in forward_linear_edges: CX(edge)
    for q in qubits: RX(q)
    for edge in reverse_linear_edges: CX(edge)
```

特点：纠缠传播方向更充分，但每层双比特门约为 `2(n-1)`，比单向 ladder 更 noisy。

### 3.10 `strongly_entangling_crx`

定义：StronglyEntanglingLayers 风格模板，局域 `U3` 后接 shifted-ring `CRX`。

每层组成：

```text
for each layer l:
    for q in qubits: U3(q)
    offset = (l mod (n - 1)) + 1
    for q in qubits: CRX(control=q, target=(q + offset) mod n)
```

特点：参数量高，连接范围随层移动，表达能力强；`CRX` 同时增加参数数量和双比特门噪声成本。

### 3.11 `ghz_ladder`

定义：GHZ ladder backbone 加轻量局域 dressing 层。

组成：

```text
H(0)
for edge in linear_edges: CX(edge)
for q in qubits: RZ(q), RY(q)
```

特点：非常浅，双比特门少，硬件效率和噪声鲁棒性通常较好；表达能力有限，适合做低噪声参考架构。

### 3.12 `mera_like`

定义：小型 MERA-like 层级 ansatz，使用 stride 递增的连接模式。

组成：

```text
for each outer layer:
    stride = 1
    while stride < n:
        for q in qubits: RY(q), RZ(q)
        for source in 0, 2*stride, 4*stride, ...:
            CX(source, source + stride) if source + stride < n
        stride *= 2
```

特点：层级结构有利于多尺度纠缠表达，门数通常低于全连接模板；适合作为 tensor-network-inspired QAS 候选。

## 4. 结果输出

展示 demo：

```powershell
C:/ProgramData/anaconda3/python.exe nexq/algorithms/qas/demo/architecture_scoring_demo.py
```

每次运行会同时：

```text
1. 在终端输出候选架构排名表和 Top 架构细节
2. 将同样内容保存到 nexq/algorithms/qas/demo/architecture_scoring_results.txt
```

输出字段包括：

```text
rank, architecture, expr, train, noise, hw, weighted, gates, 2q, params, err_budget
```

其中 `expr/train/noise/hw` 分别对应当前 active 的四组指标：

```text
expressibility.kl_haar
trainability.structure_proxy
noise_robustness.ion_trap_error_budget_proxy
hardware_efficiency.native_depth_twoq_efficiency
```