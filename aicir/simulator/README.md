# aicir.simulator — 精确张量网络模拟引擎

本模块把电路的每个门当作张量，沿电路的操作顺序把它们缩并（contract）成末态或振幅，而不是维护一份显式的 `2^n` 态矢量。适合只需要少量振幅、或电路结构带来较优缩并路径（树宽较小）时，避免整段态矢量演化的内存/算力开销。收缩建立在 `Backend` 的张量原语（`tensordot`/`transpose`/`reshape`/`conj`）之上，因此与 `NumpyBackend`/`GPUBackend`/`NPUBackend` 均兼容；精确张量网络期望在 torch/NPU 后端上保留计算图，MPS 的 NPU 梯度走 `MPSEstimator.gradient(method="psr")` 参数移位。

---

## 1. 方法选择指南

```
需要完整末态矢量，比特数不大 / 需要与 State/Measure 交互？
  -> tn_statevector(circuit, backend=...)
     一次性收缩出全部 2^n 振幅，返回 State。

只需要一个基态的振幅？
  -> single_amplitude(circuit, bitstring, backend=...)
     不构造全态矢量，标量收缩，适合大比特数下抽查个别振幅。

只关心部分比特上的（子）振幅分布，其余比特固定 |0>？
  -> partial_amplitude(circuit, open_qubits=[...], backend=...)
     或 partial_amplitude(circuit, bitstrings=[...], backend=...) 枚举给定基态。

需要期望值 <psi|O|psi>，并希望在 torch/NPU 上反传求参数梯度？
  -> tn_expectation(circuit, observable, backend=...)

只是想用 Measure 的统一入口（shots/exact/末端读出）跑一遍张量网络路径？
  -> Measure.run(circuit, method="tensor", ...)

需要更大比特数的低纠缠电路近似末态/期望（bond 截断，有损）？
  -> mps_statevector(circuit, max_bond_dim=..., cutoff=...) / mps_expectation(...)
     或 Measure.run(circuit, method="mps", max_bond_dim=..., cutoff=...)
```

---

## 2. 公共 API

### `tn_statevector(circuit, *, backend=None, optimize="auto", memory_limit=None) -> State`

收缩整段电路得到末态，返回 `State`（向量形态，`(2^n, 1)`）。等价于但不经过逐门 `State.evolve`，直接把电路的门张量网络一次性缩并。`optimize`/`memory_limit` 控制收缩路径与切片，见第 5 节。

### `single_amplitude(circuit, bitstring, *, backend=None, optimize="auto", memory_limit=None) -> complex`

求单个基态振幅 `⟨bitstring|U|0...0⟩`。`bitstring` 是长度为 `n_qubits` 的 0/1 字符串（或等价序列），只做标量收缩，不构造全态矢量，适合仅需个别振幅时节省内存。

### `partial_amplitude(circuit, *, open_qubits=None, bitstrings=None, backend=None, optimize="auto", memory_limit=None)`

求部分振幅，二选一：
- `open_qubits=[...]`：给定比特“开放”（保留为自由输出腿），其余比特的输出端固定在 `|0>`；返回长度 `2^len(open_qubits)` 的振幅向量，向量下标按开放比特**升序**排列、**首个开放比特对应最高位（MSB）**。
- `bitstrings=[...]`：枚举给定的若干基态，逐个调用 `single_amplitude` 求振幅，返回数组。

两者必须且只能提供其一。

### `tn_expectation(circuit, observable, *, backend=None, optimize="auto", memory_limit=None)`

求期望值 `⟨psi|O|psi⟩`，其中 `observable` 可以是带 `to_matrix(backend)` 方法的对象（例如 `Hamiltonian`），也可以是可直接 `backend.cast` 的矩阵。内部先用张量网络收缩出末态矢量，再走 `backend.expectation_sv`。在 `GPUBackend`/`NPUBackend` 上，若电路参数是 Torch 张量（`requires_grad=True`），返回值保留计算图，可 `.backward()` 得到参数梯度——`NPUBackend` 的复数 `tensordot` 复用 autograd-safe 的 real/imag 分解 matmul 实现，因此该可微性在 NPU 上同样成立。

### `Measure.run(circuit, method="tensor", ...)`

在统一测量入口 `Measure.run` 中传入 `method="tensor"`，即可用张量网络路径代替逐门态矢量演化求末态，再复用原有的 shots/exact、末端读出、`observables` 等逻辑。**注意以下几点与 `method="statevector"`（默认）不同：**

- **`snap`（快照）在 `method="tensor"` 下仅接受 `None`/`[]`。** 末态是经张量网络一次性求得的，没有逐操作演化轨迹可供在中间下标处拍快照；内部递归用的是 0 操作的 stripped 电路，故传入**非空** `snap` 会因下标越界抛 `ValueError`（如 `snap=[0]` → `snap 含越界操作下标 0（操作数=0）`）。
- **`method="tensor"` 忽略调用方传入的 `initial_state`/`initial_density_matrix`，始终从 `|0...0>` 出发演化。** 内部实现是先用 `tn_statevector` 求出电路作用在 `|0...0>` 上的末态，再把它当作一段空电路的 `initial_state` 交给 `method="statevector"` 路径继续处理（末端读出/shots/observables 等）；因此不支持从非零初态或指定密度矩阵起算。
- **仅支持纯态、无噪声**：若电路带 `noise_model`，或电路内嵌了 `measure(...)` 标记门，`method="tensor"` 会直接抛 `ValueError`（噪声路径必须走密度矩阵、内嵌 measure 标记与一次性收缩语义冲突）。

---

## 3. MPS 近似引擎

### `mps_statevector(circuit, *, max_bond_dim=None, cutoff=1e-10, backend=None) -> MPSState`

用矩阵乘积态（MPS）表示近似末态，适合低纠缠、大比特数的电路。返回 `MPSState` 对象，其中：

- `.to_statevector()` 还原稠密态矢量（`State` 形态，`(2^n, 1)`），用于与其他路径对比或下游消费稠密向量的函数。
- `.truncation_error` 是 float，记录 SVD 截断过程中累计丢弃的奇异值权重（相对损失）。

**截断参数**：
- `max_bond_dim`：bond 维度的硬上限（None = 不限，满秩演化）。
- `cutoff`：相对奇异值截断阈值，默认 `1e-10`；若奇异值比例 `σ_i / σ_max < cutoff`，则丢弃该及后续奇异值。两参数独立生效，取两者给定的更严格截断。

**约束与限制**：
- 仅接受 1/2 比特门；≥3 比特门需先经 `aicir.transpile.DecomposePass` 分解。
- 非相邻双比特门自动插入 SWAP 网络，并跟踪逻辑↔物理置换（`MPSState.logical_at`/`site_of` 属性）。
- 仅支持纯态、无噪声、无经典控制流（`ControlFlow`）。
- 后端支持 NumPy/GPU/NPU；NPU 支持 `mps_statevector`、`mps_expectation` 与 `MPSEstimator` 前向，梯度通过 `MPSEstimator.gradient(method="psr")` 参数移位获得。直接 autograd（`mps_expectation(...).backward()`）仅作为 CPU/GPU 能力；Ascend 缺少 complex64 梯度累加所需内核。NPU 上 `NPUBackend.svd` 走 real-embedding（Ascend 无原生 complex64 SVD），复数乘/除经 real/imag 分解，真机验收见 `demos/demo_npu_mps.py`。

### `mps_expectation(circuit, observable, *, max_bond_dim=None, cutoff=1e-10, backend=None)`

求 MPS 表示的末态对可观测量的期望值 `⟨ψ|O|ψ⟩`。行为根据 `observable` 类型调整：

- **`Hamiltonian`/`PauliString`**：用 transfer 张量网络缩并路径求解，避免稠密化，效率最优。
- **任意稠密矩阵**：回退到 `.to_statevector()` 还原后用 `backend.expectation_sv` 求解。

在 `GPUBackend` 上，若电路参数是 Torch 张量（`requires_grad=True`），返回值保留计算图，可 `.backward()` 得参数梯度。NPU 上请用 `MPSEstimator.gradient(method="psr")`，该路径只需前向计算。

### `Measure.run(circuit, method="mps", ..., max_bond_dim=..., cutoff=...)`

统一测量入口支持 `method="mps"` 路径，透传 `max_bond_dim` 与 `cutoff` 参数到底层 MPS 引擎；其余 shots/exact、末端读出逻辑与 `method="statevector"` 一致。同样的约束与限制适用（纯态无噪声、1/2 比特门、无 ControlFlow）。

---

## 4. 比特序与索引约定

- **比特序采用 msb 约定**：`qubit 0` 对应最高位（MSB）。`tn_statevector` 返回的 `State` 与直接用 `State.zero_state(...).evolve(circuit.unitary())` 得到的结果比特序一致，可直接互相比较。
- `single_amplitude` 的 `bitstring` 参数按 `qubit 0, qubit 1, ...` 顺序给出 0/1，即字符串首字符对应 `qubit 0`（MSB）。
- `partial_amplitude(open_qubits=...)` 返回向量的下标按**开放比特升序**排列，其中**第一个开放比特对应结果向量的最高位（MSB）**；未列入 `open_qubits` 的比特，其输出端固定在 `|0>`（不是求和/求迹，而是投影到 `|0>`）。

---

## 5. 收缩路径、切片与可选依赖

### 路径选择策略

**`optimize` 参数**（默认 `"auto"`）控制收缩路径寻优来源：

- **`optimize="auto"`**（默认）：网络规模较大（≥24 张量）且已安装 [`cotengra`](https://github.com/jcmgray/cotengra) 时使用 `HyperOptimizer`；否则优先使用 [`opt_einsum`](https://github.com/dgasmith/opt_einsum)（若已安装）的 `opt_path`，最后回退到内置贪心策略。三种路径均保证结果正确性，仅影响耗时与峰值内存。

- **`optimize="cotengra"|"opt_einsum"|"greedy"`**：显式指定路径来源，对应库缺失时抛 `ImportError`。

  - `"cotengra"`：需 `pip install -e ".[tn]"`（安装 `cotengra` 与 `opt_einsum` 依赖）；性能最优但耗时。
  - `"opt_einsum"`：需 `opt_einsum` 包（核心 `aicir` 安装已含）。
  - `"greedy"`：内置实现，无额外依赖。

### 切片与内存限制

**`memory_limit` 参数**（单位：中间张量最大元素数，默认 `None` 不切片）：

- 设定后，`optimize="cotengra"` 或 `optimize="auto"`（满足条件使用 cotengra）会触发 cotengra 的切片规划，返回形如 `(contraction_path, slices_dict)` 的结果。
- **执行侧**：对每个切片按 `slices_dict` 逐一固定开放指标（通过后端原语 `backend.take`），再经标准的成对 `tensordot` 收缩完成该切片，最后用 `backend.add` 累加所有切片结果。
- **开放指标也可被切片**：当输出张量（或中间张量）超过内存预算时，output indices 也会被纳入切片维度；执行仍走后端原语，通过一次性的 one-hot 置位与累加完整重建（`一次性 one-hot 放置累加`），与处理通常切片指标相同。
- **可微性保留**：执行侧采用 `backend.tensordot`、`backend.take`、`backend.add` 这些后端抽象原语，因此 Torch autograd 与 NPU 上可微性（包括复数分解）完全保留，求和操作保持梯度。
- **限制**：`memory_limit` 与 `optimize="opt_einsum"` 或 `"greedy"` 组合抛 `ValueError`（这两个路径不支持切片规划）。需 `pip install -e ".[tn]"`。

---

## 6. NPU 远程验证脚本

`demos/demo_npu_tensor.py` 是一个独立可执行脚本，用于在真实 Ascend NPU 环境上核对张量网络引擎与态矢量引擎的一致性、单/部分振幅的正确性，以及 `tn_expectation` 在 NPU 上的可微性（若安装了 `torch`）：

```bash
python demos/demo_npu_tensor.py                       # 严格要求 NPU，无卡时报错退出
python demos/demo_npu_tensor.py --allow-cpu-fallback   # 无 NPU 时允许回退到 CPU，便于开发调试
```

脚本会打印所用后端名称、设备信息，以及三项核对（全态矢量一致 / 单振幅+部分振幅 / 期望值梯度）的通过情况，并以退出码 0/1 表示整体是否全部通过（无 NPU 且未加 `--allow-cpu-fallback` 时退出码为 2）。`module.run_checks(allow_cpu_fallback=True)` 在 CPU-only 环境下亦可直接调用（见 `tests/simulator/test_npu_demo_importable.py`），可用作导入期冒烟测试。

---

## 7. 与其他子系统的关系

- `Measure`（`aicir/measure/`）：`method="tensor"` 与 `method="mps"` 复用本模块的 `tn_statevector` 与 `mps_statevector`，其余聚合/读出逻辑与 `method="statevector"` 一致。
- `Backend`（`aicir/backends/`）：本模块只依赖 `Backend` 抽象的张量原语，不感知具体后端实现；MPS/张量网络截断近似（有损压缩）见本模块的 `mps_statevector`/`mps_expectation`。
