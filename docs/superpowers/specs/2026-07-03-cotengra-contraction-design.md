# cotengra 收缩路径优化 + 切片执行设计（Spec）

日期：2026-07-03
状态：已批准设计，待实现

## 目标

给 `aicir.simulator` 张量网络引擎接入 [cotengra](https://github.com/jcmgray/cotengra)：

1. **路径优化**：用 `HyperOptimizer` 找显著优于内置贪心 / `opt_einsum` auto 的收缩路径。
2. **切片执行（slicing）**：中间张量超内存预算时，cotengra 规划切片指标；执行侧对每个切片赋值固定指标、跑同一 pairwise 收缩、累加结果——突破全收缩的内存墙。

核心架构决策：**cotengra 只做规划（planner），不做执行**。执行完全保留现有 `_pair_contract` → `backend.tensordot` 通道，因此 NPU 复数分解与 torch autograd 可微性全程不破坏（切片求和保梯度）。cotengra 自带的 array 执行接口绕开 `Backend` 抽象，明确不采用。

## 非目标

- 不用 cotengra 的执行接口（`tree.contract(arrays)` 等）。
- 不做 MPS/截断近似（有损压缩另属独立规划）。
- `Measure.run(method="tensor")` 不新增参数——内部用默认 `optimize="auto"`、`memory_limit=None`（YAGNI）。
- 不引入 kahypar/cmaes 等 cotengra 加速器依赖——纯 `cotengra` 一个包。
- 不做多 GPU/分布式切片并行；切片顺序串行执行。

## 依赖与打包

`pyproject.toml` 新增 extra：

```toml
tn = ["cotengra"]
```

并入 `all` 与 `dev`。cotengra 惰性 import（函数内），缺失时按需抛带 `pip install -e ".[tn]"` 提示的 `ImportError`。`numpy` 仍是唯一硬依赖。

## 组件 1：新 backend 原语 `take` / `add`

切片执行需要两个此前不存在的原语：

- `take(a, axis, index) -> tensor`：沿 `axis` 取下标 `index`，返回降一维张量（固定切片指标的值）。
- `add(a, b) -> tensor`：逐元素相加（切片结果累加）。

实现：

- `base.py`：两个抽象方法（`NotImplementedError`）。
- `NumpyBackend`：`np.take(a, index, axis=axis)`（axis 维退化后 squeeze）/ `a + b`。
- `GPUBackend`：`torch.select(a, axis, index)` / `a + b`。
- `NPUBackend`：**必须** autograd-safe 包装。Ascend 上 `aclnnAdd`/相关算子不支持 `DT_COMPLEX64`（真机已爆过），且直接 `torch.real/imag` 双读会造成图中 fan-out 触发复数梯度累加。照 `_NpuTransposeFn` 既有模式，各包一个 `torch.autograd.Function`（`_NpuTakeFn`/`_NpuAddFn`），forward/backward 内部 real/imag 分解，复数张量在图中保持单边。非复数或非 npu 设备走 `super()`。

## 组件 2：`contract.py` 扩展

签名：

```python
def contract(tensors, indices, open_indices, backend, *, optimize="auto", memory_limit=None)
```

- `optimize`:
  - `"auto"`（默认）：cotengra（已安装）→ `opt_einsum`（已安装）→ 内置贪心。
  - `"cotengra"` / `"opt_einsum"` / `"greedy"`：强制指定；对应库缺失抛 `ImportError`（带安装提示）。
  - 其他值：`ValueError`。
- `memory_limit`：最大中间张量元素数（int）。设定时必须走 cotengra（唯一的切片规划器）——若 cotengra 缺失抛 `ImportError`；`optimize` 为 `"opt_einsum"`/`"greedy"` 且给了 `memory_limit` 抛 `ValueError`（组合不合法）。

cotengra 规划分支：

```python
import cotengra as ctg
# indices -> inputs（每张量的指标元组），open_indices -> output，size_dict 全部维度 2
opt = ctg.HyperOptimizer(slicing_opts={"target_size": memory_limit} if memory_limit else None,
                         progbar=False, parallel=False)
tree = opt.search(inputs, output, size_dict)
path = tree.get_path()          # 线性 (i, j) 格式，与现有循环兼容
sliced = tuple(tree.sliced_inds)  # 可能为空
```

（`HyperOptimizer` 的具体参数名以安装版本实际 API 为准，实现时先探测；上面为意图描述。默认配置即可，不暴露 cotengra 内部调优参数。）

执行：

- `sliced` 为空：与现状相同，逐对收缩一次。
- `sliced` 非空：对每个切片赋值组合（`itertools.product([0, 1], repeat=len(sliced))`）：
  1. 复制张量列表；对含切片指标的张量，逐个用 `backend.take(t, axis, val)` 固定该指标（同时从该张量的指标元组中移除）；
  2. 用同一 `path` 跑 pairwise `_pair_contract` 收缩（切片后网络的指标集一致，路径可复用）；
  3. 首个切片结果作初值，其余用 `backend.add` 累加。
- open_indices 的末尾转置逻辑不变。
- 切片指标不允许出现在 `open_indices` 中（cotengra 默认也不切输出指标；实现处加断言防御）。

## 组件 3：公共 API 透传

`aicir/simulator/__init__.py` 四个函数各加两个 keyword 参数并透传到 `contract`：

```python
tn_statevector(circuit, *, backend=None, optimize="auto", memory_limit=None)
single_amplitude(circuit, bitstring, *, backend=None, optimize="auto", memory_limit=None)
partial_amplitude(circuit, *, open_qubits=None, bitstrings=None, backend=None, optimize="auto", memory_limit=None)
tn_expectation(circuit, observable, *, backend=None, optimize="auto", memory_limit=None)
```

默认值下行为与现状完全一致（cotengra 未安装时 auto 仍走 opt_einsum/greedy）。

## 错误处理

| 情形 | 行为 |
| --- | --- |
| `optimize="cotengra"`，cotengra 未装 | `ImportError`，提示 `pip install -e ".[tn]"` |
| `memory_limit` 设定，cotengra 未装 | `ImportError`，同上 |
| `memory_limit` + `optimize in ("opt_einsum", "greedy")` | `ValueError`（切片只有 cotengra 规划） |
| `optimize` 未知值 | `ValueError` |
| 切片指标含 open index | `AssertionError`（防御，正常不触发） |

## 测试

1. **原语单测**（`tests/backends/`）：`take`/`add` numpy+gpu 数值正确；`_NpuTakeFn`/`_NpuAddFn` CPU 上 forward/backward 对照 torch 原生 select/add + autograd（aclnnAdd 触发条件只在真机，CPU 测数值与梯度一致性，真机验证走 demo）。
2. **cotengra 路径 parity**（importorskip cotengra）：`optimize="cotengra"` 无切片，vs statevector 引擎，numpy+gpu。
3. **强制切片 parity**：`memory_limit` 调小到必然切片（如 3 比特电路给 `memory_limit=4` → 中间张量 8 元素必超），验证结果仍与 statevector 一致，且确认 `tree.sliced_inds` 非空（测试内探测，防"没切到片的假通过"）。
4. **切片 autograd**：torch 参数 `requires_grad=True`，强制切片下 `tn_expectation(...).backward()`，梯度对照 psr。
5. **guard 测试**：上表 4 种错误情形。
6. **NPU 远程验证**：`demos/demo_npu_tensor.py` 加一节切片检查（cotengra 装了才跑，`--allow-cpu-fallback` 兼容），覆盖真机 `_NpuTakeFn`/`_NpuAddFn`。
7. 全量回归基线不变。

实现前先 `pip install cotengra`（本地当前未装）。

## 文档

- `aicir/simulator/README.md`：§4「收缩路径与可选依赖」重写——三级路径来源、`optimize`/`memory_limit` 用法、切片语义（内存预算、串行、保可微）、`tn` extra。
- `CHANGELOG.md`：2026-07-03 Added 条目。
- `compare.md`：更新能力矩阵与 §4/§19/§20——「cotengra 级路径优化仅 WuYue」这一差距条目移除/改写。
- `CONTENTS.md`：无新目录，不动。
