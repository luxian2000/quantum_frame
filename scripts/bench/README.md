# 跨框架基准（scripts/bench）

面向论文 §8 的跨框架性能基准。与 `scripts/npu/` 的探针脚本并列：那边验真机正确性，
这边出可对外发表的性能数字。

## 快速开始

```sh
# 1) 先验有效性——各框架是否在跑同一条线路
PYTHONPATH=. python scripts/bench/run_bench.py --axis parity

# 2) 微基准：GHZ 逐比特数
PYTHONPATH=. OMP_NUM_THREADS=1 python scripts/bench/run_bench.py \
    --axis micro --min-qubits 8 --max-qubits 20

# 3) 标准线路：QFT / random / layered_ansatz
PYTHONPATH=. OMP_NUM_THREADS=1 python scripts/bench/run_bench.py \
    --axis circuits --max-qubits 20 --output-json bench.json

# 4) 全量 + 归档清单
PYTHONPATH=. OMP_NUM_THREADS=1 python scripts/bench/run_bench.py \
    --axis all --max-qubits 22 \
    --output-json docs/evidence/benchmarks/$(git rev-parse HEAD)/cpu.json
```

## 三条方法学铁律

1. **parity 先于计时。** 各框架的比特序、门约定并不一致（Qiskit 态矢量是小端、
   Qulacs 的 `RX` 用 `exp(+iθ/2 P)` 约定、Cirq 的 `CZPowGate` 以 π 为单位）。
   任何一处错了，计时表比较的都是**不同的计算**，且错得很安静。运行器在 parity
   未通过时**拒绝产出计时数据**。
2. **构建与执行分开计时。** 把 Qiskit 的线路组装/转译算进执行时间，比较的就变成了
   编译器而非模拟器。参考 *Benchmarking the performance of quantum computing software
   for quantum circuit creation, manipulation and compilation*。
3. **中位数 + IQR，预热剔除。** 均值会被一次 GC 停顿毁掉；不报离散度就无法判断
   差异是否显著；TensorCircuit 等有 JIT 的框架首次调用含编译时间。

## 精度对齐

跨精度比较是无效的（complex64 在带宽受限的态矢量演化上天然快约 2×）。
`--precision double|single` 会把各框架统一到同一精度：

| 框架 | double | single |
| --- | --- | --- |
| aicir | `NumpyBackend(complex128)` | `NumpyBackend(complex64)` |
| Qiskit Aer | `precision="double"` | `precision="single"` |
| Cirq | `complex128` | `complex64` |
| TensorCircuit | `tc.set_dtype("complex128")` | `complex64` |
| Qulacs | 仅 double | **不可用**（无单精度构建） |

Qulacs 只在 double 表中出现——论文表格需注明这是构建限制，不是它跑不动。

## 结构

```text
scripts/bench/
├── core/
│   ├── spec.py       # 框架无关的 CircuitSpec：所有适配器的唯一输入
│   ├── timing.py     # 中位数/IQR、预热剔除、构建-执行分离
│   └── manifest.py   # 证据清单（对齐 distributed-autograd 的形状）
├── adapters/         # 每框架一个适配器，比特序在此归一化
├── axes.py           # 轴 C–G（VQE / 内存 / NPU / 多卡 / 能力矩阵）
└── run_bench.py      # CLI
```

新增框架只需实现 `adapters/base.py:Adapter` 的 `is_available`/`build`/`run`，
并在 `adapters/__init__.py:ADAPTERS` 注册；parity 测试会自动把它纳入校验。

## 线路族

| 族 | 用途 |
| --- | --- |
| `ghz` | 最浅纠缠，主要压门派发开销 |
| `qft` | 稠密受控相位，双比特门数 O(n²) |
| `random` | 砖墙式随机线路，近似 supremacy 类基准结构 |
| `layered_ansatz` | 硬件高效 ansatz，代表变分工作负载 |

规格是**声明式**的：同 family 同 seed 必然逐字节复现同一条线路，与任何框架的
随机数状态无关（`tests/bench/test_bench_harness.py` 钉住了这一点）。

## 清单

每次运行产出的 JSON 记录 commit SHA、工作区是否干净、`run_id`、内容 `sha256`、
各框架版本、Python/平台/CPU、**线程环境**（`OMP_NUM_THREADS` 等）与 numpy 链接的
BLAS 实现。不记录线程数与 BLAS，跨机器的 CPU 计时无法解释。

工作区脏（`worktree_dirty=true`）时跑出的数字不可复现，归档前请先提交。

## 测量轴

| 轴 | `--axis` | 内容 | 参与者 |
| --- | --- | --- | --- |
| A | `micro` | 单门吞吐（GHZ）vs n | 全部 CPU 框架 |
| B | `circuits` | QFT / random / layered_ansatz vs n | 全部 CPU 框架 |
| C | `vqe` | 变分负载：单次能量、单次梯度（参数移位） | 目前仅 aicir |
| D | `memory` | 峰值内存 vs 理论下界 | 全部（有局限，见下） |
| E | `npu` | 昇腾 vs CPU | 仅 aicir，**需真机** |
| F | `scaling` | 多卡强/弱扩展计划 | 仅 aicir，**需真机** |
| G | `capability` | 能力矩阵：谁**跑得了**每一行 | 全部 |

A–D 用来建立可信度，E–G 承载论点。**A–D 输掉某一行可以接受并须如实报告；
悄悄不报不可接受。**

### 轴 C（VQE）为何只有 aicir

梯度走参数移位（所有框架都能做、且与硬件语义一致；自动微分只有部分框架支持，
混在一起比较不成立）。但各框架的 VQE 封装差异很大，公平的跨框架 VQE 需要为每个
框架单独实现等价的能量/梯度路径——尚未完成，是当前最大缺口。

轴 C 用 `StatevectorEstimator` primitive，即 `BasicVQE` 默认的精确能量路径，
因此测到的是真实 VQE 循环里的那条路径。返回值含能量与谱界 `Σ|cᵢ|`，
给计时结果加一道物理校验——算得快但算错没有意义。

**耗时提示**：参数移位的代价是 `2 × n_params` 次能量求值，`n=12`、2 层 HEA
约 72 个参数即 144 次求值。跑轴 C 时请调小 `--max-qubits` 与 `--repeats`。

### 轴 D（内存）的局限

`tracemalloc` 只看得见 Python 分配器。Aer/Qulacs 的 C++ 后端在其内部分配的内存
**不计入**，因此内存列**不可跨框架横比**，只能在同一框架内看 n 的增长趋势与
`overhead_ratio`（实测峰值 / `2^n·16B` 理论下界）。论文中必须写明这一点。

### 轴 E/F 在无真机时的行为

只产出**计划**并记录跳过原因，同时在本机校验参数合法性（`world_size` 为 2 的幂、
`n_qubits >= p`）——避免上真机排队数小时后才发现配置非法。

## 已知缺口

- 轴 C 尚未覆盖 Qiskit/Cirq/Aer——跨框架 VQE 比较需要为每个框架实现等价封装。
- 轴 E/F 未在真机执行（本机无昇腾设备）。
- `qulacs` / `tensorcircuit` 未安装时自动跳过并在清单中记录。
