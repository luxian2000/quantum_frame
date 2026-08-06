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

```
scripts/bench/
├── core/
│   ├── spec.py       # 框架无关的 CircuitSpec：所有适配器的唯一输入
│   ├── timing.py     # 中位数/IQR、预热剔除、构建-执行分离
│   └── manifest.py   # 证据清单（对齐 distributed-autograd 的形状）
├── adapters/         # 每框架一个适配器，比特序在此归一化
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

## 已知缺口

- 尚未覆盖的轴：VQE 端到端墙钟（轴 C）、峰值内存（轴 D）、NPU vs CPU（轴 E）、
  多卡强/弱扩展（轴 F）、能力矩阵（轴 G）。
- `qiskit-aer` / `qulacs` / `tensorcircuit` 未安装时自动跳过并在清单中记录。
