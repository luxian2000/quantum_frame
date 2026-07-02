# H2O 基态能量对比结果（QDRATS + excitation ansatz）

本次对比使用 `demos/H2O/H2O.py` 中的 6 比特 H2O 活性空间 Hamiltonian（STO-3G 基组、4 电子/3 空间轨道活性空间、Jordan-Wigner 映射），以及 `demos/H2O/H2O_qdrats.py` 用 QDRATS（QuantumDARTS 可微架构搜索，`aicir.qas` 方法名 `qdrats`）在 **excitation 门池** 上搜索得到、并保存在 `demos/H2O/H2O_qdrats_cir.py` 中的线路。

QDRATS 现支持可配置门池（`QDRATSConfig.gate_pool`）。本 demo 设 `gate_pool="excitation"`：从 closed-shell Hartree-Fock 参考态出发，每个 slot 在「施加一个参数化 `single_excitation`/`double_excitation` 算符」与「identity」之间用 Gumbel-Softmax 离散选择，激发角度交替优化。该 ansatz 粒子数/自旋守恒，与 supernet 版本（`result.md`）同款，因此能逼近化学精度——明显优于通用 `cx`/`rz·ry·rz` 门池。

## 计算结果

| 项目 | 能量 / Ha |
| --- | ---: |
| Hamiltonian 精确基态能量 | -6.1596636772 |
| QDRATS（excitation）搜索最小能量 | -6.1596274376 |
| QDRATS 能量 - 精确基态能量 | +0.0000362396 |
| 绝对误差 | 0.0000362396 |

结论：QDRATS 在 excitation 门池上搜索得到的线路能量比精确基态能量高约 `3.62e-05 Ha`（约 0.036 mHa），**处于化学精度（约 1.6 mHa）以内**，与 supernet 版本（`result.md`，约 9.5e-7 Ha）同一量级，远优于通用门池版本（旧结果约 156 mHa）。这说明把门池从通用门换成守恒型 excitation 算符后，QDRATS 也能逼近 H2O 基态。

## 搜索配置

`demos/H2O/H2O_qdrats.py` 中 `h2o_qdrats_config` 的默认设置：

- 量子比特数：6，层数：2
- 门池 `gate_pool="excitation"`
- HF 参考态占据比特：`(1, 2, 4, 5)`（线路以 4 个 `pauli_x` 制备）
- 单激发池 `single_excitations`、双激发池 `double_excitations`：取自 `demos/chemistry_ansatz.closed_shell_excitation_pools`（spin-preserving singles 与 closed-shell paired doubles）
- 搜索轮数 `search_epochs=80`，每轮角度优化步数 `theta_steps=2`，离散化后微调步数 `finetune_steps=150`
- 学习率：架构 `architecture_learning_rate=0.05`，角度 `theta_learning_rate=0.05`，微调 `finetune_learning_rate=0.03`
- Gumbel 温度 `temperature=1.0`（带退火），随机种子 `seed=2`，设备 `device="cpu"`

搜索选中的离散架构（每层每个激发 slot 选 excitation 或 identity）：

```text
layer 0: single_0_1
layer 1: single_0_1, double_0_3_2_5, double_0_3_1_4
```

本次保存线路包含 8 个门：

- 4 个 `pauli_x` 门用于制备 HF 参考态
- 2 个 `single_excitation` 门
- 2 个 `double_excitation` 门

## transpile 优化

`H2O_qdrats_cir.py` 在 `__main__` 中先用 `aicir.transpile.optimize`（默认本地流水线：cancel-inverse → merge-rotations → commute-single-qubit）优化线路，再绘图。本次：

```text
Optimized circuit: 8 -> 7 gates
```

merge-rotations 现支持 excitation 门：`single_excitation`/`double_excitation` 为固定生成元的旋转门，角度可加（`G(θ1)·G(θ2)=G(θ1+θ2)`），故相邻、同操作数的两个门按角度相加合并（角度抵消为 0 时整对消去）。本次搜索在第 0、1 层都选中了 `single_0_1` 且二者相邻，于是合并为一个 `single_excitation`，门数 `8 -> 7`（4 个 HF `pauli_x` + 1 个合并后的 `single_excitation` + 2 个 `double_excitation`）。

## 生成文件

```text
demos/H2O/H2O_qdrats_cir.py
demos/H2O/H2O_qdrats_cir.png
```

`H2O_qdrats_cir.py` 是自包含的可运行模块：导出 `build_h2o_qdrats_circuit()` 与模块级 `circuit`，并在作为 `__main__` 运行时 transpile 优化后绘图到 `H2O_qdrats_cir.png`。

## 复现方式

```bash
# 重新搜索并生成 H2O_qdrats_cir.py / H2O_qdrats_cir.png（6 比特，CPU 约数分钟）
PYTHONPATH=. python -m demos.H2O.H2O_qdrats

# 仅根据已记录的线路 transpile 优化后重新绘图
PYTHONPATH=. python -m demos.H2O.H2O_qdrats_cir
```

> 注：QDRATS 为随机化可微搜索，不同 `seed` 或更大的 `search_epochs`/`layers`/`finetune_steps` 会得到不同线路与能量；上表数值对应 `seed=2`、`gate_pool="excitation"` 的默认配置。
