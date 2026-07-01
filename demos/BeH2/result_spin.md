# BeH2 基态能量结果（NPU 自旋保持 ansatz 版）

本次结果使用 `demos/BeH2/BeH2.py` 中的 16 比特 BeH2 活性空间 Hamiltonian（3-21G 基组、6 电子/8 空间轨道活性空间、Jordan-Wigner 映射，共 1313 个 Pauli 项），以及 `demos/BeH2/BeH2_cir_spin.py` 中保存的、在 **4 块 Ascend NPU**（`device=npu:0`，`world_size=4`，搜索内分片 in-search sharding）上由 supernet QAS 搜索得到的线路。

该线路采用分子 VQE 更合适的粒子数/自旋保持 ansatz：先用 6 个 `pauli_x` 制备 closed-shell Hartree-Fock 参考态（占据比特 `5, 6, 7, 13, 14, 15`），再从 `single_excitation` 与 `double_excitation` 门池中搜索激发算符。

## 计算结果

| 项目 | 能量 / Ha |
| --- | ---: |
| Hamiltonian 精确基态能量 | -19.1421356201 |
| 线路能量 - 精确基态能量 | +0.0015367993 |
| 绝对误差 | 0.0015367993 |
| NPU 搜索报告 fine-tuned energy | -19.1405754089 |
| 固定 ansatz VQE 基线 | -19.1407012939 |
| 保存线路重新计算能量 | -19.1405988209 |
| 保存线路能量 - NPU 报告能量 | -0.0000234119 |

本次精确基态能量由对角化求得（`eigsh(Lanczos,SA)@cpu`）：构造 16 比特 Hamiltonian（2^16=65536 维）后取最小特征值。

- 保存线路能量比精确基态能量高约 `1.537 mHa`，处于化学精度（约 1.6 mHa）以内（线路能量 ≥ 精确基态能量，满足变分下界）。
- 同一次运行的固定 ansatz VQE 基线 `-19.1407012939 Ha` 相对精确基态高约 `1.434 mHa`。

结论：

- 保存线路（`BeH2_cir_spin.py`）经 statevector 复算得到 `-19.1405988209 Ha`，与 NPU 运行报告的 `fine-tuned energy = -19.1405754089 Ha` 一致，相差约 `2.34e-05 Ha`，源于 NPU complex64 精度与求值路径差异。
- 值得注意的是，同一次 NPU 运行中**固定 ansatz 的 VQE 基线 `-19.1407012939 Ha` 比 QAS 搜索到的线路更低（更优）约 `0.13 mHa`**。也就是说，在这组 16 比特 / `L=5` 配置下，supernet 搜索本身没有收敛到优于固定基线的结构——这与 H2O 的 NPU 版（`demos/H2O/result_npu.md`）观察到的现象一致：更大的搜索空间加剧了权重共享 supernet 的“竞争”，最终选出的子线路欠优化。若要改善，可考虑增大 `supernet_steps`/`finetune_steps`、调整 `supernet_num (W)` 或减小层数。

## 搜索配置

本次 NPU 运行（见 `demos/BeH2/output_spin.txt` 头部，对应 `demos/BeH2/BeH2_npu.py`）的设置：

- 设备：4 块 Ascend NPU，`device=npu:0`，`world_size=4`，`mode=safe`（与单卡数值等价的搜索内分片）
- 量子比特数：16，层数（深度）：5
- HF 参考态占据比特：`(5, 6, 7, 13, 14, 15)`
- 单比特门池：`{i}`，用于保留 supernet 每层可跳过的占位门
- 两比特门池：`{single_excitation}`，连接来自 closed-shell spin-preserving singles
- 四比特门池：`{double_excitation}`，连接来自 closed-shell paired doubles
- 超网络数 `supernet_num=6`，搜索步数 `supernet_steps=300`，排序数 `ranking_num=120`，微调步数 `finetune_steps=500`
- 学习率 `learning_rate=0.08`，微调学习率 `finetune_learning_rate=0.04`
- 梯度方式：自动微分 `ad`（`--gradient ad`）
- 随机种子 `seed=7`（所有 rank 同 seed，保证权重/候选集一致）
- 任务 `task="vqe"`
- 总墙钟时间：约 16630.1 s（约 4.62 小时）

本次保存线路包含 121 个门：

- 6 个 `pauli_x` 门用于制备 HF 参考态
- 77 个 `single_excitation` 门
- 38 个 `double_excitation` 门

（NPU 报告中的 `excitations = 115 (single=77, double=38)` 仅统计激发门，不含 6 个 HF 制备门。）

## 生成文件

```text
demos/BeH2/BeH2_cir_spin.py
demos/BeH2/output_spin.txt
demos/BeH2/BeH2_npu_cir.qasm
```

OpenQASM 3.0 导出当前会跳过，因为 QASM 导出器暂不支持 `single_excitation` / `double_excitation` 这类高层化学激发门。

## 验证方式

精确基态能量与线路能量都由本脚本算得（求解路径：`eigsh(Lanczos,SA)@cpu`）：

1. **精确基态能量**：用同一套“带符号置换”逻辑高效构造 16 比特 Hamiltonian（每个 Pauli 串每行只有一个非零，逐项向量化累加，避免 `to_matrix` 的逐项 2^16 临时矩阵），再取最小特征值。默认走稀疏 Lanczos `scipy.sparse.linalg.eigsh(k=1, which="SA")`（只求最小特征值，几秒、GB 级内存，对极端特征值收敛到精确值）。Ascend 当前没有 complex `eigvalsh` 内核，故 `--method dense` 的全谱稠密对角化会被 torch 自动回退到 CPU 运行（约 70–100 GB 工作区、耗时很长），仅建议在大内存机器上使用。
2. **线路能量**：按门逐个作用到态向量（16 比特无法构造 65536×65536 全局 unitary，故不走 `circuit.unitary()`），再对 1313 个 Pauli 项逐项计算 `<psi|P|psi>` 求和。

```bash
# 默认稀疏 Lanczos 求精确基态能量 + 复算线路能量并刷新本文件（几分钟）
python -m demos.BeH2.BeH2_result_spin

# 如需完整稠密对角化（大内存机器；Ascend 无内核会回退到 CPU 全谱）
python -m demos.BeH2.BeH2_result_spin --method dense
```

复算输出（device=npu:0）：

```text
exact   = -19.1421356201 Ha
circuit = -19.1405988209 Ha
```

## 复现方式

```bash
# 在 4 块 Ascend NPU 上重新搜索（需 torch_npu，约 5 小时）
torchrun --nproc_per_node=4 demos/BeH2/BeH2_npu.py --gradient ad --output output_spin.txt

# 仅根据已记录的线路重新绘图
python -m demos.BeH2.BeH2_cir_spin
```
