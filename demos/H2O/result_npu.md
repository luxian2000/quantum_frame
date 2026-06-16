# H2O 基态能量对比结果（NPU 版）

# NPU 实验 1

本次对比使用 `demos/H2O/H2O.py` 中的 6 比特 H2O 活性空间 Hamiltonian（STO-3G 基组、4 电子/3 空间轨道活性空间、Jordan-Wigner 映射），以及 `demos/H2O/H2O_cir_npu.py` 中保存的、在 **Ascend NPU**（`device=npu:0`）上由 supernet QAS 搜索得到的线路（`L=8` 的更深 ansatz）。

> 与 CPU 版结果（`demos/H2O/result.md`，线路能量约 -6.1553 Ha）平行，本文件记录 NPU 版搜索线路的能量复算结果，命名沿用 NPU 流水线的 `_npu` 后缀。

## 计算结果

| 项目                       |       能量 / Ha |
| -------------------------- | --------------: |
| Hamiltonian 精确基态能量   | -6.159663677216 |
| 线路态能量（NPU QAS 线路） | -5.814736366272 |
| 线路能量 - 精确基态能量    | +0.344927310944 |
| 绝对误差                   |  0.344927310944 |

该复算值与 NPU 运行报告 `demos/H2O/H2O_npu_result.txt` 中记录的 `QAS fine-tuned energy = -5.8147358894 Ha` 一致（相差约 `1e-6`，源于 complex64 精度与求值路径差异）。

结论：`H2O_cir_npu.py` 中的线路得到的能量比精确基态能量高约 `3.45e-1 Ha`（约 344.9 mHa），满足变分法（Rayleigh-Ritz）下界要求（线路能量 ≥ 精确基态能量），但**未**达到化学精度（1.6 mHa）。

值得注意的是，同一次 NPU 运行中固定 ansatz 的 VQE 基线达到 `-6.1591339111 Ha`（仅高于精确基态约 0.53 mHa，接近化学精度），明显优于本次 QAS 搜索到的线路（344.9 mHa）。也就是说，在这组更深（`L=8`）配置下，supernet 搜索本身没有收敛到一个优于固定基线的结构——更深的搜索空间反而加剧了权重共享 supernet 的“竞争”，最终选出的子线路欠优化。若要改善，可考虑增大 `supernet_steps`/`finetune_steps`、调整 `supernet_num (W)`、或减小层数后再加深。

## 搜索配置

本次 NPU 运行（`python -m demos.H2O.H2O_npu`，见 `H2O_npu_result.txt` 头部）的设置：

- 设备：`npu:0`（torch 2.4.0，torch_npu，4 卡可用）
- 量子比特数：6，层数（深度）：8
- 单比特门池：`{i, h, rx, ry, rz}`
- 双比特门池：`{cx, rzz}`（`rzz` 为带可训练参数的纠缠门）
- 双比特连接：最近邻 + 次近邻，共 9 对
- 超网络数 `supernet_num=5`，搜索步数 `supernet_steps=250`，排序数 `ranking_num=80`，微调步数 `finetune_steps=250`
- 随机种子 `seed=2`
- 训练方式：标准 autograd（`loss.backward()`，经 `NPUBackend` 自定义 `autograd.Function` 与门矩阵构造的 NPU 兼容处理）
- 墙钟时间：约 1320.5 s

选出的线路（`H2O_cir_npu.py`）共 85 个门：14×`H`、5×`rx`、6×`ry`、13×`rz`、23×`cx`、24×`rzz`，其中双比特门 47 个（`cx` 23 + `rzz` 24）。

## 验证方式

线路能量通过两种后端各自计算，结果一致：

1. `GPUBackend`（CPU，torch）：`apply_unitary(zeros_state, circuit.unitary(backend))` 演化后计算 `<psi|H|psi>` → `-5.814736366272 Ha`。
2. `NumpyBackend`：同样流程 → `-5.814737796783 Ha`。

两种后端互相吻合（`|Δ| ≈ 1.4e-6`），且与 NPU 运行报告的 `QAS fine-tuned energy` 一致。

复算命令：

```bash
PYTHONPATH=. python - <<'PY'
from demos.H2O.H2O import build_h2o_hamiltonian, exact_ground_energy
from demos.H2O.H2O_cir_npu import build_h2o_npu_qas_circuit
from aicir.channel.backends.gpu_backend import GPUBackend

ham = build_h2o_hamiltonian()
exact = exact_ground_energy(ham)
be = GPUBackend(device="cpu")
H = ham.to_matrix(be)
circuit = build_h2o_npu_qas_circuit()
state = be.apply_unitary(be.zeros_state(circuit.n_qubits), circuit.unitary(backend=be))
energy = float(be.expectation_sv(state, H).real)
print(f"exact   : {exact:.12f} Ha")
print(f"circuit : {energy:.12f} Ha")
print(f"|error| : {abs(energy-exact)*1000:.3f} mHa")
PY
```

## 复现方式

```bash
# 在 Ascend NPU 上重新搜索并生成 H2O_cir_npu.py / .qasm / .png（需 torch_npu，约 22 分钟）
python -m demos.H2O.H2O_npu

# 仅根据已记录的线路重新绘图
python -m demos.H2O.H2O_cir_npu
```
