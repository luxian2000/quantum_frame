# H2O 基态能量对比结果

本次对比使用 `demos/H2O/H2O.py` 中的 6 比特 H2O 活性空间 Hamiltonian（STO-3G 基组、4 电子/3 空间轨道活性空间、Jordan-Wigner 映射），以及 `demos/H2O/H2O_cir.py` 中保存的 supernet QAS 搜索线路。

当前线路已经从原来的硬件高效门池改为分子 VQE 更合适的粒子数/自旋保持 ansatz：先制备 closed-shell Hartree-Fock 参考态，再从 `single_excitation` 和 `double_excitation` 门池中搜索激发算符。

## 计算结果

| 项目 | 能量 / Ha |
| --- | ---: |
| Hamiltonian 精确基态能量 | -6.159663677216 |
| 保存线路重新计算能量 | -6.159662723541 |
| 线路能量 - 精确基态能量 | +0.000000953674 |
| 绝对误差 | 0.000000953674 |

结论：`H2O_cir.py` 中保存的 HF + excitation 线路得到的能量比精确基态能量高约 `9.54e-7 Ha`（约 0.000954 mHa），已经明显优于旧的硬件高效 ansatz 结果，并处于化学精度（约 1.6 mHa）以内。

运行 `demos/H2O/H2O.py` 时，supernet 搜索日志报告：

| 项目 | 能量 / Ha |
| --- | ---: |
| QAS fine-tuned energy | -6.1596651077 |
| fixed-ansatz VQE baseline | -6.1596722603 |
| `|QAS - exact|` | 1.431e-06 |

这些搜索阶段指标来自有限精度 tensor 优化，可能在微 Hartree 尺度上略低于 dense-matrix exact 值；因此上面的“保存线路重新计算能量”更适合作为 `H2O_cir.py` 的可复现线路能量。

## 搜索配置

`demos/H2O/H2O.py` 中 `h2o_vqe_qas_config` 的默认设置：

- 量子比特数：6，层数：6
- HF 参考态占据比特：`(1, 2, 4, 5)`
- 单比特门池：`{i}`，用于保留 supernet 每层可跳过的占位门
- 两比特门池：`{single_excitation}`
- 四比特门池：`{double_excitation}`
- 搜索池：spin-preserving singles 和 closed-shell paired doubles
- 超网络数 `supernet_num=5`，搜索步数 `supernet_steps=250`，排序数 `ranking_num=80`，微调步数 `finetune_steps=250`
- 学习率 `learning_rate=0.1`，微调学习率 `finetune_learning_rate=0.05`
- 随机种子 `seed=2`，任务 `task="vqe"`

本次保存线路包含 25 个门：

- 4 个 `pauli_x` 门用于制备 HF 参考态
- 14 个 `single_excitation` 门
- 7 个 `double_excitation` 门

## 生成文件

```text
demos/H2O/H2O_cir.py
demos/H2O/H2O_cir.png
```

OpenQASM 3.0 导出当前会跳过，因为 QASM 导出器暂不支持 `single_excitation` / `double_excitation` 这类高层化学激发门。

## 验证方式

保存线路能量通过 statevector 演化后计算 `<psi|H|psi>`：

```bash
PYTHONPATH=. python - <<'PY'
from demos.H2O.H2O import build_h2o_hamiltonian, exact_ground_energy
from demos.H2O.H2O_cir import build_h2o_qas_circuit
from aicir.backends.gpu_backend import TorchBackend

h = build_h2o_hamiltonian()
exact = exact_ground_energy(h)
backend = TorchBackend(device="cpu")
matrix = h.to_matrix(backend)
circuit = build_h2o_qas_circuit()
state = backend.apply_unitary(
    backend.zeros_state(circuit.n_qubits),
    circuit.unitary(backend=backend),
)
energy = float(backend.expectation_sv(state, matrix).real)

print(f"exact  = {exact:.12f}")
print(f"circuit= {energy:.12f}")
print(f"delta  = {energy - exact:+.12f}")
PY
```

输出：

```text
exact  = -6.159663677216
circuit= -6.159662723541
delta  = +0.000000953674
```

## 复现方式

```bash
# 重新搜索并生成 H2O_cir.py / H2O_cir.png（约 1 分钟）
PYTHONPATH=. python demos/H2O/H2O.py

# 仅根据已记录的线路重新绘图
PYTHONPATH=. python demos/H2O/H2O_cir.py
```
