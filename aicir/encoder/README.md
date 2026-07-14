# aicir.encoder — 经典数据 → 量子态编码器

本模块提供经典数据到量子态的编码（态准备）工具。所有编码器继承抽象基类 `BaseEncoder`，遵循统一契约：`encode(data)` 返回 `(线路表示, 量子态)` 二元组，线路是原生 `Circuit`（或按需导出为 QASM / DAG），量子态是 `aicir.core.State`，可直接接入 `Measure`、`aicir.metrics`、QML 等下游模块。

```python
from aicir.encoder import AmplitudeEncoder, AngleEncoder, BasisEncoder, IQPEncoder
```

---

## 1. 编码器选择指南

```
数据是归一化后有意义的向量（信号、图像展平、核方法特征）？
  -> AmplitudeEncoder
     N 维数据编入 2^n 个振幅，n = ceil(log2 N)，比特数最省。

每个特征独立、量纲各异，想要浅线路（QNN 输入层常用）？
  -> AngleEncoder
     x_i -> 单比特旋转角，1 特征 1 比特，线路深度 1。

数据本质是离散值 / 位串（组合优化解、类别标签）？
  -> BasisEncoder
     每个取值映射到一个计算基态，输出基态叠加（可带频次权重）。

做量子核方法 / QSVM，需要经典难模拟的特征映射？
  -> IQPEncoder
     Havlicek et al. (Nature 567, 209 (2019)) 的 IQP 特征映射，
     自带 kernel / kernel_matrix，可直接喂经典 SVM。
```

---

## 2. 公共接口一览

| 类 | 文件 | encode 输入 | 比特数 | decode |
| --- | --- | --- | --- | --- |
| `BaseEncoder` | `abstract.py` | —（抽象基类） | — | — |
| `AmplitudeEncoder` | `amplitude.py` | 长度 ≤ 2^n 的实/复向量 | `ceil(log2 N)` 或指定 | 返回振幅实部 |
| `AngleEncoder` | `angle.py` | 长度 ≤ n 的实向量 | 特征数或指定 | 由各比特 P(1) 反解角度 |
| `BasisEncoder` | `basis.py` | 任意实数列表 | 自动取最小无碰撞位宽或指定 | 返回最大概率基态位串 |
| `IQPEncoder` | `iqp.py` | 长度 ≤ n 的实向量 | 特征数或指定 | 不可逆，抛 `NotImplementedError` |

### encode 统一签名

```python
circuit_repr, state = encoder.encode(data, *, cir="dict", backend=None)
```

- `cir`：线路导出格式。`"dict"` 返回原生 `Circuit` 对象（默认，名称是历史遗留）；`"qasm"` 返回 OpenQASM 2.0 字符串；`"dag"` 返回 DAG 表示。
- `backend`：数值后端，缺省为 `NumpyBackend()`（complex64；核值/振幅精度约 1e-6）。可传 `GPUBackend` / `NPUBackend`。
- `AmplitudeEncoder` / `AngleEncoder` / `IQPEncoder` 在 `n_qubits` 指定且数据长度不足时自动补零，数据超长则抛 `ValueError`；`BasisEncoder` 的位宽语义见 §5。

---

## 3. AmplitudeEncoder — 振幅编码

把归一化后的数据向量直接写入态振幅：`x -> sum_i (x_i / ||x||) |i>`。

```python
from aicir.encoder import AmplitudeEncoder

enc = AmplitudeEncoder()                 # 比特数 = ceil(log2 N)，自动补零
circuit, state = enc.encode([0.5, 0.5, 0.5, 0.5])
enc.decode(state)                        # -> [0.5, 0.5, 0.5, 0.5]（振幅实部）
```

- 线路是单个 `unitary` 门：用 Gram–Schmidt 构造首列为目标态的酉阵。
- 输入允许复数；全零向量（范数为 0）抛 `ValueError`。
- `decode` 只返回振幅**实部**，复数据编码后解码有损。

## 4. AngleEncoder — 角度编码

每个特征作为一个单比特旋转角：`x_i -> R(x_i)` 作用在第 i 个比特。

```python
from aicir.encoder import AngleEncoder

enc = AngleEncoder(rotation="ry")        # rotation 可选 "rx" / "ry" / "rz"
circuit, state = enc.encode([0.3, 1.2, 2.5])   # 3 特征 -> 3 比特，深度 1
angles = enc.decode(state)               # 由各比特 P(|1>) 反解 2*arcsin(sqrt(p1))
```

- `rz` 编码只改相位、不改测量概率，`decode` 对 `rz` 恒返回 0——需要可反解时用 `rx`/`ry`。
- 特征值建议先缩放到 `[0, pi]`，避免 `2*arcsin` 反解分支歧义。

## 5. BasisEncoder — 基态编码

把离散数据集编码为计算基态的均匀（或频次加权）叠加，线路由分治决策树生成（`ry` + 多控 `cry`）。

```python
from aicir.encoder import BasisEncoder

enc = BasisEncoder(redundant=False)      # redundant=False：去重，均匀叠加
_, state = enc.encode([1, 2, 2, 2])      # -> (|b1> + |b2>)/sqrt(2)

enc = BasisEncoder(redundant=True)       # redundant=True：保留频次权重
_, state = enc.encode([1, 2, 2, 2])      # -> P(b2)/P(b1) = 3
```

- 实数输入先线性缩放到 `[0, 2^n - 1]` 再取整；自动模式选取"缩放后无碰撞"的最小位宽。
- `n_qubits` 指定值若小于最小无碰撞位宽，会被抬升到该位宽（不静默丢失分辨率）。
- `decode` 返回最大概率基态的位串（`list[int]`）。

## 6. IQPEncoder — IQP 特征映射（量子核方法）

实现 Havlicek et al., *Supervised learning with quantum-enhanced feature spaces*（Nature 567, 209 (2019), arXiv:1804.11326）的特征映射：

```
|Phi(x)> = (U_Phi(x) H^n)^reps |0>^n
U_Phi(x) = exp(i * sum_S phi_S(x) * prod_{i in S} Z_i),  |S| <= 2
```

默认系数取论文选择：`phi_i = x_i`，`phi_ij = (pi - x_i)(pi - x_j)`。对角层用 `rz`/`rzz` 精确实现（`exp(i*phi*Z) == rz(-2*phi)`，无全局相位差）。

```python
from aicir.encoder import IQPEncoder

enc = IQPEncoder()                       # reps=2（同论文）、全连接纠缠
circuit, state = enc.encode([0.7, 2.1])  # 2 特征 -> 2 比特

# 量子核：K(x, z) = |<Phi(x)|Phi(z)>|^2，可直接喂 sklearn.svm.SVC(kernel="precomputed")
K = enc.kernel([0.7, 2.1], [1.3, 0.4])
G = enc.kernel_matrix(X_train)                   # 对称 Gram 矩阵
K_test = enc.kernel_matrix(X_test, X_train)      # 测试 × 训练核矩阵

# 只要线路不要态（如送真机 / 转 QASM）
qc = enc.circuit([0.7, 2.1])
```

### 构造参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `n_qubits` | `None` | 缺省取特征数；指定时数据不足补零 |
| `reps` | `2` | `(H 层 + 对角层)` 重复次数，论文取 2；`reps=1` 时核可经典估计（论文 supplementary），失去量子优势 |
| `entanglement` | `"full"` | ZZ 配对拓扑：`"full"`（全对）/ `"linear"`（相邻）/ 显式配对列表如 `[(0, 2)]` |
| `data_map` | 论文映射 | 可调用对象 `f(values) -> float`，`values` 是 1 个（单比特项）或 2 个（ZZ 项）特征值的列表 |

### 专有方法

| 方法 | 返回 | 说明 |
| --- | --- | --- |
| `circuit(data)` | `Circuit` | 只构建特征映射线路，不做态演化 |
| `kernel(x, z)` | `float` | 单个核值 `\|<Phi(x)\|Phi(z)>\|^2 ∈ [0, 1]` |
| `kernel_matrix(xs, zs=None)` | `ndarray` | 批量核矩阵；`zs` 缺省时为对称 Gram 矩阵（对角元 ≈ 1） |
| `decode(state)` | — | IQP 映射不可逆，恒抛 `NotImplementedError` |

特征值建议缩放到 `(0, 2pi]`（与论文数据域一致）；`data_map` 自定义时注意保持实值输出。

---

## 7. 自定义编码器

继承 `BaseEncoder` 并实现 `encode` / `decode` 两个抽象方法即可：

```python
from aicir.encoder import BaseEncoder

class MyEncoder(BaseEncoder):
    def encode(self, data, *, cir="dict", backend=None):
        ...
        return circuit, state    # 契约：返回 (线路表示, State) 二元组

    def decode(self, quantum_state):
        ...                      # 不可逆编码可抛 NotImplementedError
```

---

## 8. 相关目录

- `demos/`：编码示例脚本与导出的 QASM 样例（`encode_1234_demo.py`）。
- 测试：`tests/circuit/test_basis_encoder.py`、`tests/circuit/test_iqp_encoder.py`（含与论文定义逐元素对照的独立 NumPy 参考实现）。
