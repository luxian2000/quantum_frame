# aicir.visual 使用手册

量子线路、量子态和密度矩阵的可视化工具库，同时支持 QAS 架构搜索结果的指标图表。文本输出与门统计不依赖图形库；绘图函数在调用时按需导入 `matplotlib`。

---

## 目录

| 文件 | 说明 |
| --- | --- |
| `circuit.py` | 文本线路图、matplotlib 线路图、门统计 |
| `plot.py` | 彩色线路图渲染与自动保存（`plot` 主入口） |
| `state.py` | 态向量概率 / 振幅 / 相位可视化 |
| `density.py` | 密度矩阵热力图 |
| `qas.py` | QAS 搜索结果 / 架构指标可视化 |
| `utils.py` | 内部工具（numpy 转换、basis 标签等） |

---

## 1  线路可视化

### 1.1  文本线路图

```python
from aicir import Circuit, hadamard, cnot
from aicir.visual import circuit_to_text

cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
print(circuit_to_text(cir))
```

输出 ASCII 线路图，无需 matplotlib。

### 1.2  门统计

```python
from aicir.visual import gate_histogram

print(gate_histogram(cir))
# {'cx': 1, 'hadamard': 1}
```

返回按门类型排序的计数字典。

### 1.3  统一线路图入口

```python
from aicir.visual import draw_circuit

# 文本模式（默认）
text = draw_circuit(cir, output="text")

# matplotlib 模式
fig, ax = draw_circuit(cir, output="mpl")
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `output` | `"text"` | `"text"` 返回 ASCII 字符串；`"mpl"` 返回 `(fig, ax)` |
| `fontsize` | `11` | mpl 模式下的字号 |
| `title` | `None` | mpl 模式下的图标题 |
| `ax` | `None` | 传入已有的 matplotlib Axes |

### 1.4  彩色线路图（`plot`）

渲染彩色、圆角方框风格的出版级线路图并保存为 PNG。

```python
from aicir.visual import plot

# 自动命名保存（脚本所在目录，文件名 = <脚本名>_<变量名>.png）
fig, ax = plot(cir)

# 显式路径
fig, ax = plot(cir, "figures/bell.png")

# 也可通过 Circuit 的方法调用
fig, ax = cir.plot()
fig, ax = cir.plot("figures/bell")
```

#### plot 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `path` | `None` | 输出路径（相对路径以调用脚本所在目录为基准，非命令行 cwd） |
| `name` | `None` | 覆盖输出基名（配合自动保存目录使用） |
| `save` | `True` | 设为 `False` 只构建 figure 不保存文件 |
| `dpi` | `200` | PNG 分辨率 |
| `layered` | `True` | 独立门打包进同一列（紧凑布局） |
| `fontsize` | `15` | 基准字号 |
| `title` | `None` | 图标题 |
| `qubit_labels` | `None` | 自定义量子比特标签列表（默认 `q0, q1, ...`） |
| `wire_color` | `"#9AA0A6"` | 量子线颜色 |
| `ax` | `None` | 传入已有的 matplotlib Axes |

#### 自动命名规则

| 场景 | 保存路径 |
| --- | --- |
| `plot(cir)` 在 `demos/demo.py` 中 | `demos/demo_cir.png` |
| `cir.plot()` 在 `demos/demo.py` 中 | `demos/demo_cir.png` |
| `plot(Circuit(...))` 非变量传入 | `demos/demo_0.png`（递增编号） |
| `plot(cir, "figures/h2")` | `demos/figures/h2.png`（相对脚本目录） |
| `plot(cir, "/abs/path/qft.png")` | `/abs/path/qft.png` |

#### 支持的输入格式

`plot` 接受任何 aicir 电路形式：

- `Circuit` 对象
- 含 `.circuit` 属性的包装器（如 QAS `ArchitectureSpec`）
- 单个或序列门字典
- circuit-JSON 字典或 JSON / OPENQASM 字符串
- `.json` / `.qasm` 文件路径

#### 门着色方案

| 门类别 | 填充色 | 说明 |
| --- | --- | --- |
| Clifford 门（H, X, Y, Z, S, CX, CY, CZ, SWAP） | 淡绿 | |
| 非参数化非 Clifford 门（T 门、Toffoli） | 淡蓝 | |
| 参数化非 Clifford 门（Rx, Ry, Rz, CRx, …, U2, U3, unitary, single/double_excitation） | 浅蓝 | 减淡 20% |
| 测量 / Reset | 粉红 | |
| 其它（未归类门名） | 灰色 | |

#### Reset 渲染细节

- Reset 门框与 Measure 同色，标注旋转箭头图案
- 紧跟在同一比特 `measure(q)` 之后的 reset 会把该测量到后续量子门之间的这段量子线渲染为一条连续实线（颜色、线宽与普通量子线相同），避免与被跳过的普通实线重叠
- 若无后续量子门则该连接线延伸到线路末端

---

## 2  量子态可视化

所有态可视化函数同时接受态向量（复数数组）、概率向量（实数数组）和 aicir `State` 对象。

### 2.1  概率柱状图

```python
import numpy as np
from aicir.visual import plot_state_probs

state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
fig, ax = plot_state_probs(state)
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `bit_order` | `"msb"` | 基标签端序：`"msb"` 或 `"lsb"` |
| `threshold` | `0.0` | 只显示概率 ≥ threshold 的基态 |
| `ax` | `None` | 传入已有 Axes |
| `title` | `None` | 图标题（默认 `"State probabilities"`） |

### 2.2  振幅图

同时绘制实部、虚部和模长：

```python
from aicir.visual import plot_state_amplitudes

fig, ax = plot_state_amplitudes(state)
```

参数与 `plot_state_probs` 相同。三组柱状图分别以蓝色（real）、橙色（imag）、绿色（abs）显示。

### 2.3  相位图

绘制各基态的振幅相位（`np.angle`）：

```python
from aicir.visual import plot_state_phase

fig, ax = plot_state_phase(state, threshold=1e-12)
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `threshold` | `1e-12` | 只显示模长 ≥ threshold 的基态相位 |

Y 轴范围固定为 `[-π, π]`。

---

## 3  密度矩阵可视化

### 3.1  单分量热力图

```python
import numpy as np
from aicir.visual import plot_density_matrix

state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
rho = np.outer(state, np.conjugate(state))

fig, ax = plot_density_matrix(rho, part="abs")
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `part` | `"abs"` | `"abs"` / `"real"` / `"imag"` / `"phase"` |
| `cmap` | `"viridis"` | matplotlib colormap 名称 |
| `colorbar` | `True` | 是否显示 colorbar |
| `ax` | `None` | 传入已有 Axes |
| `title` | `None` | 图标题 |

### 3.2  实部 / 虚部并排

```python
from aicir.visual import plot_density_real_imag

fig, axes = plot_density_real_imag(rho, cmap="coolwarm")
```

返回 `(fig, [ax_real, ax_imag])`。

---

## 4  QAS 搜索结果可视化

与 `aicir.qas` 的 `SearchResult`、`ArchitectureScore`、`ArchitectureSpec` 直接对接。

### 4.1  数据扁平化

```python
from aicir.visual import qas_scores_to_rows

rows = qas_scores_to_rows(search_result)
print(rows[0])
# {'rank': 1, 'name': 'cand_0', 'n_qubits': 3, 'n_gates': 8,
#  'weighted_score': 0.82, 'expressibility': 0.91, ...}
```

接受 `SearchResult`、`ArchitectureScore`、`ArchitectureSpec`、`dict` 或它们的序列，输出统一的扁平行列表，便于接入分析或保存。

### 4.2  搜索历史曲线

```python
from aicir.visual import plot_search_history

fig, ax = plot_search_history(
    search_result,
    metrics=["weighted_score", "expressibility", "trainability"],
    x=None,          # 自动选 rank/step 作 x 轴
)
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `metrics` | `None`（自动选前 5 个数值字段） | 要绘制的指标名列表 |
| `x` | `None` | x 轴字段名（默认按 rank 或 step 编号） |
| `ax` | `None` | 传入已有 Axes |
| `title` | `None` | 图标题（默认 `"QAS metric history"`） |

### 4.3  单架构指标条形图

```python
from aicir.visual import plot_architecture_metrics

fig, ax = plot_architecture_metrics(
    best_score,
    metrics=["weighted_score", "trainability", "hardware_efficiency"],
)
```

### 4.4  多架构对比

```python
from aicir.visual import compare_architectures

fig, ax = compare_architectures(
    search_result.scores[:5],
    metrics=["weighted_score", "n_gates", "two_qubit_gate_count"],
)
```

分组柱状图，每个候选一组，每个指标一种颜色。

### 4.5  线路 + 指标 Summary

```python
from aicir.visual import plot_qas_summary

fig, axes = plot_qas_summary(
    best_score,
    metrics=["weighted_score", "trainability", "hardware_efficiency"],
)
```

左侧面板显示文本线路图，右侧面板显示指标条形图。返回 `(fig, [ax_circuit, ax_metrics])`。

---

## 5  公共 API 速查

### 从 `aicir.visual` 直接导入

| 函数 | 来源模块 | 类别 |
| --- | --- | --- |
| `circuit_to_text` | circuit | 线路 |
| `circuit_to_mpl` | circuit | 线路 |
| `draw_circuit` | circuit | 线路 |
| `gate_histogram` | circuit | 线路 |
| `plot` | plot | 线路 |
| `plot_state_probs` | state | 态向量 |
| `plot_state_amplitudes` | state | 态向量 |
| `plot_state_phase` | state | 态向量 |
| `plot_density_matrix` | density | 密度矩阵 |
| `plot_density_real_imag` | density | 密度矩阵 |
| `qas_scores_to_rows` | qas | QAS |
| `plot_search_history` | qas | QAS |
| `plot_architecture_metrics` | qas | QAS |
| `compare_architectures` | qas | QAS |
| `plot_qas_summary` | qas | QAS |

---

## 6  依赖说明

- **无 matplotlib**：`circuit_to_text`、`gate_histogram` 可正常使用。
- **需要 matplotlib**：所有 `plot_*` / `draw_circuit(output="mpl")` / `plot` / QAS 可视化函数在首次调用时按需导入 matplotlib。未安装时抛出 `ImportError` 并提示安装。
- **输入兼容**：态/密度矩阵函数接受 numpy array、aicir `State` 对象、PyTorch tensor 或任何实现 `.to_numpy()` 的对象。
