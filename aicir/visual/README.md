# 可视化模块

`aicir.visual` 提供第一阶段的轻量可视化工具，用于查看量子线路、量子态概率/振幅和密度矩阵。文本线路图与门统计不依赖额外图形库；绘图函数会在调用时按需导入 `matplotlib`。matplotlib 图中的 `reset` 会以与 `measure` 同色、标注 `Reset` 的虚线显示；`Reset` 字号与 `Rz` 门主标签一致，虚线 dash 间距更大。虚线连接 `measure(q)` 与后续同一比特量子门，没有后续量子门时延伸到线路末端，且不再叠加普通实线。若后续门没有完整方框，reset 虚线按虚拟方框左边界停止，虚拟方框内用普通实线。

```python
import numpy as np
from aicir import Circuit, hadamard, cnot, rzz
from aicir.visual import (
    circuit_to_text,
    draw_circuit,
    gate_histogram,
    plot,
    plot_state_probs,
    plot_state_amplitudes,
    plot_density_matrix,
)

cir = Circuit(
    hadamard(0),
    cnot(1, [0]),
    rzz(np.pi / 2, 0, 1),
    n_qubits=2,
)

print(circuit_to_text(cir))
print(gate_histogram(cir))  # {'cx': 1, 'hadamard': 1, 'rzz': 1}

# 保存彩色线路图。无 path 时，默认保存到调用它的 .py 文件所在目录，
# 文件名为 <脚本名>_<电路变量名>.png，例如 demos/demo_1.py 中的
# cir.plot() 会保存为 demos/demo_1_cir.png。
fig, ax = cir.plot()

# 显式相对路径也以调用它的 .py 文件所在目录为基准，而不是命令行 cwd。
# 例如在 demos/demo_1.py 中会保存为 demos/figures/h2.png。
fig, ax = cir.plot("figures/h2")

# 也可以继续使用函数式入口。
fig, ax = plot(cir, "figures/h2_function")

# draw_circuit 默认返回文本；output="mpl" 时返回 matplotlib 的 (fig, ax)
diagram = draw_circuit(cir)
fig, ax = draw_circuit(cir, output="mpl")

# 态向量或概率向量均可输入
state = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=np.complex64)
fig, ax = plot_state_probs(state)
fig, ax = plot_state_amplitudes(state)

# 密度矩阵热力图，part 可取 "abs"、"real"、"imag"、"phase"
rho = np.outer(state, np.conjugate(state))
fig, ax = plot_density_matrix(rho, part="abs")
```

当前已实现的公共函数：

- `circuit_to_text(circuit)`：返回 ASCII 线路图
- `draw_circuit(circuit, output="text" | "mpl")`：统一线路图入口
- `gate_histogram(circuit)`：按门 `type` 统计数量
- `plot(circuit, path=None, ...)` / `Circuit.plot(path=None, ...)`：保存彩色线路图，返回 `(fig, ax)`；未提供 `path` 时默认保存到调用它的 `.py` 文件所在目录
- `plot_state_probs(state_or_probs)`：绘制计算基概率柱状图
- `plot_state_amplitudes(state)`：绘制振幅实部、虚部和模长
- `plot_state_phase(state)`：绘制振幅相位
- `plot_density_matrix(rho, part=...)`：绘制密度矩阵热力图
- `plot_density_real_imag(rho)`：并排绘制密度矩阵实部和虚部

## 9.1 QAS 与 metrics 可视化

`aicir.visual` 也可以直接消费 `aicir.qas` 的 `SearchResult`、`ArchitectureScore`、`ArchitectureSpec`，用于对比架构搜索候选、查看四类 objective group 分数，以及把线路图和指标放在一个 summary 图中。

```python
from aicir.backends.numpy_backend import NumpyBackend
from aicir.qas import ArchitectureSearch, SearchConfig
from aicir.visual import (
    qas_scores_to_rows,
    plot_search_history,
    plot_architecture_metrics,
    compare_architectures,
    plot_qas_summary,
)

backend = NumpyBackend()
search = ArchitectureSearch(backend=backend)
result = search.run(SearchConfig(n_qubits=3, candidate_layers=1, n_samples=8))

# 转成扁平行，便于打印、保存或接入其它分析流程
rows = qas_scores_to_rows(result)
print(rows[0])

# 按 rank 展示候选架构的 weighted score 和分组指标
fig, ax = plot_search_history(
    result,
    metrics=["weighted_score", "expressibility", "trainability", "hardware_efficiency"],
)

# 对比多个候选
fig, ax = compare_architectures(
    result.scores[:5],
    metrics=["weighted_score", "n_gates", "two_qubit_gate_count"],
)

# 查看单个候选的指标，或生成带线路图的 summary
best = result.best
fig, ax = plot_architecture_metrics(best)
fig, axes = plot_qas_summary(best, metrics=["weighted_score", "trainability", "hardware_efficiency"])
```

QAS/metrics 相关公共函数：

- `qas_scores_to_rows(data)`：将 `SearchResult`、`ArchitectureScore`、`ArchitectureSpec` 或 dict 记录转为扁平行
- `plot_search_history(history, metrics=None, x=None)`：绘制搜索过程或候选 ranking 的指标曲线
- `plot_architecture_metrics(item, metrics=None)`：绘制单个候选的指标条形图
- `compare_architectures(data, metrics=None)`：对比多个候选架构或评分
- `plot_qas_summary(item, metrics=None)`：左侧线路图、右侧指标图的组合视图
