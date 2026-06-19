"""MaxCut 端到端示例。

流程：
1. 生成一个 ``n_nodes`` 个节点的随机图（默认 10 个节点）。
2. 由图构造 MaxCut 对应的 Ising 哈密顿量，并把它写入
   ``maxcut_hamiltonian.py``。
3. 使用 ``aicir.qas.algorithms.supernet``（``supernet_qas`` 封装入口）搜索并微调
   该哈密顿量的基态 VQE 线路，把线路一并记录到
   ``maxcut_hamiltonian.py``。
4. 把上述 VQE 线路绘制为 ``maxcut_hamiltonian.png``。

从仓库根目录运行::

    PYTHONPATH=. python demos/MaxCut/maxcut.py
    PYTHONPATH=. python demos/MaxCut/maxcut.py --n-nodes 8 --edge-prob 0.5

MaxCut 与 Ising 的映射
----------------------
对无向带权图 ``G = (V, E)``，割的取值用每个节点的自旋 ``Z_i ∈ {+1, -1}``
表示。一条边 ``(i, j)`` 被割开当且仅当两端自旋相反，对应

    cut(i, j) = (1 - Z_i Z_j) / 2

最大化总割等价于最小化下面的 Ising 哈密顿量（VQE 求基态即最小能量）::

    H = Σ_{(i,j)∈E} w_ij/2 · (Z_i Z_j) - (Σ w_ij)/2 · I

其基态能量为 ``-max_cut``（带权时为最大权重割的相反数）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import networkx as nx

from aicir import Circuit
from aicir.backends.numpy_backend import NumpyBackend
from aicir.core.operators import Hamiltonian
from aicir.core.gates import apply_gate_to_state, gate_to_matrix
from aicir.core.io.json_io import circuit_from_json, circuit_to_json
from aicir.core.state import State
from aicir.qas import supernet_qas
from aicir.visual import plot

# 生成文件统一写在脚本同目录下。
SCRIPT_DIR = Path(__file__).resolve().parent
HAMILTONIAN_PY = SCRIPT_DIR / "maxcut_hamiltonian.py"
CIRCUIT_PNG = SCRIPT_DIR / "maxcut_hamiltonian.png"


def generate_random_graph(n_nodes: int, edge_prob: float, seed: int) -> nx.Graph:
    """生成一个连通的 Erdős–Rényi 随机图。

    为保证 MaxCut 任务非平凡，这里在采样到孤立子图时补加最小生成边，
    使整图连通；所有边权固定为 1（无权 MaxCut）。
    """
    graph = nx.gnp_random_graph(n_nodes, edge_prob, seed=seed)
    # 把所有连通分量串起来，避免出现孤立节点导致的平凡解。
    components = list(nx.connected_components(graph))
    for left, right in zip(components, components[1:]):
        graph.add_edge(min(left), min(right))
    for _, _, data in graph.edges(data=True):
        data.setdefault("weight", 1.0)
    return graph


def build_ising_hamiltonian(graph: nx.Graph) -> Tuple[Hamiltonian, List[tuple]]:
    """由图构造 MaxCut 的 Ising 哈密顿量。

    返回 ``(hamiltonian, terms)``，其中 ``terms`` 是可直接写入 Python
    源文件、并交给 ``Hamiltonian(n_qubits=..., terms=...)`` 重建的项列表。
    """
    n_qubits = graph.number_of_nodes()
    terms: List[tuple] = []
    total_weight = 0.0
    for i, j, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        total_weight += weight
        # ("ZZ", [i, j], coeff)：稀疏 Pauli 串，仅在 i、j 上作用 Z。
        terms.append(("ZZ", [int(i), int(j)], 0.5 * weight))
    # 常数项：-(Σ w)/2 · I（写成全 I 的 Pauli 串）。
    terms.append(("I" * n_qubits, -0.5 * total_weight))

    hamiltonian = Hamiltonian(n_qubits=n_qubits, terms=terms)
    return hamiltonian, terms


def circuit_energy(circuit: Circuit, hamiltonian: Hamiltonian, backend: NumpyBackend) -> float:
    """计算线路制备态对哈密顿量的期望能量 ⟨ψ|H|ψ⟩（归一化态）。

    注意：``supernet_qas`` 的 ``final_metrics`` 在微调阶段用的是未归一化的
    内部态，能量幅值不可直接采信。这里基于线路酉矩阵重新求一遍真实能量。
    """
    n_qubits = hamiltonian.n_qubits
    state = State.zero_state(n_qubits, backend)
    # 逐门作用到态向量，避免构造 2^n × 2^n 全局酉矩阵（更省内存且数值更稳）。
    data = state.data
    for gate in circuit.gates:
        evolved = apply_gate_to_state(gate, data, n_qubits, backend)
        if evolved is None:  # 局部展开失败时回退到全矩阵作用。
            data = backend.apply_unitary(data, gate_to_matrix(gate, cir_qubits=n_qubits, backend=backend))
        else:
            data = evolved
    final = State(data, n_qubits, backend)
    return float(hamiltonian.expectation(final, backend))


def exact_ground_energy(hamiltonian: Hamiltonian, backend: NumpyBackend) -> float:
    """对角化哈密顿量得到精确基态能量（用于核对，n 较小时可行）。"""
    matrix = np.asarray(backend.to_numpy(hamiltonian.to_matrix(backend)))
    return float(np.linalg.eigvalsh(matrix).min())


def evaluate_vqe_cut_metrics(
    circuit: Circuit,
    graph: nx.Graph,
    hamiltonian: Hamiltonian,
    backend: NumpyBackend,
) -> Tuple[dict, dict]:
    """计算 VQE 线路的 MaxCut 指标。

    ``-vqe_energy`` 是量子态概率分布上的期望割值，不一定是实际读出的某个
    割。``achieved_cut`` 使用概率显著比特串中的最佳割值，更符合 MaxCut
    读出结果；``expected_cut`` 单独保留能量期望对应的值。
    """
    vqe_energy = circuit_energy(circuit, hamiltonian, backend)
    ground_energy = exact_ground_energy(hamiltonian, backend)
    partition, sampled_cut = vqe_cut_partition(circuit, graph, backend)
    max_cut = -ground_energy
    expected_cut = -vqe_energy
    approx_ratio = float(sampled_cut / max_cut) if max_cut > 0 else None
    expected_approx_ratio = float(expected_cut / max_cut) if max_cut > 0 else None
    metrics = {
        "vqe_energy": vqe_energy,
        "exact_ground_energy": ground_energy,
        "max_cut": max_cut,
        "expected_cut": expected_cut,
        "achieved_cut": sampled_cut,
        "approx_ratio": approx_ratio,
        "expected_approx_ratio": expected_approx_ratio,
        "n_gates": len(circuit.gates),
    }
    return metrics, partition


def vqe_cut_partition(
    circuit: Circuit, graph: nx.Graph, backend: NumpyBackend
) -> Tuple[dict, float]:
    """从 VQE 末态读出割划分（采样/取整方式）。

    构造线路末态后，在概率显著的比特串中挑选割值最大的那个作为划分结果，
    比单纯用能量期望 ⟨H⟩ 更贴近 MaxCut 的实际读出方式。返回
    ``(assignment, sampled_cut)``：``assignment[node] ∈ {0, 1}`` 为节点所属集合。
    """
    n_qubits = graph.number_of_nodes()
    state = State.zero_state(n_qubits, backend)
    data = state.data
    for gate in circuit.gates:
        evolved = apply_gate_to_state(gate, data, n_qubits, backend)
        if evolved is None:
            data = backend.apply_unitary(data, gate_to_matrix(gate, cir_qubits=n_qubits, backend=backend))
        else:
            data = evolved
    probs = np.abs(np.asarray(backend.to_numpy(data)).reshape(-1)) ** 2

    def bits_of(index: int) -> List[int]:
        # 量子比特 i 对应基态串的第 i 位（高位在前）。
        return [(index >> (n_qubits - 1 - k)) & 1 for k in range(n_qubits)]

    def cut_value(bits: List[int]) -> float:
        return sum(
            float(d.get("weight", 1.0))
            for i, j, d in graph.edges(data=True)
            if bits[int(i)] != bits[int(j)]
        )

    best_cut = -1.0
    best_bits = bits_of(int(np.argmax(probs)))
    for index in np.flatnonzero(probs > 1e-6):
        bits = bits_of(int(index))
        cut = cut_value(bits)
        if cut > best_cut:
            best_cut = cut
            best_bits = bits
    assignment = {node: best_bits[int(node)] for node in graph.nodes()}
    return assignment, float(best_cut)


def plot_graph_and_circuit(
    graph: nx.Graph,
    circuit: Circuit,
    path: Path,
    metrics: dict,
    partition: dict | None = None,
) -> None:
    """把随机图和 VQE 线路画在同一张 PNG 里：上方为图，下方为线路。

    若给定 ``partition``（``node -> {0, 1}`` 的割划分），则按所属集合为
    节点上两种颜色，并把被割开的边高亮显示。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.1], hspace=0.18)
    graph_ax = fig.add_subplot(grid[0])
    circuit_ax = fig.add_subplot(grid[1])

    # 上：随机图（节点即量子比特，边即 ZZ 项）。
    layout = nx.spring_layout(graph, seed=1)

    # 两个集合的配色：集合 0 绿色，集合 1 橙色。
    SET_COLORS = ("#A7D2A1", "#F2B66D")
    SET_EDGES = ("#4F8A45", "#C77B1F")
    if partition is not None:
        node_color = [SET_COLORS[int(partition[node])] for node in graph.nodes()]
        node_edge = [SET_EDGES[int(partition[node])] for node in graph.nodes()]
        # 被割开的边（两端属于不同集合）用实线高亮，其余边淡化。
        cut_edges = [(i, j) for i, j in graph.edges() if partition[i] != partition[j]]
        uncut_edges = [(i, j) for i, j in graph.edges() if partition[i] == partition[j]]
        nx.draw_networkx_edges(graph, layout, ax=graph_ax, edgelist=uncut_edges,
                               edge_color="#D5D8DC", width=1.5)
        nx.draw_networkx_edges(graph, layout, ax=graph_ax, edgelist=cut_edges,
                               edge_color="#5B6066", width=2.4, style="dashed")
    else:
        node_color = "#A7D2A1"
        node_edge = "#4F8A45"
        nx.draw_networkx_edges(graph, layout, ax=graph_ax, edge_color="#9AA0A6", width=2.0)

    nx.draw_networkx_nodes(graph, layout, ax=graph_ax, node_color=node_color,
                           edgecolors=node_edge, linewidths=2.0, node_size=900)
    nx.draw_networkx_labels(graph, layout, ax=graph_ax, font_size=14, font_color="#1B3A16")
    graph_ax.set_title(
        f"Random graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges  "
        f"(max-cut={metrics['max_cut']:.0f}, readout cut={metrics['achieved_cut']:.2f}, "
        f"expected={metrics['expected_cut']:.2f}, ratio={metrics['approx_ratio']:.3f})",
        fontsize=14,
    )
    graph_ax.set_axis_off()

    # 下：VQE 基态线路（复用 aicir.visual.plot，写到同一个 ax，不单独保存）。
    plot(circuit, ax=circuit_ax, save=False, title="MaxCut VQE ground-state circuit")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_hamiltonian_module(
    graph: nx.Graph,
    terms: List[tuple],
    circuit_json: dict,
    metrics: dict,
) -> str:
    """渲染 ``maxcut_hamiltonian.py`` 的源码文本。"""
    import pprint

    edges = [(int(i), int(j), float(d.get("weight", 1.0))) for i, j, d in graph.edges(data=True)]
    terms_repr = pprint.pformat(terms, width=88, sort_dicts=False)
    circuit_repr = pprint.pformat(circuit_json, width=88, sort_dicts=False)
    metrics_repr = pprint.pformat(metrics, width=88, sort_dicts=False)

    return f'''"""MaxCut Ising 哈密顿量与对应的 VQE 基态线路（自动生成）。

本文件由 ``demos/MaxCut/maxcut.py`` 生成，请勿手工编辑。

包含：
- 随机图的节点数与边集；
- MaxCut Ising 哈密顿量项 ``MAXCUT_HAMILTONIAN_TERMS``；
- ``aicir.qas.algorithms.supernet`` 搜索 + 微调得到的 VQE 线路 ``VQE_CIRCUIT_JSON``；
- 重建辅助函数 ``build_hamiltonian()`` / ``build_vqe_circuit()``。
"""

from __future__ import annotations

from aicir import Circuit
from aicir.core.operators import Hamiltonian
from aicir.core.io.json_io import circuit_from_json

# ── 随机图 ────────────────────────────────────────────────────────────────
N_NODES = {graph.number_of_nodes()}
# (节点 i, 节点 j, 边权 w)
EDGES = {pprint.pformat(edges, width=88)}

# ── MaxCut Ising 哈密顿量 ───────────────────────────────────────────────────
# 项格式：("ZZ", [i, j], coeff) 或 ("I"*N_NODES, const)。
MAXCUT_HAMILTONIAN_TERMS = {terms_repr}


def build_hamiltonian() -> Hamiltonian:
    """重建 MaxCut Ising 哈密顿量。"""
    return Hamiltonian(n_qubits=N_NODES, terms=MAXCUT_HAMILTONIAN_TERMS)


# ── VQE 基态线路（QAS supernet 搜索并微调）────────────────────────────────────
# 指标（基于归一化态重新计算）：vqe_energy 为线路能量，expected_cut 为能量
# 期望对应的割值，achieved_cut 为概率显著比特串中的最佳读出割值。
VQE_METRICS = {metrics_repr}

# 线路以 circuit-JSON 形式记录，可用 circuit_from_json 无损重建。
VQE_CIRCUIT_JSON = {circuit_repr}


def build_vqe_circuit() -> Circuit:
    """重建 QAS 搜索得到的 VQE 基态线路。"""
    return circuit_from_json(VQE_CIRCUIT_JSON)
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="MaxCut: 随机图 → Ising 哈密顿量 → QAS VQE 线路。")
    parser.add_argument("--n-nodes", type=int, default=10, help="随机图节点数（量子比特数），默认 10。")
    parser.add_argument("--edge-prob", type=float, default=0.4, help="ER 随机图连边概率，默认 0.4。")
    parser.add_argument("--graph-seed", type=int, default=7, help="随机图种子。")
    parser.add_argument("--layers", type=int, default=None,
                        help="ansatz 层数 L，默认 min(量子比特数, 6)。")
    parser.add_argument("--supernet-num", type=int, default=3, help="权重共享超网络数量 W。")
    parser.add_argument("--supernet-steps", type=int, default=80, help="超网络优化步数。")
    parser.add_argument("--finetune-steps", type=int, default=120, help="选中架构微调步数。")
    parser.add_argument("--ranking-num", type=int, default=40, help="排序阶段采样候选架构数。")
    parser.add_argument("--qas-seed", type=int, default=2, help="QAS 随机种子。")
    parser.add_argument("--device", type=str, default="cpu", help="Torch 设备，如 cpu/cuda/npu:0。")
    parser.add_argument("--disable-rzz", action="store_true", help="在 supernet 搜索中禁用 rzz，只保留 cx 双比特门。")
    args = parser.parse_args()

    # 1) 随机图
    graph = generate_random_graph(args.n_nodes, args.edge_prob, args.graph_seed)
    print(f"[1/4] 随机图：{graph.number_of_nodes()} 节点，{graph.number_of_edges()} 条边。")

    # 层数 L：未显式指定时取 min(量子比特数, 6)，随问题规模自适应又设上限，
    # 避免过深线路带来的贫瘠高原与开销。
    n_qubits = graph.number_of_nodes()
    layers = args.layers if args.layers is not None else min(n_qubits, 6)

    # 2) Ising 哈密顿量
    hamiltonian, terms = build_ising_hamiltonian(graph)
    print(f"[2/4] Ising 哈密顿量：{len(terms)} 项（含常数项）。")

    # 3) QAS supernet 搜索 + 微调 VQE 基态线路
    print(f"[3/4] 运行 QAS supernet 搜索 VQE 基态线路（L={layers} 层，可能较慢）……")
    result = supernet_qas(
        hamiltonian,
        layers=layers,
        supernet_num=args.supernet_num,
        supernet_steps=args.supernet_steps,
        finetune_steps=args.finetune_steps,
        ranking_num=args.ranking_num,
        two_qubit_gates=("cx",) if args.disable_rzz else ("cx", "rzz"),
        seed=args.qas_seed,
        device=args.device,
    )
    best_circuit: Circuit = result.best_circuit

    # 重新计算线路能量与 MaxCut 读出指标，避免把能量期望误当作单次读出的割。
    backend = NumpyBackend()
    metrics, partition = evaluate_vqe_cut_metrics(best_circuit, graph, hamiltonian, backend)
    print(
        f"       VQE 能量 = {metrics['vqe_energy']:.6f}，精确基态 = {metrics['exact_ground_energy']:.6f}"
        f"（最大割 = {metrics['max_cut']:.3f}，期望割 = {metrics['expected_cut']:.3f}，"
        f"读出割 = {metrics['achieved_cut']:.3f}，近似比 = {metrics['approx_ratio']:.4f}）；"
        f"线路门数 = {metrics['n_gates']}。"
    )

    # 写入 Hamiltonian + VQE 线路到 maxcut_hamiltonian.py
    circuit_json = circuit_to_json(best_circuit)
    HAMILTONIAN_PY.write_text(
        render_hamiltonian_module(graph, terms, circuit_json, metrics),
        encoding="utf-8",
    )
    print(f"       已写入 {HAMILTONIAN_PY}")

    # 4) 把随机图和 VQE 线路画到同一张 PNG（按割划分给节点上两种颜色）
    print(f"       采样读出割值（最佳比特串）= {metrics['achieved_cut']:.3f}。")
    plot_graph_and_circuit(graph, best_circuit, CIRCUIT_PNG, metrics, partition=partition)
    print(f"[4/4] 已绘制随机图 + VQE 线路到 {CIRCUIT_PNG}")


if __name__ == "__main__":
    main()
