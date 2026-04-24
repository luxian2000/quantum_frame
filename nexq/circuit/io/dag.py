"""
nexq/circuit/io/dag.py

将 nexq Circuit 转换为 DAG 图表示。
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..model import Circuit


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _gate_name(gate: dict) -> str:
    """从 nexq 门字典中返回门类型名称（即 gate["type"]）。"""
    return gate["type"]


def _gate_qubits(gate: dict) -> List[int]:
    """
    从 nexq 门字典中提取所有作用的量子比特索引列表。

    nexq 门字典的比特字段因门类型而异：
    - 单比特门：target_qubit
    - cx/cy/cz/crx/cry/crz/toffoli/ccnot：control_qubits + target_qubit
    - swap/rzz：qubit_1, qubit_2
    - identity/I：range(n_qubits)
    """
    gate_type = gate["type"]

    if gate_type in ("swap", "rzz"):
        return [gate["qubit_1"], gate["qubit_2"]]

    if gate_type in ("identity", "I"):
        return list(range(gate["n_qubits"]))

    if gate_type in ("cx", "cnot", "cy", "cz", "crx", "cry", "crz", "toffoli", "ccnot"):
        return list(gate["control_qubits"]) + [gate["target_qubit"]]

    # 其余均为单比特门
    return [gate["target_qubit"]]


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def circuit_to_dag(
    circuit: Circuit,
    gate_types: List[str],
):
    """
    将 nexq Circuit 转换为 DAG 图表示，
    输出节点特征矩阵 X、有向邻接矩阵 A 以及门类型 one-hot 矩阵。

    Args:
        circuit:    nexq Circuit 对象（circuit.gates 为 list of dict，
                    circuit.n_qubits 为量子比特总数）。
        gate_types: list of str，所有可能的门类型集合
                    （例如 ['hadamard', 'rx', 'ry', 'rz', 'cx', 'cz']）。

    Returns:
        X:           np.ndarray，形状 (N+2, len(gate_types)+n_qubits)，节点特征矩阵。
        A:           np.ndarray，形状 (N+2, N+2)，有向邻接矩阵。
        type_onehot: np.ndarray，形状 (N+2, len(gate_types))，
                     仅类型 one-hot 部分（START/END 行全零）。
    """
    n_qubits = circuit.n_qubits
    gate_list = list(circuit.gates)

    # ---- 1. 节点列表初始化 ----
    # 节点索引：0 = START，1..N = 门节点，N+1 = END
    N = len(gate_list)
    nodes: List[dict] = [None] * (N + 2)  # type: ignore[list-item]
    nodes[0] = {"name": "START", "gate": None, "qubits": []}
    for i, gate in enumerate(gate_list, start=1):
        nodes[i] = {
            "name": "gate",
            "gate": _gate_name(gate),
            "qubits": _gate_qubits(gate),
        }
    nodes[N + 1] = {"name": "END", "gate": None, "qubits": []}

    # ---- 2. 构建有向边 ----
    # 为每个量子比特维护上一次经过它的节点索引，初始指向 START（索引 0）
    last_node_on_qubit = [0] * n_qubits
    edges: set = set()

    # 按时间顺序处理门节点（索引 1 到 N）
    for idx in range(1, N + 1):
        for q in nodes[idx]["qubits"]:
            from_idx = last_node_on_qubit[q]
            edges.add((from_idx, idx))
            last_node_on_qubit[q] = idx

    # 最后每个量子比特连接到 END 节点
    end_idx = N + 1
    for q in range(n_qubits):
        edges.add((last_node_on_qubit[q], end_idx))

    # ---- 3. 构建邻接矩阵 A ----
    total_nodes = N + 2
    A = np.zeros((total_nodes, total_nodes), dtype=np.float32)
    for (i, j) in edges:
        A[i, j] = 1.0

    # ---- 4. 构建特征矩阵 X 和类型 one-hot 矩阵 ----
    # 列布局：[gate_type one-hot (F_type 列) | qubit position (n_qubits 列)]
    F_type = len(gate_types)
    F = F_type + n_qubits
    X = np.zeros((total_nodes, F), dtype=np.float32)
    type_onehot = np.zeros((total_nodes, F_type), dtype=np.float32)

    gate_to_idx = {g: i for i, g in enumerate(gate_types)}

    # START（索引 0）和 END（索引 N+1）保持全零
    for idx in range(1, N + 1):
        gate_name = nodes[idx]["gate"]
        qubits = nodes[idx]["qubits"]

        # 类型 one-hot
        if gate_name in gate_to_idx:
            type_idx = gate_to_idx[gate_name]
            X[idx, type_idx] = 1.0
            type_onehot[idx, type_idx] = 1.0

        # 位置向量：门作用的每个 qubit 对应列置 1
        for q in qubits:
            X[idx, F_type + q] = 1.0

    return X, A, type_onehot