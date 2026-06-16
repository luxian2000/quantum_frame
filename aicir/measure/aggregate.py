"""sm="avg" 聚合：把多条 TrajectoryResult 折叠为 Result 的字段字典。"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def _as_density(state) -> np.ndarray:
    """将 State 对象转换为密度矩阵 numpy 数组。"""
    arr = np.asarray(state.to_numpy())
    if arr.ndim == 1 or arr.shape[0] != arr.shape[1]:
        vec = arr.reshape(-1, 1)
        return vec @ vec.conj().T
    return arr


def _bitstr(bits: Sequence[int]) -> str:
    """将本征值列表（+1/-1）转换为比特串（+1→0, -1→1）。"""
    return "".join("0" if b == 1 else "1" for b in bits)


def aggregate_avg(trajectories, n_qubits: int, measurement_specs,
                  terminal_qubits: Optional[List[int]]) -> Dict[str, object]:
    """将多条 TrajectoryResult 按 avg 模式聚合为 Result 字段字典。

    参数:
        trajectories:      TrajectoryResult 列表，每条对应一次采样轨迹
        n_qubits:          线路量子比特数
        measurement_specs: 测量规格（仅用于签名兼容，实际 op 索引从轨迹中提取）
        terminal_qubits:   末端测量的比特列表，None 表示不聚合末端结果

    返回:
        包含聚合字段的字典，供 Measure.run 构造 Result 使用。
    """
    M = len(trajectories)
    dim = 1 << n_qubits

    # 对所有轨迹的 pre/post 态做密度矩阵平均
    pre_sum = np.zeros((dim, dim), dtype=complex)
    post_sum = np.zeros((dim, dim), dtype=complex)
    for tr in trajectories:
        pre_sum += _as_density(tr.pre)
        post_sum += _as_density(tr.post)
    state = pre_sum / M
    final_state = post_sum / M

    # 从轨迹的 incircuit 键集合推导待聚合的 op 索引（不依赖 measurement_specs）
    op_indices = (
        sorted(set().union(*[set(tr.incircuit) for tr in trajectories]))
        if trajectories else []
    )

    # 每个 op 的本征值栈（M×1）和计数字典
    incircuit_outputs: Dict[int, np.ndarray] = {}
    incircuit_counts: Dict[int, Dict[int, int]] = {}
    for op in op_indices:
        col = np.array([[int(tr.incircuit[op])] for tr in trajectories], dtype=int)
        incircuit_outputs[op] = col
        c: Dict[int, int] = {}
        for tr in trajectories:
            lam = int(tr.incircuit[op])
            c[lam] = c.get(lam, 0) + 1
        incircuit_counts[op] = c

    # 末端测量聚合
    terminal_output = None
    terminal_counts = None
    if terminal_qubits is not None and trajectories and trajectories[0].terminal is not None:
        k = len(terminal_qubits)
        terminal_output = np.array(
            [tr.terminal for tr in trajectories], dtype=int
        ).reshape(M, k)
        terminal_counts: Dict[str, int] = {}
        for tr in trajectories:
            key = _bitstr(tr.terminal)
            terminal_counts[key] = terminal_counts.get(key, 0) + 1

    # snap 态平均
    snap_states: Dict[int, np.ndarray] = {}
    snap_keys = (
        set().union(*[set(tr.snaps) for tr in trajectories])
        if trajectories else set()
    )
    for t in snap_keys:
        snap_states[t] = sum(_as_density(tr.snaps[t]) for tr in trajectories) / M

    # 对角元即基态概率
    probabilities = np.real(np.diag(state)).astype(np.float64)

    return {
        "state": state,
        "final_state": final_state,
        "final_state_kind": "density_matrix",
        "incircuit_outputs": incircuit_outputs,
        "incircuit_counts": incircuit_counts,
        "terminal_output": terminal_output,
        "terminal_counts": terminal_counts,
        "snapshot_states": snap_states,
        "probabilities": probabilities,
    }
