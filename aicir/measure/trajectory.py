"""单条测量轨迹：逐操作执行 cir，处理线路内 measure/reset、snap 与末端测量。

执行语义：
- in-circuit measure 门 → 联合 Pauli 投影测量，本征值记录于 incircuit[op_index]
- in-circuit reset 门  → 重置信道（无需事先测量）
- 酉门              → 演化态；可选在每门后施加噪声（密度矩阵路径）
- snap_ops          → 记录指定 op_index 操作完成后的完整态快照
- 末端测量          → tm=True 时对 measure_qubits 逐比特 Z 基测量
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from ..core.gates import apply_gate_to_state, gate_to_matrix
from ..core.state import State
from ..ir import (
    circuit_instructions,
    instruction_name,
    instruction_qubits,
)
from . import projector


def _is_measure(gate) -> bool:
    """判断操作是否为线路内嵌 measure/measurement 标记。"""
    return instruction_name(gate).lower() in {"measure", "measurement"}


def _is_reset(gate) -> bool:
    """判断操作是否为线路内嵌 reset 标记。"""
    return instruction_name(gate).lower() == "reset"


def _marker_qubits(gate, n: int) -> List[int]:
    """返回 marker 门作用的量子比特；空列表意味着作用全部比特。"""
    qs = [int(q) for q in instruction_qubits(gate)]
    return qs if qs else list(range(n))


def _gate_basis(gate) -> str:
    """从 Measurement 类型对象或旧式 dict 中读取 basis 字段，缺省 'Z'。"""
    # circuit_instructions 返回的是类型化 Measurement 对象，有 .basis 属性
    basis = getattr(gate, "basis", None)
    if basis is None:
        # 兼容仍为 dict 的极端情况
        basis = gate.get("basis", "Z") if hasattr(gate, "get") else "Z"
    return str(basis)


@dataclass
class TrajectoryResult:
    """单条轨迹执行的结果。

    属性:
        pre:       末端测量前的量子态（即所有酉门和线路内 measure/reset 执行完后）
        post:      末端测量后的坍缩态（tm=False 时与 pre 相同）
        incircuit: 线路内 measure 门的测量本征值，键为 op_index
        terminal:  末端测量本征值列表（tm=False 时为 None）
        snaps:     snap_ops 指定位置的量子态快照，键为 op_index
    """
    pre: State
    post: State
    incircuit: Dict[int, int] = field(default_factory=dict)
    terminal: Optional[List[int]] = None
    snaps: Dict[int, State] = field(default_factory=dict)


def run_trajectory(
    circuit,
    init_state: State,
    backend,
    *,
    tm: bool,
    measure_qubits: Optional[Sequence[int]],
    snap_ops: Set[int],
    rng,
    noise_model=None,
) -> TrajectoryResult:
    """执行单条量子测量轨迹。

    参数:
        circuit:        待执行的量子电路
        init_state:     初始量子态（State 对象）
        backend:        后端实例
        tm:             是否在电路执行完后进行末端测量
        measure_qubits: 末端测量的比特列表；None 表示全部比特；
                        tm=False 时忽略此参数
        snap_ops:       需要记录快照的 op_index 集合
        rng:            NumPy 随机数生成器
        noise_model:    可选噪声模型（若提供，在每个酉门后施加噪声，走密度矩阵路径）

    返回:
        TrajectoryResult 对象
    """
    state = init_state
    n = state.n_qubits
    incircuit: Dict[int, int] = {}
    snaps: Dict[int, State] = {}

    for op_index, gate in enumerate(circuit_instructions(circuit)):
        if _is_measure(gate):
            # 线路内嵌测量：联合 Pauli 投影，记录本征值
            qubits = _marker_qubits(gate, n)
            basis = _gate_basis(gate)
            state, lam = projector.measure_joint_pauli(state, qubits, basis, rng)
            incircuit[op_index] = lam
        elif _is_reset(gate):
            # 重置信道：直接作用，无需先测量
            qubits = _marker_qubits(gate, n)
            state = projector.reset_channel(state, qubits)
        else:
            # 酉门演化：密度矩阵形态直接走 gate_to_matrix + evolve（UρU†），
            # 纯态形态先尝试快速路径，回退到 evolve。
            if state.is_density:
                gm = gate_to_matrix(gate, cir_qubits=n, backend=backend)
                state = state.evolve(gm)
            else:
                new_data = apply_gate_to_state(gate, state.data, n, backend)
                if new_data is None:
                    gm = gate_to_matrix(gate, cir_qubits=n, backend=backend)
                    state = state.evolve(gm)
                else:
                    state = State(new_data, n, backend, bit_order=state.bit_order)
            # 可选噪声（密度矩阵路径）
            if noise_model is not None:
                rho_data = (
                    state.data if state.is_density
                    else state.to_density_matrix().data
                )
                rho_noisy = noise_model.apply(
                    rho_data,
                    n_qubits=n,
                    backend=backend,
                    gate_type=instruction_name(gate),
                )
                state = State(rho_noisy, n, backend)

        # 快照：在本 op 执行完后记录
        if op_index in snap_ops:
            snaps[op_index] = state

    # 末端测量
    pre = state
    if tm and measure_qubits is not None and len(measure_qubits) > 0:
        post, terminal = projector.terminal_z_measure(state, measure_qubits, rng)
    elif tm and measure_qubits is None:
        post, terminal = projector.terminal_z_measure(state, list(range(n)), rng)
    else:
        post, terminal = pre, None

    return TrajectoryResult(
        pre=pre,
        post=post,
        incircuit=incircuit,
        terminal=terminal,
        snaps=snaps,
    )
