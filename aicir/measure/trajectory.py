"""单条测量轨迹：递归执行 cir，处理线路内 measure/reset、if/while 控制流、
measure→经典寄存器写入、snap 与末端测量。

执行语义：
- in-circuit measure 门（无经典目标）→ 联合 Pauli 投影测量，本征值记录于 incircuit[op_index]
- in-circuit measure 门（有经典目标 classical_register）→ 逐比特 Z 投影，
  |0>/|1> 写入 trajectory 本地经典 store 的 classical[reg_name][clbit]
- in-circuit reset 门  → 重置信道（无需事先测量）
- if/while 控制流      → 依据 Condition.evaluate(classical) 递归执行 body/else_body；
                        while 超过 max_iterations 仍满足条件抛 RuntimeError
- 酉门              → 演化态；可选在每门后施加噪声（密度矩阵路径）
- snap_ops          → 记录指定 op_index 操作完成后的完整态快照（仅顶层操作下标）
- 末端测量          → tm=True 时对 measure_qubits 逐比特 Z 基测量
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from ..core.gates import apply_gate_to_state, gate_to_matrix
from ..core.state import State
from ..ir import (
    ControlFlow,
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


def _apply_unitary(gate, state: State, backend, n: int, noise_model) -> State:
    """施加一个酉门（可选噪声）。原 run_trajectory 中的酉门演化逻辑原样抽出。"""
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
    return state


def _measure_into_creg(state, qubits, reg_name, clbits, classical, rng):
    """per-qubit Z 投影，比特 |0>->0/|1>->1 写入 classical[reg_name][clbit]。

    复用 projector.terminal_z_measure 做逐比特 Z 采样（按 qubits 入参顺序坍缩并
    返回本征值 ±1，约定 0 比特 -> +1、1 比特 -> -1；此处转换回 0/1 比特值）。
    """
    state, eig = projector.terminal_z_measure(state, list(qubits), rng)
    bits = [0 if lam == 1 else 1 for lam in eig]
    slot = classical.setdefault(reg_name, [0] * (max(clbits) + 1))
    if len(slot) < max(clbits) + 1:
        slot.extend([0] * (max(clbits) + 1 - len(slot)))
    for cb, b in zip(clbits, bits):
        slot[cb] = int(b)
    return state


def _exec_ops(ops, state, classical, backend, n, *, rng, noise_model,
              snap_ops, incircuit, snaps, op_index_ref):
    """递归执行一段操作序列，处理控制流 / measure→creg / 酉门 / reset。

    op_index_ref 是单元素可变列表，充当跨递归层级共享的顶层操作计数器：
    body/else_body 内的操作不消耗顶层 op_index（snap_ops 语义只针对顶层线路）。
    """
    for gate in circuit_instructions(ops):
        if isinstance(gate, ControlFlow):
            cond = gate.condition
            if gate.name == "if":
                if cond.evaluate(classical):
                    state = _exec_ops(gate.body, state, classical, backend, n,
                                      rng=rng, noise_model=noise_model, snap_ops=set(),
                                      incircuit=incircuit, snaps=snaps, op_index_ref=op_index_ref)
                elif gate.else_gates is not None:
                    state = _exec_ops(gate.else_body, state, classical, backend, n,
                                      rng=rng, noise_model=noise_model, snap_ops=set(),
                                      incircuit=incircuit, snaps=snaps, op_index_ref=op_index_ref)
            else:  # while
                iters = 0
                while cond.evaluate(classical):
                    iters += 1
                    if iters > gate.max_iterations:
                        raise RuntimeError(
                            f"while 超过 max_iterations={gate.max_iterations} 仍满足条件")
                    state = _exec_ops(gate.body, state, classical, backend, n,
                                      rng=rng, noise_model=noise_model, snap_ops=set(),
                                      incircuit=incircuit, snaps=snaps, op_index_ref=op_index_ref)
            op_index_ref[0] += 1
            continue

        op_index = op_index_ref[0]
        if _is_measure(gate):
            reg_name = gate.get("classical_register")
            if reg_name is not None:
                clbits = list(getattr(gate, "classical_bits", ()) or gate.get("classical_bits", ()))
                qubits = _marker_qubits(gate, n)
                state = _measure_into_creg(state, qubits, reg_name, clbits, classical, rng)
            else:
                qubits = _marker_qubits(gate, n)
                basis = _gate_basis(gate)
                state, lam = projector.measure_joint_pauli(state, qubits, basis, rng)
                incircuit[op_index] = lam
        elif _is_reset(gate):
            state = projector.reset_channel(state, _marker_qubits(gate, n))
        else:
            state = _apply_unitary(gate, state, backend, n, noise_model)

        if op_index in snap_ops:
            snaps[op_index] = state
        op_index_ref[0] += 1
    return state


@dataclass
class TrajectoryResult:
    """单条轨迹执行的结果。

    属性:
        pre:       末端测量前的量子态（即所有酉门和线路内 measure/reset 执行完后）
        post:      末端测量后的坍缩态（tm=False 时与 pre 相同）
        incircuit: 线路内 measure 门（无经典目标）的测量本征值，键为 op_index
        terminal:  末端测量本征值列表（tm=False 时为 None）
        snaps:     snap_ops 指定位置的量子态快照，键为 op_index
        classical: 轨迹本地经典 store，键为寄存器名，值为该寄存器的位列表（0/1）
    """
    pre: State
    post: State
    incircuit: Dict[int, int] = field(default_factory=dict)
    terminal: Optional[List[int]] = None
    snaps: Dict[int, State] = field(default_factory=dict)
    classical: Dict[str, list] = field(default_factory=dict)


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
    n = init_state.n_qubits
    classical: Dict[str, list] = {}
    incircuit: Dict[int, int] = {}
    snaps: Dict[int, State] = {}

    state = _exec_ops(circuit, init_state, classical, backend, n, rng=rng,
                      noise_model=noise_model, snap_ops=snap_ops, incircuit=incircuit,
                      snaps=snaps, op_index_ref=[0])

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
        classical=classical,
    )
