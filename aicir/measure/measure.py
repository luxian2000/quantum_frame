"""
aicir/measure/measure.py

统一测量模型入口：把"电路 -> 轨迹引擎 -> 校验 -> shot 策略 -> 聚合装配"
串成单一 `Measure.run`。

测量语义（见 README §4 与设计文档）：
- 线路内嵌 measure 门 = 投影测量（由轨迹引擎逐操作执行、坍缩态）
- 末端读出 = 由 tm / measure_qubits 控制的 Z 基逐比特测量
- shots ∈ {None, 0} = exact 模式：单条精确轨迹、不做末端测量（覆盖 tm）
- shots ≥ 1 = M 条轨迹，按 sm（默认 avg）聚合
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..core.state import State
from ..ir import (
    circuit_instructions,
    has_circuit_instructions,
    instruction_name,
    instruction_qubits,
)
from .aggregate import aggregate_avg
from .result import MeasureSpec, Result
from .trajectory import run_trajectory


def _is_measure(g) -> bool:
    """判断操作是否为线路内嵌 measure/measurement 标记。"""
    return instruction_name(g).lower() in {"measure", "measurement"}


def _is_reset(g) -> bool:
    """判断操作是否为线路内嵌 reset 标记。"""
    return instruction_name(g).lower() == "reset"


class Measure:
    """统一测量模型入口。

    使用方式::

        measure = Measure(backend)
        result = measure.run(circuit, shots=1024)
        print(result.counts(-1))
    """

    def __init__(self, backend):
        self.backend = backend

    def _resolve_backend(self, circuit):
        return getattr(circuit, "backend", None) or self.backend

    # ---------- 校验 ----------
    @staticmethod
    def _validate_shots(shots) -> Optional[int]:
        """归一化 shots：None/0 → None（exact 模式）；其余须为非负整数。"""
        if isinstance(shots, bool):  # 须在 == 0 之前：False == 0 为真
            raise ValueError(f"shots 必须是正整数、0 或 None，收到 {shots!r}")
        if shots is None or shots == 0:
            return None  # exact 模式
        if not isinstance(shots, (int, np.integer)) or shots < 0:
            raise ValueError(f"shots 必须是正整数、0 或 None，收到 {shots!r}")
        return int(shots)

    def _collect_specs(self, circuit, n: int) -> List[MeasureSpec]:
        """遍历线路登记每个 measure 操作的 MeasureSpec，并做越界/重复/重名校验。"""
        specs: List[MeasureSpec] = []
        ids = set()
        for op_index, gate in enumerate(circuit_instructions(circuit)):
            qs = [int(q) for q in instruction_qubits(gate)] or list(range(n))
            for q in qs:
                if q < 0 or q >= n:
                    raise ValueError(f"{instruction_name(gate)} 含越界比特 {q}（n={n}）")
            if len(set(qs)) != len(qs):
                raise ValueError(f"{instruction_name(gate)} 含重复比特：{qs}")
            if _is_measure(gate):
                # circuit_instructions 返回类型化 Measurement 对象（非 dict），
                # 须用 getattr 读取 id/basis，否则会被错误地默认。
                mid = getattr(gate, "id", None)
                if mid is not None:
                    if mid in ids:
                        raise ValueError(f"重复的 measure id={mid!r}")
                    ids.add(mid)
                basis = getattr(gate, "basis", "Z")
                specs.append(MeasureSpec(op_index=op_index, id=mid, qubits=qs, basis=str(basis)))
        return specs

    @staticmethod
    def _normalize_measure_qubits(mq, n: int) -> List[int]:
        """归一化显式末端读出比特：越界/重复校验，保留输入顺序（不排序）。"""
        if isinstance(mq, (int, np.integer)):
            mq = [int(mq)]
        out = [int(q) for q in mq]
        for q in out:
            if q < 0 or q >= n:
                raise ValueError(f"measure_qubits 含越界比特 {q}（n={n}）")
        if len(set(out)) != len(out):
            raise ValueError(f"measure_qubits 含重复比特：{out}")
        return out  # 保留输入顺序，不排序

    @staticmethod
    def _normalize_snap(snap, n_ops: int) -> set:
        """归一化 snap 操作下标集合：0<=t<n_ops，去重。"""
        if snap is None:
            return set()
        if isinstance(snap, (int, np.integer)) and not isinstance(snap, bool):
            snap = [snap]
        out = set()
        for t in snap:
            t = int(t)
            if t < 0 or t >= n_ops:
                raise ValueError(f"snap 含越界操作下标 {t}（操作数={n_ops}）")
            out.add(t)
        return out

    # ---------- 主入口 ----------
    def run(self, circuit, shots=1, measure_qubits=None, snap=None,
            tm=True, sm="avg", seed=None, *,
            initial_state=None, observables=None, return_state=True) -> Result:
        """统一测量入口。

        参数:
            circuit:        待测电路（需具备 n_qubits 属性）
            shots:          采样次数；None 或 0 表示 exact 模式（单条精确轨迹，
                            覆盖 tm、不做末端测量）；≥1 表示 M 条轨迹按 sm 聚合
            measure_qubits: 显式末端读出比特（保留顺序）。与 tm=False、exact 模式互斥
            snap:           需记录完整态快照的操作下标集合
            tm:             是否在电路执行完后进行末端测量（exact 模式下被覆盖）
            sm:             多轨迹聚合模式，目前仅支持 'avg'（'shot'/'cond' 暂未实现）
            seed:           随机种子（用于复现）
            initial_state:  初始态（None 表示 |0...0>）
            observables:    可观测量字典 {name: operator_matrix}
            return_state:   是否在结果中附带 state / final_state
        """
        if not hasattr(circuit, "n_qubits"):
            raise TypeError("circuit 需要具备 n_qubits 属性")
        n = int(circuit.n_qubits)
        if n <= 0:
            raise ValueError("n_qubits 必须为正整数")
        backend = self._resolve_backend(circuit)

        norm_shots = self._validate_shots(shots)             # None=exact
        exact = norm_shots is None
        n_ops = len(list(circuit_instructions(circuit))) if has_circuit_instructions(circuit) else 0
        specs = self._collect_specs(circuit, n)
        snap_ops = self._normalize_snap(snap, n_ops)

        if sm not in ("avg", "shot", "cond"):
            raise ValueError(f"sm 必须是 avg/shot/cond，收到 {sm!r}")
        if sm in ("shot", "cond") and snap_ops:
            raise NotImplementedError(f"sm={sm!r} 暂未实现（仅支持 avg）")

        # 末端测量解析（exact 模式覆盖 tm；与显式 measure_qubits 冲突报错）
        mq_explicit = measure_qubits is not None
        if mq_explicit:
            norm_mq = self._normalize_measure_qubits(measure_qubits, n)
        else:
            norm_mq = None
        if not tm and mq_explicit and len(norm_mq) > 0:
            raise ValueError("tm=False 与非空 measure_qubits 冲突")
        if exact and mq_explicit and len(norm_mq) > 0:
            raise ValueError("shots∈{None,0}（exact 模式）覆盖 tm、不做末端测量，"
                             "与显式 measure_qubits 冲突；如需末端读出请用 shots≥1")

        do_terminal = tm and not exact and not (mq_explicit and len(norm_mq) == 0)
        terminal_qubits = (norm_mq if (mq_explicit and len(norm_mq) > 0) else list(range(n))) if do_terminal else None

        noise_model = getattr(circuit, "noise_model", None)
        seed_seq = np.random.SeedSequence(seed) if seed is not None else np.random.SeedSequence()

        def fresh_state() -> State:
            if initial_state is None:
                return State.zero_state(n, backend)
            if isinstance(initial_state, State):
                return initial_state
            return State(initial_state, n, backend)

        has_incircuit = any(True for g in circuit_instructions(circuit) if _is_measure(g)) if n_ops else False
        M = 1 if exact else norm_shots

        rng = np.random.default_rng(seed_seq)
        trajectories = []
        if has_incircuit or noise_model is not None:
            for _ in range(M):
                trajectories.append(run_trajectory(
                    circuit, fresh_state(), backend, tm=do_terminal,
                    measure_qubits=terminal_qubits, snap_ops=snap_ops, rng=rng, noise_model=noise_model))
        else:
            # 无线路中途随机源：ρ_pre 算一次，末端采样 M 次
            base = run_trajectory(circuit, fresh_state(), backend, tm=False,
                                  measure_qubits=None, snap_ops=snap_ops, rng=rng, noise_model=None)
            from .projector import terminal_z_measure
            for _ in range(M):
                if do_terminal:
                    post, terminal = terminal_z_measure(base.pre, terminal_qubits, rng)
                else:
                    post, terminal = base.pre, None
                trajectories.append(type(base)(pre=base.pre, post=post, incircuit={},
                                               terminal=terminal, snaps=base.snaps))

        result = self._build_result(trajectories, n, backend, norm_shots, exact, specs,
                                     terminal_qubits, do_terminal, observables, return_state)
        return result

    def _build_result(self, trajectories, n, backend, norm_shots, exact, specs,
                      terminal_qubits, do_terminal, observables, return_state) -> Result:
        """把轨迹集合按 shots 语义折叠成 Result。"""
        if exact or norm_shots == 1:
            tr = trajectories[0]
            state = np.asarray(tr.pre.to_numpy())
            final = np.asarray(tr.post.to_numpy())
            incircuit_outputs = ({op: tr.incircuit[op] for op in (s.op_index for s in specs)}
                                 if exact else
                                 {op: np.array([[tr.incircuit[op]]]) for op in (s.op_index for s in specs)})
            terminal_output = None
            if do_terminal and tr.terminal is not None:
                terminal_output = (np.array(tr.terminal) if exact
                                   else np.array(tr.terminal).reshape(1, -1))
            snap_states = {t: np.asarray(s.to_numpy()) for t, s in tr.snaps.items()}
            probabilities = np.asarray(tr.pre.probabilities()).reshape(-1).astype(np.float64) \
                if hasattr(tr.pre, "probabilities") else np.abs(state.reshape(-1)) ** 2
            incircuit_counts = {}
            terminal_counts = None
            if not exact:  # shots=1 仍可统计
                incircuit_counts = {op: {int(tr.incircuit[op]): 1} for op in (s.op_index for s in specs)}
                if terminal_output is not None:
                    key = "".join("0" if b == 1 else "1" for b in tr.terminal)
                    terminal_counts = {key: 1}
        else:
            agg = aggregate_avg(trajectories, n, specs, terminal_qubits if do_terminal else None)
            state = agg["state"]; final = agg["final_state"]
            incircuit_outputs = agg["incircuit_outputs"]; incircuit_counts = agg["incircuit_counts"]
            terminal_output = agg["terminal_output"]; terminal_counts = agg["terminal_counts"]
            snap_states = agg["snapshot_states"]; probabilities = agg["probabilities"]

        exp_vals: Dict[str, float] = {}
        exp_vars: Dict[str, float] = {}
        if observables:
            rho = state if (np.asarray(state).ndim == 2 and state.shape[0] == state.shape[1]) else None
            vec = None if rho is not None else np.asarray(state).reshape(-1, 1)
            for name, op in observables.items():
                op = np.asarray(op)
                if rho is not None:
                    exp_vals[name] = float(np.real(np.trace(rho @ op)))
                else:
                    exp_vals[name] = float(np.real((vec.conj().T @ op @ vec)[0, 0]))

        return Result(
            n_qubits=n, backend_name=type(backend).__name__,
            probabilities=probabilities, shots=norm_shots,
            measurement_specs=specs, incircuit_outputs=incircuit_outputs,
            incircuit_counts=incircuit_counts, terminal_output=terminal_output,
            terminal_counts=terminal_counts, terminal_qubits=terminal_qubits,
            state=(state if return_state else None),
            final_state=(final if return_state else None),
            final_state_kind=("density_matrix" if np.asarray(final).ndim == 2 and final.shape[0] == final.shape[1] else "state_vector") if return_state else None,
            expectation_values=exp_vals, expectation_variances=exp_vars,
            snapshot_states=snap_states,
        )
