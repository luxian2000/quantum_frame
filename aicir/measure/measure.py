"""
aicir/measure/measure.py

统一测量模型入口：把"电路 -> 轨迹引擎 -> 校验 -> shot 策略 -> 聚合装配"
串成单一 `Measure.run`。

测量语义（见 README §4 与设计文档）：
- 线路内嵌 measure 门 = 投影测量（由轨迹引擎逐操作执行、坍缩态）
- 末端读出 = 由 `measure_qubits` 控制的 Z 基逐比特测量（None=不测；[]=全部；[list]=子集）
- shots ∈ {None, 0} = exact 模式：单条精确轨迹、不做末端测量（忽略 measure_qubits）
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
    def run(self, circuit, shots=1, measure_qubits=[], snap=None,
            sm="avg", seed=None, *,
            initial_state=None, initial_density_matrix=None,
            observables=None, return_state=True) -> Result:
        """统一测量入口。

        参数:
            circuit:                 待测电路（需具备 n_qubits 属性）
            shots:                   采样次数；None 或 0 表示 exact 模式（单条精确轨迹，
                                     不做末端测量，且忽略 measure_qubits）；≥1 表示 M 条轨迹按 sm 聚合
            measure_qubits:          末端读出比特控制（仅 shot 模式生效）：
                                     None=不做末端测量；[]（默认）=读出全部比特；
                                     [q0, q1, …]=读出该子集（保留输入顺序）。
                                     exact 模式下该参数被忽略、不报错
            snap:                    需记录完整态快照的操作下标集合
            sm:                      多轨迹聚合模式，目前仅支持 'avg'（'shot'/'cond' 暂未实现）
            seed:                    随机种子（用于复现）
            initial_state:           初始态（None 表示 |0...0>）
            initial_density_matrix:  初始密度矩阵（提供时以密度矩阵模式初始化量子态；
                                     与 initial_state 互斥）
            observables:             可观测量字典 {name: operator_matrix}
            return_state:            是否在结果中附带 state / final_state
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
        if sm in ("shot", "cond"):
            raise NotImplementedError(f"sm={sm!r} 暂未实现（仅支持 avg）")

        # 末端读出解析：exact 模式永不测量且忽略 measure_qubits；
        # shot 模式下 None=off、[]=全部、[list]=子集。
        if exact or measure_qubits is None:
            do_terminal = False
            terminal_qubits = None
        else:
            norm_mq = self._normalize_measure_qubits(measure_qubits, n)
            terminal_qubits = norm_mq if len(norm_mq) > 0 else list(range(n))
            do_terminal = True

        if initial_state is not None and initial_density_matrix is not None:
            raise ValueError("initial_state 与 initial_density_matrix 互斥，只能提供其一")

        noise_model = getattr(circuit, "noise_model", None)
        seed_seq = np.random.SeedSequence(seed) if seed is not None else np.random.SeedSequence()

        def fresh_state() -> State:
            if initial_density_matrix is not None:
                dm = np.asarray(initial_density_matrix)
                return State(backend.cast(dm), n, backend)
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
                                     terminal_qubits, do_terminal, observables, return_state,
                                     noise_model=noise_model,
                                     initial_density_matrix=initial_density_matrix)
        return result

    def _build_result(self, trajectories, n, backend, norm_shots, exact, specs,
                      terminal_qubits, do_terminal, observables, return_state,
                      *, noise_model=None, initial_density_matrix=None) -> Result:
        """把轨迹集合按 shots 语义折叠成 Result。"""
        if exact or norm_shots == 1:
            tr = trajectories[0]
            state = tr.pre
            final = tr.post
            incircuit_outputs = ({op: tr.incircuit[op] for op in (s.op_index for s in specs)}
                                 if exact else
                                 {op: np.array([[tr.incircuit[op]]]) for op in (s.op_index for s in specs)})
            terminal_output = None
            if do_terminal and tr.terminal is not None:
                terminal_output = (np.array(tr.terminal) if exact
                                   else np.array(tr.terminal).reshape(1, -1))
            snap_states = dict(tr.snaps)
            probabilities = np.asarray(tr.pre.probabilities()).reshape(-1).astype(np.float64) \
                if hasattr(tr.pre, "probabilities") else np.abs(np.asarray(state).reshape(-1)) ** 2
            incircuit_counts = {}
            terminal_counts = None
            if not exact:  # shots=1 仍可统计
                incircuit_counts = {op: {int(tr.incircuit[op]): 1} for op in (s.op_index for s in specs)}
                if terminal_output is not None:
                    key = "".join("0" if b == 1 else "1" for b in tr.terminal)
                    terminal_counts = {key: 1}
        else:
            agg = aggregate_avg(trajectories, n, specs, terminal_qubits if do_terminal else None)
            state = State.from_matrix(np.asarray(agg["state"]), n)
            final = State.from_matrix(np.asarray(agg["final_state"]), n)
            incircuit_outputs = agg["incircuit_outputs"]; incircuit_counts = agg["incircuit_counts"]
            terminal_output = agg["terminal_output"]; terminal_counts = agg["terminal_counts"]
            snap_states = {t: State.from_matrix(np.asarray(s), n) for t, s in agg["snapshot_states"].items()}
            probabilities = agg["probabilities"]

        exp_vals: Dict[str, float] = {}
        exp_vars: Dict[str, float] = {}
        if observables:
            state_arr = np.asarray(state)
            rho = state_arr if (state_arr.ndim == 2 and state_arr.shape[0] == state_arr.shape[1]) else None
            vec = None if rho is not None else state_arr.reshape(-1, 1)
            for name, op in observables.items():
                op = np.asarray(op)
                if rho is not None:
                    exp_vals[name] = float(np.real(np.trace(rho @ op)))
                else:
                    exp_vals[name] = float(np.real((vec.conj().T @ op @ vec)[0, 0]))

        # 判断末态是否为密度矩阵（噪声路径 / 初始密度矩阵输入）
        is_dm = bool(return_state and final.is_density) if return_state else False
        state_mode = "density_matrix" if (noise_model is not None or initial_density_matrix is not None or is_dm) else "state_vector"

        meta: Dict[str, object] = {"state_mode": state_mode}
        if noise_model is not None:
            meta["noise_model"] = type(noise_model).__name__

        return Result(
            n_qubits=n, backend_name=type(backend).__name__,
            probabilities=probabilities, shots=norm_shots,
            measurement_specs=specs, incircuit_outputs=incircuit_outputs,
            incircuit_counts=incircuit_counts, terminal_output=terminal_output,
            terminal_counts=terminal_counts, terminal_qubits=terminal_qubits,
            state=(state if return_state else None),
            final_state=(final if return_state else None),
            final_state_kind=("density_matrix" if is_dm else "state_vector") if return_state else None,
            expectation_values=exp_vals, expectation_variances=exp_vars,
            snapshot_states=snap_states,
            metadata=meta,
        )
