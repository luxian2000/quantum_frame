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
    ControlFlow,
    circuit_instructions,
    has_circuit_instructions,
    instruction_name,
    instruction_qubits,
)
from .aggregate import aggregate_avg, terminal_mixture
from .result import MeasureSpec, Result
from .trajectory import run_trajectory


def _is_measure(g) -> bool:
    """判断操作是否为线路内嵌 measure/measurement 标记。"""
    return instruction_name(g).lower() in {"measure", "measurement"}


def _is_reset(g) -> bool:
    """判断操作是否为线路内嵌 reset 标记。"""
    return instruction_name(g).lower() == "reset"


def _needs_trajectory(circuit) -> bool:
    """判断电路是否必须走逐轨迹执行（含 in-circuit measure 或控制流指令）。"""
    for g in circuit_instructions(circuit):
        if isinstance(g, ControlFlow) or _is_measure(g):
            return True
    return False


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
            if _is_measure(gate) and gate.get("classical_register") is None:
                # creg 目标的 measure（携带 classical_register）只做逐比特 Z 投影、
                # 写入 trajectory 本地经典 store，不产生联合 Pauli 本征值，
                # 不应注册为 MeasureSpec（否则 exact/shots=1 路径按 op_index 读
                # tr.incircuit 时会 KeyError，因为 _exec_ops 从不为它写 incircuit）。
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
    def run(self, circuit, shots=1, measure_qubits=(), snap=None,
            sm="avg", seed=None, *,
            initial_state=None, initial_density_matrix=None,
            observables=None, return_state=True, method="statevector",
            max_bond_dim=None, cutoff=1e-10) -> Result:
        """统一测量入口。

        参数:
            circuit:                 待测电路（需具备 n_qubits 属性）
            shots:                   采样次数；None 或 0 表示 exact 模式（单条精确轨迹，
                                     不做末端测量，且忽略 measure_qubits）；≥1 表示 M 条轨迹按 sm 聚合
            measure_qubits:          末端读出比特控制（仅 shot 模式生效）：
                                     None=不做末端测量；空（默认）=读出全部比特；
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
            method:                  "statevector"（默认，逐门态矢量演化）、"tensor"（张量网络求末态后
                                     复用既有测量机制；仅纯态、无噪声）或 "mps"（bond 截断的矩阵乘积态
                                     演化求末态后复用既有测量机制；仅纯态、无噪声、不支持 snap/initial_state）
            max_bond_dim:            仅 method="mps" 生效：bond 维上限（None 表示无硬上限，仅按 cutoff 截断）
            cutoff:                  仅 method="mps" 生效：相对最大奇异值的截断阈值（默认 1e-10）
        """
        if not hasattr(circuit, "n_qubits"):
            raise TypeError("circuit 需要具备 n_qubits 属性")
        n = int(circuit.n_qubits)
        if n <= 0:
            raise ValueError("n_qubits 必须为正整数")
        backend = self._resolve_backend(circuit)

        if method not in ("statevector", "tensor", "mps"):
            raise ValueError(f"method 必须是 statevector/tensor/mps，收到 {method!r}")
        if method == "tensor":
            if getattr(circuit, "noise_model", None) is not None:
                raise ValueError("method='tensor' 仅支持纯态，无法用于含噪线路")
            if any(_is_measure(g) for g in circuit_instructions(circuit)):
                raise ValueError("method='tensor' 不支持线路内嵌 measure 标记")
            from ..simulator import tn_statevector
            psi = tn_statevector(circuit, backend=backend)
            from ..core.circuit import Circuit as _Circuit
            stripped = _Circuit(n_qubits=n)
            return self.run(
                stripped, shots=shots, measure_qubits=measure_qubits, snap=snap,
                sm=sm, seed=seed, initial_state=psi, observables=observables,
                return_state=return_state, method="statevector",
            )

        if method == "mps":
            if getattr(circuit, "noise_model", None) is not None:
                raise ValueError("method='mps' 仅支持纯态，无法用于含噪线路")
            if any(_is_measure(g) for g in circuit_instructions(circuit)):
                raise ValueError("method='mps' 不支持线路内嵌 measure 标记")
            if initial_state is not None or initial_density_matrix is not None:
                raise ValueError("method='mps' 始终从 |0...0> 出发，不接受 initial_state/initial_density_matrix")
            if snap not in (None, [], ()):  # 无逐门快照语义
                raise ValueError("method='mps' 不支持非空 snap")
            from ..simulator import mps_statevector
            psi = mps_statevector(circuit, max_bond_dim=max_bond_dim, cutoff=cutoff, backend=backend).to_statevector()
            from ..core.circuit import Circuit as _Circuit
            stripped = _Circuit(n_qubits=n)
            return self.run(
                stripped, shots=shots, measure_qubits=measure_qubits, snap=None,
                sm=sm, seed=seed, initial_state=psi, observables=observables,
                return_state=return_state, method="statevector",
            )

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
            # 空（()/[]）归一化后为空列表，展开为全比特
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

        has_incircuit = _needs_trajectory(circuit) if n_ops else False
        M = 1 if exact else norm_shots

        rng = np.random.default_rng(seed_seq)
        trajectories = []
        if has_incircuit or noise_model is not None:
            for _ in range(M):
                trajectories.append(run_trajectory(
                    circuit, fresh_state(), backend, tm=do_terminal,
                    measure_qubits=terminal_qubits, snap_ops=snap_ops, rng=rng, noise_model=noise_model))
        else:
            # 无线路中途随机源：ρ_pre 算一次，末端读出批量采样（分布只算一次，
            # O(2^n + M)），坍缩后完整态只按不同读出结果构造、且仅在下游需要
            # 聚合末态（return_state / observables）时才构造
            base = run_trajectory(circuit, fresh_state(), backend, tm=False,
                                  measure_qubits=None, snap_ops=snap_ops, rng=rng, noise_model=None)
            if do_terminal:
                from .projector import sample_terminal_batch
                need_posts = bool(return_state or observables)
                outcomes, posts = sample_terminal_batch(base.pre, terminal_qubits, M, rng,
                                                        collapse=need_posts)
                for eig in outcomes:
                    post = posts[tuple(eig)] if need_posts else base.pre
                    trajectories.append(type(base)(pre=base.pre, post=post, incircuit={},
                                                   terminal=eig, snaps=base.snaps))
            else:
                for _ in range(M):
                    trajectories.append(type(base)(pre=base.pre, post=base.pre, incircuit={},
                                                   terminal=None, snaps=base.snaps))

        result = self._build_result(trajectories, n, backend, norm_shots, exact, specs,
                                     terminal_qubits, do_terminal, observables, return_state,
                                     noise_model=noise_model,
                                     initial_density_matrix=initial_density_matrix)
        return result

    def _build_result(self, trajectories, n, backend, norm_shots, exact, specs,
                      terminal_qubits, do_terminal, observables, return_state,
                      *, noise_model=None, initial_density_matrix=None) -> Result:
        classical_trajectories = [dict(getattr(tr, "classical", {}) or {}) for tr in trajectories]
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
            probabilities = backend.to_numpy(tr.pre.probabilities()).reshape(-1).astype(np.float64) \
                if hasattr(tr.pre, "probabilities") else np.abs(backend.to_numpy(state).reshape(-1)) ** 2
            incircuit_counts = {}
            terminal_counts = None
            if not exact:  # shots=1 仍可统计
                incircuit_counts = {op: {int(tr.incircuit[op]): 1} for op in (s.op_index for s in specs)}
                if terminal_output is not None:
                    key = "".join("0" if b == 1 else "1" for b in tr.terminal)
                    terminal_counts = {key: 1}
        else:
            # 仅当调用方需要聚合态（return_state 或 observables）时才做
            # (2^n,2^n) 密度矩阵平均；否则跳过密度矩阵构造以省内存。
            # 共享纯态前态（无噪声、无线路内随机源：全轨迹 pre 为同一对象）时
            # avg(|ψ><ψ|) == |ψ><ψ|，聚合 state 直接保持向量形态；final 仅在
            # 有末端测量时才是真混合态（按读出结果分组构造，见 terminal_mixture）
            tr0 = trajectories[0]
            want_states = bool(return_state or observables)
            shared_pure_pre = (not tr0.pre.is_density) and all(tr.pre is tr0.pre for tr in trajectories)
            agg = aggregate_avg(trajectories, n, specs,
                                terminal_qubits if do_terminal else None,
                                include_states=want_states and not shared_pure_pre)
            if not want_states:
                state = None
                final = None
            elif shared_pure_pre:
                state = tr0.pre
                final = (State.from_matrix(terminal_mixture(trajectories, n), n)
                         if do_terminal else tr0.pre)
            else:
                state = State.from_matrix(np.asarray(agg["state"]), n)
                final = State.from_matrix(np.asarray(agg["final_state"]), n)
            incircuit_outputs = agg["incircuit_outputs"]; incircuit_counts = agg["incircuit_counts"]
            terminal_output = agg["terminal_output"]; terminal_counts = agg["terminal_counts"]
            snap_states = {t: State.from_matrix(np.asarray(s), n) for t, s in agg["snapshot_states"].items()}
            probabilities = agg["probabilities"]

        exp_vals: Dict[str, float] = {}
        exp_vars: Dict[str, float] = {}
        if observables:
            state_arr = backend.to_numpy(state)
            rho = state_arr if (state_arr.ndim == 2 and state_arr.shape[0] == state_arr.shape[1]) else None
            vec = None if rho is not None else state_arr.reshape(-1, 1)
            for name, op in observables.items():
                op = backend.to_numpy(op)
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
            classical_trajectories=classical_trajectories,
        )
