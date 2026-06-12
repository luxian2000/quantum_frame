"""
aicir/measure/measure.py

测量入口：统一调度 "电路 -> 末态 -> 概率/采样/期望值" 的流程。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core.gates import apply_gate_to_state, gate_to_matrix
from ..core.state import State
from ..ir import circuit_instructions, has_circuit_instructions, instruction_name, instruction_qubits
from .result import Result
from .sampler import Sampler


def _is_measure_gate(gate) -> bool:
    """True for in-circuit measurement markers (see :func:`aicir.measure`)."""
    return instruction_name(gate).lower() in {"measure", "measurement"}


def _readout_qubits(circuit, n_qubits: int) -> Tuple[bool, List[int]]:
    """Resolve which qubits to read out from a circuit's measure gates.

    Returns ``(has_measure_gate, qubits)``. With no measure gate the first
    measurement mechanism applies: read out every qubit. An empty ``measure()``
    also reads out every qubit. Otherwise only the marked qubits are read out,
    sorted ascending so their bitstring is a substring of the full-register one.
    """
    measured: set[int] = set()
    has_measure = False
    if not has_circuit_instructions(circuit):
        return False, list(range(n_qubits))
    for gate in circuit_instructions(circuit):
        if not _is_measure_gate(gate):
            continue
        has_measure = True
        measured.update(int(q) for q in instruction_qubits(gate))
    if not has_measure or not measured:
        return has_measure, list(range(n_qubits))
    return True, sorted(measured)


def _normalize_measure_qubits(measure_qubits, n_qubits: int) -> List[int]:
    """Validate and sort an explicit (Approach 1) ``measure_qubits`` argument."""
    if isinstance(measure_qubits, (int, np.integer)):
        measure_qubits = [measure_qubits]
    try:
        qubits = sorted({int(q) for q in measure_qubits})
    except TypeError as exc:
        raise TypeError("measure_qubits 必须是整数或整数序列") from exc
    for q in qubits:
        if q < 0 or q >= n_qubits:
            raise ValueError(
                f"measure_qubits 含越界比特下标 {q}（n_qubits={n_qubits}）"
            )
    return qubits


def _normalize_snap_indices(snap, n_gates: int) -> set[int]:
    """Validate gate indices whose post-gate states should be recorded."""
    if snap is None:
        return set()
    if isinstance(snap, (int, np.integer)) and not isinstance(snap, bool):
        snap = [snap]
    try:
        raw_indices = list(snap)
    except TypeError as exc:
        raise TypeError("snap 必须是整数或整数序列") from exc

    indices: set[int] = set()
    for raw in raw_indices:
        if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
            raise TypeError("snap 必须只包含整数门序号")
        idx = int(raw)
        if idx < 0 or idx >= n_gates:
            raise ValueError(f"snap 含越界门序号 {idx}（门数量={n_gates}）")
        indices.add(idx)
    return indices


def _resolve_readout(circuit, n_qubits: int, measure_qubits=None) -> Tuple[Optional[List[int]], List[int]]:
    """Resolve readout qubits while enforcing measurement-mechanism exclusivity.

    The two measurement mechanisms are mutually exclusive:

    * **Approach 1 (standalone)** — the caller passes ``measure_qubits`` to
      :meth:`Measure.run`; the circuit carries no ``measure()`` gate.
    * **Approach 2 (in-circuit)** — the circuit embeds ``measure()`` gates and
      ``run`` reads them out automatically; no ``measure_qubits`` is given.

    Once a circuit contains ``measure()`` gates, Approach 1 can no longer be
    applied to it: combining the two raises :class:`ValueError`.

    Returns ``(report_qubits, readout)`` where ``readout`` is the ascending list
    of qubits to sample and ``report_qubits`` is what to surface in
    ``metadata['measured_qubits']`` (``None`` when every qubit is read out by the
    plain default, so existing Approach-1 callers see unchanged metadata).
    """
    has_measure, gate_readout = _readout_qubits(circuit, n_qubits)

    if measure_qubits is not None:
        if has_measure:
            raise ValueError(
                "measure_qubits 不能与电路内嵌的 measure() 门同时使用："
                "独立测量（机制一/Approach 1）与线路内嵌测量（机制二/Approach 2）"
                "互斥。请移除电路中的 measure() 门，或不要传入 measure_qubits。"
            )
        explicit = _normalize_measure_qubits(measure_qubits, n_qubits)
        if not explicit:
            return None, list(range(n_qubits))
        return explicit, explicit

    if has_measure:
        return gate_readout, gate_readout
    return None, gate_readout


def _marginal_counts(probs_np, n_qubits: int, qubits: List[int], backend, shots: int) -> Dict[str, int]:
    """Sample ``shots`` outcomes over a subset of qubits (MSB convention)."""
    k = len(qubits)
    index = np.arange(1 << n_qubits)
    sub = np.zeros(1 << n_qubits, dtype=np.int64)
    for qubit in qubits:  # qubit q occupies bit (n-1-q) of the full index
        sub = (sub << 1) | ((index >> (n_qubits - 1 - qubit)) & 1)
    marginal = np.zeros(1 << k, dtype=np.float64)
    np.add.at(marginal, sub, np.asarray(probs_np).real.astype(np.float64).reshape(-1))

    counts_vec = backend.sample(backend.cast(marginal), shots)
    counts_np = backend.to_numpy(counts_vec).astype(int).reshape(-1)
    return {f"|{idx:0{k}b}>": int(c) for idx, c in enumerate(counts_np) if c > 0}


def _single_outcome_index(counts: Dict[str, int]) -> int:
    """从 shots=1 的计数字典中取出唯一测量结果对应的整数下标。"""
    label = next(iter(counts))
    return int(label.strip("|>"), 2)


def _parity_eigenvalue(outcome: int) -> int:
    """Z⊗...⊗Z 关联测量本征值：测得 1 的个数为偶数时 +1，否则 -1。"""
    return 1 if bin(outcome).count("1") % 2 == 0 else -1


def _collapse_state_vector(psi_np, n_qubits: int, readout: List[int], outcome: int):
    """单次投影测量后的坍缩态向量。

    全比特读出时返回保留原相位的基态；子集读出时各被测比特分别沿 Z 基
    坍缩，返回其余比特（下标升序）的归一化纯态，shape ``(2^m, 1)``。
    """
    psi = np.asarray(psi_np).reshape(-1)
    k = len(readout)
    if k == n_qubits:
        amp = psi[outcome]
        phase = amp / abs(amp) if abs(amp) > 0 else 1.0
        collapsed = np.zeros_like(psi)
        collapsed[outcome] = phase
        return collapsed.reshape(-1, 1)

    tensor = psi.reshape([2] * n_qubits)
    slices: List[object] = [slice(None)] * n_qubits
    for j, qubit in enumerate(readout):  # 子串 MSB 对应 readout 中最小的比特
        slices[qubit] = (outcome >> (k - 1 - j)) & 1
    sub = np.ascontiguousarray(tensor[tuple(slices)]).reshape(-1, 1)
    return sub / np.linalg.norm(sub)


def _reduced_density_from_state(psi_np, n_qubits: int, traced_out: List[int]):
    """纯态对 traced_out 比特求偏迹，返回其余比特的约化密度矩阵 ``(2^m, 2^m)``。"""
    psi = np.asarray(psi_np).reshape([2] * n_qubits)
    remaining = [q for q in range(n_qubits) if q not in set(traced_out)]
    mat = np.transpose(psi, remaining + list(traced_out)).reshape(1 << len(remaining), -1)
    return mat @ mat.conj().T


def _collapse_density_matrix(rho_np, n_qubits: int, readout: List[int], outcome: int):
    """单次投影测量后的密度矩阵坍缩。

    全比特读出时返回 ``|outcome><outcome|``；子集读出时把被测比特投影到
    测得值并归一化，同时迹掉被测比特，返回其余比特的密度矩阵。
    """
    dim = 1 << n_qubits
    rho = np.asarray(rho_np).reshape(dim, dim)
    k = len(readout)
    if k == n_qubits:
        collapsed = np.zeros_like(rho)
        collapsed[outcome, outcome] = 1.0
        return collapsed

    tensor = rho.reshape([2] * (2 * n_qubits))
    slices: List[object] = [slice(None)] * (2 * n_qubits)
    for j, qubit in enumerate(readout):
        bit = (outcome >> (k - 1 - j)) & 1
        slices[qubit] = bit
        slices[n_qubits + qubit] = bit
    m = n_qubits - k
    sub = np.ascontiguousarray(tensor[tuple(slices)]).reshape(1 << m, 1 << m)
    return sub / np.trace(sub)


def _reduced_density_from_density(rho_np, n_qubits: int, traced_out: List[int]):
    """密度矩阵对 traced_out 比特求偏迹，返回其余比特的约化密度矩阵。"""
    dim = 1 << n_qubits
    rho = np.asarray(rho_np).reshape(dim, dim).reshape([2] * (2 * n_qubits))
    remaining = [q for q in range(n_qubits) if q not in set(traced_out)]
    perm = (
        remaining
        + list(traced_out)
        + [n_qubits + q for q in remaining]
        + [n_qubits + q for q in traced_out]
    )
    m, k = len(remaining), len(traced_out)
    t = np.transpose(rho, perm).reshape(1 << m, 1 << k, 1 << m, 1 << k)
    return np.einsum("akbk->ab", t)


def _resolve_post_measurement(
    use_shots: int,
    n_qubits: int,
    readout: List[int],
    counts: Optional[Dict[str, int]],
    state_np,
    state_kind: str,
    collapse_fn,
    reduce_fn,
):
    """按 shots 语义解析 ``(final_state, output, final_state_kind, final_state_qubits)``。

    - use_shots=0：不测量，final_state 即测量前末态 state；
    - use_shots=1：单次投影测量；output 为被测比特上 Z⊗...⊗Z 的本征值（±1），
      子集读出时 final_state 仅含未被测比特（坍缩到的基态见 counts）；
    - use_shots>1：final_state 为对被测比特求偏迹的约化密度矩阵；
      全比特读出时无剩余比特，final_state 为 None。

    ``state_np`` 为 None（return_state=False）时不计算任何末态，仅产出 output。
    """
    full_readout = len(readout) == n_qubits
    remaining = [q for q in range(n_qubits) if q not in set(readout)]

    if use_shots == 0:
        return state_np, None, state_kind, list(range(n_qubits))

    if use_shots == 1:
        outcome = _single_outcome_index(counts)
        output = _parity_eigenvalue(outcome)
        qubits = list(range(n_qubits)) if full_readout else remaining
        final = collapse_fn(outcome) if state_np is not None else None
        return final, output, state_kind, qubits

    if full_readout:
        return None, None, None, []
    final = reduce_fn() if state_np is not None else None
    return final, None, "density_matrix", remaining


class Measure:
    """
    量子测量与结果生成入口。

    使用方式:
        measure = Measure(backend)
        result = measure.run(circuit, shots=1024)
        print(result.counts)
    """

    def __init__(self, backend):
        self.backend = backend
        self.sampler = Sampler(backend)
        self._variance_eps = 1e-7

    def _resolve_backend(self, circuit):
        circuit_backend = getattr(circuit, "backend", None)
        return circuit_backend if circuit_backend is not None else self.backend

    @staticmethod
    def _has_gate_sequence(circuit) -> bool:
        return has_circuit_instructions(circuit)

    def _build_initial_state(self, n_qubits: int, backend, initial_state=None) -> State:
        if initial_state is None:
            return State.zero_state(n_qubits, backend)

        if isinstance(initial_state, State):
            if initial_state.is_density:
                raise TypeError(
                    "initial_state 为密度矩阵形态 State；态矢演化需要向量形态"
                )
            if initial_state.n_qubits != n_qubits:
                raise ValueError("initial_state.n_qubits 与电路 n_qubits 不一致")
            return initial_state

        return State(initial_state, n_qubits, backend)

    def _build_initial_density_matrix(
        self,
        n_qubits: int,
        backend,
        initial_density_matrix=None,
    ) -> State:
        if initial_density_matrix is None:
            dim = 1 << n_qubits
            rho = np.zeros((dim, dim), dtype=np.complex64)
            rho[0, 0] = 1.0 + 0j
            return State.from_matrix(rho, n_qubits, backend)

        if isinstance(initial_density_matrix, State):
            if not initial_density_matrix.is_density:
                raise TypeError(
                    "initial_density_matrix 为向量形态 State；请先调用 .to_density_matrix()"
                )
            if initial_density_matrix.n_qubits != n_qubits:
                raise ValueError("initial_density_matrix.n_qubits 与电路 n_qubits 不一致")
            return initial_density_matrix

        return State.from_matrix(initial_density_matrix, n_qubits, backend)

    def _circuit_unitary_on_backend(self, circuit, backend):
        """
        优先走 `unitary(backend=...)` 路径以在设备端组装/累乘矩阵。

        兼容旧式或外部电路实现：若不支持 backend 参数，则回退到 unitary()。
        """
        try:
            unitary_raw = circuit.unitary(backend=backend)
        except TypeError:
            unitary_raw = circuit.unitary()
        # Fast path: when unitary_raw is already a backend-native tensor on the
        # target device/dtype, backend.cast can return without host round-trip.
        return backend.cast(unitary_raw)

    def _evolve_state_vector_gatewise(
        self,
        instructions,
        sv0: State,
        backend,
        snap_indices: Optional[set[int]] = None,
    ) -> Tuple[State, Dict[int, np.ndarray]]:
        sv = sv0
        snap_indices = snap_indices or set()
        snapshot_states: Dict[int, np.ndarray] = {}
        for gate_index, gate in enumerate(instructions):
            if _is_measure_gate(gate):
                if gate_index in snap_indices:
                    snapshot_states[gate_index] = sv.to_numpy().copy()
                continue
            new_data = apply_gate_to_state(gate, sv.data, sv.n_qubits, backend)
            if new_data is None:
                gm = gate_to_matrix(gate, cir_qubits=sv.n_qubits, backend=backend)
                sv = sv.evolve(gm)
            else:
                sv = State(new_data, sv.n_qubits, backend, bit_order=sv.bit_order)
            if gate_index in snap_indices:
                snapshot_states[gate_index] = sv.to_numpy().copy()
        return sv, snapshot_states

    def _evolve_density_matrix_gatewise(
        self,
        circuit,
        rho0: State,
        backend,
        noise_model=None,
    ) -> State:
        rho = rho0
        for gate in circuit_instructions(circuit):
            if _is_measure_gate(gate):
                continue
            gate_unitary = gate_to_matrix(gate, cir_qubits=rho.n_qubits, backend=backend)
            rho = rho.evolve(gate_unitary)
            if noise_model is not None:
                rho_noisy = noise_model.apply(
                    rho.data,
                    n_qubits=rho.n_qubits,
                    backend=backend,
                    gate_type=instruction_name(gate),
                )
                rho = State(rho_noisy, rho.n_qubits, backend)
        return rho

    def run(
        self,
        circuit,
        shots: Optional[int] = 1,
        initial_state=None,
        observables: Optional[Dict[str, object]] = None,
        return_state: bool = True,
        measure_qubits: Optional[Sequence[int]] = None,
        snap: Optional[Sequence[int]] = None,
    ) -> Result:
        """
        测量一个电路，返回统一结果对象。

        参数:
            circuit: 必须具有 `n_qubits` 属性和 `unitary()` 方法
            shots: 采样次数，默认 1。为 None 或 0 时不测量（仅返回概率与末态），
                此时 result.final_state 与 result.state 相同；
                为 1 时做单次投影测量，result.final_state 为坍缩后的态，
                result.output 为被测比特上 Z⊗...⊗Z 的本征值（±1）；
                大于 1 时 result.final_state 为对被测比特求偏迹的约化密度矩阵
                （读出全部比特时无剩余比特，为 None），其余统计字段照常
            initial_state: 初始态（None 表示 |0...0>）
            observables: 可观测量字典 {name: operator_matrix}
            return_state: 是否在结果中附带 state / final_state
            measure_qubits: 机制一（Approach 1）显式读出比特；None 表示读取全部。
                与电路内嵌的 measure() 门（机制二/Approach 2）互斥，二者不可同时使用。
                shots=1 且指定子集时，对这些比特做 Z⊗...⊗Z 关联投影测量，
                final_state 仅含未被测比特
            snap: 可选门序号列表（从 0 开始）；记录这些门作用结束后的完整态，
                可通过 result.snap(index) 读取。None 表示不记录。
        """
        if not hasattr(circuit, "n_qubits"):
            raise TypeError("circuit 需要具备 n_qubits 属性")
        if not self._has_gate_sequence(circuit) and not hasattr(circuit, "unitary"):
            raise TypeError("circuit 需要具备 gates 序列或 unitary() 方法")

        n_qubits = int(circuit.n_qubits)
        if n_qubits <= 0:
            raise ValueError("n_qubits 必须为正整数")

        backend = self._resolve_backend(circuit)
        sampler = Sampler(backend)

        sv0 = self._build_initial_state(n_qubits, backend, initial_state=initial_state)

        snapshot_states: Dict[int, np.ndarray] = {}
        if self._has_gate_sequence(circuit):
            # Preferred execution path: apply each gate directly on the state.
            instructions = circuit_instructions(circuit)
            snap_indices = _normalize_snap_indices(snap, len(instructions))
            sv, snapshot_states = self._evolve_state_vector_gatewise(
                instructions,
                sv0,
                backend,
                snap_indices=snap_indices,
            )
        else:
            if snap is not None:
                raise ValueError("snap 需要电路提供 gates/operations 门序列")
            unitary = self._circuit_unitary_on_backend(circuit, backend)
            sv = sv0.evolve(unitary)
        probs_backend = sv.probabilities()
        probs = backend.to_numpy(probs_backend).real

        # Resolve readout qubits, enforcing that the explicit (Approach 1)
        # measure_qubits and in-circuit measure() gates (Approach 2) are not
        # combined. With neither present every qubit is read out as before.
        report_qubits, readout = _resolve_readout(circuit, n_qubits, measure_qubits)

        counts = None
        use_shots = int(shots) if shots else 0
        if use_shots < 0:
            raise ValueError("shots 不能为负数")
        if use_shots > 0:
            if len(readout) < n_qubits:
                counts = _marginal_counts(probs, n_qubits, readout, backend, use_shots)
            else:
                counts = sampler.sample_counts(probs_backend, n_qubits=n_qubits, shots=use_shots)

        exp_vals: Dict[str, float] = {}
        exp_vars: Dict[str, float] = {}
        if observables:
            for name, op in observables.items():
                op_backend = backend.cast(backend.to_numpy(op))
                exp_val = float(backend.to_numpy(sv.expectation(op_backend)))
                op2 = backend.matmul(op_backend, op_backend)
                exp2 = float(backend.to_numpy(sv.expectation(op2)))
                var = exp2 - exp_val * exp_val
                if abs(var) < self._variance_eps:
                    var = 0.0
                exp_vals[name] = exp_val
                exp_vars[name] = max(var, 0.0)

        state_np = sv.to_numpy() if return_state else None
        final_state_np, output, final_kind, final_qubits = _resolve_post_measurement(
            use_shots,
            n_qubits,
            readout,
            counts,
            state_np,
            "state_vector",
            collapse_fn=lambda outcome: _collapse_state_vector(state_np, n_qubits, readout, outcome),
            reduce_fn=lambda: _reduced_density_from_state(state_np, n_qubits, readout),
        )

        return Result(
            n_qubits=n_qubits,
            backend_name=backend.name,
            probabilities=probs,
            counts=counts,
            shots=use_shots if use_shots > 0 else None,
            expectation_values=exp_vals,
            expectation_variances=exp_vars,
            final_state=final_state_np,
            state=state_np,
            output=output,
            snapshot_states=snapshot_states,
            metadata={
                "measure": "Measure",
                "circuit_type": type(circuit).__name__,
                "state_mode": "state_vector",
                "measured_qubits": report_qubits,
                "final_state_kind": final_kind,
                "final_state_qubits": final_qubits,
                "snap_indices": sorted(snapshot_states),
            },
        )

    def run_density_matrix(
        self,
        circuit,
        shots: Optional[int] = 1,
        initial_density_matrix=None,
        observables: Optional[Dict[str, object]] = None,
        noise_model=None,
        return_state: bool = True,
        measure_qubits: Optional[Sequence[int]] = None,
    ) -> Result:
        """
        以密度矩阵路径测量一个电路，返回统一结果对象。

        shots 语义与 :meth:`run` 一致（None/0 不测量、1 单次坍缩、>1 偏迹），
        差别仅在于本路径的 state / final_state 均为 flatten 一维的密度矩阵数据。

        参数:
            circuit: 必须具有 `n_qubits` 属性和 `unitary()` 方法
            shots: 采样次数，默认 1；为 None 或 0 时不测量，仅返回概率
            initial_density_matrix: 初始密度矩阵（None 表示 |0...0><0...0|）
            observables: 可观测量字典 {name: operator_matrix}
            noise_model: NoiseModel，可选。若提供且电路具有 gates，则在每个门后施加噪声。
            return_state: 是否在结果中附带 state / final_state（flatten 一维）
            measure_qubits: 机制一（Approach 1）显式读出比特；None 表示读取全部。
                与电路内嵌的 measure() 门（机制二/Approach 2）互斥，二者不可同时使用。
        """
        if not hasattr(circuit, "n_qubits"):
            raise TypeError("circuit 需要具备 n_qubits 属性")
        if not self._has_gate_sequence(circuit) and not hasattr(circuit, "unitary"):
            raise TypeError("circuit 需要具备 gates 序列或 unitary() 方法")

        n_qubits = int(circuit.n_qubits)
        if n_qubits <= 0:
            raise ValueError("n_qubits 必须为正整数")

        backend = self._resolve_backend(circuit)
        sampler = Sampler(backend)

        rho0 = self._build_initial_density_matrix(
            n_qubits,
            backend,
            initial_density_matrix=initial_density_matrix,
        )

        if self._has_gate_sequence(circuit):
            rho = self._evolve_density_matrix_gatewise(
                circuit,
                rho0,
                backend,
                noise_model=noise_model,
            )
        else:
            rho = rho0
            unitary = self._circuit_unitary_on_backend(circuit, backend)
            rho = rho.evolve(unitary)

        probs = rho.probabilities()
        probs_backend = backend.cast(probs)

        report_qubits, readout = _resolve_readout(circuit, n_qubits, measure_qubits)

        counts = None
        use_shots = int(shots) if shots else 0
        if use_shots < 0:
            raise ValueError("shots 不能为负数")
        if use_shots > 0:
            if len(readout) < n_qubits:
                counts = _marginal_counts(
                    backend.to_numpy(probs_backend), n_qubits, readout, backend, use_shots
                )
            else:
                counts = sampler.sample_counts(probs_backend, n_qubits=n_qubits, shots=use_shots)

        exp_vals: Dict[str, float] = {}
        exp_vars: Dict[str, float] = {}
        if observables:
            for name, op in observables.items():
                op_backend = backend.cast(backend.to_numpy(op))
                exp_val = float(backend.to_numpy(rho.expectation(op_backend)))
                op2 = backend.matmul(op_backend, op_backend)
                exp2 = float(backend.to_numpy(rho.expectation(op2)))
                var = exp2 - exp_val * exp_val
                if abs(var) < self._variance_eps:
                    var = 0.0
                exp_vals[name] = exp_val
                exp_vars[name] = max(var, 0.0)

        rho_np = rho.to_numpy() if return_state else None
        state_np = rho_np.reshape(-1) if rho_np is not None else None
        final_state_np, output, final_kind, final_qubits = _resolve_post_measurement(
            use_shots,
            n_qubits,
            readout,
            counts,
            state_np,
            "density_matrix",
            collapse_fn=lambda outcome: _collapse_density_matrix(
                rho_np, n_qubits, readout, outcome
            ).reshape(-1),
            reduce_fn=lambda: _reduced_density_from_density(rho_np, n_qubits, readout).reshape(-1),
        )

        return Result(
            n_qubits=n_qubits,
            backend_name=backend.name,
            probabilities=np.asarray(probs, dtype=np.float64),
            counts=counts,
            shots=use_shots if use_shots > 0 else None,
            expectation_values=exp_vals,
            expectation_variances=exp_vars,
            final_state=final_state_np,
            state=state_np,
            output=output,
            metadata={
                "measure": "Measure",
                "circuit_type": type(circuit).__name__,
                "state_mode": "density_matrix",
                "noise_model": type(noise_model).__name__ if noise_model is not None else None,
                "measured_qubits": report_qubits,
                "final_state_kind": final_kind,
                "final_state_qubits": final_qubits,
            },
        )

    def run_batch(
        self,
        circuits: Sequence[object],
        shots: Optional[int] = None,
        observables: Optional[Dict[str, object]] = None,
        mode: str = "state_vector",
        per_circuit_options: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Result]:
        """
        批量运行多个电路。

        参数:
            circuits: 电路序列，每个元素需具备 n_qubits 和 unitary()
            shots: 全局采样次数（可被 per_circuit_options 覆盖）
            observables: 全局可观测量字典（可被 per_circuit_options 覆盖）
            mode: "state_vector" 或 "density_matrix"
            per_circuit_options: 每个电路的附加参数字典序列，长度需与 circuits 相同
        返回:
            Result 列表，顺序与输入 circuits 一致
        """
        if not isinstance(circuits, Sequence) or len(circuits) == 0:
            raise ValueError("circuits 必须是非空序列")

        if per_circuit_options is not None and len(per_circuit_options) != len(circuits):
            raise ValueError("per_circuit_options 长度必须与 circuits 一致")

        mode_norm = mode.strip().lower()
        if mode_norm not in {"state_vector", "density_matrix"}:
            raise ValueError("mode 仅支持 'state_vector' 或 'density_matrix'")

        indexed_results = []
        for idx, circ in enumerate(circuits):
            if hasattr(self.backend, "should_run_batch_index") and not self.backend.should_run_batch_index(idx):
                continue

            opts = per_circuit_options[idx] if per_circuit_options is not None else {}
            run_shots = opts.get("shots", shots)
            run_observables = opts.get("observables", observables)
            run_return_state = opts.get("return_state", True)
            label = opts.get("label")

            if mode_norm == "state_vector":
                result = self.run(
                    circ,
                    shots=run_shots,
                    initial_state=opts.get("initial_state"),
                    observables=run_observables,
                    return_state=run_return_state,
                    measure_qubits=opts.get("measure_qubits"),
                    snap=opts.get("snap"),
                )
            else:
                result = self.run_density_matrix(
                    circ,
                    shots=run_shots,
                    initial_density_matrix=opts.get("initial_density_matrix"),
                    observables=run_observables,
                    noise_model=opts.get("noise_model"),
                    measure_qubits=opts.get("measure_qubits"),
                    return_state=run_return_state,
                )

            result.metadata["batch_index"] = idx
            if label is not None:
                result.metadata["label"] = str(label)
            indexed_results.append((idx, result))

        if hasattr(self.backend, "gather_indexed_results"):
            indexed_results = self.backend.gather_indexed_results(indexed_results)
        else:
            indexed_results = sorted(indexed_results, key=lambda item: item[0])

        return [result for _, result in indexed_results]

    def scan_parameters(
        self,
        circuit_builder: Callable[[Any], object],
        param_values: Iterable[Any],
        shots: Optional[int] = None,
        observables: Optional[Dict[str, object]] = None,
        mode: str = "state_vector",
        return_state: bool = False,
    ) -> List[Result]:
        """
        参数扫描：针对一组参数值构造电路并批量执行。

        返回:
            Result 列表，每个结果 metadata 包含 `scan_param`
        """
        params = list(param_values)
        if len(params) == 0:
            raise ValueError("param_values 不能为空")

        circuits = [circuit_builder(p) for p in params]
        options = [{"return_state": return_state, "label": f"scan_{i}"} for i in range(len(params))]
        results = self.run_batch(
            circuits,
            shots=shots,
            observables=observables,
            mode=mode,
            per_circuit_options=options,
        )

        for i, p in enumerate(params):
            results[i].metadata["scan_param"] = p
            results[i].metadata["scan_index"] = i
        return results
