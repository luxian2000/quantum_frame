"""交错 simulate↔decode 运行器。

本文件在 Task 4 只提供无噪声综合征采集（verify_schedule 依赖），
完整的逐 shot 在线解码循环在 Task 7 补齐。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from aicir.backends import NumpyBackend
from aicir.core.state import State
from aicir.measure.trajectory import run_trajectory

from .decoders import resolve_decoder
from .errors import PauliErrorModel
from .record import QECResult, QECShotRecord

_PAULI_GATE = {"X": "pauli_x", "Y": "pauli_y", "Z": "pauli_z"}


def _read_creg(classical: dict, name: str, size: int) -> np.ndarray:
    """从轨迹经典 store 读出 size 位，缺位补 0。"""
    bits = list(classical.get(name, []))
    bits.extend([0] * (size - len(bits)))
    return np.array(bits[:size], dtype=np.uint8)


def collect_noiseless_syndromes(code, schedule, rounds: int, *, logical_state: str = "0",
                                backend=None, seed: int = 0) -> np.ndarray:
    """无噪声运行 rounds 轮，返回 raw_syndromes，shape (rounds, m) uint8。

    **不返回 reference**：轮 0 参考值恒为 zeros(m)（见 verify_schedule 的说明）。
    非确定生成元的轮 0 读数逐 shot 随机，任何「从一次运行实测 reference」的做法
    都会与其他 shot 错配。
    """
    from .schedules import resolve_schedule

    schedule = resolve_schedule(schedule)
    backend = backend or NumpyBackend()
    rng = np.random.default_rng(seed)
    n_total = code.n + code.m

    state = run_trajectory(
        schedule.build_encode(code, logical_state), State.zero_state(n_total, backend),
        backend, tm=False, measure_qubits=None, snap_ops=set(), rng=rng,
    ).pre

    raw = np.zeros((int(rounds), code.m), dtype=np.uint8)
    for t in range(int(rounds)):
        rc = schedule.build_round(code, t)
        res = run_trajectory(rc.circuit, state, backend, tm=False,
                             measure_qubits=None, snap_ops=set(), rng=rng)
        state = res.pre
        raw[t] = _read_creg(res.classical, rc.creg_name, code.m)

    return raw


def _error_gates(events):
    """data 错误事件 → 门列表。"""
    from aicir.core.circuit import pauli_x, pauli_y, pauli_z
    factory = {"X": pauli_x, "Y": pauli_y, "Z": pauli_z}
    return [factory[e.pauli](e.qubit) for e in events if e.source == "data"]


def _apply_pauli_gates(state, pairs, backend, n_total, rng):
    """把 [(qubit, pauli), ...] 作为真实门施加到量子态上。"""
    from aicir.core.circuit import Circuit, pauli_x, pauli_y, pauli_z
    if not pairs:
        return state
    factory = {"X": pauli_x, "Y": pauli_y, "Z": pauli_z}
    cir = Circuit(*[factory[p](q) for q, p in pairs], n_qubits=n_total)
    return run_trajectory(cir, state, backend, tm=False, measure_qubits=None,
                          snap_ops=set(), rng=rng).pre


def run(code, *, schedule="bare", errors=None, decoder=None, rounds=2, shots=1,
        logical_state="0", correction_mode="frame", timing=None, backend=None,
        seed=None, keep_records=100, keep_failures=100) -> QECResult:
    """逐 shot 交错「模拟 ↔ 解码」主循环。"""
    from .schedules import build_layout, resolve_schedule

    rounds = int(rounds)
    if rounds < 1:
        raise ValueError(f"rounds 必须 ≥1，收到 {rounds}")
    shots = int(shots)
    if shots < 1:
        raise ValueError(f"shots 必须 ≥1，收到 {shots}")
    if correction_mode not in ("frame", "active"):
        raise ValueError(f"correction_mode 只支持 'frame' / 'active'，收到 {correction_mode!r}")

    if decoder is None:
        raise ValueError("decoder 为必填参数；请传入实现 OnlineDecoder 协议的实例或已注册的名字")
    schedule = resolve_schedule(schedule)
    decoder = resolve_decoder(decoder)
    errors = errors if errors is not None else PauliErrorModel()
    backend = backend or NumpyBackend()
    layout = build_layout(code, schedule, rounds, logical_state=logical_state)
    n_total = code.n + code.m

    # 轮 0 参考值恒为 zeros：轮 0 确定的生成元读数确定为 0，其余已被 layout 掩掉。
    reference = np.zeros(code.m, dtype=np.uint8)

    kept, failures, verdicts = [], [], {}
    for shot in range(shots):
        shot_seed = shot if seed is None else int(seed) * 100003 + shot
        record = _run_one_shot(
            code, schedule, errors, decoder, rounds, layout, reference,
            logical_state, correction_mode, timing, backend, shot, shot_seed, n_total,
        )
        verdicts[record.verdict] = verdicts.get(record.verdict, 0) + 1
        if len(kept) < int(keep_records):
            kept.append(record)
        if record.verdict != "corrected" and len(failures) < int(keep_failures):
            failures.append(record)

    n_fail = shots - verdicts.get("corrected", 0)
    p = n_fail / shots
    result = QECResult(
        code_name=code.name, decoder_name=getattr(decoder, "name", type(decoder).__name__),
        schedule_name=getattr(schedule, "name", type(schedule).__name__),
        rounds=rounds, shots=shots, records=kept, failure_records=failures,
        logical_error_rate=p, logical_error_rate_stderr=float(np.sqrt(p * (1 - p) / shots)),
        verdict_counts=verdicts,
    )
    if timing is not None:
        _fill_timing_aggregates(result, kept, timing)
    return result


def _run_one_shot(code, schedule, errors, decoder, rounds, layout, reference,
                  logical_state, correction_mode, timing, backend, shot, shot_seed, n_total):
    rng = np.random.default_rng(shot_seed)
    decoder.reset(layout)

    state = run_trajectory(
        schedule.build_encode(code, logical_state), State.zero_state(n_total, backend),
        backend, tm=False, measure_qubits=None, snap_ops=set(), rng=rng,
    ).pre

    raw = np.zeros((rounds, code.m), dtype=np.uint8)
    events_log = np.zeros((rounds, code.m), dtype=np.uint8)
    injected, steps, wall = [], [], np.zeros(rounds, dtype=float)
    frame = np.zeros(2 * code.k, dtype=np.uint8)
    applied = np.zeros(2 * code.n, dtype=np.uint8)     # active 模式下已施加修正的累积
    committed = -1

    for t in range(rounds):
        # 轮 0 是投影式制备（|0…0⟩ 一般不在码空间内），不注入错误；从轮 1 开始注入。
        round_errors = [] if t == 0 else errors.sample_round(t, code.n, code.m, rng)
        injected.extend(round_errors)

        rc = schedule.build_round(code, t)
        cir = rc.circuit
        cir.gates[:0] = _error_gates(round_errors)     # data 错误在该轮提取之前施加

        res = run_trajectory(cir, state, backend, tm=False, measure_qubits=None,
                             snap_ops=set(), rng=rng)
        state = res.pre
        bits = _read_creg(res.classical, rc.creg_name, code.m)

        for e in round_errors:                          # 测量误差翻转经典记录
            if e.source == "measurement":
                bits[e.qubit] ^= 1
        raw[t] = bits

        ev = layout.detection_events(raw, t, reference)
        if correction_mode == "active":
            # active 模式下已施加的修正会把原始稳定子读数复位，若不扣除，
            # 下一轮的朴素差分会放出一个虚假 detection event。
            ev = (ev ^ _applied_syndrome_delta(code, applied, raw, t)).astype(np.uint8)
        events_log[t] = ev

        t0 = time.perf_counter()
        step = decoder.update(t, ev)
        wall[t] = time.perf_counter() - t0

        if step.committed_through < committed:
            raise ValueError(
                f"解码器 {getattr(decoder, 'name', '?')} 的 committed_through 回退："
                f"轮 {t} 报 {step.committed_through}，此前已提交到 {committed}"
            )
        committed = step.committed_through
        steps.append(step)

        if step.frame_flips is not None:
            frame ^= np.asarray(step.frame_flips, dtype=np.uint8).ravel()[:2 * code.k]
        if correction_mode == "active" and step.corrections:
            state = _apply_pauli_gates(state, step.corrections, backend, n_total, rng)
            for q, p in step.corrections:
                if p in ("X", "Y"):
                    applied[q] ^= 1
                if p in ("Z", "Y"):
                    applied[code.n + q] ^= 1

    final = decoder.flush()
    if final.committed_through < committed:
        raise ValueError("flush() 的 committed_through 不得低于此前已提交轮次")
    if final.frame_flips is not None:
        frame ^= np.asarray(final.frame_flips, dtype=np.uint8).ravel()[:2 * code.k]

    ro = schedule.build_readout(code, logical_state)
    ro_res = run_trajectory(ro.circuit, state, backend, tm=False, measure_qubits=None,
                            snap_ops=set(), rng=rng)
    readout = _read_creg(ro_res.classical, ro.creg_name, code.n)

    residual = _residual_from_readout(code, readout, frame, logical_state)
    verdict = code.verdict(residual)

    record = QECShotRecord(
        shot=shot, seed=shot_seed, injected_errors=injected, raw_syndromes=raw,
        detection_events=events_log, decode_steps=steps, wall_clock=wall,
        observable_raw=readout, frame_flips=frame, verdict=verdict,
    )
    if timing is not None:
        _fill_shot_timing(record, decoder, steps, timing, rounds)
    return record


def _applied_syndrome_delta(code, applied, raw, t):
    """active 模式：已施加修正自身对各稳定子的贡献。"""
    if t == 0 or not applied.any():
        return np.zeros(code.m, dtype=np.uint8)
    return code.syndrome(applied).astype(np.uint8)


def _residual_from_readout(code, readout, frame, logical_state):
    """由 data 比特读数与已提交 frame 还原残余逻辑 Pauli。

    读数给出逻辑算符的奇偶；frame 是解码器声称已抵消的部分。二者异或后
    仍非零的分量，即真正逃逸的逻辑错误。
    """
    residual = np.zeros(2 * code.n, dtype=np.uint8)
    for i in range(code.k):
        support = code.logical_z[i][code.n:] if logical_state in ("0", "1") \
            else code.logical_x[i][:code.n]
        parity = int(np.dot(readout, support) % 2)
        flip = int(frame[2 * i]) if logical_state in ("0", "1") else int(frame[2 * i + 1])
        if parity ^ flip:
            residual ^= code.logical_x[i] if logical_state in ("0", "1") else code.logical_z[i]
    return residual


@dataclass
class TimingModel:
    """实时预算模型。

    用**声明代价**而非 Python wall-clock：Python 解码器比 FPGA 慢约 10⁴ 倍，
    wall-clock 对实时可行性毫无意义。解码器经 cost_of() 声明自己的复杂度度量，
    cost_to_seconds 把它映射到用户掌控的硬件模型（例如「每单位代价对应多少纳秒」）。

    wall-clock 也照常记录（免费），但在**独立字段** QECShotRecord.wall_clock 里，
    绝不与建模时间混同——一个是 Python 解释器实测的秒数，一个是研究者假想中
    目标硬件（FPGA/ASIC）会花费的秒数，二者数量级可以相差 10⁴ 倍以上。
    """
    round_duration: float
    cost_to_seconds: Callable[[float], float] = float

    def __post_init__(self):
        if float(self.round_duration) <= 0.0:
            raise ValueError(f"round_duration 必须为正，收到 {self.round_duration}")


def backlog_sequence(decode_times, round_duration: float) -> list[float]:
    """确定性到达的单服务台排队：backlog[t] = max(0, backlog[t−1] + decode[t] − 轮时长)。

    每轮综合征都在轮时长 round_duration 之后「到达」，解码器要花 decode[t] 秒处理它；
    backlog 是尚未被消化的积压时间。backlog 线性增长（斜率为正）即**吞吐失败模式**：
    解码器永久落后，无论运行多久都追不上新到来的轮次——这是比「单轮偶尔超时」更
    根本的判据。
    """
    out, backlog = [], 0.0
    for dt in decode_times:
        backlog = max(0.0, backlog + float(dt) - float(round_duration))
        out.append(backlog)
    return out


def commit_latency_sequence(decode_times, round_duration: float, commit_lag: int = 0) -> list[float]:
    """由声明代价与轮时长算出每轮的提交延迟（FIFO 单服务台 sojourn time）。

    `backlog_sequence` 实现的是 Lindley 递推：`backlog[t]` 是第 t 轮**处理完之后**
    遗留的积压，也就是第 t+1 轮到达时看到的排队延迟——不是第 t 轮自己的排队延迟。
    第 t 轮自己的排队延迟是 `backlog[t-1]`（`t=0` 时前面没有历史，取 0）。
    错用 `backlog[t]` 会把每个拥堵轮次的延迟低估恰好一个 `round_duration`。

    提交延迟 = 排队延迟（`backlog[t-1]`）+ 该轮解码时长 + 滞后提交带来的
    `commit_lag × round_duration`。
    """
    backlog = backlog_sequence(decode_times, round_duration)
    out, prev_backlog = [], 0.0
    for t, dt in enumerate(decode_times):
        queueing_delay = prev_backlog          # backlog[t-1]；t=0 时记 0
        out.append(queueing_delay + float(dt) + int(commit_lag) * float(round_duration))
        prev_backlog = backlog[t]
    return out


def _fill_shot_timing(record, decoder, steps, timing: TimingModel, rounds: int) -> None:
    """按声明代价填充该 shot 的 backlog 与提交延迟。"""
    decode_times = [float(timing.cost_to_seconds(step.cost)) for step in steps]
    lag = int(getattr(decoder, "commit_lag", 0))
    backlog = backlog_sequence(decode_times, timing.round_duration)
    latency = commit_latency_sequence(decode_times, timing.round_duration, lag)
    record.backlog = np.asarray(backlog, dtype=float)
    record.commit_latency = np.asarray(latency, dtype=float)
    record.decode_times = np.asarray(decode_times, dtype=float)


def _fill_timing_aggregates(result, records, timing: TimingModel) -> None:
    """把逐 shot timing 汇总到 QECResult。

    `records` 为空（例如 `keep_records=0`）时没有任何数据可汇总——保持
    `None` 而非编造 `0.0`/`0`，理由与 `TimingModel=None → None` 一致：
    「没测」和「测出来是零」是两件事，不能混同。
    """
    if not records:
        result.max_backlog = None
        result.mean_commit_latency = None
        result.budget_violations = None
        return
    result.max_backlog = float(max(r.backlog.max() for r in records))
    result.mean_commit_latency = float(
        np.mean(np.concatenate([r.commit_latency for r in records]))
    )
    result.budget_violations = int(sum(
        int((r.decode_times > timing.round_duration).sum()) for r in records
    ))
