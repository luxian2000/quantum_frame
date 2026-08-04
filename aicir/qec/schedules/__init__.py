"""syndrome 提取调度：把 StabilizerCode 编译成逐轮线路 + DetectorLayout。

按**轮**构建而非构建单一整体线路——运行器必须执行轮 t、暂停解码、再继续。
run_trajectory 接受 init_state 并返回 .pre，运行器据此在轮间串联量子态。

全局比特编号：data 0..n−1，ancilla n..n+m−1（ancilla j 测量生成元 j）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from ..detectors import Detector, DetectorLayout, Observable


@dataclass
class RoundCircuit:
    """一轮 syndrome 提取的产物。"""
    circuit: object
    creg_name: str
    ancilla_qubits: tuple[int, ...]
    data_qubits: tuple[int, ...]
    record_offset: int          # 该轮首个 measurement 在全局 record 序列中的下标


@dataclass
class ReadoutCircuit:
    """末端逻辑读出的产物。"""
    circuit: object
    creg_name: str
    observable_records: tuple[tuple[int, ...], ...]   # 每个逻辑比特对应的 record 下标组


class Schedule(Protocol):
    def build_encode(self, code, logical_state: str): ...
    def build_round(self, code, round_index: int, *, creg_name: str = "syn") -> RoundCircuit: ...
    def build_readout(self, code, logical_state: str) -> ReadoutCircuit: ...


SCHEDULES: dict[str, Callable[[], Schedule]] = {}


def register_schedule(name: str, factory: Callable[[], Schedule]) -> None:
    SCHEDULES[str(name)] = factory


def resolve_schedule(name_or_obj) -> Schedule:
    """名字 → 调度实例；已经是实例则原样返回。"""
    if not isinstance(name_or_obj, str):
        return name_or_obj
    if name_or_obj not in SCHEDULES:
        raise KeyError(f"未知调度 {name_or_obj!r}；可用：{sorted(SCHEDULES)}")
    return SCHEDULES[name_or_obj]()


def deterministic_round0(code, logical_state: str = "0") -> tuple[int, ...]:
    """哪些生成元在轮 0 读数确定 —— 即在制备基下 |0…0⟩/|+…+⟩ 是其本征态者。

    |0…0⟩ 制备：x 块全零的生成元（纯 Z 型），读数确定为 0。
    |+…+⟩ 制备：z 块全零的生成元（纯 X 型），读数确定为 0。
    其余生成元轮 0 读数是 50/50 随机的，**不构成 detector**。

    实测各码的确定生成元个数：repetition 2/2、Steane 3/6、Shor 6/8、
    surface_d3 4/8、five_qubit **0/4**（非 CSS，无纯 Z 型生成元）。
    """
    state = str(logical_state)
    if state in ("0", "1"):
        block = code.generators[:, :code.n]          # x 块须全零
    elif state in ("+", "-"):
        block = code.generators[:, code.n:]          # z 块须全零
    else:
        raise ValueError(f"未知逻辑初态 {state!r}")
    return tuple(int(j) for j in range(code.m) if not block[j].any())


def build_layout(code, schedule, rounds: int, *, logical_state: str = "0") -> DetectorLayout:
    """由码与调度构造 DetectorLayout。

    detector (s, t)：轮 t 的稳定子 s 读数 XOR 轮 t−1 的读数。
    t=0 **只对 deterministic_round0 内的生成元建 detector**（其余轮 0 读数随机）。

    Observable(i) 的 records 取该逻辑比特**实际逻辑算符的支持**，而非笼统的
    「全部 n 个 data 比特」：Z 基读出（'0'/'1'）用 logical_z[i] 的 z 块，
    X 基读出（'+'/'-'）用 logical_x[i] 的 x 块（GF(2) 向量布局 x 块在前、
    z 块在后，各宽 n）。k=1 的内置码里「全 n 比特奇偶」恰好是稳定子等价的
    合法代表元，故数值不变；但 k>1 时（如 Task 10 的 [[4,2,2]] 码）不同逻辑
    比特的算符支持不同，笼统写法会让所有逻辑比特共用同一组 record、彼此不可
    区分，必须按各自算符的真实支持取值。
    """
    schedule = resolve_schedule(schedule)
    m = code.m
    round0 = deterministic_round0(code, logical_state)
    detectors, idx = [], 0
    for t in range(int(rounds)):
        for s in range(m):
            if t == 0 and s not in round0:
                continue
            cur = t * m + s
            recs = (cur,) if t == 0 else ((t - 1) * m + s, cur)
            detectors.append(Detector(index=idx, records=recs, stabilizer=s, round_index=t))
            idx += 1
    base = int(rounds) * m
    state = str(logical_state)
    if state in ("0", "1"):
        support = code.logical_z[:, code.n:]         # z 块给出逻辑 Z 算符支持
    else:                                              # "+"/"-"；非法值已由上面的
        support = code.logical_x[:, :code.n]           # deterministic_round0 抛出
    observables = tuple(
        Observable(index=i, records=tuple(base + q for q in range(code.n) if support[i, q]))
        for i in range(code.k)
    )
    return DetectorLayout(
        n_detectors=idx, n_rounds=int(rounds), n_stabilizers=m,
        detectors=tuple(detectors), observables=observables, coords=dict(code.coords),
        round0_stabilizers=round0,
    )


def verify_schedule(code, schedule, rounds: int, *, logical_state: str = "0",
                    backend=None, shots: int = 4) -> None:
    """无噪声运行，断言每个 detector 恒为 0。不满足则抛 ValueError 并指名违规项。

    这是提取调度**唯一最有力的结构性检验**：它抓 CNOT 顺序错、漏掉 ancilla reset、
    轮 0 确定集合推错。公开它，使用户验证自己写的调度时享有与内置调度同等的保障。

    参考值恒为 zeros(m)：轮 0 确定的生成元（|0…0⟩ 制备下即纯 Z 型）读数确定为 0。
    **不从某次运行中实测 reference**——非确定生成元的轮 0 读数逐 shot 随机，
    用某一次的实测值当参考会让检验对该 bug 视而不见（且轮 0 变成什么都不断言）。
    """
    from ..runner import collect_noiseless_syndromes   # 延迟导入，避免循环

    schedule = resolve_schedule(schedule)
    layout = build_layout(code, schedule, int(rounds), logical_state=logical_state)
    reference = np.zeros(code.m, dtype=np.uint8)
    for shot in range(int(shots)):
        raw = collect_noiseless_syndromes(
            code, schedule, int(rounds), logical_state=logical_state,
            backend=backend, seed=shot,
        )
        for t in range(int(rounds)):
            events = layout.detection_events(raw, t, reference)
            bad = np.nonzero(events)[0]
            if bad.size:
                raise ValueError(
                    f"[{code.name}] 调度不满足 detector 确定性："
                    f"shot {shot} 的轮 {t} 稳定子 {int(bad[0])} 触发了 detector"
                    f"（无噪声下应恒为 0）"
                )


from .bare import BareAncillaSchedule  # noqa: E402  自注册

__all__ = [
    "RoundCircuit", "ReadoutCircuit", "Schedule", "BareAncillaSchedule",
    "SCHEDULES", "register_schedule", "resolve_schedule",
    "build_layout", "verify_schedule", "deterministic_round0",
]
