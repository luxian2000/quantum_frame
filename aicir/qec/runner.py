"""交错 simulate↔decode 运行器。

本文件在 Task 4 只提供无噪声综合征采集（verify_schedule 依赖），
完整的逐 shot 在线解码循环在 Task 7 补齐。
"""

from __future__ import annotations

import numpy as np

from aicir.backends import NumpyBackend
from aicir.core.state import State
from aicir.measure.trajectory import run_trajectory


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
