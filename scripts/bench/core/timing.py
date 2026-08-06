"""计时统计。

三条方法学约束，都是为了让跨框架数字站得住：

1. **中位数而非均值**——一次 GC 或调度抖动就能把均值拉到毫无意义；
2. **上报 IQR**——不给离散度的计时表无法判断差异是否显著；
3. **预热不计入**——TensorCircuit/JAX 之类有 JIT，首次调用包含编译时间，
   把它算进去测的就不是执行速度。

另外把**构建**与**执行**分开计时：Qiskit 的 transpile 开销若混进执行时间，
比较就变成了在比编译器。参考 *Benchmarking the performance of quantum computing
software for quantum circuit creation, manipulation and compilation*。
"""

from __future__ import annotations

import gc
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Callable

__all__ = ["TimingStats", "time_callable", "BuildRunTiming"]


@dataclass(frozen=True)
class TimingStats:
    """一组重复计时的统计摘要（单位：秒）。"""

    median: float
    minimum: float
    maximum: float
    iqr: float
    repeats: int
    warmup: int

    def to_dict(self) -> dict:
        return asdict(self)


def _interquartile_range(samples: list[float]) -> float:
    if len(samples) < 4:
        return max(samples) - min(samples)
    ordered = sorted(samples)
    mid = len(ordered) // 2
    lower = ordered[:mid]
    upper = ordered[-mid:]
    return statistics.median(upper) - statistics.median(lower)


def time_callable(
    fn: Callable[[], object],
    *,
    repeats: int = 7,
    warmup: int = 1,
    disable_gc: bool = True,
    _clock: Callable[[], float] = time.perf_counter,
) -> TimingStats:
    """重复调用 ``fn`` 并返回计时统计。

    参数:
        repeats:    计入统计的重复次数（建议 ≥7 才能谈中位数与 IQR）。
        warmup:     预热次数，不计入统计（吸收 JIT/缓存冷启动）。
        disable_gc: 计时期间关闭 GC，避免回收停顿混入样本。
        _clock:     时钟注入点，仅供测试使用。
    """

    if repeats < 1:
        raise ValueError(f"repeats 必须 ≥1，收到 {repeats}")

    for _ in range(warmup):
        fn()

    samples: list[float] = []
    gc_was_enabled = gc.isenabled()
    if disable_gc and gc_was_enabled:
        gc.disable()
    try:
        for _ in range(repeats):
            start = _clock()
            fn()
            samples.append(_clock() - start)
    finally:
        if disable_gc and gc_was_enabled:
            gc.enable()

    return TimingStats(
        median=statistics.median(samples),
        minimum=min(samples),
        maximum=max(samples),
        iqr=_interquartile_range(samples),
        repeats=repeats,
        warmup=warmup,
    )


@dataclass(frozen=True)
class BuildRunTiming:
    """构建与执行分离的计时结果。

    分开报告是必须的：把线路构建/转译算进执行时间，比较的就不是模拟器速度。
    """

    build: TimingStats
    run: TimingStats

    def to_dict(self) -> dict:
        return {"build": self.build.to_dict(), "run": self.run.to_dict()}
