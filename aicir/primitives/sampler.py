"""有限 shots 采样 primitive：包装现有 :class:`~aicir.measure.measure.Measure`。"""

from __future__ import annotations

import math

from ..channel.backends import NumpyBackend
from ..measure import Measure
from .base import BaseSampler, normalize_run_inputs
from .results import SampleResult


def _sample_result_from_measure(result) -> SampleResult:
    """把 Measure 的 Result 转换为统一 SampleResult。"""

    measured = result.metadata.get("measured_qubits")
    if measured is None:
        measured = tuple(range(int(result.n_qubits)))
    else:
        measured = tuple(int(q) for q in measured)

    width = int(round(math.log2(len(result.probabilities)))) if len(result.probabilities) else 0
    probs = {
        f"|{index:0{width}b}>": float(p)
        for index, p in enumerate(result.probabilities)
        if float(p) > 0.0
    }
    return SampleResult(
        counts=dict(result.counts or {}),
        probs=probs,
        shots=result.shots,
        measured_qubits=measured,
        metadata=dict(result.metadata),
    )


class ShotSampler(BaseSampler):
    """有限 shots 采样；支持显式 ``measure_qubits`` 与电路内嵌 measure 门。"""

    def __init__(self, backend=None, *, shots: int = 1024) -> None:
        self.backend = backend if backend is not None else NumpyBackend()
        self.shots = int(shots)
        if self.shots <= 0:
            raise ValueError("shots must be positive")

    def run(self, circuits, *, shots: int | None = None, measure_qubits=None):
        items, single = normalize_run_inputs(circuits)
        use_shots = self.shots if shots is None else int(shots)
        measure = Measure(self.backend)
        results = [
            _sample_result_from_measure(
                measure.run(
                    circuit,
                    shots=use_shots,
                    return_state=False,
                    measure_qubits=measure_qubits,
                )
            )
            for circuit in items
        ]
        return results[0] if single else results
