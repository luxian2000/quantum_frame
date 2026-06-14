"""有限 shots 采样 primitive：包装现有 :class:`~aicir.measure.measure.Measure`。"""

from __future__ import annotations

import math

from ..backends import NumpyBackend
from ..measure import Measure
from .base import BaseSampler, normalize_run_inputs
from .results import SampleResult


def _eigenvalue_to_bit(lam: int) -> str:
    """本征值转比特：+1 → '0'，-1 → '1'（与 aggregate.py 约定一致）。"""
    return "0" if lam == 1 else "1"


def _sample_result_from_measure(result) -> SampleResult:
    """把 Measure 的 Result 转换为统一 SampleResult。

    若电路内嵌了 measure 门（measurement_specs 非空），则从线路内计数
    中读取各 measure 操作的比特并拼接为 |..> 格式的比特串；
    否则从末端测量（counts(-1)）读取。
    """
    specs = result.measurement_specs  # List[MeasureSpec]

    if specs:
        # 线路内嵌 measure 门路径：收集每个 measure 操作的比特列表
        # 各 spec 按 op_index 升序排列（collect_specs 按遍历顺序登记）
        all_qubits: list[int] = []
        for spec in specs:
            all_qubits.extend(spec.qubits)
        measured = tuple(all_qubits)

        # 从 incircuit_counts 汇总各 op 的本征值计数，拼成联合比特串
        # 当前仅支持各 measure op 独立单比特情形（多 qubit 联合测量暂不处理）
        if len(specs) == 1:
            spec = specs[0]
            raw = result.incircuit_counts.get(spec.op_index, {})
            # raw 键为本征值整数 +1/-1，转换为 "|0>" / "|1>" 格式
            width = len(spec.qubits)
            # 单比特 measure：本征值直接映射
            counts: dict[str, int] = {}
            for lam, cnt in raw.items():
                bit = _eigenvalue_to_bit(int(lam))
                key = f"|{bit}>"
                counts[key] = counts.get(key, 0) + cnt
        else:
            # 多个 measure 操作：拼接各操作的本征值为联合比特串
            # 取各操作中 shots 次独立采样的笛卡尔积（当前实现：按轨迹顺序重建）
            # 此处退化为对各操作独立统计（联合分布未存储）
            counts = {}
            for spec in specs:
                raw = result.incircuit_counts.get(spec.op_index, {})
                for lam, cnt in raw.items():
                    bit = _eigenvalue_to_bit(int(lam))
                    key = f"|{bit}>"
                    counts[key] = counts.get(key, 0) + cnt
    else:
        # 末端测量路径：counts(-1) 返回裸比特串如 "00"，包装为 |..> 格式
        measured = tuple(result.terminal_qubits) if result.terminal_qubits is not None else tuple(range(int(result.n_qubits)))
        if result.shots is not None:
            raw_counts = result.counts(-1)
            counts = {f"|{k}>": v for k, v in raw_counts.items()}
        else:
            counts = {}

    # 概率分布：保持原有 |index> 格式
    width = int(round(math.log2(len(result.probabilities)))) if len(result.probabilities) else 0
    probs = {
        f"|{index:0{width}b}>": float(p)
        for index, p in enumerate(result.probabilities)
        if float(p) > 0.0
    }
    return SampleResult(
        counts=counts,
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
