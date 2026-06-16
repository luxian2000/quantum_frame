"""primitives 统一结果对象（NEXT.md 第 9 节的最小切片）。

只承载 primitives 需要的字段；后续 `GradientResult`/`TranspileResult`
等按需补充。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SampleResult:
    """一次采样执行的统一结果。"""

    counts: dict[str, int]
    probs: dict[str, float]
    shots: int | None
    measured_qubits: tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EstimateResult:
    """一次期望值估计的统一结果。

    ``shots`` 为 ``None`` 表示精确（态向量）路径；``term_results``
    携带逐 Pauli 项明细（精确路径为 ``None``）。
    """

    value: float
    variance: float | None = None
    shots: int | None = None
    term_results: tuple[Any, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
