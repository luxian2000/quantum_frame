"""编译执行的统一结果对象 ``TranspileResult``（NEXT.md 第 9 节）。

与 primitives 的 ``SampleResult``/``EstimateResult``/``GradientResult`` 同属第 9 节
统一结果模型，但置于 transpile 域内，避免 primitives ↔ transpile 耦合。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TranspileResult:
    """一次编译（PassManager 运行）的统一结果。

    - ``circuit``：编译后的线路。
    - ``layout``：``logical -> physical`` 比特映射（来自 ``LayoutPass``）；
      无布局 pass 时为 ``None``（平凡/恒等）。
    - ``passes``：实际运行的 pass 名序列。
    - ``depth_before`` / ``depth_after``：编译前后的线路深度（ASAP 层数）。
    - ``metadata``：附加信息。
    """

    circuit: Any
    layout: dict[int, int] | None
    passes: tuple[str, ...]
    depth_before: int
    depth_after: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["TranspileResult"]
