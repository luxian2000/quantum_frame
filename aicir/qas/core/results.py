"""QAS 统一结果类型。

各方法（``supernet``/``pporb``/``pprdql``/``crlqas``/``qdrats``/``dqas``/``vqe_loop``）
当前各自返回互不相同的 dataclass（例如 ``SupernetResult``、``PPRDQLResult``，甚至
``pporb`` 直接返回裸 ``(theta_dict, Circuit)`` 元组）。:class:`QASResult` 是可选的、
方法无关的包装：把任意底层结果映射为统一字段，同时通过 ``raw`` 保留原始对象不丢信息。

3a 阶段仅新增本类型，不改动任何方法的返回值；各策略接入 ``QASResult`` 留给 3b。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QASResult:
    """跨 QAS 方法的统一结果包装。

    字段:
        method: 产生该结果的方法规范名（如 ``"supernet"``/``"crlqas"``）。
        value: 目标标量（能量、损失或保真度等，视方法语义而定）；不适用时为 ``None``。
        circuit: 搜索/训练得到的最终 :class:`~aicir.core.circuit.Circuit`；无则为 ``None``。
        parameters: 最优参数，形态随方法而异（``dict[str, float]``/``ndarray``/``None``），
            调用方需按 ``method`` 或 ``raw`` 的具体类型解读。
        history: 逐步记录列表（默认空列表），各方法自行决定元素形态
            （``HistoryRecord``/dict 等）。
        metadata: 附加信息字典（默认空字典），例如后端 provenance、终止原因等。
        raw: 保留的方法特定原始结果对象（如 ``SupernetResult``/``PPRDQLResult``），
            不做任何转换；需要完整信息时从这里取。

    满足 :class:`aicir.protocols.AlgorithmResult` 协议（``value``/``parameters``/
    ``history``/``metadata`` 四个只读成员均已具备，``isinstance`` 检查可直接通过）。
    """

    method: str
    value: float | None
    circuit: Any = None
    parameters: Any = None
    history: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    raw: Any = None


__all__ = ["QASResult"]
