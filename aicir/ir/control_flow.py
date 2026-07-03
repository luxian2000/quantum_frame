"""控制流指令 ControlFlow：携带嵌套 body 的 if/while。

body 以 gate-dict 元组 + n_qubits 存储（不直接持有 Circuit，避免 ir<->core 循环
导入）；.body/.else_body 属性在访问时懒重建 Circuit。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.classical import Condition
from .operation import LegacyGateView

_CF_NAMES = {"if", "while"}


class ControlFlow(LegacyGateView):
    """if / while 控制流指令。"""

    def __init__(self, name, condition, body_gates, n_qubits,
                 else_gates=None, max_iterations=None):
        name = str(name).lower()
        if name not in _CF_NAMES:
            raise ValueError(f"控制流 name 必须是 if/while，收到 {name!r}")
        if not isinstance(condition, Condition):
            raise TypeError("condition 必须是 Condition")
        if name == "while" and max_iterations is None:
            raise ValueError("while 必须提供 max_iterations")
        self.name = name
        self.condition = condition
        self.body_gates = tuple(dict(g) for g in body_gates)
        self.else_gates = None if else_gates is None else tuple(dict(g) for g in else_gates)
        self.n_qubits = int(n_qubits)
        self.max_iterations = None if max_iterations is None else int(max_iterations)

    @property
    def body(self):
        from ..core.circuit import Circuit
        return Circuit(*self.body_gates, n_qubits=self.n_qubits)

    @property
    def else_body(self):
        if self.else_gates is None:
            return None
        from ..core.circuit import Circuit
        return Circuit(*self.else_gates, n_qubits=self.n_qubits)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.name,
            "condition": self.condition.to_dict(),
            "body": [dict(g) for g in self.body_gates],
            "n_qubits": self.n_qubits,
        }
        if self.else_gates is not None:
            d["else_body"] = [dict(g) for g in self.else_gates]
        if self.name == "while":
            d["max_iterations"] = self.max_iterations
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ControlFlow":
        return cls(
            name=d["type"],
            condition=Condition.from_dict(d["condition"]),
            body_gates=d["body"],
            n_qubits=int(d["n_qubits"]),
            else_gates=d.get("else_body"),
            max_iterations=d.get("max_iterations"),
        )
