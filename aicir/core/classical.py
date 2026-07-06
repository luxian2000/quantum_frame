"""经典寄存器与条件模型：支撑测量反馈的 if/while 控制流。

Condition 内部只存 (register_name, index, op, value)，不引用活寄存器对象，
因此求值与 JSON 序列化都与具体 ClassicalRegister 实例解耦。
"""

from __future__ import annotations

from collections.abc import Mapping

_VALID_OPS = ("==", "!=")


class Condition:
    """经典条件：位或整个寄存器整数值与常量的 == / != 比较。"""

    __slots__ = ("register_name", "index", "op", "value")

    def __init__(self, register_name: str, index: int | None, op: str, value: int):
        if op not in _VALID_OPS:
            raise ValueError(f"op 必须是 {_VALID_OPS} 之一，收到 {op!r}")
        self.register_name = str(register_name)
        self.index = None if index is None else int(index)
        self.op = op
        self.value = int(value)

    def evaluate(self, store: Mapping[str, list]) -> bool:
        bits = store.get(self.register_name)
        if bits is None:
            actual = 0  # 从未写入的寄存器默认全 0
        elif self.index is None:
            actual = sum(int(b) << i for i, b in enumerate(bits))  # LSB=bit0
        else:
            actual = int(bits[self.index])
        return actual == self.value if self.op == "==" else actual != self.value

    def to_dict(self) -> dict:
        return {
            "target": {"register": self.register_name, "index": self.index},
            "op": self.op,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "Condition":
        t = d["target"]
        return cls(t["register"], t.get("index"), d["op"], int(d["value"]))

    def __repr__(self) -> str:
        tgt = self.register_name if self.index is None else f"{self.register_name}[{self.index}]"
        return f"Condition({tgt} {self.op} {self.value})"


class Bit:
    """经典寄存器中的单个位引用（register_name, index）。"""

    __slots__ = ("register_name", "index")

    def __init__(self, register_name: str, index: int):
        self.register_name = str(register_name)
        self.index = int(index)

    def _cond(self, op: str, value: int) -> Condition:
        if value not in (0, 1):
            raise ValueError(f"位比较值必须是 0 或 1，收到 {value!r}")
        return Condition(self.register_name, self.index, op, value)

    def __eq__(self, value):  # type: ignore[override]
        return self._cond("==", value)

    def __ne__(self, value):  # type: ignore[override]
        return self._cond("!=", value)

    def __hash__(self):
        return hash((self.register_name, self.index))


class ClassicalRegister:
    """经典寄存器：size 个位，creg[0] 为 LSB（整数值 = Σ bit_i << i）。"""

    __slots__ = ("name", "size")

    def __init__(self, size: int, name: str):
        if int(size) <= 0:
            raise ValueError(f"size 必须为正，收到 {size}")
        if not str(name).strip():
            raise ValueError("name 不能为空")
        self.size = int(size)
        self.name = str(name)

    def __getitem__(self, index: int) -> Bit:
        idx = int(index)
        if idx < 0 or idx >= self.size:
            raise IndexError(f"位下标 {idx} 越界（size={self.size}）")
        return Bit(self.name, idx)

    def __len__(self) -> int:
        return self.size

    def __eq__(self, value):  # type: ignore[override]
        return Condition(self.name, None, "==", int(value))

    def __ne__(self, value):  # type: ignore[override]
        return Condition(self.name, None, "!=", int(value))

    def __hash__(self):
        return hash((self.name, self.size))
