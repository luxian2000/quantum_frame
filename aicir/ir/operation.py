"""Typed operation IR with compatibility helpers for gate dictionaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..gates import get_gate_spec


_PAIR_QUBIT_GATES = {"swap", "rzz", "rxx", "single_excitation"}
_KNOWN_GATE_KEYS = {
    "type",
    "target_qubit",
    "qubit_1",
    "qubit_2",
    "qubits",
    "targets",
    "control_qubits",
    "control_states",
    "parameter",
    "label",
}


def _as_int_tuple(values: Any, *, label: str) -> tuple[int, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must contain integer qubit indices")
    try:
        return tuple(int(value) for value in values)
    except TypeError:
        return (int(values),)


def _parameter_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


class LegacyGateView:
    """旧门字典的只读访问兼容层。

    工厂函数返回类型化 IR 之后，仍有少量代码以旧字典键
    （``gate["type"]``、``gate.get("parameter")`` 等）读取门信息。
    本混入类把这些读取委托给 :meth:`to_dict`，使类型化对象在
    *只读* 场景下可以当旧字典使用；写入仍被禁止（对象不可变）。
    """

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - 由子类实现
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self.to_dict()

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError(
            f"{type(self).__name__} is immutable; use dataclasses.replace or to_dict()"
        )

    def __delitem__(self, key: str) -> None:
        raise TypeError(
            f"{type(self).__name__} is immutable; use dataclasses.replace or to_dict()"
        )


@dataclass(frozen=True)
class Operation(LegacyGateView):
    """Typed representation of one circuit operation.

    ``Operation`` is intentionally small: it captures the common gate fields
    needed by the current dict-based ``Circuit`` surface and can round-trip back
    to that surface through :meth:`to_dict`.
    """

    name: str
    qubits: tuple[int, ...] = ()
    params: tuple[Any, ...] = ()
    controls: tuple[int, ...] = ()
    control_states: tuple[int, ...] = ()
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("Operation name cannot be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "qubits", _as_int_tuple(self.qubits, label="qubits"))
        object.__setattr__(self, "controls", _as_int_tuple(self.controls, label="controls"))
        states = _as_int_tuple(self.control_states, label="control_states")
        if states and len(states) != len(self.controls):
            raise ValueError("control_states length must match controls length")
        object.__setattr__(self, "control_states", states)
        object.__setattr__(self, "params", tuple(self.params))
        object.__setattr__(self, "label", None if self.label is None else str(self.label))
        object.__setattr__(self, "metadata", dict(self.metadata))
        self._validate_against_spec()

    def _validate_against_spec(self) -> None:
        """对已注册门按 GateSpec 校验目标比特数/参数个数/控制位；未注册门保持宽松。"""

        spec = get_gate_spec(self.name)
        if spec is None:
            return
        if spec.num_qubits is not None and len(self.qubits) != spec.num_qubits:
            raise ValueError(
                f"gate '{self.name}' expects {spec.num_qubits} target qubit(s), "
                f"got {len(self.qubits)}"
            )
        if spec.num_params is not None and len(self.params) != spec.num_params:
            raise ValueError(
                f"gate '{self.name}' expects {spec.num_params} parameter(s), "
                f"got {len(self.params)}"
            )
        if spec.controlled and not self.controls:
            raise ValueError(f"gate '{self.name}' requires at least one control qubit")

    @classmethod
    def from_dict(cls, gate: Mapping[str, Any]) -> "Operation":
        """Build an operation from the current gate-dictionary format."""

        if not isinstance(gate, Mapping):
            raise TypeError("Operation.from_dict expects a mapping")
        if "type" not in gate:
            raise ValueError("gate dict must contain a 'type' field")

        qubits: tuple[int, ...]
        if "target_qubit" in gate:
            qubits = (int(gate["target_qubit"]),)
        elif "qubit_1" in gate or "qubit_2" in gate:
            if "qubit_1" not in gate or "qubit_2" not in gate:
                raise ValueError("pair-qubit gates must provide qubit_1 and qubit_2")
            qubits = (int(gate["qubit_1"]), int(gate["qubit_2"]))
        elif "qubits" in gate:
            qubits = _as_int_tuple(gate["qubits"], label="qubits")
        elif "targets" in gate:
            qubits = _as_int_tuple(gate["targets"], label="targets")
        else:
            qubits = ()

        controls = _as_int_tuple(gate.get("control_qubits"), label="control_qubits")
        control_states = _as_int_tuple(gate.get("control_states"), label="control_states")
        params = _parameter_tuple(gate["parameter"]) if gate.get("parameter") is not None else ()
        label = gate.get("label")
        metadata = {key: value for key, value in gate.items() if key not in _KNOWN_GATE_KEYS}

        return cls(
            str(gate["type"]),
            qubits=qubits,
            params=params,
            controls=controls,
            control_states=control_states,
            label=None if label is None else str(label),
            metadata=metadata,
        )

    def __eq__(self, other: object) -> bool:
        # 与旧门字典可直接比较相等（双向，经由反射比较），类型化对象之间按字段比较。
        if isinstance(other, Operation):
            return (
                self.name,
                self.qubits,
                self.params,
                self.controls,
                self.control_states,
                self.label,
                self.metadata,
            ) == (
                other.name,
                other.qubits,
                other.params,
                other.controls,
                other.control_states,
                other.label,
                other.metadata,
            )
        if isinstance(other, Mapping):
            return self.to_dict() == dict(other)
        return NotImplemented

    def to_dict(self) -> dict[str, Any]:
        """Convert this operation to the existing gate-dictionary surface."""

        gate: dict[str, Any] = {"type": self.name}
        low_name = self.name.lower()

        if low_name in {"measure", "measurement"}:
            gate["qubits"] = list(self.qubits)
        elif low_name in _PAIR_QUBIT_GATES and len(self.qubits) == 2 and not self.controls:
            gate["qubit_1"] = self.qubits[0]
            gate["qubit_2"] = self.qubits[1]
        elif len(self.qubits) == 1:
            gate["target_qubit"] = self.qubits[0]
        elif len(self.qubits) > 1:
            gate["qubits"] = list(self.qubits)

        if self.controls:
            gate["control_qubits"] = list(self.controls)
            if self.control_states:
                gate["control_states"] = list(self.control_states)

        if self.params:
            gate["parameter"] = self.params[0] if len(self.params) == 1 else list(self.params)

        if self.label is not None:
            gate["label"] = self.label

        for key, value in self.metadata.items():
            gate[key] = value
        return gate


def normalize_gate(gate: Operation | Mapping[str, Any]) -> dict[str, Any]:
    """Return a gate dictionary from a typed IR object or mapping."""

    if isinstance(gate, Operation):
        return gate.to_dict()
    from .control_flow import ControlFlow
    from .measurement import Measurement

    if isinstance(gate, (Measurement, ControlFlow)):
        return gate.to_dict()
    if isinstance(gate, Mapping):
        return dict(gate)
    raise TypeError("gate must be an Operation, Measurement, ControlFlow, or mapping")
