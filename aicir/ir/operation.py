"""Typed operation IR with compatibility helpers for gate dictionaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


_PAIR_QUBIT_GATES = {"swap", "rzz", "rxx"}
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


@dataclass(frozen=True)
class Operation:
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
        object.__setattr__(self, "metadata", dict(self.metadata))

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
        if controls and not control_states:
            control_states = tuple(1 for _ in controls)

        params = _parameter_tuple(gate["parameter"]) if "parameter" in gate else ()
        metadata = {key: value for key, value in gate.items() if key not in _KNOWN_GATE_KEYS}

        return cls(
            str(gate["type"]),
            qubits=qubits,
            params=params,
            controls=controls,
            control_states=control_states,
            metadata=metadata,
        )

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
            states = self.control_states or tuple(1 for _ in self.controls)
            gate["control_states"] = list(states)

        if self.params:
            gate["parameter"] = self.params[0] if len(self.params) == 1 else list(self.params)

        for key, value in self.metadata.items():
            gate[key] = value
        return gate


def normalize_gate(gate: Operation | Mapping[str, Any]) -> dict[str, Any]:
    """Return a gate dictionary from either an ``Operation`` or mapping."""

    if isinstance(gate, Operation):
        return gate.to_dict()
    if isinstance(gate, Mapping):
        return dict(gate)
    raise TypeError("gate must be an Operation or mapping")
