"""Typed measurement/reset IR with compatibility helpers for circuit markers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .operation import LegacyGateView, _as_int_tuple


_KNOWN_MEASUREMENT_KEYS = {
    "type",
    "target_qubit",
    "qubits",
    "return_type",
    "classical_bit",
    "classical_bits",
    "clbits",
}


@dataclass(frozen=True)
class Measurement(LegacyGateView):
    """Typed representation of an in-circuit measurement-like declaration."""

    qubits: tuple[int, ...] = ()
    measurement_type: str = "measure"
    return_type: str = "counts"
    classical_bits: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        measurement_type = str(self.measurement_type).strip()
        if not measurement_type:
            raise ValueError("measurement_type cannot be empty")
        return_type = str(self.return_type).strip()
        if not return_type:
            raise ValueError("return_type cannot be empty")

        qubits = _as_int_tuple(self.qubits, label="qubits")
        classical_bits = _as_int_tuple(self.classical_bits, label="classical_bits")
        if classical_bits and qubits and len(classical_bits) != len(qubits):
            raise ValueError("classical_bits length must match qubits length")

        object.__setattr__(self, "measurement_type", measurement_type)
        object.__setattr__(self, "return_type", return_type)
        object.__setattr__(self, "qubits", qubits)
        object.__setattr__(self, "classical_bits", classical_bits)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_dict(cls, gate: Mapping[str, Any]) -> "Measurement":
        """Build a measurement from the current measure-gate dictionary."""

        if not isinstance(gate, Mapping):
            raise TypeError("Measurement.from_dict expects a mapping")
        measurement_type = str(gate.get("type", "measure"))
        if measurement_type.lower() not in {"measure", "measurement", "reset"}:
            raise ValueError("measurement gate type must be 'measure', 'measurement', or 'reset'")

        if "qubits" in gate:
            qubits = _as_int_tuple(gate["qubits"], label="qubits")
        elif "target_qubit" in gate:
            qubits = (int(gate["target_qubit"]),)
        else:
            qubits = ()

        if "classical_bits" in gate:
            classical_bits = _as_int_tuple(gate["classical_bits"], label="classical_bits")
        elif "clbits" in gate:
            classical_bits = _as_int_tuple(gate["clbits"], label="clbits")
        elif "classical_bit" in gate:
            classical_bits = (int(gate["classical_bit"]),)
        else:
            classical_bits = ()

        metadata = {
            key: value
            for key, value in gate.items()
            if key not in _KNOWN_MEASUREMENT_KEYS
        }
        return cls(
            qubits=qubits,
            measurement_type=measurement_type,
            return_type=str(gate.get("return_type", "counts")),
            classical_bits=classical_bits,
            metadata=metadata,
        )

    def __eq__(self, other: object) -> bool:
        # 与旧 measure 门字典可直接比较相等；类型化对象之间按字段比较。
        if isinstance(other, Measurement):
            return (
                self.qubits,
                self.measurement_type,
                self.return_type,
                self.classical_bits,
                self.metadata,
            ) == (
                other.qubits,
                other.measurement_type,
                other.return_type,
                other.classical_bits,
                other.metadata,
            )
        if isinstance(other, Mapping):
            return self.to_dict() == dict(other)
        return NotImplemented

    def to_dict(self) -> dict[str, Any]:
        """Convert this measurement to the existing measure-gate surface."""

        gate: dict[str, Any] = {
            "type": self.measurement_type,
            "qubits": list(self.qubits),
        }
        if self.return_type != "counts":
            gate["return_type"] = self.return_type
        if self.classical_bits:
            gate["classical_bits"] = list(self.classical_bits)
        for key, value in self.metadata.items():
            gate[key] = value
        return gate
