"""Circuit-level typed IR that can round-trip existing Circuit objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .measurement import Measurement
from .operation import Operation, _as_int_tuple


CircuitInstruction = Operation | Measurement


def _is_measurement_dict(gate: Mapping[str, Any]) -> bool:
    return str(gate.get("type", "")).lower() in {"measure", "measurement"}


def _instruction_from_object(value: CircuitInstruction | Mapping[str, Any]) -> CircuitInstruction:
    if isinstance(value, (Operation, Measurement)):
        return value
    if isinstance(value, Mapping):
        if _is_measurement_dict(value):
            return Measurement.from_dict(value)
        return Operation.from_dict(value)
    raise TypeError("CircuitIR operations must be Operation, Measurement, or gate mappings")


@dataclass(frozen=True)
class CircuitIR:
    """Typed circuit representation with an ordered operation sequence."""

    operations: Sequence[CircuitInstruction] = ()
    n_qubits: int = 0
    classical_bits: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_qubits = int(self.n_qubits)
        if n_qubits < 0:
            raise ValueError("n_qubits must be non-negative")
        operations = tuple(_instruction_from_object(operation) for operation in self.operations)
        classical_bits = _as_int_tuple(self.classical_bits, label="classical_bits")

        object.__setattr__(self, "n_qubits", n_qubits)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "classical_bits", classical_bits)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_circuit(
        cls,
        circuit,
        *,
        classical_bits: Sequence[int] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "CircuitIR":
        """Build CircuitIR from the current dict-based Circuit surface."""

        if not hasattr(circuit, "n_qubits"):
            raise TypeError("circuit must provide n_qubits")
        gates = getattr(circuit, "gates", None)
        if gates is None:
            raise TypeError("circuit must provide gates")
        return cls(
            tuple(_instruction_from_object(gate) for gate in gates),
            n_qubits=int(circuit.n_qubits),
            classical_bits=tuple(classical_bits),
            metadata=metadata or {},
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CircuitIR":
        """Build CircuitIR from a plain serializable mapping."""

        if not isinstance(payload, Mapping):
            raise TypeError("CircuitIR.from_dict expects a mapping")
        return cls(
            payload.get("operations", ()),
            n_qubits=int(payload.get("n_qubits", 0)),
            classical_bits=tuple(payload.get("classical_bits", ())),
            metadata=payload.get("metadata", {}),
        )

    def to_gate_dicts(self) -> list[dict[str, Any]]:
        """Return operation dictionaries compatible with current Circuit."""

        return [operation.to_dict() for operation in self.operations]

    def to_circuit(self, *, backend=None):
        """Convert this IR back to the existing Circuit class."""

        from ..core.circuit import Circuit

        return Circuit(*self.to_gate_dicts(), n_qubits=self.n_qubits, backend=backend)

    def to_dict(self) -> dict[str, Any]:
        """Convert this IR to a plain serializable mapping."""

        return {
            "n_qubits": self.n_qubits,
            "operations": self.to_gate_dicts(),
            "classical_bits": list(self.classical_bits),
            "metadata": dict(self.metadata),
        }
