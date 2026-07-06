"""Hardware-efficient ansatz templates.

The public constructor returns an :class:`aicir.core.circuit.Circuit` built from
the repository's native gate dictionaries and symbolic ``Parameter`` objects.
Numeric parameter values can be passed directly, or bound later with
``Circuit.bind_parameters``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.circuit import (
    Circuit,
    Parameter,
    crx,
    cry,
    crz,
    cx,
    cy,
    cz,
    rx,
    ry,
    rz,
    rxx,
    rzz,
    swap,
    u2,
    u3,
)


Edge = tuple[int, int]

_ROTATION_PARAMETER_COUNTS = {
    "rx": 1,
    "ry": 1,
    "rz": 1,
    "u2": 2,
    "u3": 3,
}
_SUPPORTED_ENTANGLERS = {"cx", "cnot", "cy", "cz", "crx", "cry", "crz", "rzz", "rxx", "swap"}


class _ParameterStream:
    def __init__(self, values: Sequence[Any] | None, prefix: str) -> None:
        self._values = values
        self._prefix = prefix
        self._index = 0

    def next(self) -> Any:
        index = self._index
        self._index += 1
        if self._values is None:
            return Parameter(f"{self._prefix}_{index}")
        if index >= len(self._values):
            raise ValueError(f"Expected at least {self._index} parameter value(s), got {len(self._values)}")
        return self._values[index]

    def finish(self) -> None:
        if self._values is not None and self._index != len(self._values):
            raise ValueError(f"Expected {self._index} parameter value(s), got {len(self._values)}")


def _flatten_parameters(parameters: Sequence[Any] | None) -> list[Any] | None:
    if parameters is None:
        return None
    if isinstance(parameters, (str, bytes)):
        raise TypeError("parameters must be a non-string sequence")
    if hasattr(parameters, "reshape"):
        flat = parameters.reshape(-1)
        return [flat[index] for index in range(len(flat))]
    return list(parameters)


def _validate_n_qubits(n_qubits: int) -> int:
    value = int(n_qubits)
    if value <= 0:
        raise ValueError(f"n_qubits must be positive, got {n_qubits}")
    return value


def _validate_layers(layers: int) -> int:
    value = int(layers)
    if value < 0:
        raise ValueError(f"layers must be non-negative, got {layers}")
    return value


def _normalize_rotation_gates(gates: str | Sequence[str], *, label: str) -> tuple[str, ...]:
    raw_gates = (gates,) if isinstance(gates, str) else tuple(gates)
    if not raw_gates:
        raise ValueError(f"{label} must contain at least one rotation gate")

    normalized = tuple(str(gate).lower() for gate in raw_gates)
    unsupported = sorted(set(normalized) - set(_ROTATION_PARAMETER_COUNTS))
    if unsupported:
        allowed = ", ".join(sorted(_ROTATION_PARAMETER_COUNTS))
        raise ValueError(f"Unsupported {label}: {', '.join(unsupported)}. Supported gates: {allowed}")
    return normalized


def _linear_edges(n_qubits: int) -> list[Edge]:
    return [(control, control + 1) for control in range(n_qubits - 1)]


def _ring_edges(n_qubits: int) -> list[Edge]:
    edges = _linear_edges(n_qubits)
    if n_qubits > 2:
        edges.append((n_qubits - 1, 0))
    return edges


def _all_to_all_edges(n_qubits: int) -> list[Edge]:
    return [(control, target) for control in range(n_qubits) for target in range(control + 1, n_qubits)]


def _validate_edges(edges: Sequence[Edge], n_qubits: int) -> list[Edge]:
    normalized: list[Edge] = []
    for edge in edges:
        if len(edge) != 2:
            raise ValueError(f"Entangling edge must contain two qubit indices, got {edge!r}")
        control, target = int(edge[0]), int(edge[1])
        if control == target:
            raise ValueError(f"Entangling edge cannot connect a qubit to itself: {edge!r}")
        if control < 0 or control >= n_qubits or target < 0 or target >= n_qubits:
            raise ValueError(f"Entangling edge {edge!r} is out of range for n_qubits={n_qubits}")
        normalized.append((control, target))
    return normalized


def entangling_edges(n_qubits: int, topology: str | Sequence[Edge] = "linear") -> list[Edge]:
    """Return entangling edges for a supported topology.

    Supported string topologies are ``"linear"``, ``"ring"``, ``"all_to_all"``
    and ``"full"``. A custom sequence of ``(control, target)`` edges can be
    provided directly.
    """

    n_qubits = _validate_n_qubits(n_qubits)
    if isinstance(topology, str):
        key = topology.lower()
        if key == "linear":
            return _linear_edges(n_qubits)
        if key == "ring":
            return _ring_edges(n_qubits)
        if key in {"all_to_all", "full"}:
            return _all_to_all_edges(n_qubits)
        raise ValueError("topology must be 'linear', 'ring', 'all_to_all', 'full', or a custom edge sequence")
    return _validate_edges(topology, n_qubits)


def _append_rotation(gates: list[dict[str, Any]], gate: str, qubit: int, params: _ParameterStream) -> None:
    if gate == "rx":
        gates.append(rx(params.next(), qubit))
    elif gate == "ry":
        gates.append(ry(params.next(), qubit))
    elif gate == "rz":
        gates.append(rz(params.next(), qubit))
    elif gate == "u2":
        gates.append(u2(params.next(), params.next(), qubit))
    elif gate == "u3":
        gates.append(u3(params.next(), params.next(), params.next(), qubit))
    else:  # pragma: no cover - guarded by _normalize_rotation_gates
        raise ValueError(f"Unsupported rotation gate: {gate}")


def _append_rotation_layer(
    gates: list[dict[str, Any]],
    n_qubits: int,
    rotation_gates: Sequence[str],
    params: _ParameterStream,
) -> None:
    for qubit in range(n_qubits):
        for gate in rotation_gates:
            _append_rotation(gates, gate, qubit, params)


def _append_entangler(gates: list[dict[str, Any]], entangler: str, edge: Edge, params: _ParameterStream) -> None:
    control, target = edge
    if entangler in {"cx", "cnot"}:
        gates.append(cx(target, [control]))
    elif entangler == "cy":
        gates.append(cy(target, [control]))
    elif entangler == "cz":
        gates.append(cz(target, [control]))
    elif entangler == "crx":
        gates.append(crx(params.next(), target, [control]))
    elif entangler == "cry":
        gates.append(cry(params.next(), target, [control]))
    elif entangler == "crz":
        gates.append(crz(params.next(), target, [control]))
    elif entangler == "rzz":
        gates.append(rzz(params.next(), control, target))
    elif entangler == "rxx":
        gates.append(rxx(params.next(), control, target))
    elif entangler == "swap":
        gates.append(swap(control, target))
    else:  # pragma: no cover - guarded by hardware_efficient_ansatz
        raise ValueError(f"Unsupported entangler: {entangler}")


def _append_entangler_layer(
    gates: list[dict[str, Any]],
    edges: Sequence[Edge],
    entangler: str,
    params: _ParameterStream,
) -> None:
    for edge in edges:
        _append_entangler(gates, entangler, edge, params)


def hea_parameter_count(
    n_qubits: int,
    layers: int = 1,
    *,
    rotation_gates: str | Sequence[str] = ("ry", "rz"),
    entangler: str = "cx",
    topology: str | Sequence[Edge] = "linear",
    final_rotation_layer: bool = True,
    final_rotation_gates: str | Sequence[str] | None = None,
) -> int:
    """Return the number of trainable parameters used by the HEA template."""

    n_qubits = _validate_n_qubits(n_qubits)
    layers = _validate_layers(layers)
    rotations = _normalize_rotation_gates(rotation_gates, label="rotation_gates")
    final_rotations = rotations if final_rotation_gates is None else _normalize_rotation_gates(
        final_rotation_gates,
        label="final_rotation_gates",
    )

    entangler_key = str(entangler).lower()
    if entangler_key not in _SUPPORTED_ENTANGLERS:
        allowed = ", ".join(sorted(_SUPPORTED_ENTANGLERS))
        raise ValueError(f"Unsupported entangler: {entangler}. Supported entanglers: {allowed}")

    edges = entangling_edges(n_qubits, topology)
    rotation_params = n_qubits * sum(_ROTATION_PARAMETER_COUNTS[gate] for gate in rotations)
    final_params = n_qubits * sum(_ROTATION_PARAMETER_COUNTS[gate] for gate in final_rotations)
    entangler_params = len(edges) if entangler_key in {"crx", "cry", "crz", "rzz", "rxx"} else 0

    total = layers * (rotation_params + entangler_params)
    if final_rotation_layer:
        total += final_params
    return total


def hardware_efficient_ansatz(
    n_qubits: int,
    layers: int = 1,
    *,
    rotation_gates: str | Sequence[str] = ("ry", "rz"),
    entangler: str = "cx",
    topology: str | Sequence[Edge] = "linear",
    final_rotation_layer: bool = True,
    final_rotation_gates: str | Sequence[str] | None = None,
    parameter_prefix: str = "theta",
    parameters: Sequence[Any] | None = None,
    backend: Any = None,
) -> Circuit:
    """Build a standard hardware-efficient ansatz circuit.

    Each layer applies local trainable rotations on every qubit, followed by an
    entangling layer over the selected topology. By default, a final local
    rotation layer is appended after the last entangling layer.

    Args:
        n_qubits: Number of qubits in the circuit.
        layers: Number of rotation-entangler blocks.
        rotation_gates: Local rotation block, such as ``("ry", "rz")`` or
            ``"u3"``.
        entangler: Entangling gate. Supported values are ``cx``, ``cnot``,
            ``cy``, ``cz``, ``crx``, ``cry``, ``crz``, ``rzz``, ``rxx`` and ``swap``.
        topology: ``linear``, ``ring``, ``all_to_all``/``full``, or custom
            ``(control, target)`` edges.
        final_rotation_layer: Whether to append a trailing local rotation layer.
        final_rotation_gates: Local rotation block for the trailing layer.
            Defaults to ``rotation_gates``.
        parameter_prefix: Prefix for generated symbolic parameters.
        parameters: Optional flat sequence of parameter values/placeholders.
            When omitted, symbolic ``Parameter`` objects are generated.
        backend: Optional backend to bind to the returned ``Circuit``.

    Returns:
        A parameterized ``Circuit``.
    """

    n_qubits = _validate_n_qubits(n_qubits)
    layers = _validate_layers(layers)
    rotations = _normalize_rotation_gates(rotation_gates, label="rotation_gates")
    final_rotations = rotations if final_rotation_gates is None else _normalize_rotation_gates(
        final_rotation_gates,
        label="final_rotation_gates",
    )

    entangler_key = str(entangler).lower()
    if entangler_key not in _SUPPORTED_ENTANGLERS:
        allowed = ", ".join(sorted(_SUPPORTED_ENTANGLERS))
        raise ValueError(f"Unsupported entangler: {entangler}. Supported entanglers: {allowed}")

    edges = entangling_edges(n_qubits, topology)
    params = _ParameterStream(_flatten_parameters(parameters), parameter_prefix)
    gates: list[dict[str, Any]] = []

    for _ in range(layers):
        _append_rotation_layer(gates, n_qubits, rotations, params)
        _append_entangler_layer(gates, edges, entangler_key, params)

    if final_rotation_layer:
        _append_rotation_layer(gates, n_qubits, final_rotations, params)

    params.finish()
    return Circuit(*gates, n_qubits=n_qubits, backend=backend)


hea = hardware_efficient_ansatz


__all__ = [
    "Edge",
    "entangling_edges",
    "hea",
    "hea_parameter_count",
    "hardware_efficient_ansatz",
]
