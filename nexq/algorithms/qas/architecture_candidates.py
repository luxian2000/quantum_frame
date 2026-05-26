"""Common candidate architectures for architecture-level QAS scoring."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ...channel.backends.base import Backend
from ._types import ArchitectureSpec


Edge = Tuple[int, int]


class _ParameterCursor:
    def __init__(self) -> None:
        self._index = 0

    def next(self) -> float:
        self._index += 1
        return 0.071 * self._index


def _validate_n_qubits(n_qubits: int, min_qubits: int = 1) -> None:
    if n_qubits < min_qubits:
        raise ValueError(f"n_qubits must be at least {min_qubits}, got {n_qubits}")


def _linear_edges(n_qubits: int) -> List[Edge]:
    return [(source_qubit, source_qubit + 1) for source_qubit in range(n_qubits - 1)]


def _ring_edges(n_qubits: int) -> List[Edge]:
    edges = _linear_edges(n_qubits)
    if n_qubits > 2:
        edges.append((n_qubits - 1, 0))
    return edges


def _all_to_all_edges(n_qubits: int) -> List[Edge]:
    return [
        (source_qubit, target_qubit)
        for source_qubit in range(n_qubits)
        for target_qubit in range(source_qubit + 1, n_qubits)
    ]


def _topology_edges(n_qubits: int, topology: str) -> List[Edge]:
    if topology == "linear":
        return _linear_edges(n_qubits)
    if topology == "ring":
        return _ring_edges(n_qubits)
    if topology == "all_to_all":
        return _all_to_all_edges(n_qubits)
    raise ValueError(f"Unsupported topology: {topology}")


def _controlled_gate(gate_type: str, control_qubit: int, target_qubit: int, cursor: _ParameterCursor) -> Dict[str, Any]:
    gate: Dict[str, Any] = {
        "type": gate_type,
        "target_qubit": target_qubit,
        "control_qubits": [control_qubit],
        "control_states": [1],
    }
    if gate_type in {"crx", "cry", "crz"}:
        gate["parameter"] = cursor.next()
    return gate


def _two_qubit_gate(gate_type: str, edge: Edge, cursor: _ParameterCursor) -> Dict[str, Any]:
    control_qubit, target_qubit = edge
    if gate_type == "rzz":
        return {
            "type": "rzz",
            "qubit_1": control_qubit,
            "qubit_2": target_qubit,
            "parameter": cursor.next(),
        }
    if gate_type == "swap":
        return {"type": "swap", "qubit_1": control_qubit, "qubit_2": target_qubit}
    return _controlled_gate(gate_type, control_qubit, target_qubit, cursor)


def _append_rotation_layer(
    gates: List[Dict[str, Any]],
    n_qubits: int,
    rotation_blocks: Sequence[str],
    cursor: _ParameterCursor,
) -> None:
    for target_qubit in range(n_qubits):
        for gate_type in rotation_blocks:
            if gate_type == "u3":
                gates.append(
                    {
                        "type": "u3",
                        "target_qubit": target_qubit,
                        "parameter": [cursor.next(), cursor.next(), cursor.next()],
                    }
                )
            elif gate_type in {"rx", "ry", "rz"}:
                gates.append({"type": gate_type, "target_qubit": target_qubit, "parameter": cursor.next()})
            elif gate_type == "hadamard":
                gates.append({"type": "hadamard", "target_qubit": target_qubit})
            else:
                raise ValueError(f"Unsupported rotation block: {gate_type}")


def _append_entangler_layer(
    gates: List[Dict[str, Any]],
    edges: Sequence[Edge],
    entangler: str,
    cursor: _ParameterCursor,
) -> None:
    for edge in edges:
        gates.append(_two_qubit_gate(entangler, edge, cursor))


def _spec(
    name: str,
    gates: Sequence[Dict[str, Any]],
    n_qubits: int,
    backend: Optional[Backend],
    description: str,
    tags: Sequence[str],
    metadata: Dict[str, Any],
) -> ArchitectureSpec:
    return ArchitectureSpec.from_gates(
        name=name,
        gates=gates,
        n_qubits=n_qubits,
        backend=backend,
        description=description,
        tags=list(tags),
        metadata=dict(metadata),
    )


def hardware_efficient_ansatz(
    n_qubits: int,
    layers: int = 2,
    topology: str = "linear",
    entangler: str = "cx",
    backend: Optional[Backend] = None,
) -> ArchitectureSpec:
    """Hardware-efficient ansatz: local rotations plus repeated entanglers."""
    _validate_n_qubits(n_qubits, 2)
    cursor = _ParameterCursor()
    gates: List[Dict[str, Any]] = []
    edges = _topology_edges(n_qubits, topology)
    for layer_index in range(layers):
        _append_rotation_layer(gates, n_qubits, ("ry", "rz"), cursor)
        _append_entangler_layer(gates, edges, entangler, cursor)
    _append_rotation_layer(gates, n_qubits, ("ry",), cursor)
    return _spec(
        name=f"hea_{topology}_{entangler}_L{layers}",
        gates=gates,
        n_qubits=n_qubits,
        backend=backend,
        description="Hardware-efficient ansatz with RY/RZ layers and repeated entanglers.",
        tags=("HEA", "hardware_efficient", topology, entangler),
        metadata={"family": "HEA", "layers": layers, "topology": topology, "entangler": entangler},
    )


def real_amplitudes_ansatz(
    n_qubits: int,
    layers: int = 2,
    topology: str = "linear",
    backend: Optional[Backend] = None,
) -> ArchitectureSpec:
    """RealAmplitudes-style ansatz with RY rotations and CX entanglers."""
    _validate_n_qubits(n_qubits, 2)
    cursor = _ParameterCursor()
    gates: List[Dict[str, Any]] = []
    edges = _topology_edges(n_qubits, topology)
    for layer_index in range(layers):
        _append_rotation_layer(gates, n_qubits, ("ry",), cursor)
        _append_entangler_layer(gates, edges, "cx", cursor)
    _append_rotation_layer(gates, n_qubits, ("ry",), cursor)
    return _spec(
        name=f"real_amplitudes_{topology}_L{layers}",
        gates=gates,
        n_qubits=n_qubits,
        backend=backend,
        description="Real-valued RY ansatz with CX entanglement blocks.",
        tags=("RealAmplitudes", "two_local", topology, "cx"),
        metadata={"family": "RealAmplitudes", "layers": layers, "topology": topology, "entangler": "cx"},
    )


def efficient_su2_ansatz(
    n_qubits: int,
    layers: int = 2,
    topology: str = "ring",
    entangler: str = "cx",
    backend: Optional[Backend] = None,
) -> ArchitectureSpec:
    """EfficientSU2-style ansatz with SU(2) local rotations and entanglers."""
    _validate_n_qubits(n_qubits, 2)
    cursor = _ParameterCursor()
    gates: List[Dict[str, Any]] = []
    edges = _topology_edges(n_qubits, topology)
    for layer_index in range(layers):
        _append_rotation_layer(gates, n_qubits, ("rx", "ry", "rz"), cursor)
        _append_entangler_layer(gates, edges, entangler, cursor)
    _append_rotation_layer(gates, n_qubits, ("ry", "rz"), cursor)
    return _spec(
        name=f"efficient_su2_{topology}_{entangler}_L{layers}",
        gates=gates,
        n_qubits=n_qubits,
        backend=backend,
        description="EfficientSU2-like ansatz with RX/RY/RZ rotations and entanglers.",
        tags=("EfficientSU2", "two_local", topology, entangler),
        metadata={"family": "EfficientSU2", "layers": layers, "topology": topology, "entangler": entangler},
    )


def two_local_ansatz(
    n_qubits: int,
    layers: int = 2,
    topology: str = "all_to_all",
    entangler: str = "cz",
    backend: Optional[Backend] = None,
) -> ArchitectureSpec:
    """General TwoLocal template with dense rotation blocks and configurable entanglement."""
    _validate_n_qubits(n_qubits, 2)
    cursor = _ParameterCursor()
    gates: List[Dict[str, Any]] = []
    edges = _topology_edges(n_qubits, topology)
    for layer_index in range(layers):
        _append_rotation_layer(gates, n_qubits, ("rx", "ry", "rz"), cursor)
        _append_entangler_layer(gates, edges, entangler, cursor)
    return _spec(
        name=f"two_local_{topology}_{entangler}_L{layers}",
        gates=gates,
        n_qubits=n_qubits,
        backend=backend,
        description="TwoLocal ansatz with RX/RY/RZ rotation blocks and configurable entanglers.",
        tags=("TwoLocal", topology, entangler),
        metadata={"family": "TwoLocal", "layers": layers, "topology": topology, "entangler": entangler},
    )


def qaoa_ansatz(
    n_qubits: int,
    layers: int = 2,
    topology: str = "linear",
    backend: Optional[Backend] = None,
) -> ArchitectureSpec:
    """QAOA-like Ising cost ansatz with RZZ cost terms and RX mixers."""
    _validate_n_qubits(n_qubits, 2)
    cursor = _ParameterCursor()
    gates: List[Dict[str, Any]] = []
    edges = _topology_edges(n_qubits, topology)
    for target_qubit in range(n_qubits):
        gates.append({"type": "hadamard", "target_qubit": target_qubit})
    for layer_index in range(layers):
        _append_entangler_layer(gates, edges, "rzz", cursor)
        _append_rotation_layer(gates, n_qubits, ("rx",), cursor)
    return _spec(
        name=f"qaoa_{topology}_rzz_rx_p{layers}",
        gates=gates,
        n_qubits=n_qubits,
        backend=backend,
        description="QAOA-style ansatz with Ising RZZ cost layers and RX mixer layers.",
        tags=("QAOA", "Ising", topology, "rzz"),
        metadata={"family": "QAOA", "layers": layers, "topology": topology, "entangler": "rzz"},
    )


def brickwork_ansatz(
    n_qubits: int,
    layers: int = 2,
    entangler: str = "cx",
    backend: Optional[Backend] = None,
) -> ArchitectureSpec:
    """Brickwork ansatz with alternating even/odd nearest-neighbor entanglers."""
    _validate_n_qubits(n_qubits, 2)
    cursor = _ParameterCursor()
    gates: List[Dict[str, Any]] = []
    even_edges = [(source_qubit, source_qubit + 1) for source_qubit in range(0, n_qubits - 1, 2)]
    odd_edges = [(source_qubit, source_qubit + 1) for source_qubit in range(1, n_qubits - 1, 2)]
    for layer_index in range(layers):
        _append_rotation_layer(gates, n_qubits, ("ry", "rz"), cursor)
        _append_entangler_layer(gates, even_edges, entangler, cursor)
        _append_rotation_layer(gates, n_qubits, ("rx",), cursor)
        _append_entangler_layer(gates, odd_edges, entangler, cursor)
    return _spec(
        name=f"brickwork_{entangler}_L{layers}",
        gates=gates,
        n_qubits=n_qubits,
        backend=backend,
        description="Alternating brickwork nearest-neighbor ansatz.",
        tags=("brickwork", "nearest_neighbor", entangler),
        metadata={"family": "Brickwork", "layers": layers, "topology": "brickwork", "entangler": entangler},
    )


def cascade_entangler_ansatz(
    n_qubits: int,
    layers: int = 2,
    backend: Optional[Backend] = None,
) -> ArchitectureSpec:
    """Cascade entangler template with forward and reverse CX ladders."""
    _validate_n_qubits(n_qubits, 2)
    cursor = _ParameterCursor()
    gates: List[Dict[str, Any]] = []
    forward_edges = _linear_edges(n_qubits)
    reverse_edges = [(target_qubit, control_qubit) for control_qubit, target_qubit in reversed(forward_edges)]
    for layer_index in range(layers):
        _append_rotation_layer(gates, n_qubits, ("ry", "rz"), cursor)
        _append_entangler_layer(gates, forward_edges, "cx", cursor)
        _append_rotation_layer(gates, n_qubits, ("rx",), cursor)
        _append_entangler_layer(gates, reverse_edges, "cx", cursor)
    return _spec(
        name=f"cascade_cx_L{layers}",
        gates=gates,
        n_qubits=n_qubits,
        backend=backend,
        description="Forward/reverse CX cascade ansatz.",
        tags=("cascade", "ladder", "cx"),
        metadata={"family": "Cascade", "layers": layers, "topology": "linear", "entangler": "cx"},
    )


def strongly_entangling_layers_ansatz(
    n_qubits: int,
    layers: int = 2,
    backend: Optional[Backend] = None,
) -> ArchitectureSpec:
    """Strongly-entangling-layer template with U3 blocks and shifting CRX ranges."""
    _validate_n_qubits(n_qubits, 2)
    cursor = _ParameterCursor()
    gates: List[Dict[str, Any]] = []
    for layer_index in range(layers):
        _append_rotation_layer(gates, n_qubits, ("u3",), cursor)
        offset = (layer_index % (n_qubits - 1)) + 1
        edges = [(control_qubit, (control_qubit + offset) % n_qubits) for control_qubit in range(n_qubits)]
        _append_entangler_layer(gates, edges, "crx", cursor)
    return _spec(
        name=f"strongly_entangling_crx_L{layers}",
        gates=gates,
        n_qubits=n_qubits,
        backend=backend,
        description="StronglyEntanglingLayers-style template with U3 and shifted CRX ranges.",
        tags=("StronglyEntanglingLayers", "u3", "crx"),
        metadata={"family": "StronglyEntanglingLayers", "layers": layers, "topology": "shifted_ring", "entangler": "crx"},
    )


def ghz_ladder_ansatz(
    n_qubits: int,
    backend: Optional[Backend] = None,
) -> ArchitectureSpec:
    """GHZ-style ladder with a light parameterized dressing layer."""
    _validate_n_qubits(n_qubits, 2)
    cursor = _ParameterCursor()
    gates: List[Dict[str, Any]] = [{"type": "hadamard", "target_qubit": 0}]
    _append_entangler_layer(gates, _linear_edges(n_qubits), "cx", cursor)
    _append_rotation_layer(gates, n_qubits, ("rz", "ry"), cursor)
    return _spec(
        name="ghz_ladder_dressed",
        gates=gates,
        n_qubits=n_qubits,
        backend=backend,
        description="GHZ ladder state-preparation backbone with a parameterized local dressing layer.",
        tags=("GHZ", "ladder", "cx"),
        metadata={"family": "GHZ", "layers": 1, "topology": "linear", "entangler": "cx"},
    )


def mera_like_ansatz(
    n_qubits: int,
    layers: int = 1,
    backend: Optional[Backend] = None,
) -> ArchitectureSpec:
    """Small MERA-like hierarchical ansatz using local rotations and CX disentanglers."""
    _validate_n_qubits(n_qubits, 2)
    cursor = _ParameterCursor()
    gates: List[Dict[str, Any]] = []
    for layer_index in range(layers):
        stride = 1
        while stride < n_qubits:
            _append_rotation_layer(gates, n_qubits, ("ry", "rz"), cursor)
            edges = [
                (source_qubit, source_qubit + stride)
                for source_qubit in range(0, n_qubits, 2 * stride)
                if source_qubit + stride < n_qubits
            ]
            _append_entangler_layer(gates, edges, "cx", cursor)
            stride *= 2
    return _spec(
        name=f"mera_like_cx_L{layers}",
        gates=gates,
        n_qubits=n_qubits,
        backend=backend,
        description="Hierarchical MERA-like ansatz with increasing-range CX entanglers.",
        tags=("MERA", "hierarchical", "cx"),
        metadata={"family": "MERA-like", "layers": layers, "topology": "hierarchical", "entangler": "cx"},
    )


def common_architecture_names() -> List[str]:
    """Return the preset candidate names accepted by build_common_architectures."""
    return [
        "hea_linear",
        "hea_ring",
        "real_amplitudes_linear",
        "efficient_su2_ring",
        "two_local_all_to_all",
        "qaoa_chain",
        "qaoa_complete",
        "brickwork_cx",
        "cascade_cx",
        "strongly_entangling_crx",
        "ghz_ladder",
        "mera_like",
    ]


def build_common_architectures(
    n_qubits: int = 4,
    layers: int = 2,
    backend: Optional[Backend] = None,
    names: Optional[Sequence[str]] = None,
) -> List[ArchitectureSpec]:
    """Build a reusable library of common QAS candidate architectures.

    The presets intentionally mix shallow/deep, sparse/dense, and fixed/parameterized
    templates so the architecture evaluator can rank real tradeoffs instead of one
    narrow family.
    """
    _validate_n_qubits(n_qubits, 2)
    builders: Dict[str, Callable[[], ArchitectureSpec]] = {
        "hea_linear": lambda: hardware_efficient_ansatz(n_qubits, layers, "linear", "cx", backend),
        "hea_ring": lambda: hardware_efficient_ansatz(n_qubits, layers, "ring", "cx", backend),
        "real_amplitudes_linear": lambda: real_amplitudes_ansatz(n_qubits, layers, "linear", backend),
        "efficient_su2_ring": lambda: efficient_su2_ansatz(n_qubits, layers, "ring", "cx", backend),
        "two_local_all_to_all": lambda: two_local_ansatz(n_qubits, layers, "all_to_all", "cz", backend),
        "qaoa_chain": lambda: qaoa_ansatz(n_qubits, layers, "linear", backend),
        "qaoa_complete": lambda: qaoa_ansatz(n_qubits, layers, "all_to_all", backend),
        "brickwork_cx": lambda: brickwork_ansatz(n_qubits, layers, "cx", backend),
        "cascade_cx": lambda: cascade_entangler_ansatz(n_qubits, layers, backend),
        "strongly_entangling_crx": lambda: strongly_entangling_layers_ansatz(n_qubits, layers, backend),
        "ghz_ladder": lambda: ghz_ladder_ansatz(n_qubits, backend),
        "mera_like": lambda: mera_like_ansatz(n_qubits, max(1, layers // 2), backend),
    }
    selected_names = list(names) if names is not None else common_architecture_names()
    unknown_names = [name for name in selected_names if name not in builders]
    if unknown_names:
        raise ValueError(f"Unknown architecture preset(s): {unknown_names}")
    return [builders[name]() for name in selected_names]


__all__ = [
    "build_common_architectures",
    "common_architecture_names",
    "hardware_efficient_ansatz",
    "real_amplitudes_ansatz",
    "efficient_su2_ansatz",
    "two_local_ansatz",
    "qaoa_ansatz",
    "brickwork_ansatz",
    "cascade_entangler_ansatz",
    "strongly_entangling_layers_ansatz",
    "ghz_ladder_ansatz",
    "mera_like_ansatz",
]