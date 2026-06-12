"""Hardware-efficiency metrics for quantum circuits."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..core.circuit import Circuit
from ..ir import circuit_instruction_count, circuit_instructions, instruction_controls, instruction_qubits
from ._utils import count_two_qubit_gates, depth_proxy, gate_type


DEFAULT_NATIVE_GATES = ("hadamard", "rx", "ry", "rz", "cx", "cnot")


@dataclass(frozen=True)
class HardwareProfile:
    """Minimal hardware profile for zero-cost topology mapping efficiency."""

    native_gates: Sequence[str] = DEFAULT_NATIVE_GATES
    coupling_map: Sequence[tuple[int, int]] = field(default_factory=tuple)
    edge_fidelity: Optional[Dict[tuple[int, int], float] | float] = None
    gate_durations: Dict[str, float] = field(default_factory=dict)
    max_depth: Optional[int] = None


def native_depth_twoq_efficiency(
    circuit: Circuit,
    native_gates: Optional[Sequence[str]] = None,
    max_depth: int = 100,
) -> float:
    """Native-gate, depth, and two-qubit-density hardware efficiency proxy."""
    native_set = {str(gate).lower() for gate in (native_gates or DEFAULT_NATIVE_GATES)}
    n_gates = circuit_instruction_count(circuit)
    native_count = sum(1 for gate in circuit_instructions(circuit) if gate_type(gate) in native_set)
    native_ratio = native_count / n_gates if n_gates > 0 else 1.0
    circuit_depth_proxy = depth_proxy(circuit)
    depth_score = min(1.0, float(max_depth) / max(1.0, circuit_depth_proxy * 10.0))
    two_qubit_ratio = count_two_qubit_gates(circuit) / max(1, int(circuit.n_qubits))
    twoq_efficiency = np.exp(-two_qubit_ratio / 3.0)
    return float(np.clip(0.4 * native_ratio + 0.3 * depth_score + 0.3 * twoq_efficiency, 0.0, 1.0))


def native_depth_twoq_efficiency_details(
    circuit: Circuit,
    native_gates: Optional[Sequence[str]] = None,
    max_depth: int = 100,
) -> Dict[str, Any]:
    native_set = {str(gate).lower() for gate in (native_gates or DEFAULT_NATIVE_GATES)}
    n_gates = circuit_instruction_count(circuit)
    native_count = sum(1 for gate in circuit_instructions(circuit) if gate_type(gate) in native_set)
    return {
        "native_depth_twoq_efficiency_score": native_depth_twoq_efficiency(circuit, native_gates, max_depth),
        "native_gate_count": native_count,
        "native_gate_ratio": native_count / n_gates if n_gates > 0 else 1.0,
        "two_qubit_gate_count": count_two_qubit_gates(circuit),
        "depth_proxy": depth_proxy(circuit),
    }


def _two_qubit_edge(gate) -> Optional[tuple[int, int]]:
    controls = instruction_controls(gate)
    qubits = instruction_qubits(gate)
    if controls and qubits:
        if len(controls) == 1 and len(qubits) == 1:
            return tuple(sorted((int(controls[0]), int(qubits[0]))))
    if len(qubits) == 2:
        return tuple(sorted((int(qubits[0]), int(qubits[1]))))
    return None


def _adjacency(coupling_map: Sequence[tuple[int, int]]) -> Dict[int, set[int]]:
    graph: Dict[int, set[int]] = {}
    for source, target in coupling_map:
        i, j = int(source), int(target)
        graph.setdefault(i, set()).add(j)
        graph.setdefault(j, set()).add(i)
    return graph


def _shortest_path_distance(graph: Dict[int, set[int]], source: int, target: int) -> Optional[int]:
    if source == target:
        return 0
    if source not in graph or target not in graph:
        return None
    queue = deque([(source, 0)])
    seen = {source}
    while queue:
        node, distance = queue.popleft()
        for next_node in graph.get(node, ()):
            if next_node == target:
                return distance + 1
            if next_node not in seen:
                seen.add(next_node)
                queue.append((next_node, distance + 1))
    return None


def _edge_fidelity(edge: tuple[int, int], edge_fidelity: Optional[Dict[tuple[int, int], float] | float]) -> Optional[float]:
    if edge_fidelity is None:
        return None
    if isinstance(edge_fidelity, (int, float)):
        return float(edge_fidelity)
    normalized = tuple(sorted(edge))
    if normalized in edge_fidelity:
        return float(edge_fidelity[normalized])
    reversed_edge = (normalized[1], normalized[0])
    if reversed_edge in edge_fidelity:
        return float(edge_fidelity[reversed_edge])
    return None


def topology_mapping_efficiency(circuit: Circuit, profile: Optional[HardwareProfile] = None) -> float:
    """Zero-cost hardware score from native gates, topology, routing, and depth."""
    details = topology_mapping_efficiency_details(circuit, profile=profile)
    return float(details["topology_mapping_efficiency_score"])


def topology_mapping_efficiency_details(
    circuit: Circuit,
    profile: Optional[HardwareProfile] = None,
) -> Dict[str, Any]:
    profile = profile or HardwareProfile()
    native_set = {str(gate).lower() for gate in profile.native_gates}
    coupling_edges = {tuple(sorted((int(i), int(j)))) for i, j in profile.coupling_map}
    graph = _adjacency(tuple(coupling_edges))
    n_gates = circuit_instruction_count(circuit)
    native_count = sum(1 for gate in circuit_instructions(circuit) if gate_type(gate) in native_set)
    non_native_ratio = 1.0 - (native_count / n_gates if n_gates > 0 else 1.0)

    two_qubit_edges = [edge for edge in (_two_qubit_edge(gate) for gate in circuit_instructions(circuit)) if edge is not None]
    two_qubit_count = len(two_qubit_edges)
    connectivity_violations = 0
    routing_distance_cost = 0.0
    mapped_fidelities = []
    for edge in two_qubit_edges:
        if coupling_edges and edge not in coupling_edges:
            connectivity_violations += 1
        if coupling_edges:
            distance = _shortest_path_distance(graph, edge[0], edge[1])
            if distance is None:
                routing_distance_cost += max(1, int(circuit.n_qubits))
            else:
                routing_distance_cost += max(0, distance - 1)
        fidelity = _edge_fidelity(edge, profile.edge_fidelity)
        if fidelity is not None:
            mapped_fidelities.append(float(np.clip(fidelity, 0.0, 1.0)))

    circuit_depth = depth_proxy(circuit)
    max_depth = profile.max_depth if profile.max_depth is not None else max(1.0, 4.0 * max(1, int(circuit.n_qubits)))
    depth_norm = circuit_depth / max(1.0, float(max_depth))
    routing_norm = routing_distance_cost / max(1.0, float(two_qubit_count))
    twoq_density = two_qubit_count / max(1, int(circuit.n_qubits))
    score = np.exp(-0.8 * routing_norm - 0.5 * depth_norm - 0.4 * non_native_ratio - 0.2 * twoq_density)
    mapping_fidelity_score = None if not mapped_fidelities else float(np.mean(mapped_fidelities))
    return {
        "topology_mapping_efficiency_score": float(np.clip(score, 0.0, 1.0)),
        "native_gate_count": native_count,
        "native_gate_ratio": native_count / n_gates if n_gates > 0 else 1.0,
        "non_native_ratio": non_native_ratio,
        "two_qubit_gate_count": two_qubit_count,
        "two_qubit_density": twoq_density,
        "depth_proxy": circuit_depth,
        "depth_norm": depth_norm,
        "connectivity_violation_count": connectivity_violations,
        "routing_distance_cost": routing_distance_cost,
        "routing_distance_per_twoq": routing_norm,
        "mapping_fidelity_score": mapping_fidelity_score,
        "mapping_fidelity_note": "reported for mapping preference only; not included in the primary score",
    }


__all__ = [
    "DEFAULT_NATIVE_GATES",
    "HardwareProfile",
    "native_depth_twoq_efficiency",
    "native_depth_twoq_efficiency_details",
    "topology_mapping_efficiency",
    "topology_mapping_efficiency_details",
]
