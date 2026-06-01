"""NexQ-native architecture generation strategies for QAS."""

from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..channel.backends.base import Backend
from ..core.circuit import Circuit
from ..metrics.hardware import HardwareProfile
from ._types import ArchitectureSpec, SearchConfig


@dataclass(frozen=True)
class SuperCircuitBlock:
    """One architectural choice position inside a SuperCircuit search space."""

    name: str
    choices: Sequence[str]


class _ParameterCursor:
    def __init__(self) -> None:
        self._index = 0

    def next(self) -> float:
        self._index += 1
        return 0.071 * self._index


def _linear_edges(n_qubits: int) -> List[tuple[int, int]]:
    return [(i, i + 1) for i in range(n_qubits - 1)]


def _ring_edges(n_qubits: int) -> List[tuple[int, int]]:
    edges = _linear_edges(n_qubits)
    if n_qubits > 2:
        edges.append((n_qubits - 1, 0))
    return edges


def _append_rotation_choice(gates: List[Dict[str, Any]], choice: str, n_qubits: int, cursor: _ParameterCursor) -> None:
    if choice == "skip":
        return
    if choice == "h":
        for q in range(n_qubits):
            gates.append({"type": "hadamard", "target_qubit": q})
        return
    rotations = {
        "rx": ("rx",),
        "ry": ("ry",),
        "rz": ("rz",),
        "ry_rz": ("ry", "rz"),
        "rx_ry_rz": ("rx", "ry", "rz"),
    }.get(choice)
    if rotations is None:
        raise ValueError(f"Unsupported rotation choice: {choice}")
    for q in range(n_qubits):
        for gate_type in rotations:
            gates.append({"type": gate_type, "target_qubit": q, "parameter": cursor.next()})


def _has_mixing_choice(mask: Sequence[int], blocks: Sequence[SuperCircuitBlock]) -> bool:
    mixing_choices = {"h", "rx", "ry", "ry_rz", "rx_ry_rz"}
    for index, block in zip(mask, blocks):
        if not (block.name.startswith("rot") or block.name == "final_rot"):
            continue
        if block.choices[int(index) % len(block.choices)] in mixing_choices:
            return True
    return False


def _is_valid_supercircuit_mask(mask: Sequence[int], blocks: Sequence[SuperCircuitBlock]) -> bool:
    return not all(block.choices[int(index)] == "skip" for index, block in zip(mask, blocks)) and _has_mixing_choice(mask, blocks)


def _append_entangler_choice(gates: List[Dict[str, Any]], choice: str, n_qubits: int, cursor: _ParameterCursor) -> None:
    if choice == "skip" or n_qubits < 2:
        return
    parts = choice.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unsupported entangler choice: {choice}")
    gate_type = parts[0]
    topology = "_".join(parts[1:])
    if topology == "linear":
        edges = _linear_edges(n_qubits)
    elif topology == "ring":
        edges = _ring_edges(n_qubits)
    else:
        raise ValueError(f"Unsupported entangler topology: {topology}")

    for control, target in edges:
        if gate_type == "rzz":
            gates.append({"type": "rzz", "qubit_1": control, "qubit_2": target, "parameter": cursor.next()})
        elif gate_type in {"cx", "cz"}:
            gates.append(
                {
                    "type": gate_type,
                    "target_qubit": target,
                    "control_qubits": [control],
                    "control_states": [1],
                }
            )
        else:
            raise ValueError(f"Unsupported entangler gate: {gate_type}")


def supercircuit_blocks(layers: int) -> List[SuperCircuitBlock]:
    """Build a block-level SuperCircuit search space."""
    blocks: List[SuperCircuitBlock] = []
    for layer_index in range(max(1, int(layers))):
        blocks.append(
            SuperCircuitBlock(
                name=f"rot_{layer_index}",
                choices=("skip", "h", "rx", "ry", "rz", "ry_rz", "rx_ry_rz"),
            )
        )
        blocks.append(
            SuperCircuitBlock(
                name=f"ent_{layer_index}",
                choices=("skip", "cx_linear", "cz_linear", "rzz_linear", "cx_ring", "rzz_ring"),
            )
        )
    blocks.append(SuperCircuitBlock(name="final_rot", choices=("skip", "rx", "ry", "rz", "ry_rz")))
    return blocks


def _gate_qubits(gate: Dict[str, Any]) -> List[int]:
    if "target_qubit" in gate:
        qubits = [int(gate["target_qubit"])]
        qubits.extend(int(q) for q in gate.get("control_qubits", []) or [])
        return sorted(set(qubits))
    qubits = []
    if "qubit_1" in gate:
        qubits.append(int(gate["qubit_1"]))
    if "qubit_2" in gate:
        qubits.append(int(gate["qubit_2"]))
    return sorted(set(qubits))


def _two_qubit_edge(gate: Dict[str, Any]) -> Optional[tuple[int, int]]:
    qubits = _gate_qubits(gate)
    if len(qubits) == 2:
        return tuple(sorted((qubits[0], qubits[1])))
    return None


def dag_path_count(circuit: Circuit) -> int:
    """Count input-output paths in a dependency DAG built from circuit wires."""
    n_qubits = int(circuit.n_qubits)
    input_node = 0
    output_node = 1
    next_node = 2
    last_on_wire = {qubit: input_node for qubit in range(n_qubits)}
    predecessors: Dict[int, List[int]] = {input_node: [], output_node: []}
    order = [input_node]

    for gate in circuit.gates:
        qubits = _gate_qubits(gate)
        if not qubits:
            continue
        node = next_node
        next_node += 1
        preds = sorted({last_on_wire[qubit] for qubit in qubits})
        predecessors[node] = preds
        order.append(node)
        for qubit in qubits:
            last_on_wire[qubit] = node

    predecessors[output_node] = sorted(set(last_on_wire.values()))
    order.append(output_node)
    paths = {input_node: 1}
    for node in order[1:]:
        paths[node] = sum(paths.get(pred, 0) for pred in predecessors.get(node, []))
    return int(paths.get(output_node, 0))


def progressive_structure_score(
    candidate: ArchitectureSpec,
    hardware_profile: Optional[HardwareProfile] = None,
) -> float:
    """Cheap structural proxy used before expensive zero-cost metrics."""
    path_score = log1p(max(0, dag_path_count(candidate.circuit)))
    n_gates = max(1, candidate.n_gates)
    twoq_ratio = candidate.two_qubit_gate_count / max(1, int(candidate.n_qubits))
    twoq_balance = 1.0 / (1.0 + abs(twoq_ratio - 1.0))
    param_balance = 1.0 / (1.0 + abs(candidate.parameter_count / max(1, candidate.n_qubits) - 2.0))

    topology_score = 1.0
    if hardware_profile is not None and hardware_profile.coupling_map:
        coupling_edges = {tuple(sorted((int(i), int(j)))) for i, j in hardware_profile.coupling_map}
        twoq_edges = [_two_qubit_edge(gate) for gate in candidate.circuit.gates]
        twoq_edges = [edge for edge in twoq_edges if edge is not None]
        if twoq_edges:
            native_edges = sum(1 for edge in twoq_edges if edge in coupling_edges)
            topology_score = native_edges / len(twoq_edges)

    compactness = 1.0 / (1.0 + n_gates / max(1, int(candidate.n_qubits) * 6.0))
    return float(0.40 * path_score + 0.25 * twoq_balance + 0.20 * topology_score + 0.10 * param_balance + 0.05 * compactness)


def subcircuit_from_mask(
    n_qubits: int,
    mask: Sequence[int],
    blocks: Sequence[SuperCircuitBlock],
    backend: Optional[Backend] = None,
    name: Optional[str] = None,
) -> ArchitectureSpec:
    """Convert a SuperCircuit choice mask into a NexQ ArchitectureSpec."""
    cursor = _ParameterCursor()
    gates: List[Dict[str, Any]] = []
    choices: List[str] = []
    for choice_index, block in zip(mask, blocks):
        choice = block.choices[int(choice_index) % len(block.choices)]
        choices.append(choice)
        if block.name.startswith("rot") or block.name == "final_rot":
            _append_rotation_choice(gates, choice, n_qubits, cursor)
        elif block.name.startswith("ent"):
            _append_entangler_choice(gates, choice, n_qubits, cursor)
        else:
            raise ValueError(f"Unsupported SuperCircuit block: {block.name}")

    mask_tuple = tuple(int(x) for x in mask)
    suffix = "_".join(str(x) for x in mask_tuple)
    return ArchitectureSpec.from_gates(
        name=name or f"supercircuit_sub_{suffix}",
        gates=gates,
        n_qubits=n_qubits,
        backend=backend,
        description="SubCircuit sampled from a block-level SuperCircuit search space.",
        tags=["SuperCircuit", "SubCircuit", "zero_cost"],
        metadata={
            "family": "SuperCircuit",
            "supercircuit_mask": mask_tuple,
            "supercircuit_choices": choices,
            "supercircuit_blocks": [block.name for block in blocks],
        },
    )


def _sample_supercircuit_candidates(
    config: SearchConfig,
    backend: Optional[Backend] = None,
    sample_count: Optional[int] = None,
) -> List[ArchitectureSpec]:
    blocks = supercircuit_blocks(config.candidate_layers)
    rng = np.random.default_rng(int(config.seed))
    target_count = max(0, int(sample_count if sample_count is not None else config.population_size))
    candidates: List[ArchitectureSpec] = []
    seen: set[tuple[int, ...]] = set()

    seed_masks = [
        tuple(1 if block.name.startswith("rot") or block.name == "final_rot" else 1 for block in blocks),
        tuple(min(3, len(block.choices) - 1) for block in blocks),
    ]
    for mask in seed_masks:
        if len(candidates) >= target_count:
            break
        if mask in seen or not _is_valid_supercircuit_mask(mask, blocks):
            continue
        seen.add(mask)
        candidates.append(
            subcircuit_from_mask(config.n_qubits, mask, blocks, backend=backend, name=f"supercircuit_seed_{len(candidates)}")
        )

    attempts = 0
    max_attempts = max(50, target_count * 20)
    while len(candidates) < target_count and attempts < max_attempts:
        attempts += 1
        mask = tuple(int(rng.integers(0, len(block.choices))) for block in blocks)
        if mask in seen or not _is_valid_supercircuit_mask(mask, blocks):
            continue
        seen.add(mask)
        candidates.append(subcircuit_from_mask(config.n_qubits, mask, blocks, backend=backend))
    return candidates


def generate_supercircuit_subcircuits(
    config: SearchConfig,
    backend: Optional[Backend] = None,
) -> List[ArchitectureSpec]:
    """Sample SubCircuit masks from a SuperCircuit search space."""
    return _sample_supercircuit_candidates(config, backend=backend)


def generate_progressive_supercircuit_subcircuits(
    config: SearchConfig,
    backend: Optional[Backend] = None,
    hardware_profile: Optional[HardwareProfile] = None,
) -> List[ArchitectureSpec]:
    """Sample many masks, prefilter with a cheap DAG/topology proxy, then return the best subset."""
    sample_count = max(int(config.population_size), int(config.n_samples))
    keep_count = config.progressive_keep
    if keep_count is None:
        keep_count = config.candidate_budget or config.top_k or max(1, sample_count // 4)
    keep_count = max(0, min(int(keep_count), sample_count))
    candidates = _sample_supercircuit_candidates(config, backend=backend, sample_count=sample_count)
    ranked: List[tuple[float, int, ArchitectureSpec]] = []
    for candidate in candidates:
        path_count = dag_path_count(candidate.circuit)
        cheap_score = progressive_structure_score(candidate, hardware_profile=hardware_profile)
        candidate.metadata.update(
            {
                "progressive_path_count": path_count,
                "progressive_structure_score": cheap_score,
                "progressive_prefilter_rank": None,
            }
        )
        ranked.append((cheap_score, path_count, candidate))
    ranked.sort(key=lambda item: (item[0], item[1], -item[2].n_gates), reverse=True)
    selected = [candidate for _, _, candidate in ranked[:keep_count]]
    for rank, candidate in enumerate(selected, start=1):
        candidate.metadata["progressive_prefilter_rank"] = rank
    return selected


__all__ = [
    "SuperCircuitBlock",
    "dag_path_count",
    "generate_progressive_supercircuit_subcircuits",
    "generate_supercircuit_subcircuits",
    "progressive_structure_score",
    "subcircuit_from_mask",
    "supercircuit_blocks",
]
