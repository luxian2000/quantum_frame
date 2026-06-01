"""NexQ-native architecture generation strategies for QAS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..channel.backends.base import Backend
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


def generate_supercircuit_subcircuits(
    config: SearchConfig,
    backend: Optional[Backend] = None,
) -> List[ArchitectureSpec]:
    """Sample SubCircuit masks from a SuperCircuit search space."""
    blocks = supercircuit_blocks(config.candidate_layers)
    rng = np.random.default_rng(int(config.seed))
    target_count = max(0, int(config.population_size))
    candidates: List[ArchitectureSpec] = []
    seen: set[tuple[int, ...]] = set()

    # Include a few structured masks so the search space is not only random.
    seed_masks = [
        tuple(1 if block.name.startswith("rot") or block.name == "final_rot" else 1 for block in blocks),
        tuple(min(3, len(block.choices) - 1) for block in blocks),
    ]
    for mask in seed_masks:
        if len(candidates) >= target_count:
            break
        if mask in seen or all(block.choices[index] == "skip" for index, block in zip(mask, blocks)):
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
        if mask in seen or all(block.choices[index] == "skip" for index, block in zip(mask, blocks)):
            continue
        seen.add(mask)
        candidates.append(subcircuit_from_mask(config.n_qubits, mask, blocks, backend=backend))
    return candidates


__all__ = [
    "SuperCircuitBlock",
    "generate_supercircuit_subcircuits",
    "subcircuit_from_mask",
    "supercircuit_blocks",
]
