"""QAS ansatz encodings and circuit builders.

This module only maps architecture genes to ``ArchitectureSpec`` circuits; it
does not know about Hamiltonians, oracle scores, or VQE labels.  ``HEAMask`` is
kept for existing queues, while ``LayerwiseAnsatzGene`` is the larger per-layer
search space used by the current planner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ...backends.numpy_backend import NumpyBackend
from ..core._types import ArchitectureSpec


ROTATION_BLOCKS = ("ry", "ry_rz", "rx_ry_rz")
ENTANGLERS = ("cx", "cz", "rzz")
LAYERWISE_SINGLE_BLOCKS = ("none", "rx", "ry", "rz", "ry_rz", "rx_ry_rz")
LAYERWISE_TWO_QUBIT_GATES = ("none", "cx", "cz", "rxx", "rzz")
FINAL_ROTATIONS = ("ry", "ry_rz")
ENTANGLE_PATTERNS = ("linear", "ring")
LAYER_CHOICES = (1, 2, 3)


@dataclass(frozen=True)
class HEAMask:
    """Compact legacy HEA mask used by Stage-0 coverage and existing queues."""

    n_qubits: int = 2
    layers: int = 1
    rotation_block: str = "ry_rz"
    entangler: str = "cx"
    final_rotation: str = "ry"
    entangle_pattern: str = "linear"

    def key(self) -> tuple[Any, ...]:
        return (
            self.n_qubits,
            self.layers,
            self.rotation_block,
            self.entangler,
            self.final_rotation,
            self.entangle_pattern,
        )

    def label(self) -> str:
        return (
            f"hea_mask_L{self.layers}_{self.rotation_block}_{self.entangler}_"
            f"{self.entangle_pattern}_{self.final_rotation}"
        )


@dataclass(frozen=True)
class LayerwiseAnsatzGene:
    """Layer-wise ansatz gene with independent per-layer and per-edge choices."""

    n_qubits: int
    single_blocks: tuple[str, ...]
    edge_entanglers: tuple[tuple[str, ...], ...]
    entangle_pattern: str = "linear"

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_qubits", int(self.n_qubits))
        single_blocks = tuple(str(item).lower() for item in self.single_blocks)
        edge_entanglers = tuple(tuple(str(item).lower() for item in layer) for layer in self.edge_entanglers)
        object.__setattr__(self, "single_blocks", single_blocks)
        object.__setattr__(self, "edge_entanglers", edge_entanglers)
        object.__setattr__(self, "entangle_pattern", str(self.entangle_pattern).lower())
        if self.n_qubits < 1:
            raise ValueError("LayerwiseAnsatzGene requires at least one qubit")
        if len(single_blocks) != len(edge_entanglers) + 1:
            raise ValueError("single_blocks must contain one entry per entangler layer plus a final block")
        edge_count = len(_edges(self.n_qubits, self.entangle_pattern))
        for block in single_blocks:
            if block not in LAYERWISE_SINGLE_BLOCKS:
                raise ValueError(f"unsupported single block: {block!r}")
        for layer in edge_entanglers:
            if len(layer) != edge_count:
                raise ValueError("each edge_entanglers layer must match the topology edge count")
            for gate in layer:
                if gate not in LAYERWISE_TWO_QUBIT_GATES:
                    raise ValueError(f"unsupported two-qubit gate: {gate!r}")

    @property
    def layers(self) -> int:
        return len(self.edge_entanglers)

    def key(self) -> tuple[Any, ...]:
        return (
            self.n_qubits,
            self.entangle_pattern,
            self.single_blocks,
            self.edge_entanglers,
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "single_blocks": list(self.single_blocks),
            "edge_entanglers": [list(layer) for layer in self.edge_entanglers],
            "entangle_pattern": self.entangle_pattern,
        }

    @classmethod
    def from_jsonable(cls, raw: Any) -> "LayerwiseAnsatzGene":
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            raise ValueError("layerwise ansatz gene must be a JSON object")
        return cls(
            n_qubits=int(raw["n_qubits"]),
            single_blocks=tuple(raw["single_blocks"]),
            edge_entanglers=tuple(tuple(layer) for layer in raw["edge_entanglers"]),
            entangle_pattern=str(raw.get("entangle_pattern", "linear")),
        )

    def label(self) -> str:
        singles = "-".join(self.single_blocks)
        twoq = "-".join("_".join(layer) for layer in self.edge_entanglers)
        return f"layerwise_L{self.layers}_{self.entangle_pattern}_{singles}_{twoq}"


def enumerate_hea_masks(n_qubits: int = 2) -> List[HEAMask]:
    masks = []
    for layers in LAYER_CHOICES:
        for rotation in ROTATION_BLOCKS:
            for entangler in ENTANGLERS:
                for final_rotation in FINAL_ROTATIONS:
                    for pattern in ENTANGLE_PATTERNS:
                        masks.append(
                            HEAMask(
                                n_qubits=n_qubits,
                                layers=layers,
                                rotation_block=rotation,
                                entangler=entangler,
                                final_rotation=final_rotation,
                                entangle_pattern=pattern,
                            )
                        )
    return masks


def sample_layerwise_genes(
    n_qubits: int = 4,
    layers: int = 3,
    count: int = 16,
    seed: int = 2026,
    entangle_pattern: str = "linear",
    single_blocks: Sequence[str] = ("rx", "ry", "rz", "ry_rz", "rx_ry_rz"),
    two_qubit_gates: Sequence[str] = ("none", "cx", "cz", "rxx", "rzz"),
) -> List[LayerwiseAnsatzGene]:
    """Sample a bounded layer-wise gene pool without enumerating the huge product space."""

    rng = np.random.default_rng(int(seed))
    edge_count = len(_edges(int(n_qubits), entangle_pattern))
    singles = tuple(str(item).lower() for item in single_blocks)
    twoq = tuple(str(item).lower() for item in two_qubit_gates)
    genes: list[LayerwiseAnsatzGene] = []
    seen: set[tuple[Any, ...]] = set()

    def add(gene: LayerwiseAnsatzGene) -> None:
        key = gene.key()
        if key not in seen:
            genes.append(gene)
            seen.add(key)

    seed_patterns = [
        (("rx_ry_rz", "ry_rz", "rx_ry_rz", "ry_rz"), ("rxx", "rzz", "cz")),
        (("ry_rz", "rx", "rz", "rx_ry_rz"), ("rzz", "rxx", "cx")),
        (("rx", "ry", "rz", "ry_rz"), ("rxx", "none", "rzz")),
    ]
    for single_pattern, edge_cycle in seed_patterns:
        if len(single_pattern) != int(layers) + 1:
            single_pattern = tuple((list(single_pattern) * (int(layers) + 1))[: int(layers) + 1])
        edge_layers = []
        for layer_index in range(int(layers)):
            edge_layers.append(
                tuple(edge_cycle[(layer_index + edge_index) % len(edge_cycle)] for edge_index in range(edge_count))
            )
        add(
            LayerwiseAnsatzGene(
                n_qubits=int(n_qubits),
                single_blocks=tuple(single_pattern),
                edge_entanglers=tuple(edge_layers),
                entangle_pattern=entangle_pattern,
            )
        )

    attempts = 0
    max_attempts = max(100, int(count) * 100)
    while len(genes) < int(count) and attempts < max_attempts:
        attempts += 1
        sampled_singles = tuple(str(rng.choice(singles)) for _ in range(int(layers) + 1))
        sampled_edges = tuple(
            tuple(str(rng.choice(twoq)) for _edge in range(edge_count))
            for _layer in range(int(layers))
        )
        add(
            LayerwiseAnsatzGene(
                n_qubits=int(n_qubits),
                single_blocks=sampled_singles,
                edge_entanglers=sampled_edges,
                entangle_pattern=entangle_pattern,
            )
        )
    return genes[: max(0, int(count))]


def _edges(n_qubits: int, pattern: str) -> List[tuple[int, int]]:
    edges = [(i, i + 1) for i in range(max(0, n_qubits - 1))]
    if pattern == "ring":
        if n_qubits == 2:
            edges.append((1, 0))
        elif n_qubits > 2:
            edges.append((n_qubits - 1, 0))
    return edges


def _append_rotation(gates: List[Dict[str, Any]], n_qubits: int, block: str, cursor: List[int]) -> None:
    rotations = {
        "none": (),
        "ry": ("ry",),
        "rx": ("rx",),
        "rz": ("rz",),
        "ry_rz": ("ry", "rz"),
        "rx_ry_rz": ("rx", "ry", "rz"),
    }[block]
    for qubit in range(n_qubits):
        for gate_type in rotations:
            cursor[0] += 1
            gates.append({"type": gate_type, "target_qubit": qubit, "parameter": 0.071 * cursor[0]})


def _append_entangler(gates: List[Dict[str, Any]], edges: Sequence[tuple[int, int]], entangler: str, cursor: List[int]) -> None:
    for control, target in edges:
        if entangler in {"none", "skip", "identity"}:
            continue
        if entangler in {"rzz", "rxx"}:
            cursor[0] += 1
            gates.append({"type": entangler, "qubit_1": control, "qubit_2": target, "parameter": 0.071 * cursor[0]})
        else:
            gates.append(
                {
                    "type": entangler,
                    "target_qubit": target,
                    "control_qubits": [control],
                    "control_states": [1],
                }
            )


def architecture_from_layerwise_gene(
    gene: LayerwiseAnsatzGene,
    backend: Optional[NumpyBackend] = None,
) -> ArchitectureSpec:
    gates: List[Dict[str, Any]] = []
    cursor = [0]
    edges = _edges(gene.n_qubits, gene.entangle_pattern)
    for layer_index, edge_choices in enumerate(gene.edge_entanglers):
        _append_rotation(gates, gene.n_qubits, gene.single_blocks[layer_index], cursor)
        for edge, entangler in zip(edges, edge_choices):
            _append_entangler(gates, [edge], entangler, cursor)
    _append_rotation(gates, gene.n_qubits, gene.single_blocks[-1], cursor)
    return ArchitectureSpec.from_gates(
        name=gene.label(),
        gates=gates,
        n_qubits=gene.n_qubits,
        backend=backend,
        description="Layer-wise VQE-QAS ansatz gene with independent edge entanglers.",
        tags=["VQE", "layerwise-gene", gene.entangle_pattern],
        metadata={
            "ansatz_gene": gene.to_jsonable(),
            "family": "layerwise_gene",
            "layers": gene.layers,
            "topology": gene.entangle_pattern,
        },
    )


def architecture_from_hea_mask(mask: HEAMask, backend: Optional[NumpyBackend] = None) -> ArchitectureSpec:
    gates: List[Dict[str, Any]] = []
    cursor = [0]
    edges = _edges(mask.n_qubits, mask.entangle_pattern)
    for _ in range(mask.layers):
        _append_rotation(gates, mask.n_qubits, mask.rotation_block, cursor)
        _append_entangler(gates, edges, mask.entangler, cursor)
    _append_rotation(gates, mask.n_qubits, mask.final_rotation, cursor)
    return ArchitectureSpec.from_gates(
        name=mask.label(),
        gates=gates,
        n_qubits=mask.n_qubits,
        backend=backend,
        description="HEA mask candidate for VQE-QAS.",
        tags=["VQE", "HEA"],
        metadata={"hea_mask": mask.key(), "family": "HEA-mask"},
    )


__all__ = [
    "ENTANGLERS",
    "ENTANGLE_PATTERNS",
    "FINAL_ROTATIONS",
    "HEAMask",
    "LAYERWISE_SINGLE_BLOCKS",
    "LAYERWISE_TWO_QUBIT_GATES",
    "LAYER_CHOICES",
    "LayerwiseAnsatzGene",
    "ROTATION_BLOCKS",
    "architecture_from_hea_mask",
    "architecture_from_layerwise_gene",
    "enumerate_hea_masks",
    "sample_layerwise_genes",
]
