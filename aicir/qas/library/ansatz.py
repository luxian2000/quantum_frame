"""QAS ansatz encodings and circuit builders.

This module only maps architecture genes to ``ArchitectureSpec`` circuits; it
does not know about Hamiltonians, oracle scores, or VQE labels.  ``HEAMask`` is
kept for existing queues, while ``LayerwiseAnsatzGene`` is the larger per-layer
search space used by the current planner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
import hashlib
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


@dataclass(frozen=True)
class SupernetAnsatzGene:
    """Exact vqe_loop representation of a native :mod:`supernet` architecture."""

    n_qubits: int
    single_qubit_layers: tuple[tuple[str, ...], ...]
    two_qubit_layers: tuple[tuple[str, ...], ...]
    two_qubit_pairs: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_qubits", int(self.n_qubits))
        single_layers = tuple(tuple(str(gate).lower() for gate in layer) for layer in self.single_qubit_layers)
        two_layers = tuple(tuple(str(gate).lower() for gate in layer) for layer in self.two_qubit_layers)
        pairs = tuple((int(left), int(right)) for left, right in self.two_qubit_pairs)
        object.__setattr__(self, "single_qubit_layers", single_layers)
        object.__setattr__(self, "two_qubit_layers", two_layers)
        object.__setattr__(self, "two_qubit_pairs", pairs)
        if self.n_qubits < 1:
            raise ValueError("SupernetAnsatzGene requires at least one qubit")
        if len(single_layers) != len(two_layers):
            raise ValueError("single_qubit_layers and two_qubit_layers must have the same number of layers")
        for layer in single_layers:
            if len(layer) != self.n_qubits:
                raise ValueError("each single_qubit_layers entry must match n_qubits")
            for gate in layer:
                if gate not in {"i", "h", "rx", "ry", "rz"}:
                    raise ValueError(f"unsupported supernet single-qubit gate: {gate!r}")
        for layer in two_layers:
            if len(layer) != len(pairs):
                raise ValueError("each two_qubit_layers entry must match two_qubit_pairs")
            for gate in layer:
                if gate not in {"none", "cx", "rzz"}:
                    raise ValueError(f"unsupported supernet two-qubit gate: {gate!r}")
        for left, right in pairs:
            if not (0 <= left < self.n_qubits and 0 <= right < self.n_qubits):
                raise ValueError("two_qubit_pairs contain a qubit outside [0, n_qubits)")
            if left == right:
                raise ValueError("two_qubit_pairs cannot contain self edges")

    @property
    def layers(self) -> int:
        return len(self.single_qubit_layers)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "kind": "supernet_native",
            "n_qubits": self.n_qubits,
            "single_qubit_layers": [list(layer) for layer in self.single_qubit_layers],
            "two_qubit_layers": [list(layer) for layer in self.two_qubit_layers],
            "two_qubit_pairs": [list(pair) for pair in self.two_qubit_pairs],
        }

    @classmethod
    def from_jsonable(cls, raw: Any) -> "SupernetAnsatzGene":
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            raise ValueError("supernet ansatz gene must be a JSON object")
        if str(raw.get("kind", "supernet_native")).lower() != "supernet_native":
            raise ValueError("supernet ansatz gene kind must be 'supernet_native'")
        return cls(
            n_qubits=int(raw["n_qubits"]),
            single_qubit_layers=tuple(tuple(layer) for layer in raw["single_qubit_layers"]),
            two_qubit_layers=tuple(tuple(layer) for layer in raw["two_qubit_layers"]),
            two_qubit_pairs=tuple(tuple(pair) for pair in raw["two_qubit_pairs"]),
        )

    def label(self) -> str:
        payload = json.dumps(self.to_jsonable(), ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"supernet_native_L{self.layers}_{digest}"


@dataclass(frozen=True)
class ChemistryExcitationAnsatzGene:
    """Chemistry-preserving HF plus excitation sequence ansatz gene."""

    n_qubits: int
    hf_occupied_qubits: tuple[int, ...]
    excitations: tuple[dict[str, Any], ...]
    active_electrons: int | None = None
    active_spatial_orbitals: int | None = None
    name: str = "chemistry_excitation"

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_qubits", int(self.n_qubits))
        if self.n_qubits < 1:
            raise ValueError("ChemistryExcitationAnsatzGene requires at least one qubit")
        occupied = tuple(int(qubit) for qubit in self.hf_occupied_qubits)
        if len(set(occupied)) != len(occupied):
            raise ValueError("hf_occupied_qubits contains duplicate qubits")
        for qubit in occupied:
            if not (0 <= qubit < self.n_qubits):
                raise ValueError("hf_occupied_qubits contains a qubit outside [0, n_qubits)")
        normalized: list[dict[str, Any]] = []
        for excitation in self.excitations:
            if not isinstance(excitation, dict):
                raise ValueError("chemistry excitations must be JSON objects")
            gate_type = str(excitation.get("type", "")).strip().lower()
            qubits = tuple(int(qubit) for qubit in excitation.get("qubits", ()))
            expected_width = 2 if gate_type == "single_excitation" else 4 if gate_type == "double_excitation" else 0
            if expected_width == 0:
                raise ValueError(f"unsupported chemistry excitation type: {gate_type!r}")
            if len(qubits) != expected_width:
                raise ValueError(f"{gate_type} requires {expected_width} qubits")
            if len(set(qubits)) != len(qubits):
                raise ValueError(f"{gate_type} cannot repeat qubits")
            for qubit in qubits:
                if not (0 <= qubit < self.n_qubits):
                    raise ValueError(f"{gate_type} contains a qubit outside [0, n_qubits)")
            normalized.append({"type": gate_type, "qubits": list(qubits)})
        object.__setattr__(self, "hf_occupied_qubits", occupied)
        object.__setattr__(self, "excitations", tuple(normalized))
        object.__setattr__(self, "name", str(self.name or "chemistry_excitation"))
        if self.active_electrons is not None:
            object.__setattr__(self, "active_electrons", int(self.active_electrons))
        if self.active_spatial_orbitals is not None:
            object.__setattr__(self, "active_spatial_orbitals", int(self.active_spatial_orbitals))

    @property
    def layers(self) -> int:
        return len(self.excitations)

    def to_jsonable(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": "chemistry_excitation",
            "n_qubits": self.n_qubits,
            "hf_occupied_qubits": list(self.hf_occupied_qubits),
            "excitations": [dict(excitation) for excitation in self.excitations],
            "name": self.name,
        }
        if self.active_electrons is not None:
            payload["active_electrons"] = int(self.active_electrons)
        if self.active_spatial_orbitals is not None:
            payload["active_spatial_orbitals"] = int(self.active_spatial_orbitals)
        return payload

    @classmethod
    def from_jsonable(cls, raw: Any) -> "ChemistryExcitationAnsatzGene":
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            raise ValueError("chemistry excitation ansatz gene must be a JSON object")
        if str(raw.get("kind", "chemistry_excitation")).lower() != "chemistry_excitation":
            raise ValueError("chemistry excitation ansatz gene kind must be 'chemistry_excitation'")
        return cls(
            n_qubits=int(raw["n_qubits"]),
            hf_occupied_qubits=tuple(int(qubit) for qubit in raw.get("hf_occupied_qubits", ())),
            excitations=tuple(dict(excitation) for excitation in raw.get("excitations", ())),
            active_electrons=None if raw.get("active_electrons") is None else int(raw.get("active_electrons")),
            active_spatial_orbitals=None
            if raw.get("active_spatial_orbitals") is None
            else int(raw.get("active_spatial_orbitals")),
            name=str(raw.get("name", "chemistry_excitation")),
        )

    def label(self) -> str:
        payload = json.dumps(self.to_jsonable(), ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"chemistry_excitation_L{self.layers}_{digest}"

@dataclass(frozen=True)
class ExplicitGateAnsatzGene:
    """Task-agnostic ansatz gene storing an already decomposed gate sequence.

    This is the bridge format for external architecture generators such as
    ADAPT-VQE.  It deliberately stores gates in the same dictionary format used
    by :class:`ArchitectureSpec` so fair labeling can evaluate the circuit
    without forcing it into a layerwise or supernet template.
    """

    n_qubits: int
    gates: tuple[dict[str, Any], ...]
    name: str = "explicit_gate_sequence"

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_qubits", int(self.n_qubits))
        if self.n_qubits < 1:
            raise ValueError("ExplicitGateAnsatzGene requires at least one qubit")
        copied_gates: list[dict[str, Any]] = []
        for gate in self.gates:
            if not isinstance(gate, dict):
                raise ValueError("explicit gate entries must be JSON objects")
            if not str(gate.get("type", "")).strip():
                raise ValueError("explicit gate entries require a non-empty type")
            copied_gates.append(json.loads(json.dumps(gate)))
        object.__setattr__(self, "gates", tuple(copied_gates))
        object.__setattr__(self, "name", str(self.name or "explicit_gate_sequence"))

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "kind": "explicit_gate_sequence",
            "n_qubits": self.n_qubits,
            "gates": [dict(gate) for gate in self.gates],
            "name": self.name,
        }

    @classmethod
    def from_jsonable(cls, raw: Any) -> "ExplicitGateAnsatzGene":
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            raise ValueError("explicit gate ansatz gene must be a JSON object")
        if str(raw.get("kind", "explicit_gate_sequence")).lower() != "explicit_gate_sequence":
            raise ValueError("explicit gate ansatz gene kind must be 'explicit_gate_sequence'")
        return cls(
            n_qubits=int(raw["n_qubits"]),
            gates=tuple(dict(gate) for gate in raw.get("gates", ())),
            name=str(raw.get("name", "explicit_gate_sequence")),
        )

    def label(self) -> str:
        payload = json.dumps(self.to_jsonable(), ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"explicit_gate_sequence_{digest}"


@dataclass(frozen=True)
class OperatorSequenceAnsatzGene:
    """ADAPT-style variable-length ansatz represented as Pauli evolutions.

    Each entry in ``operators`` is a Pauli string over ``I/X/Y/Z`` and maps to
    one trainable Pauli-evolution angle in the generated architecture.
    """

    n_qubits: int
    operators: tuple[str, ...]
    name: str = "operator_sequence"

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_qubits", int(self.n_qubits))
        if self.n_qubits < 1:
            raise ValueError("OperatorSequenceAnsatzGene requires at least one qubit")
        normalized: list[str] = []
        for operator in self.operators:
            pauli = str(operator).strip().upper()
            if len(pauli) != self.n_qubits:
                raise ValueError("operator Pauli string width must match n_qubits")
            if any(symbol not in {"I", "X", "Y", "Z"} for symbol in pauli):
                raise ValueError(f"unsupported Pauli symbol in operator: {operator!r}")
            if all(symbol == "I" for symbol in pauli):
                raise ValueError("operator sequence cannot contain the identity-only Pauli string")
            normalized.append(pauli)
        object.__setattr__(self, "operators", tuple(normalized))
        object.__setattr__(self, "name", str(self.name or "operator_sequence"))

    @property
    def layers(self) -> int:
        return len(self.operators)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "kind": "operator_sequence",
            "n_qubits": self.n_qubits,
            "operators": list(self.operators),
            "name": self.name,
        }

    @classmethod
    def from_jsonable(cls, raw: Any) -> "OperatorSequenceAnsatzGene":
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            raise ValueError("operator sequence ansatz gene must be a JSON object")
        if str(raw.get("kind", "operator_sequence")).lower() != "operator_sequence":
            raise ValueError("operator sequence ansatz gene kind must be 'operator_sequence'")
        return cls(
            n_qubits=int(raw["n_qubits"]),
            operators=tuple(str(operator) for operator in raw.get("operators", ())),
            name=str(raw.get("name", "operator_sequence")),
        )

    def label(self) -> str:
        payload = json.dumps(self.to_jsonable(), ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"operator_sequence_L{self.layers}_{digest}"


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


def architecture_from_supernet_gene(
    gene: SupernetAnsatzGene,
    backend: Optional[NumpyBackend] = None,
) -> ArchitectureSpec:
    gates: List[Dict[str, Any]] = []
    cursor = [0]
    for single_layer, two_layer in zip(gene.single_qubit_layers, gene.two_qubit_layers):
        for qubit, gate_type in enumerate(single_layer):
            if gate_type == "i":
                continue
            if gate_type == "h":
                gates.append({"type": "h", "target_qubit": qubit})
                continue
            cursor[0] += 1
            gates.append({"type": gate_type, "target_qubit": qubit, "parameter": 0.071 * cursor[0]})
        for (left, right), gate_type in zip(gene.two_qubit_pairs, two_layer):
            if gate_type == "none":
                continue
            if gate_type == "rzz":
                cursor[0] += 1
                gates.append({"type": "rzz", "qubit_1": left, "qubit_2": right, "parameter": 0.071 * cursor[0]})
            elif gate_type == "cx":
                gates.append(
                    {
                        "type": "cx",
                        "target_qubit": right,
                        "control_qubits": [left],
                        "control_states": [1],
                    }
                )
            else:
                raise ValueError(f"unsupported supernet two-qubit gate: {gate_type!r}")
    return ArchitectureSpec.from_gates(
        name=gene.label(),
        gates=gates,
        n_qubits=gene.n_qubits,
        backend=backend,
        description="Native supernet-ranked ansatz gene.",
        tags=["VQE", "supernet-native"],
        metadata={
            "ansatz_gene": gene.to_jsonable(),
            "family": "supernet_native",
            "layers": gene.layers,
            "topology": "supernet_pairs",
        },
    )


def architecture_from_chemistry_excitation_gene(
    gene: ChemistryExcitationAnsatzGene,
    backend: Optional[NumpyBackend] = None,
) -> ArchitectureSpec:
    gates: List[Dict[str, Any]] = []
    cursor = [0]
    for qubit in gene.hf_occupied_qubits:
        gates.append({"type": "pauli_x", "target_qubit": int(qubit)})
    for excitation in gene.excitations:
        cursor[0] += 1
        gate_type = str(excitation["type"])
        qubits = [int(qubit) for qubit in excitation["qubits"]]
        if gate_type == "single_excitation":
            gates.append(
                {
                    "type": "single_excitation",
                    "qubit_1": qubits[0],
                    "qubit_2": qubits[1],
                    "qubits": qubits,
                    "parameter": 0.071 * cursor[0],
                }
            )
        elif gate_type == "double_excitation":
            gates.append(
                {
                    "type": "double_excitation",
                    "qubits": qubits,
                    "parameter": 0.071 * cursor[0],
                }
            )
        else:
            raise ValueError(f"unsupported chemistry excitation type: {gate_type!r}")
    return ArchitectureSpec.from_gates(
        name=gene.label(),
        gates=gates,
        n_qubits=gene.n_qubits,
        backend=backend,
        description="Hartree-Fock initialized chemistry excitation ansatz gene.",
        tags=["VQE", "chemistry-excitation"],
        metadata={
            "ansatz_gene": gene.to_jsonable(),
            "family": "chemistry_excitation",
            "layers": gene.layers,
            "topology": "chemistry_excitation_pool",
            "hf_occupied_qubits": list(gene.hf_occupied_qubits),
            "active_electrons": gene.active_electrons,
            "active_spatial_orbitals": gene.active_spatial_orbitals,
        },
    )

def architecture_from_explicit_gate_gene(
    gene: ExplicitGateAnsatzGene,
    backend: Optional[NumpyBackend] = None,
) -> ArchitectureSpec:
    return ArchitectureSpec.from_gates(
        name=gene.label(),
        gates=gene.gates,
        n_qubits=gene.n_qubits,
        backend=backend,
        description="Explicit gate-sequence ansatz gene.",
        tags=["QAS", "explicit-gate-sequence"],
        metadata={
            "ansatz_gene": gene.to_jsonable(),
            "family": "explicit_gate_sequence",
            "topology": "explicit",
        },
    )


def _append_cx(gates: List[Dict[str, Any]], control: int, target: int) -> None:
    gates.append({"type": "cx", "target_qubit": target, "control_qubits": [control], "control_states": [1]})


def _append_pauli_basis_forward(gates: List[Dict[str, Any]], symbol: str, qubit: int) -> None:
    if symbol == "X":
        gates.append({"type": "hadamard", "target_qubit": qubit})
    elif symbol == "Y":
        gates.extend({"type": "s_gate", "target_qubit": qubit} for _ in range(3))
        gates.append({"type": "hadamard", "target_qubit": qubit})


def _append_pauli_basis_inverse(gates: List[Dict[str, Any]], symbol: str, qubit: int) -> None:
    if symbol == "X":
        gates.append({"type": "hadamard", "target_qubit": qubit})
    elif symbol == "Y":
        gates.append({"type": "hadamard", "target_qubit": qubit})
        gates.append({"type": "s_gate", "target_qubit": qubit})


def _append_pauli_evolution(gates: List[Dict[str, Any]], pauli: str, theta: float) -> None:
    active = [(index, symbol) for index, symbol in enumerate(pauli) if symbol != "I"]
    if not active:
        return
    for qubit, symbol in active:
        _append_pauli_basis_forward(gates, symbol, qubit)
    pivot = active[-1][0]
    for qubit, _symbol in active[:-1]:
        _append_cx(gates, qubit, pivot)
    gates.append({"type": "rz", "target_qubit": pivot, "parameter": float(theta)})
    for qubit, _symbol in reversed(active[:-1]):
        _append_cx(gates, qubit, pivot)
    for qubit, symbol in reversed(active):
        _append_pauli_basis_inverse(gates, symbol, qubit)


def architecture_from_operator_sequence_gene(
    gene: OperatorSequenceAnsatzGene,
    backend: Optional[NumpyBackend] = None,
) -> ArchitectureSpec:
    gates: List[Dict[str, Any]] = []
    for index, pauli in enumerate(gene.operators, start=1):
        _append_pauli_evolution(gates, pauli, 0.071 * index)
    return ArchitectureSpec.from_gates(
        name=gene.label(),
        gates=gates,
        n_qubits=gene.n_qubits,
        backend=backend,
        description="Variable-length ADAPT-style Pauli operator-sequence ansatz gene.",
        tags=["QAS", "operator-sequence", "ADAPT-style"],
        metadata={
            "ansatz_gene": gene.to_jsonable(),
            "family": "operator_sequence",
            "layers": gene.layers,
            "topology": "pauli_operator_sequence",
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
    "ChemistryExcitationAnsatzGene",
    "ExplicitGateAnsatzGene",
    "FINAL_ROTATIONS",
    "HEAMask",
    "LAYERWISE_SINGLE_BLOCKS",
    "LAYERWISE_TWO_QUBIT_GATES",
    "LAYER_CHOICES",
    "LayerwiseAnsatzGene",
    "OperatorSequenceAnsatzGene",
    "SupernetAnsatzGene",
    "ROTATION_BLOCKS",
    "architecture_from_chemistry_excitation_gene",
    "architecture_from_explicit_gate_gene",
    "architecture_from_hea_mask",
    "architecture_from_layerwise_gene",
    "architecture_from_operator_sequence_gene",
    "architecture_from_supernet_gene",
    "enumerate_hea_masks",
    "sample_layerwise_genes",
]



