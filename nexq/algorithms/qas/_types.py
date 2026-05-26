"""Shared data types for QAS architecture search and scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ...channel.backends.base import Backend
from ...core.circuit import Circuit
from ._utils import count_parameters, count_two_qubit_gates


@dataclass
class MetricDefinition:
    """Metadata for one metric option inside a metric group."""

    name: str
    purpose: str
    status: str = "todo"
    active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "status": self.status,
            "active": self.active,
        }


@dataclass
class MetricGroupScore:
    """Score for one orthogonal objective group."""

    name: str
    active_metric: str
    metrics: List[MetricDefinition]
    score: float
    raw_values: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "active_metric": self.active_metric,
            "score": self.score,
            "raw_values": dict(self.raw_values),
            "notes": list(self.notes),
            "metrics": [metric.to_dict() for metric in self.metrics],
        }


@dataclass
class ArchitectureSpec:
    """A candidate quantum architecture/ansatz to be scored by QAS."""

    name: str
    circuit: Circuit
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_gates(
        cls,
        name: str,
        gates: Sequence[Dict[str, Any]],
        n_qubits: int,
        backend: Optional[Backend] = None,
        **kwargs: Any,
    ) -> "ArchitectureSpec":
        return cls(
            name=name,
            circuit=Circuit(*[dict(gate) for gate in gates], n_qubits=n_qubits, backend=backend),
            **kwargs,
        )

    @property
    def n_qubits(self) -> int:
        return int(self.circuit.n_qubits)

    @property
    def n_gates(self) -> int:
        return len(self.circuit.gates)

    @property
    def two_qubit_gate_count(self) -> int:
        return count_two_qubit_gates(self.circuit)

    @property
    def parameter_count(self) -> int:
        return count_parameters(self.circuit)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "n_qubits": self.n_qubits,
            "n_gates": self.n_gates,
            "n_parameters": self.parameter_count,
            "two_qubit_gate_count": self.two_qubit_gate_count,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


@dataclass
class ArchitectureScore:
    """Unified score report for one candidate architecture."""

    architecture: ArchitectureSpec
    expressibility: MetricGroupScore
    trainability: MetricGroupScore
    noise_robustness: MetricGroupScore
    hardware_efficiency: MetricGroupScore
    weights: Dict[str, float]
    weighted_score: float
    rank: Optional[int] = None

    def groups(self) -> Dict[str, MetricGroupScore]:
        return {
            "expressibility": self.expressibility,
            "trainability": self.trainability,
            "noise_robustness": self.noise_robustness,
            "hardware_efficiency": self.hardware_efficiency,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "architecture": self.architecture.to_dict(),
            "expressibility": self.expressibility.to_dict(),
            "trainability": self.trainability.to_dict(),
            "noise_robustness": self.noise_robustness.to_dict(),
            "hardware_efficiency": self.hardware_efficiency.to_dict(),
            "weights": dict(self.weights),
            "weighted_score": self.weighted_score,
        }

    def to_row(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "name": self.architecture.name,
            "n_qubits": self.architecture.n_qubits,
            "n_gates": self.architecture.n_gates,
            "n_parameters": self.architecture.parameter_count,
            "two_qubit_gate_count": self.architecture.two_qubit_gate_count,
            "expressibility": self.expressibility.score,
            "trainability": self.trainability.score,
            "noise_robustness": self.noise_robustness.score,
            "hardware_efficiency": self.hardware_efficiency.score,
            "weighted_score": self.weighted_score,
        }


@dataclass
class SearchConfig:
    """Configuration for architecture candidate generation and scoring."""

    n_qubits: int = 4
    candidate_layers: int = 2
    n_samples: int = 200
    include_common_candidates: bool = True
    active_metrics: Dict[str, str] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Result of an architecture-search run."""

    candidates: List[ArchitectureSpec]
    scores: List[ArchitectureScore]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def best(self) -> Optional[ArchitectureScore]:
        return self.scores[0] if self.scores else None


__all__ = [
    "MetricDefinition",
    "MetricGroupScore",
    "ArchitectureSpec",
    "ArchitectureScore",
    "SearchConfig",
    "SearchResult",
]
