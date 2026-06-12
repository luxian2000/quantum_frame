"""Typed observable IR for Pauli, Hamiltonian, and dense-matrix observables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from ..channel.operators import Hamiltonian, PauliOp, PauliString


_KIND_ALIASES = {
    "pauli": "pauli",
    "pauli_string": "pauli",
    "paulistring": "pauli",
    "hamiltonian": "hamiltonian",
    "matrix": "matrix",
    "dense_matrix": "matrix",
    "dense": "matrix",
}


def _normalize_kind(kind: str) -> str:
    normalized = str(kind).strip().lower()
    if normalized not in _KIND_ALIASES:
        raise ValueError("Observable kind must be 'pauli', 'hamiltonian', or 'matrix'")
    return _KIND_ALIASES[normalized]


def _infer_matrix_n_qubits(matrix: Any) -> int:
    shape = getattr(matrix, "shape", None)
    if shape is None:
        shape = np.asarray(matrix).shape
    shape = tuple(int(dim) for dim in shape)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("matrix observable must be a square matrix")
    dim = shape[0]
    if dim <= 0:
        raise ValueError("matrix observable dimension must be positive")
    n_qubits = int(round(math.log2(dim)))
    if (1 << n_qubits) != dim:
        raise ValueError("matrix observable dimension must be a power of two")
    return n_qubits


@dataclass(frozen=True)
class Observable:
    """Typed descriptor for an observable used by estimators and measurements."""

    kind: str
    value: Any
    n_qubits: int | None = None
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = _normalize_kind(self.kind)
        n_qubits = self.n_qubits
        if n_qubits is None:
            if kind == "matrix":
                n_qubits = _infer_matrix_n_qubits(self.value)
            elif hasattr(self.value, "n_qubits"):
                n_qubits = int(self.value.n_qubits)
        if n_qubits is not None:
            n_qubits = int(n_qubits)
            if n_qubits <= 0:
                raise ValueError("n_qubits must be a positive integer")

        name = None if self.name is None else str(self.name)
        if name is not None and not name:
            raise ValueError("name cannot be empty")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "n_qubits", n_qubits)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def pauli(
        cls,
        paulistring: str | Mapping[str, Sequence[int]],
        coefficient: complex = 1.0,
        n_qubits: int | None = None,
        *,
        qubits: Sequence[int] | None = None,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Observable":
        """Build a Pauli-string observable."""

        value = PauliString(
            paulistring,
            coefficient=coefficient,
            n_qubits=n_qubits,
            qubits=qubits,
        )
        return cls("pauli", value, n_qubits=value.n_qubits, name=name, metadata=metadata or {})

    @classmethod
    def hamiltonian(
        cls,
        hamiltonian: Hamiltonian,
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Observable":
        """Wrap an existing Hamiltonian observable."""

        if not isinstance(hamiltonian, Hamiltonian):
            raise TypeError("Observable.hamiltonian expects a Hamiltonian")
        return cls(
            "hamiltonian",
            hamiltonian,
            n_qubits=hamiltonian.n_qubits,
            name=name,
            metadata=metadata or {},
        )

    @classmethod
    def matrix(
        cls,
        matrix: Any,
        *,
        n_qubits: int | None = None,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Observable":
        """Build a dense-matrix observable."""

        inferred = _infer_matrix_n_qubits(matrix)
        if n_qubits is not None and int(n_qubits) != inferred:
            raise ValueError("n_qubits does not match matrix dimension")
        return cls("matrix", matrix, n_qubits=inferred, name=name, metadata=metadata or {})

    @classmethod
    def from_object(
        cls,
        value: Any,
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Observable":
        """Wrap a supported existing observable object."""

        if isinstance(value, Observable):
            if name is None and metadata is None:
                return value
            return cls(value.kind, value.value, value.n_qubits, name=name or value.name, metadata=metadata or value.metadata)
        if isinstance(value, Hamiltonian):
            return cls.hamiltonian(value, name=name, metadata=metadata)
        if isinstance(value, PauliString):
            return cls("pauli", value, n_qubits=value.n_qubits, name=name, metadata=metadata or {})
        if isinstance(value, PauliOp):
            return cls("pauli", value, n_qubits=value.qubit + 1, name=name, metadata=metadata or {})
        return cls.matrix(value, name=name, metadata=metadata)

    def to_operator(self) -> Any:
        """Return the wrapped existing observable object."""

        return self.value

    def to_matrix(self, backend):
        """Convert this observable to a backend-native dense matrix."""

        if self.kind == "matrix":
            return backend.cast(self.value)
        if isinstance(self.value, PauliOp):
            if self.n_qubits is None:
                raise ValueError("n_qubits is required for PauliOp observables")
            return self.value.to_matrix(self.n_qubits, backend)
        return self.value.to_matrix(backend)
