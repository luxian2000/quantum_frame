from __future__ import annotations

from dataclasses import dataclass

from .registry import VariableMetadata


@dataclass(frozen=True)
class IsingModel:
    h: dict[int, float]
    J: dict[tuple[int, int], float]
    offset: float = 0.0
    variable_names: list[str] | None = None
    variable_metadata: list[VariableMetadata] | None = None

    def named(self) -> dict[str, object]:
        if self.variable_names is None:
            raise ValueError("Named Ising output requires variable_names.")
        return {
            "h": {self.variable_names[index]: coeff for index, coeff in self.h.items()},
            "J": {
                (self.variable_names[left], self.variable_names[right]): coeff
                for (left, right), coeff in self.J.items()
            },
            "offset": self.offset,
        }

    def to_qaoa_terms(self) -> list[QAOATerm]:
        terms = [QAOATerm((index,), coeff) for index, coeff in sorted(self.h.items())]
        terms.extend(QAOATerm((left, right), coeff) for (left, right), coeff in sorted(self.J.items()))
        return terms


@dataclass(frozen=True)
class QAOATerm:
    qubits: tuple[int, ...]
    coefficient: float

    @property
    def pauli(self) -> str:
        return "Z" * len(self.qubits)


def qubo_to_ising_indices(
    qubo: dict[tuple[int, int], float],
    offset: float = 0.0,
    variable_names: list[str] | None = None,
    variable_metadata: list[VariableMetadata] | None = None,
) -> IsingModel:
    h: dict[int, float] = {}
    couplings: dict[tuple[int, int], float] = {}
    ising_offset = offset

    for (left, right), coeff in qubo.items():
        if left == right:
            h[left] = h.get(left, 0.0) + coeff / 2.0
            ising_offset += coeff / 2.0
        else:
            key = (left, right) if left <= right else (right, left)
            h[left] = h.get(left, 0.0) + coeff / 4.0
            h[right] = h.get(right, 0.0) + coeff / 4.0
            couplings[key] = couplings.get(key, 0.0) + coeff / 4.0
            ising_offset += coeff / 4.0

    return IsingModel(
        h={key: value for key, value in h.items() if abs(value) > 1e-12},
        J={key: value for key, value in couplings.items() if abs(value) > 1e-12},
        offset=ising_offset,
        variable_names=variable_names,
        variable_metadata=variable_metadata,
    )

