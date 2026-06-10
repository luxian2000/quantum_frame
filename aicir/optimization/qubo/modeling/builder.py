from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Iterable

from .backends import IsingModel, QAOATerm, qubo_to_ising_indices
from .integer import bounded_log_weights
from .matrix import SparseMatrixCOO
from .polynomial import Polynomial
from .registry import GLOBAL_REGISTRY, VariableRegistry


@dataclass
class QuboBuilder:
    """Low-level sparse QUBO coefficient builder.

    The public expression API is convenient for research code. This builder is
    the lower-level path for high-throughput model generation and framework
    adapters that already know the variable IDs they want to write.
    """

    registry: VariableRegistry = field(default_factory=lambda: GLOBAL_REGISTRY)
    qubo: dict[tuple[int, int], float] = field(default_factory=dict)
    offset: float = 0.0

    def add_offset(self, value: float) -> None:
        self.offset += float(value)

    def add_linear(self, var_id: int, coeff: float) -> None:
        self._add_pair(var_id, var_id, coeff)

    def add_quadratic(self, left_id: int, right_id: int, coeff: float) -> None:
        self._add_pair(left_id, right_id, coeff)

    def add_linear_terms(self, weighted_ids: Iterable[tuple[float, int]]) -> None:
        qubo = self.qubo
        for coeff, var_id in weighted_ids:
            coeff = float(coeff)
            if abs(coeff) <= 1e-12:
                continue
            key = (var_id, var_id)
            qubo[key] = qubo.get(key, 0.0) + coeff

    def add_quadratic_terms(self, weighted_ids: Iterable[tuple[float, int, int]]) -> None:
        qubo = self.qubo
        for coeff, left_id, right_id in weighted_ids:
            coeff = float(coeff)
            if abs(coeff) <= 1e-12:
                continue
            key = (left_id, right_id) if left_id <= right_id else (right_id, left_id)
            qubo[key] = qubo.get(key, 0.0) + coeff

    def add_polynomial(self, polynomial: Polynomial, scale: float = 1.0) -> None:
        if polynomial.registry is not self.registry:
            raise ValueError("Polynomial registry does not match QuboBuilder registry.")
        for term, coeff in polynomial.terms.items():
            scaled = scale * coeff
            if len(term) == 0:
                self.add_offset(scaled)
            elif len(term) == 1:
                self.add_linear(term[0], scaled)
            elif len(term) == 2:
                self.add_quadratic(term[0], term[1], scaled)
            else:
                raise ValueError("QuboBuilder only supports degree <= 2 polynomials.")

    def add_cardinality_penalty(self, var_ids: Iterable[int], count: int, penalty: float = 1.0) -> None:
        var_ids = list(var_ids)
        if len(set(var_ids)) != len(var_ids):
            raise ValueError("Cardinality penalties require distinct variable IDs.")
        penalty = float(penalty)
        self.add_offset(penalty * count * count)
        linear = penalty * (1 - 2 * count)
        qubo = self.qubo
        for var_id in var_ids:
            key = (var_id, var_id)
            qubo[key] = qubo.get(key, 0.0) + linear
        pair_coeff = 2.0 * penalty
        for left_index, left_id in enumerate(var_ids):
            for right_index in range(left_index + 1, len(var_ids)):
                right_id = var_ids[right_index]
                key = (left_id, right_id) if left_id <= right_id else (right_id, left_id)
                qubo[key] = qubo.get(key, 0.0) + pair_coeff

    def add_at_most_one_penalty(self, var_ids: Iterable[int], penalty: float = 1.0) -> None:
        var_ids = list(var_ids)
        if len(set(var_ids)) != len(var_ids):
            raise ValueError("At-most-one penalties require distinct variable IDs.")
        penalty = float(penalty)
        qubo = self.qubo
        for left_index, left_id in enumerate(var_ids):
            for right_id in var_ids[left_index + 1 :]:
                key = (left_id, right_id) if left_id <= right_id else (right_id, left_id)
                qubo[key] = qubo.get(key, 0.0) + penalty

    def add_at_least_one_penalty(
        self,
        var_ids: Iterable[int],
        slack_prefix: str = "at_least_slack",
        penalty: float = 1.0,
    ) -> list[int]:
        var_ids = list(var_ids)
        if not var_ids:
            raise ValueError("At-least-one penalties require at least one variable ID.")
        if len(set(var_ids)) != len(var_ids):
            raise ValueError("At-least-one penalties require distinct variable IDs.")
        slack_capacity = len(var_ids) - 1
        bit_count = max(1, slack_capacity.bit_length())
        slack_ids = [
            self.registry.get_or_create(
                f"{slack_prefix}[{bit}]",
                kind="binary",
                role="auxiliary",
                source="at_least_one",
            )
            for bit in range(bit_count)
        ]
        weighted_ids = [(1.0, var_id) for var_id in var_ids]
        weighted_ids.extend((-float(2**bit), var_id) for bit, var_id in enumerate(slack_ids))
        self.add_weighted_equality_penalty(weighted_ids, target=1.0, penalty=penalty)
        return slack_ids

    def add_weighted_equality_penalty(
        self,
        weighted_ids: Iterable[tuple[float, int]],
        target: float,
        penalty: float = 1.0,
    ) -> None:
        weights_by_id: dict[int, float] = {}
        for weight, var_id in weighted_ids:
            if not isinstance(weight, Real):
                raise ValueError("Weighted equality weights must be real numbers.")
            weights_by_id[var_id] = weights_by_id.get(var_id, 0.0) + float(weight)

        penalty = float(penalty)
        target = float(target)
        items = [(var_id, weight) for var_id, weight in weights_by_id.items() if abs(weight) > 1e-12]
        self.add_offset(penalty * target * target)
        qubo = self.qubo
        for var_id, weight in items:
            coeff = penalty * (weight * weight - 2.0 * target * weight)
            if abs(coeff) > 1e-12:
                key = (var_id, var_id)
                qubo[key] = qubo.get(key, 0.0) + coeff
        for left_index, (left_id, left_weight) in enumerate(items):
            for right_id, right_weight in items[left_index + 1 :]:
                coeff = penalty * 2.0 * left_weight * right_weight
                if abs(coeff) > 1e-12:
                    key = (left_id, right_id) if left_id <= right_id else (right_id, left_id)
                    qubo[key] = qubo.get(key, 0.0) + coeff

    def add_linear_inequality_penalty(
        self,
        weighted_ids: Iterable[tuple[float, int]],
        upper_bound: int,
        slack_prefix: str = "slack",
        penalty: float = 1.0,
    ) -> list[int]:
        if upper_bound < 0:
            raise ValueError("Linear inequality upper_bound must be non-negative.")
        weighted_ids = list(weighted_ids)
        for weight, _ in weighted_ids:
            if weight < 0:
                raise ValueError("Linear inequality weights must be non-negative.")
        slack_weights = bounded_log_weights(upper_bound)
        slack_ids = [
            self.registry.get_or_create(
                f"{slack_prefix}[{bit}]",
                kind="binary",
                role="auxiliary",
                source="linear_inequality",
            )
            for bit in range(len(slack_weights))
        ]
        weighted_with_slack = list(weighted_ids)
        weighted_with_slack.extend((float(weight), var_id) for weight, var_id in zip(slack_weights, slack_ids))
        self.add_weighted_equality_penalty(weighted_with_slack, target=float(upper_bound), penalty=penalty)
        return slack_ids

    def to_qubo_indices(
        self,
        clean: bool = True,
        copy: bool = True,
    ) -> tuple[dict[tuple[int, int], float], float]:
        if clean:
            return ({key: value for key, value in self.qubo.items() if abs(value) > 1e-12}, self.offset)
        if copy:
            return dict(self.qubo), self.offset
        return self.qubo, self.offset

    def to_qubo(self) -> tuple[dict[tuple[str, str], float], float]:
        qubo_ids, offset = self.to_qubo_indices()
        names = self.registry.id_to_name
        return ({(names[left], names[right]): coeff for (left, right), coeff in qubo_ids.items()}, offset)

    def to_sparse_matrix(self, symmetric: bool = False, compact: bool = True) -> SparseMatrixCOO:
        qubo, offset = self.to_qubo_indices()
        used_ids = sorted({var_id for key in qubo for var_id in key})
        names = self.registry.id_to_name
        if compact:
            id_map = {var_id: index for index, var_id in enumerate(used_ids)}
            variable_names = [names[var_id] for var_id in used_ids]
            variable_metadata = [self.registry.metadata(var_id) for var_id in used_ids]
        else:
            id_map = {var_id: var_id for var_id in used_ids}
            variable_names = list(names)
            variable_metadata = [self.registry.metadata(var_id) for var_id in range(len(names))]

        row: list[int] = []
        col: list[int] = []
        data: list[float] = []
        for (left, right), coeff in sorted(qubo.items()):
            mapped_left = id_map[left]
            mapped_right = id_map[right]
            if symmetric and mapped_left != mapped_right:
                half = coeff / 2.0
                row.extend([mapped_left, mapped_right])
                col.extend([mapped_right, mapped_left])
                data.extend([half, half])
            else:
                row.append(mapped_left)
                col.append(mapped_right)
                data.append(coeff)

        size = len(variable_names) if compact else len(names)
        return SparseMatrixCOO(
            row=row,
            col=col,
            data=data,
            shape=(size, size),
            offset=offset,
            variable_names=variable_names,
            variable_metadata=variable_metadata,
        )

    def to_ising_indices(self, compact: bool = True) -> IsingModel:
        qubo, offset = self.to_qubo_indices()
        if not compact:
            metadata = [self.registry.metadata(var_id) for var_id in range(len(self.registry.id_to_name))]
            return qubo_to_ising_indices(
                qubo,
                offset=offset,
                variable_names=list(self.registry.id_to_name),
                variable_metadata=metadata,
            )

        used_ids = sorted({var_id for key in qubo for var_id in key})
        id_map = {var_id: index for index, var_id in enumerate(used_ids)}
        compact_qubo = {(id_map[left], id_map[right]): coeff for (left, right), coeff in qubo.items()}
        variable_names = [self.registry.id_to_name[var_id] for var_id in used_ids]
        variable_metadata = [self.registry.metadata(var_id) for var_id in used_ids]
        return qubo_to_ising_indices(
            compact_qubo,
            offset=offset,
            variable_names=variable_names,
            variable_metadata=variable_metadata,
        )

    def to_qaoa_terms(self, compact: bool = True) -> tuple[list[QAOATerm], float, list[str] | None]:
        ising = self.to_ising_indices(compact=compact)
        return ising.to_qaoa_terms(), ising.offset, ising.variable_names

    def _add_pair(self, left_id: int, right_id: int, coeff: float) -> None:
        if abs(coeff) <= 1e-12:
            return
        key = (left_id, right_id) if left_id <= right_id else (right_id, left_id)
        self.qubo[key] = self.qubo.get(key, 0.0) + float(coeff)

