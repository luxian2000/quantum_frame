from __future__ import annotations

from dataclasses import dataclass, field

from .backends import IsingExport, IsingModel, QAOATerm
from .builder import QuboBuilder
from .constraints import Constraint
from .matrix import SparseMatrixCOO
from .objective import ObjectiveFragment
from .polynomial import Polynomial


@dataclass
class Model:
    objective: Polynomial
    constraints: list[Constraint] = field(default_factory=list)
    objective_fragments: list[ObjectiveFragment] = field(default_factory=list)

    def add_constraint(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)

    def add_constraints(self, constraints: list[Constraint]) -> None:
        self.constraints.extend(constraints)

    def add_objective(self, objective: ObjectiveFragment | Polynomial) -> None:
        if isinstance(objective, Polynomial):
            objective = ObjectiveFragment(objective)
        if objective.registry is not self.objective.registry:
            raise ValueError("Objective fragment registry does not match model registry.")
        self.objective_fragments.append(objective)

    def polynomial(self) -> Polynomial:
        total = self.objective
        for objective in self.objective_fragments:
            total = total + objective.expression
        for constraint in self.constraints:
            total = total + constraint.as_penalty()
        return total

    def to_qubo(self) -> tuple[dict[tuple[str, str], float], float]:
        builder = self.to_qubo_builder()
        return builder.to_qubo()

    def to_qubo_indices(
        self,
        clean: bool = True,
        copy: bool = True,
    ) -> tuple[dict[tuple[int, int], float], float]:
        builder = self.to_qubo_builder()
        return builder.to_qubo_indices(clean=clean, copy=copy)

    def to_sparse_matrix(self, symmetric: bool = False, compact: bool = True) -> SparseMatrixCOO:
        return self.to_qubo_builder().to_sparse_matrix(symmetric=symmetric, compact=compact)

    def to_qubo_builder(self) -> QuboBuilder:
        builder = QuboBuilder(registry=self.objective.registry)
        builder.add_polynomial(self.objective)
        for objective in self.objective_fragments:
            objective.add_to_builder(builder)
        for constraint in self.constraints:
            constraint.add_to_builder(builder)
        return builder

    def to_ising_indices(self, compact: bool = True) -> IsingModel:
        return self.to_qubo_builder().to_ising_indices(compact=compact)

    def to_qaoa_terms(self, compact: bool = True) -> tuple[list[QAOATerm], float, list[str] | None]:
        return self.to_qubo_builder().to_qaoa_terms(compact=compact)

    def to_ising(self) -> dict[str, object]:
        """按变量名映射的 Ising 模型 dict（``h``/``J``/``offset``）。

        已弃用（deprecated）：新代码请使用类型化的 :meth:`to_ising_export`，字段语义完全一致。
        """
        return self.to_ising_indices(compact=False).named()

    def to_ising_export(self) -> IsingExport:
        """``to_ising()`` 的强类型版本：``linear``/``quadratic`` 按变量名映射，语义与 dict 版一致。"""
        ising = self.to_ising_indices(compact=False)
        named = ising.named()
        return IsingExport(
            linear=named["h"],
            quadratic=named["J"],
            offset=named["offset"],
            variable_names=tuple(ising.variable_names) if ising.variable_names is not None else None,
            variable_metadata=(
                tuple(ising.variable_metadata) if ising.variable_metadata is not None else None
            ),
        )

