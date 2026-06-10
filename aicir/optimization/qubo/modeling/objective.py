from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable

from .polynomial import Polynomial
from .registry import VariableRegistry

if TYPE_CHECKING:
    from .builder import QuboBuilder

ObjectiveBuilderAction = Callable[["QuboBuilder"], None]
ObjectiveExpressionFactory = Callable[[], Polynomial]


@dataclass(frozen=True, init=False)
class ObjectiveFragment:
    _expression: Polynomial | None
    builder_action: ObjectiveBuilderAction | None
    expression_factory: ObjectiveExpressionFactory | None
    registry: VariableRegistry

    def __init__(
        self,
        expression: Polynomial | None = None,
        builder_action: ObjectiveBuilderAction | None = None,
        expression_factory: ObjectiveExpressionFactory | None = None,
        registry: VariableRegistry | None = None,
    ) -> None:
        if expression is None and expression_factory is None:
            raise ValueError("ObjectiveFragment requires either an expression or an expression_factory.")
        if registry is None:
            if expression is None:
                raise ValueError("Lazy objective fragments require an explicit registry.")
            registry = expression.registry

        object.__setattr__(self, "_expression", expression)
        object.__setattr__(self, "builder_action", builder_action)
        object.__setattr__(self, "expression_factory", expression_factory)
        object.__setattr__(self, "registry", registry)

    @property
    def expression(self) -> Polynomial:
        expression = self._expression
        if expression is None:
            if self.expression_factory is None:
                raise ValueError("Lazy objective fragment is missing an expression factory.")
            expression = self.expression_factory()
            if expression.registry is not self.registry:
                raise ValueError("Lazy objective expression registry does not match its registry.")
            object.__setattr__(self, "_expression", expression)
        return expression

    def add_to_builder(self, builder: QuboBuilder) -> None:
        if self.registry is not builder.registry:
            raise ValueError("Objective fragment registry does not match QuboBuilder registry.")
        if self.builder_action is None:
            builder.add_polynomial(self.expression)
            return
        self.builder_action(builder)


def linear_objective(weighted_variables: Iterable[tuple[float, Polynomial]]) -> ObjectiveFragment:
    weighted_variables = list(weighted_variables)
    items = [(float(weight), _variable_id(variable)) for weight, variable in weighted_variables]
    registry = _registry_from_variables([variable for _, variable in weighted_variables])
    return ObjectiveFragment(
        registry=registry,
        expression_factory=lambda weighted_ids=tuple(items), objective_registry=registry: _linear_polynomial(
            weighted_ids,
            objective_registry,
        ),
        builder_action=lambda builder, weighted_ids=tuple(items): _add_linear_terms(builder, weighted_ids),
    )


def quadratic_objective(weighted_pairs: Iterable[tuple[float, Polynomial, Polynomial]]) -> ObjectiveFragment:
    pairs = list(weighted_pairs)
    variables = [variable for _, left, right in pairs for variable in (left, right)]
    registry = _registry_from_variables(variables)
    items = [(float(weight), _variable_id(left), _variable_id(right)) for weight, left, right in pairs]
    return ObjectiveFragment(
        registry=registry,
        expression_factory=lambda weighted_ids=tuple(items), objective_registry=registry: _quadratic_polynomial(
            weighted_ids,
            objective_registry,
        ),
        builder_action=lambda builder, weighted_ids=tuple(items): _add_quadratic_terms(builder, weighted_ids),
    )


def _add_linear_terms(builder: QuboBuilder, weighted_ids: tuple[tuple[float, int], ...]) -> None:
    builder.add_linear_terms(weighted_ids)


def _add_quadratic_terms(builder: QuboBuilder, weighted_ids: tuple[tuple[float, int, int], ...]) -> None:
    builder.add_quadratic_terms(weighted_ids)


def _linear_polynomial(weighted_ids: tuple[tuple[float, int], ...], registry: VariableRegistry) -> Polynomial:
    terms: dict[tuple[int, ...], float] = {}
    for weight, var_id in weighted_ids:
        terms[(var_id,)] = terms.get((var_id,), 0.0) + weight
    return Polynomial(terms, registry).clean()


def _quadratic_polynomial(
    weighted_ids: tuple[tuple[float, int, int], ...],
    registry: VariableRegistry,
) -> Polynomial:
    terms: dict[tuple[int, ...], float] = {}
    for weight, left_id, right_id in weighted_ids:
        key = tuple(sorted({left_id, right_id}))
        terms[key] = terms.get(key, 0.0) + weight
    return Polynomial(terms, registry).clean()


def _registry_from_variables(variables: list[Polynomial]) -> VariableRegistry:
    if not variables:
        raise ValueError("Objective fragments require at least one variable.")
    registry = variables[0].registry
    for variable in variables:
        if variable.registry is not registry:
            raise ValueError("All variables in an objective fragment must share the same registry.")
    return registry


def _variable_id(variable: Polynomial) -> int:
    if len(variable.terms) != 1:
        raise ValueError("Objective fragments require plain binary variables.")
    ((term, coeff),) = variable.terms.items()
    if len(term) != 1 or coeff != 1.0:
        raise ValueError("Objective fragments require plain binary variables.")
    return term[0]

