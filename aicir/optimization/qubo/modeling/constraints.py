from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable

from .integer import EncodedInteger, LogEncodedInteger
from .linear import LinearExpression
from .polynomial import Polynomial
from .registry import VariableRegistry

if TYPE_CHECKING:
    from .builder import QuboBuilder

BuilderAction = Callable[["QuboBuilder", float], None]


ExpressionFactory = Callable[[], Polynomial]


@dataclass(frozen=True, init=False)
class Constraint:
    _expression: Polynomial | None
    penalty: float
    label: str | None
    builder_action: BuilderAction | None
    expression_factory: ExpressionFactory | None
    registry: VariableRegistry

    def __init__(
        self,
        expression: Polynomial | None = None,
        penalty: float = 1.0,
        label: str | None = None,
        builder_action: BuilderAction | None = None,
        expression_factory: ExpressionFactory | None = None,
        registry: VariableRegistry | None = None,
    ) -> None:
        if expression is None and expression_factory is None:
            raise ValueError("Constraint requires either an expression or an expression_factory.")
        if registry is None:
            if expression is None:
                raise ValueError("Lazy constraints require an explicit registry.")
            registry = expression.registry

        object.__setattr__(self, "_expression", expression)
        object.__setattr__(self, "penalty", float(penalty))
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "builder_action", builder_action)
        object.__setattr__(self, "expression_factory", expression_factory)
        object.__setattr__(self, "registry", registry)

    @property
    def expression(self) -> Polynomial:
        expression = self._expression
        if expression is None:
            if self.expression_factory is None:
                raise ValueError("Lazy constraint is missing an expression factory.")
            expression = self.expression_factory()
            if expression.registry is not self.registry:
                raise ValueError("Lazy constraint expression registry does not match its registry.")
            object.__setattr__(self, "_expression", expression)
        return expression

    def as_penalty(self) -> Polynomial:
        return self.penalty * self.expression

    def add_to_builder(self, builder: QuboBuilder) -> None:
        if self.registry is not builder.registry:
            raise ValueError("Constraint registry does not match QuboBuilder registry.")
        if self.builder_action is None:
            builder.add_polynomial(self.expression, scale=self.penalty)
            return
        self.builder_action(builder, self.penalty)


def one_hot(variables: Iterable[Polynomial], penalty: float = 1.0, label: str | None = None) -> Constraint:
    return cardinality(variables, count=1, penalty=penalty, label=label)


def one_hot_rows(
    matrix: Iterable[Iterable[Polynomial]],
    penalty: float = 1.0,
    label: str | None = None,
) -> list[Constraint]:
    rows = _validate_matrix(matrix, "Row one-hot")
    return [
        one_hot(row, penalty=penalty, label=None if label is None else f"{label}_row_{row_index}")
        for row_index, row in enumerate(rows)
    ]


def one_hot_columns(
    matrix: Iterable[Iterable[Polynomial]],
    penalty: float = 1.0,
    label: str | None = None,
) -> list[Constraint]:
    rows = _validate_matrix(matrix, "Column one-hot")
    col_count = len(rows[0])
    return [
        one_hot(
            [rows[row_index][col_index] for row_index in range(len(rows))],
            penalty=penalty,
            label=None if label is None else f"{label}_col_{col_index}",
        )
        for col_index in range(col_count)
    ]


def assignment_matrix(
    matrix: Iterable[Iterable[Polynomial]],
    penalty: float = 1.0,
    label: str | None = None,
) -> list[Constraint]:
    """Constrain every row and every column of a binary matrix to be one-hot."""

    rows = _validate_matrix(matrix, "Assignment matrix")
    return one_hot_rows(rows, penalty=penalty, label=label) + one_hot_columns(rows, penalty=penalty, label=label)


def permutation(
    matrix: Iterable[Iterable[Polynomial]],
    penalty: float = 1.0,
    label: str | None = None,
) -> list[Constraint]:
    """Constrain a square binary matrix to represent a permutation matrix."""

    rows = _validate_matrix(matrix, "Permutation")
    if len(rows) != len(rows[0]):
        raise ValueError("Permutation constraints require a square binary matrix.")
    return assignment_matrix(rows, penalty=penalty, label=label)


def at_most_one(variables: Iterable[Polynomial], penalty: float = 1.0, label: str | None = None) -> Constraint:
    variables = list(variables)
    if not variables:
        return Constraint(Polynomial.constant(0.0), penalty=penalty, label=label)

    registry, var_ids = _plain_variable_ids(variables, "At-most-one")
    if len(set(var_ids)) != len(var_ids):
        raise ValueError("At-most-one constraints require distinct binary variables.")

    return Constraint(
        registry=registry,
        penalty=penalty,
        label=label,
        expression_factory=lambda ids=tuple(var_ids), constraint_registry=registry: _at_most_one_polynomial(
            ids,
            constraint_registry,
        ),
        builder_action=lambda builder, scale, ids=tuple(var_ids): builder.add_at_most_one_penalty(ids, penalty=scale),
    )


def at_least_one(
    variables: Iterable[Polynomial],
    slack_prefix: str = "at_least_slack",
    penalty: float = 1.0,
    label: str | None = None,
) -> tuple[Constraint, list[Polynomial]]:
    """Encode sum(x_i) >= 1 with binary slack variables."""

    variables = list(variables)
    if not variables:
        raise ValueError("At-least-one constraints require at least one variable.")
    registry, _ = _plain_variable_ids(variables, "At-least-one")
    slack_capacity = len(variables) - 1
    bit_count = max(1, slack_capacity.bit_length())
    source = label or "at_least_one"
    slack = [
        Polynomial.variable(f"{slack_prefix}[{bit}]", registry=registry, role="auxiliary", source=source)
        for bit in range(bit_count)
    ]
    weighted_variables = [(1.0, variable) for variable in variables]
    weighted_variables.extend((-float(2**bit), slack_var) for bit, slack_var in enumerate(slack))
    return weighted_equality(weighted_variables, target=1.0, penalty=penalty, label=label), slack


def cardinality(
    variables: Iterable[Polynomial],
    count: int,
    penalty: float = 1.0,
    label: str | None = None,
) -> Constraint:
    variables = list(variables)
    if not variables:
        return Constraint(Polynomial.constant(count**2), penalty=penalty, label=label)

    registry, var_ids = _plain_variable_ids(variables, "Cardinality")
    if len(set(var_ids)) != len(var_ids):
        raise ValueError("Cardinality constraints require distinct binary variables.")

    return Constraint(
        registry=registry,
        penalty=penalty,
        label=label,
        expression_factory=lambda ids=tuple(var_ids), constraint_registry=registry, constraint_count=count: _cardinality_polynomial(
            ids,
            constraint_count,
            constraint_registry,
        ),
        builder_action=lambda builder, scale, ids=tuple(var_ids), count=count: builder.add_cardinality_penalty(
            ids,
            count=count,
            penalty=scale,
        ),
    )


def weighted_equality(
    weighted_variables: Iterable[tuple[float, Polynomial]] | LinearExpression,
    target: float,
    penalty: float = 1.0,
    label: str | None = None,
) -> Constraint:
    """Fast path for (sum(weight_i * x_i) - target)^2."""

    linear = _as_linear_expression(weighted_variables)
    adjusted_target = float(target) - linear.offset
    if not linear.terms:
        return Constraint(Polynomial.constant(adjusted_target**2, registry=linear.registry), penalty=penalty, label=label)

    items = [(var_id, weight) for var_id, weight in linear.terms.items() if abs(weight) > 1e-12]
    weighted_ids = tuple((weight, var_id) for var_id, weight in items)
    return Constraint(
        registry=linear.registry,
        penalty=penalty,
        label=label,
        expression_factory=lambda ids=weighted_ids, target=adjusted_target, constraint_registry=linear.registry: _weighted_equality_polynomial(
            ids,
            target,
            constraint_registry,
        ),
        builder_action=lambda builder, scale, ids=weighted_ids, adjusted_target=adjusted_target: builder.add_weighted_equality_penalty(
            ids,
            target=adjusted_target,
            penalty=scale,
        ),
    )


def linear_inequality(
    weighted_variables: Iterable[tuple[float, Polynomial]] | LinearExpression,
    upper_bound: int,
    slack_prefix: str = "slack",
    penalty: float = 1.0,
    label: str | None = None,
) -> tuple[Constraint, list[Polynomial]]:
    """Encode sum(weight_i * x_i) <= upper_bound with binary slack variables."""

    if upper_bound < 0:
        raise ValueError("Linear inequality upper_bound must be non-negative.")
    linear = _as_linear_expression(weighted_variables)
    for weight in linear.terms.values():
        if weight < 0:
            raise ValueError("Linear inequality weights must be non-negative.")
    adjusted_upper_bound = upper_bound - linear.offset
    if adjusted_upper_bound < 0:
        raise ValueError("Linear inequality upper_bound must be >= expression offset.")
    if not linear.terms:
        return Constraint(Polynomial.constant(float(adjusted_upper_bound**2), registry=linear.registry), penalty=penalty, label=label), []

    registry = linear.registry
    source = label or "linear_inequality"
    slack_integer = LogEncodedInteger(
        slack_prefix,
        lower_bound=0,
        upper_bound=int(adjusted_upper_bound),
        registry=registry,
        role="auxiliary",
        source=source,
    )
    slack = slack_integer.bits
    weighted_with_slack = linear.weighted_terms()
    weighted_with_slack.extend(slack_integer.weighted_terms())
    return weighted_equality(weighted_with_slack, target=float(adjusted_upper_bound), penalty=penalty, label=label), slack


def integer_equality(
    expression: LinearExpression | EncodedInteger | Iterable[tuple[float, Polynomial]],
    target: float,
    penalty: float = 1.0,
    label: str | None = None,
) -> Constraint:
    return weighted_equality(_as_linear_expression(expression), target=target, penalty=penalty, label=label)


def integer_less_equal(
    expression: LinearExpression | EncodedInteger | Iterable[tuple[float, Polynomial]],
    upper_bound: int,
    slack_prefix: str = "integer_le_slack",
    penalty: float = 1.0,
    label: str | None = None,
) -> tuple[Constraint, list[Polynomial]]:
    return linear_inequality(
        _as_linear_expression(expression),
        upper_bound=upper_bound,
        slack_prefix=slack_prefix,
        penalty=penalty,
        label=label,
    )


def _plain_variable_ids(variables: list[Polynomial], constraint_name: str) -> tuple[VariableRegistry, list[int]]:
    registry = variables[0].registry
    var_ids: list[int] = []
    for variable in variables:
        if variable.registry is not registry:
            raise ValueError(f"All variables in a {constraint_name} constraint must share the same registry.")
        if len(variable.terms) != 1:
            raise ValueError(f"{constraint_name} constraints require plain binary variables.")
        ((term, coeff),) = variable.terms.items()
        if len(term) != 1 or coeff != 1.0:
            raise ValueError(f"{constraint_name} constraints require plain binary variables.")
        var_ids.append(term[0])
    return registry, var_ids


def _validate_matrix(matrix: Iterable[Iterable[Polynomial]], constraint_name: str) -> list[list[Polynomial]]:
    rows = [list(row) for row in matrix]
    if not rows:
        raise ValueError(f"{constraint_name} constraints require at least one row.")
    col_count = len(rows[0])
    if col_count == 0:
        raise ValueError(f"{constraint_name} constraints require at least one column.")
    if any(len(row) != col_count for row in rows):
        raise ValueError(f"{constraint_name} constraints require a rectangular matrix.")
    return rows


def _variable_id(variable: Polynomial) -> int:
    ((term, coeff),) = variable.terms.items()
    if len(term) != 1 or coeff != 1.0:
        raise ValueError("Expected a plain binary variable.")
    return term[0]


def _at_most_one_polynomial(var_ids: tuple[int, ...], registry: VariableRegistry) -> Polynomial:
    terms: dict[tuple[int, ...], float] = {}
    for left_index, left_id in enumerate(var_ids):
        for right_id in var_ids[left_index + 1 :]:
            key = (left_id, right_id) if left_id <= right_id else (right_id, left_id)
            terms[key] = terms.get(key, 0.0) + 1.0
    return Polynomial(terms, registry).clean()


def _cardinality_polynomial(var_ids: tuple[int, ...], count: int, registry: VariableRegistry) -> Polynomial:
    terms = {(): float(count**2)}
    linear_coeff = float(1 - 2 * count)
    for var_id in var_ids:
        terms[(var_id,)] = terms.get((var_id,), 0.0) + linear_coeff

    for left_index, left_id in enumerate(var_ids):
        for right_index in range(left_index + 1, len(var_ids)):
            right_id = var_ids[right_index]
            key = (left_id, right_id) if left_id <= right_id else (right_id, left_id)
            terms[key] = terms.get(key, 0.0) + 2.0

    return Polynomial(terms, registry).clean()


def _weighted_equality_polynomial(
    weighted_ids: tuple[tuple[float, int], ...],
    target: float,
    registry: VariableRegistry,
) -> Polynomial:
    terms = {(): float(target**2)}
    for weight, var_id in weighted_ids:
        terms[(var_id,)] = terms.get((var_id,), 0.0) + weight * weight - 2.0 * target * weight

    for left_index, (left_weight, left_id) in enumerate(weighted_ids):
        for right_weight, right_id in weighted_ids[left_index + 1 :]:
            key = (left_id, right_id) if left_id <= right_id else (right_id, left_id)
            terms[key] = terms.get(key, 0.0) + 2.0 * left_weight * right_weight

    return Polynomial(terms, registry).clean()


def _as_linear_expression(
    value: Iterable[tuple[float, Polynomial]] | LinearExpression | EncodedInteger,
) -> LinearExpression:
    if isinstance(value, LinearExpression):
        return value
    if isinstance(value, EncodedInteger):
        return value.linear_expression()
    return LinearExpression.from_terms(value)

