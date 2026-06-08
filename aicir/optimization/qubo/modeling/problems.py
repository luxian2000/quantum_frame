from __future__ import annotations

from typing import Sequence

from .builder import QuboBuilder
from .constraints import one_hot, permutation, weighted_equality
from .integer import LogEncodedInteger, bounded_log_weights
from .model import Model
from .objective import quadratic_objective
from .polynomial import Binary, Sum, binary_array
from .registry import GLOBAL_REGISTRY, VariableRegistry


def tsp_model(
    distances: Sequence[Sequence[float]],
    penalty: float = 10.0,
    prefix: str = "x",
    registry: VariableRegistry = GLOBAL_REGISTRY,
) -> Model:
    """Build a TSP QUBO with x[city][position] binary variables."""

    city_count = len(distances)
    if city_count == 0:
        raise ValueError("TSP requires at least one city.")
    if any(len(row) != city_count for row in distances):
        raise ValueError("TSP distance matrix must be square.")

    x = binary_array(prefix, (city_count, city_count), registry=registry)
    objective = quadratic_objective(
        (distances[left][right], x[left][position], x[right][(position + 1) % city_count])
        for position in range(city_count)
        for left in range(city_count)
        for right in range(city_count)
        if left != right
    )

    model = Model(0 * x[0][0])
    model.add_objective(objective)
    model.add_constraints(permutation(x, penalty=penalty, label="tour"))
    return model


def tsp_qubo_builder(
    distances: Sequence[Sequence[float]],
    penalty: float = 10.0,
    prefix: str = "x",
    registry: VariableRegistry = GLOBAL_REGISTRY,
) -> QuboBuilder:
    """Build a TSP QUBO directly with QuboBuilder."""

    city_count = len(distances)
    if city_count == 0:
        raise ValueError("TSP requires at least one city.")
    if any(len(row) != city_count for row in distances):
        raise ValueError("TSP distance matrix must be square.")

    x = _grid_ids(prefix, city_count, city_count, registry)
    builder = QuboBuilder(registry=registry)
    for position in range(city_count):
        next_position = (position + 1) % city_count
        for left in range(city_count):
            for right in range(city_count):
                if left != right:
                    builder.add_quadratic(x[left][position], x[right][next_position], distances[left][right])

    for city in range(city_count):
        builder.add_cardinality_penalty(x[city], count=1, penalty=penalty)
    for position in range(city_count):
        builder.add_cardinality_penalty([x[city][position] for city in range(city_count)], count=1, penalty=penalty)
    return builder


def graph_coloring_model(
    node_count: int,
    edges: Sequence[tuple[int, int]],
    color_count: int,
    penalty: float = 10.0,
    prefix: str = "x",
    registry: VariableRegistry = GLOBAL_REGISTRY,
) -> Model:
    """Build a graph-coloring QUBO with x[node][color] binary variables."""

    if node_count <= 0:
        raise ValueError("Graph coloring requires at least one node.")
    if color_count <= 0:
        raise ValueError("Graph coloring requires at least one color.")

    x = binary_array(prefix, (node_count, color_count), registry=registry)
    normalized_edges = []
    for left, right in edges:
        if left < 0 or left >= node_count or right < 0 or right >= node_count:
            raise ValueError("Graph coloring edge contains an invalid node index.")
        normalized_edges.append((left, right))
    objective = Sum(
        penalty * x[left][color] * x[right][color]
        for left, right in normalized_edges
        for color in range(color_count)
    )

    model = Model(objective)
    for node in range(node_count):
        model.add_constraint(one_hot(x[node], penalty=penalty, label=f"node_{node}_one_color"))
    return model


def graph_coloring_qubo_builder(
    node_count: int,
    edges: Sequence[tuple[int, int]],
    color_count: int,
    penalty: float = 10.0,
    prefix: str = "x",
    registry: VariableRegistry = GLOBAL_REGISTRY,
) -> QuboBuilder:
    """Build a graph-coloring QUBO directly with QuboBuilder."""

    if node_count <= 0:
        raise ValueError("Graph coloring requires at least one node.")
    if color_count <= 0:
        raise ValueError("Graph coloring requires at least one color.")

    x = _grid_ids(prefix, node_count, color_count, registry)
    builder = QuboBuilder(registry=registry)
    for left, right in edges:
        if left < 0 or left >= node_count or right < 0 or right >= node_count:
            raise ValueError("Graph coloring edge contains an invalid node index.")
        for color in range(color_count):
            builder.add_quadratic(x[left][color], x[right][color], penalty)

    for node in range(node_count):
        builder.add_cardinality_penalty(x[node], count=1, penalty=penalty)
    return builder


def knapsack_model(
    values: Sequence[float],
    weights: Sequence[int],
    capacity: int,
    penalty: float = 10.0,
    item_prefix: str = "x",
    slack_prefix: str = "s",
    registry: VariableRegistry = GLOBAL_REGISTRY,
) -> Model:
    """Build a 0/1 knapsack QUBO using binary slack variables.

    The constraint is encoded as:

        sum(weights[i] * x[i]) + slack == capacity

    so infeasible overweight and underweight assignments are penalized. This is
    simple and reliable for benchmarking, though not the most compact encoding.
    """

    if len(values) != len(weights):
        raise ValueError("Knapsack values and weights must have the same length.")
    if capacity < 0:
        raise ValueError("Knapsack capacity must be non-negative.")
    if any(weight < 0 for weight in weights):
        raise ValueError("Knapsack weights must be non-negative.")

    item_count = len(values)
    x = [Binary(f"{item_prefix}[{i}]", registry=registry) for i in range(item_count)]
    objective = Sum(-values[i] * x[i] for i in range(item_count))

    slack_integer = LogEncodedInteger(
        slack_prefix,
        lower_bound=0,
        upper_bound=capacity,
        registry=registry,
        role="auxiliary",
        source="knapsack_capacity",
    )
    weighted_variables = [(float(weights[i]), x[i]) for i in range(item_count)]
    weighted_variables.extend(slack_integer.weighted_terms())

    model = Model(objective)
    model.add_constraint(
        weighted_equality(weighted_variables, target=float(capacity), penalty=penalty, label="capacity")
    )
    return model


def knapsack_qubo_builder(
    values: Sequence[float],
    weights: Sequence[int],
    capacity: int,
    penalty: float = 10.0,
    item_prefix: str = "x",
    slack_prefix: str = "s",
    registry: VariableRegistry = GLOBAL_REGISTRY,
) -> QuboBuilder:
    """Build a 0/1 knapsack QUBO directly with QuboBuilder."""

    if len(values) != len(weights):
        raise ValueError("Knapsack values and weights must have the same length.")
    if capacity < 0:
        raise ValueError("Knapsack capacity must be non-negative.")
    if any(weight < 0 for weight in weights):
        raise ValueError("Knapsack weights must be non-negative.")

    builder = QuboBuilder(registry=registry)
    item_ids = [registry.get_or_create(f"{item_prefix}[{item}]") for item in range(len(values))]
    for var_id, value in zip(item_ids, values):
        builder.add_linear(var_id, -float(value))

    slack_weights = bounded_log_weights(capacity)
    slack_ids = [
        registry.get_or_create(
            f"{slack_prefix}[{bit}]",
            kind="binary",
            role="auxiliary",
            source="knapsack_capacity",
        )
        for bit in range(len(slack_weights))
    ]
    weighted_ids = [(float(weight), var_id) for weight, var_id in zip(weights, item_ids)]
    weighted_ids.extend((float(weight), var_id) for weight, var_id in zip(slack_weights, slack_ids))
    builder.add_weighted_equality_penalty(weighted_ids, target=float(capacity), penalty=penalty)
    return builder


def _grid_ids(prefix: str, rows: int, cols: int, registry: VariableRegistry) -> list[list[int]]:
    return [[registry.get_or_create(f"{prefix}[{row}][{col}]") for col in range(cols)] for row in range(rows)]

