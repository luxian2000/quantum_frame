from .analysis import (
    BruteForceResult,
    brute_force_builder,
    brute_force_model,
    brute_force_qubo,
    decode_best_solutions,
    qubo_energy,
)
from .backends import IsingModel, QAOATerm, qubo_to_ising_indices
from .builder import QuboBuilder
from .constraints import (
    Constraint,
    assignment_matrix,
    at_least_one,
    at_most_one,
    cardinality,
    integer_equality,
    integer_less_equal,
    linear_inequality,
    one_hot,
    one_hot_columns,
    one_hot_rows,
    permutation,
    weighted_equality,
)
from .integer import EncodedInteger, Integer, LogEncodedInteger, UnaryEncodedInteger
from .linear import Linear, LinearExpression
from .matrix import SparseMatrixCOO
from .model import Model
from .objective import ObjectiveFragment, linear_objective, quadratic_objective
from .polynomial import Binary, Polynomial, Sum, binary_array
from .problems import (
    graph_coloring_model,
    graph_coloring_qubo_builder,
    knapsack_model,
    knapsack_qubo_builder,
    tsp_model,
    tsp_qubo_builder,
)
from .registry import ModelContext, VariableMetadata, VariableRegistry
from .solution import DecodedSolution, decode_integer, decode_solution

__version__ = "0.1.0"

__all__ = [
    "Binary",
    "BruteForceResult",
    "Constraint",
    "DecodedSolution",
    "EncodedInteger",
    "Integer",
    "IsingModel",
    "Linear",
    "LinearExpression",
    "LogEncodedInteger",
    "Model",
    "ModelContext",
    "ObjectiveFragment",
    "Polynomial",
    "QAOATerm",
    "QuboBuilder",
    "SparseMatrixCOO",
    "Sum",
    "UnaryEncodedInteger",
    "VariableMetadata",
    "VariableRegistry",
    "__version__",
    "assignment_matrix",
    "at_least_one",
    "at_most_one",
    "binary_array",
    "brute_force_builder",
    "brute_force_model",
    "brute_force_qubo",
    "cardinality",
    "decode_integer",
    "decode_best_solutions",
    "decode_solution",
    "graph_coloring_model",
    "graph_coloring_qubo_builder",
    "integer_equality",
    "integer_less_equal",
    "knapsack_model",
    "knapsack_qubo_builder",
    "linear_inequality",
    "linear_objective",
    "one_hot",
    "one_hot_columns",
    "one_hot_rows",
    "permutation",
    "qubo_energy",
    "quadratic_objective",
    "tsp_model",
    "tsp_qubo_builder",
    "qubo_to_ising_indices",
    "weighted_equality",
]

