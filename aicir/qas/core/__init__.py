"""Core architecture-search workflow for QAS.

This package contains traditional QAS types, scoring, reward composition,
search orchestration, and runner/config helpers.
"""

from ._types import (
    ArchitectureScore,
    ArchitectureSpec,
    MetricDefinition,
    MetricGroupScore,
    SearchConfig,
    SearchResult,
)
from .architecture_search import ArchitectureSearch, NoiseAdaptiveQAS
from .evaluator import ArchitectureEvaluator, evaluate_architectures, metric_catalog
from .reward import RewardComposer, RewardWeights
from .search_env import NoisyQASEnv, QASState
from .strategies import (
    SearchStrategy,
    StrategySpec,
    get_spec,
    get_strategy,
    register_strategy,
    registered_strategies,
    unregister_strategy,
)

__all__ = [
    "ArchitectureEvaluator",
    "ArchitectureScore",
    "ArchitectureSearch",
    "ArchitectureSpec",
    "MetricDefinition",
    "MetricGroupScore",
    "NoiseAdaptiveQAS",
    "NoisyQASEnv",
    "QASState",
    "RewardComposer",
    "RewardWeights",
    "SearchConfig",
    "SearchResult",
    "SearchStrategy",
    "StrategySpec",
    "evaluate_architectures",
    "get_spec",
    "get_strategy",
    "metric_catalog",
    "register_strategy",
    "registered_strategies",
    "unregister_strategy",
]
