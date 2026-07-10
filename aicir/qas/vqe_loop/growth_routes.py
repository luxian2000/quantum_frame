"""Executable defaults for the two supported P1 growth routes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GrowthRouteConfig:
    name: str
    ansatz_family: str
    p0_initializer: str
    rounds: int
    parent_count: int
    diversity_count: int
    children_per_parent: int
    fair_top_k: int
    selector: str
    baseline_selectors: tuple[str, ...]
    mutation_types: tuple[str, ...]
    genetic_mutation_weights: tuple[tuple[str, float], ...]
    genetic_weight: float
    adapt_growth_weight: float
    min_layers: int
    max_layers: int | None
    early_stop_epsilon: float
    early_stop_patience: int
    max_total_fair_calls: int
    operator_pool_limit: int | None = None
    chemistry_adapt_append_k: int = 1
    chemistry_adapt_pool_limit: int | None = None

    def genetic_weights(self) -> dict[str, float]:
        return dict(self.genetic_mutation_weights)


LINE_A_OPERATOR_SEQUENCE = GrowthRouteConfig(
    name="line_a_operator_sequence",
    ansatz_family="operator_sequence",
    p0_initializer="external_operator_fair_labels",
    rounds=5,
    parent_count=4,
    diversity_count=0,
    children_per_parent=8,
    fair_top_k=4,
    selector="e5",
    baseline_selectors=("E2",),
    mutation_types=(
        "operator_insert",
        "operator_delete",
        "operator_swap",
        "operator_big_mutation",
        "operator_adapt_growth",
    ),
    genetic_mutation_weights=(
        ("operator_insert", 0.4),
        ("operator_delete", 0.2),
        ("operator_swap", 0.2),
        ("operator_big_mutation", 0.2),
    ),
    genetic_weight=0.5,
    adapt_growth_weight=0.5,
    min_layers=0,
    max_layers=None,
    early_stop_epsilon=1.0e-4,
    early_stop_patience=2,
    max_total_fair_calls=100,
    operator_pool_limit=128,
)


LINE_B_CHEMISTRY_EXCITATION = GrowthRouteConfig(
    name="line_b_chemistry_excitation",
    ansatz_family="chemistry_excitation",
    p0_initializer="hf_empty_then_excitation_population",
    rounds=6,
    parent_count=8,
    diversity_count=0,
    children_per_parent=16,
    fair_top_k=8,
    selector="e2",
    baseline_selectors=("E2",),
    mutation_types=(
        "chemistry_insert",
        "chemistry_delete",
        "chemistry_swap",
        "chemistry_change",
        "chemistry_adapt_growth",
    ),
    genetic_mutation_weights=(
        ("chemistry_insert", 0.4),
        ("chemistry_delete", 0.15),
        ("chemistry_swap", 0.15),
        ("chemistry_change", 0.3),
    ),
    genetic_weight=0.3,
    adapt_growth_weight=0.7,
    min_layers=0,
    max_layers=32,
    early_stop_epsilon=1.0e-4,
    early_stop_patience=3,
    max_total_fair_calls=100,
    chemistry_adapt_append_k=4,
    chemistry_adapt_pool_limit=24,
)


GROWTH_ROUTES = {
    LINE_A_OPERATOR_SEQUENCE.name: LINE_A_OPERATOR_SEQUENCE,
    LINE_B_CHEMISTRY_EXCITATION.name: LINE_B_CHEMISTRY_EXCITATION,
}


def get_growth_route_config(name: str) -> GrowthRouteConfig:
    normalized = str(name).strip().lower()
    try:
        return GROWTH_ROUTES[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported growth route: {name}") from exc


__all__ = [
    "GROWTH_ROUTES",
    "GrowthRouteConfig",
    "LINE_A_OPERATOR_SEQUENCE",
    "LINE_B_CHEMISTRY_EXCITATION",
    "get_growth_route_config",
]
