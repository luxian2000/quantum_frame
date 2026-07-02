"""Reusable oracle helpers for mutation-driven VQE-QAS search.

The oracle predicts the final fair-VQE energy from previously fair-labeled
architectures.  It never predicts E2/E5 proxy energy and it abstains outside
the local trust region.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Callable, Mapping, Sequence, TypeVar

from aicir.qas.primitives.ansatz import ChemistryExcitationAnsatzGene, OperatorSequenceAnsatzGene, SupernetAnsatzGene
from aicir.qas.vqe_loop.benchmark_table import decoded_ansatz_gene_payload
from aicir.qas.vqe_loop.benchmark_table import as_float as _as_float


T = TypeVar("T")
OracleGene = SupernetAnsatzGene | OperatorSequenceAnsatzGene | ChemistryExcitationAnsatzGene


@dataclass(frozen=True)
class OraclePrediction:
    prediction: float | None
    trusted: bool
    reason: str
    neighbor_count: int
    nearest_distance: float
    kth_distance: float
    mutation_type: str = ""
    neighbor_target_std: float = 0.0

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction,
            "trusted": self.trusted,
            "reason": self.reason,
            "neighbor_count": self.neighbor_count,
            "nearest_distance": self.nearest_distance,
            "kth_distance": self.kth_distance,
            "mutation_type": self.mutation_type,
            "neighbor_target_std": self.neighbor_target_std,
        }


def _gene_from_row(row: Mapping[str, Any]) -> OracleGene | None:
    raw = row.get("ansatz_gene")
    if isinstance(raw, (SupernetAnsatzGene, OperatorSequenceAnsatzGene, ChemistryExcitationAnsatzGene)):
        return raw
    try:
        payload = decoded_ansatz_gene_payload(row)
    except (TypeError, ValueError, KeyError):
        return None
    if payload is None:
        return None
    for parser in (SupernetAnsatzGene.from_jsonable, OperatorSequenceAnsatzGene.from_jsonable, ChemistryExcitationAnsatzGene.from_jsonable):
        try:
            return parser(payload)
        except (TypeError, ValueError, KeyError):
            continue
    return None


def _layer_slot_distance(
    left_layers: Sequence[Sequence[str]],
    right_layers: Sequence[Sequence[str]],
) -> float:
    max_layers = max(len(left_layers), len(right_layers))
    if max_layers == 0:
        return 0.0
    max_width = max(
        [len(layer) for layer in left_layers] + [len(layer) for layer in right_layers] + [0]
    )
    if max_width == 0:
        return 0.0

    mismatches = 0
    total = max_layers * max_width
    for layer_index in range(max_layers):
        left_layer = tuple(left_layers[layer_index]) if layer_index < len(left_layers) else ()
        right_layer = tuple(right_layers[layer_index]) if layer_index < len(right_layers) else ()
        for slot_index in range(max_width):
            left_gate = left_layer[slot_index] if slot_index < len(left_layer) else "__missing__"
            right_gate = right_layer[slot_index] if slot_index < len(right_layer) else "__missing__"
            if left_gate != right_gate:
                mismatches += 1
    return mismatches / float(total)


def _pair_distance(
    left_pairs: Sequence[Sequence[int]],
    right_pairs: Sequence[Sequence[int]],
) -> float:
    left = {tuple(pair) for pair in left_pairs}
    right = {tuple(pair) for pair in right_pairs}
    if not left and not right:
        return 0.0
    union = left | right
    return len(left ^ right) / float(len(union))


def _supernet_distance(left: SupernetAnsatzGene, right: SupernetAnsatzGene) -> float:
    """Return a local structural distance for native supernet genes.

    Single-gate mutations are intentionally much smaller than whole-layer or
    depth mutations, matching the P1 oracle trust-region use case.
    """

    qubit_distance = 0.0 if left.n_qubits == right.n_qubits else 1.0
    max_layers = max(left.layers, right.layers, 1)
    depth_distance = abs(left.layers - right.layers) / float(max_layers)
    single_distance = _layer_slot_distance(left.single_qubit_layers, right.single_qubit_layers)
    two_gate_distance = _layer_slot_distance(left.two_qubit_layers, right.two_qubit_layers)
    pair_distance = _pair_distance(left.two_qubit_pairs, right.two_qubit_pairs)
    return min(
        1.0,
        0.10 * qubit_distance
        + 0.25 * depth_distance
        + 0.35 * single_distance
        + 0.20 * two_gate_distance
        + 0.10 * pair_distance,
    )


def _operator_sequence_distance(left: OperatorSequenceAnsatzGene, right: OperatorSequenceAnsatzGene) -> float:
    qubit_distance = 0.0 if left.n_qubits == right.n_qubits else 1.0
    max_layers = max(left.layers, right.layers, 1)
    length_distance = abs(left.layers - right.layers) / float(max_layers)
    max_length = max(len(left.operators), len(right.operators), 1)
    ordered_mismatches = 0
    for index in range(max_length):
        left_op = left.operators[index] if index < len(left.operators) else "__missing__"
        right_op = right.operators[index] if index < len(right.operators) else "__missing__"
        if left_op != right_op:
            ordered_mismatches += 1
    ordered_distance = ordered_mismatches / float(max_length)
    left_set = set(left.operators)
    right_set = set(right.operators)
    set_distance = 0.0 if not left_set and not right_set else len(left_set ^ right_set) / float(len(left_set | right_set))
    return min(
        1.0,
        0.10 * qubit_distance
        + 0.25 * length_distance
        + 0.35 * ordered_distance
        + 0.30 * set_distance,
    )



def _excitation_key(excitation: Mapping[str, Any]) -> tuple[Any, ...]:
    return (str(excitation.get("type", "")), tuple(int(qubit) for qubit in excitation.get("qubits", ())))


def _chemistry_excitation_distance(
    left: ChemistryExcitationAnsatzGene,
    right: ChemistryExcitationAnsatzGene,
) -> float:
    qubit_distance = 0.0 if left.n_qubits == right.n_qubits else 1.0
    hf_left = set(left.hf_occupied_qubits)
    hf_right = set(right.hf_occupied_qubits)
    hf_distance = 0.0 if not (hf_left or hf_right) else len(hf_left ^ hf_right) / float(len(hf_left | hf_right))
    max_layers = max(left.layers, right.layers, 1)
    length_distance = abs(left.layers - right.layers) / float(max_layers)
    max_length = max(len(left.excitations), len(right.excitations), 1)
    ordered_mismatches = 0
    for index in range(max_length):
        left_item = _excitation_key(left.excitations[index]) if index < len(left.excitations) else ("__missing__",)
        right_item = _excitation_key(right.excitations[index]) if index < len(right.excitations) else ("__missing__",)
        if left_item != right_item:
            ordered_mismatches += 1
    ordered_distance = ordered_mismatches / float(max_length)
    left_set = {_excitation_key(item) for item in left.excitations}
    right_set = {_excitation_key(item) for item in right.excitations}
    set_distance = 0.0 if not (left_set or right_set) else len(left_set ^ right_set) / float(len(left_set | right_set))
    return min(
        1.0,
        0.10 * qubit_distance
        + 0.15 * hf_distance
        + 0.25 * length_distance
        + 0.30 * ordered_distance
        + 0.20 * set_distance,
    )
def gene_aware_distance(left: OracleGene, right: OracleGene) -> float:
    if isinstance(left, SupernetAnsatzGene) and isinstance(right, SupernetAnsatzGene):
        return _supernet_distance(left, right)
    if isinstance(left, OperatorSequenceAnsatzGene) and isinstance(right, OperatorSequenceAnsatzGene):
        return _operator_sequence_distance(left, right)
    if isinstance(left, ChemistryExcitationAnsatzGene) and isinstance(right, ChemistryExcitationAnsatzGene):
        return _chemistry_excitation_distance(left, right)
    return 1.0


def weighted_knn_prediction(
    neighbors: Sequence[tuple[float, T, float]],
    k: int,
    *,
    epsilon: float = 1e-9,
) -> float | None:
    """Inverse-distance weighted kNN prediction shared by P1 and Stage-2."""

    usable = list(neighbors[: max(1, int(k))])
    if not usable:
        return None
    weighted_sum = 0.0
    weight_total = 0.0
    for distance, _item, target in usable:
        weight = 1.0 / max(float(distance), float(epsilon))
        weighted_sum += weight * float(target)
        weight_total += weight
    return weighted_sum / weight_total if weight_total > 0 else None


def _target_std(neighbors: Sequence[tuple[float, T, float]]) -> float:
    values = [float(target) for _distance, _item, target in neighbors]
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / float(len(values))
    variance = sum((value - mean) ** 2 for value in values) / float(len(values))
    return sqrt(variance)


def nearest_labeled_by_distance(
    query: T,
    labeled: Sequence[tuple[T, float]],
    distance: Callable[[T, T], float],
    *,
    identity: Callable[[T], str] | None = None,
) -> list[tuple[float, T, float]]:
    """Sort labeled items by distance to a query, excluding identical IDs."""

    query_id = identity(query) if identity is not None else ""
    scored: list[tuple[float, T, float]] = []
    for candidate, target in labeled:
        if identity is not None and identity(candidate) == query_id:
            continue
        scored.append((float(distance(query, candidate)), candidate, float(target)))
    return sorted(scored, key=lambda item: (item[0], identity(item[1]) if identity is not None else ""))


def _gene_neighbors(
    query: OracleGene,
    labeled_rows: Sequence[Mapping[str, Any]],
) -> list[tuple[float, Mapping[str, Any], float]]:
    neighbors: list[tuple[float, Mapping[str, Any], float]] = []
    for row in labeled_rows:
        energy = _as_float(row.get("fair_best_energy"))
        candidate_gene = _gene_from_row(row)
        if energy is None or candidate_gene is None:
            continue
        neighbors.append((gene_aware_distance(query, candidate_gene), row, energy))
    return sorted(neighbors, key=lambda item: (item[0], str(item[1].get("architecture_id", ""))))


def predict_fair_energy(
    query_row: Mapping[str, Any],
    *,
    labeled_rows: Sequence[Mapping[str, Any]],
    k_min: int,
    d_max: float,
    max_neighbor_std: float | None = None,
) -> OraclePrediction:
    """Predict fair_best_energy for a child row or abstain.

    The prediction target is the fair COBYLA label (`fair_best_energy`).  E2/E5
    proxy energies are deliberately ignored.
    """

    mutation_type = str(query_row.get("mutation_type", "") or "")
    query_gene = _gene_from_row(query_row)
    if query_gene is None:
        return OraclePrediction(None, False, "missing_gene", 0, float("inf"), float("inf"), mutation_type)

    neighbors = _gene_neighbors(query_gene, labeled_rows)
    if not neighbors:
        return OraclePrediction(None, False, "no_labeled_neighbors", 0, float("inf"), float("inf"), mutation_type)

    nearest_distance = float(neighbors[0][0])
    k = max(1, int(k_min))
    kth_distance = float(neighbors[k - 1][0]) if len(neighbors) >= k else float("inf")

    exact = next((item for item in neighbors if float(item[0]) <= 1e-12), None)
    if exact is not None:
        return OraclePrediction(
            float(exact[2]),
            True,
            "exact_match",
            len(neighbors),
            nearest_distance,
            kth_distance,
            mutation_type,
            0.0,
        )

    radius = max(0.0, float(d_max))
    in_region = [item for item in neighbors if float(item[0]) <= radius]
    if len(in_region) < k:
        return OraclePrediction(
            None,
            False,
            "insufficient_neighbors",
            len(in_region),
            nearest_distance,
            kth_distance,
            mutation_type,
            _target_std(in_region),
        )

    k_neighbors = neighbors[:k]
    neighbor_target_std = _target_std(k_neighbors)
    if max_neighbor_std is not None and neighbor_target_std > float(max_neighbor_std):
        return OraclePrediction(
            None,
            False,
            "high_neighbor_variance",
            len(in_region),
            nearest_distance,
            kth_distance,
            mutation_type,
            neighbor_target_std,
        )

    prediction = weighted_knn_prediction(neighbors, k)
    if prediction is None:
        return OraclePrediction(
            None,
            False,
            "no_prediction",
            len(in_region),
            nearest_distance,
            kth_distance,
            mutation_type,
            neighbor_target_std,
        )
    return OraclePrediction(
        float(prediction),
        True,
        "trusted_knn",
        len(in_region),
        nearest_distance,
        kth_distance,
        mutation_type,
        neighbor_target_std,
    )


__all__ = [
    "OraclePrediction",
    "gene_aware_distance",
    "nearest_labeled_by_distance",
    "predict_fair_energy",
    "weighted_knn_prediction",
]



