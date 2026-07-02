"""Lightweight graph-style fair-energy predictor for P1.

This is intentionally dependency-light: it exposes the same fit/predict shape
as a future GNN but uses fixed graph/structure features plus ridge regression.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from aicir.qas.vqe_loop.benchmark_table import decoded_ansatz_gene_payload
from aicir.qas.vqe_loop.benchmark_table import as_float as _as_float
from aicir.qas.vqe_loop.task_proxy import (
    ansatz_pauli_support_coverage,
    gradient_sensitivity_proxy,
    hamiltonian_ansatz_overlap,
)


@dataclass(frozen=True)
class GraphPrediction:
    prediction: float | None
    confidence: float
    reason: str



def _operator_features(payload: Mapping[str, Any]) -> list[float]:
    operators = [str(op).upper() for op in payload.get("operators", ()) or ()]
    n_ops = len(operators)
    support_sizes = [sum(1 for letter in op if letter != "I") for op in operators]
    letters = "".join(operators)
    denom = max(1, len(letters))
    return [
        float(n_ops),
        float(sum(support_sizes)),
        float(max(support_sizes) if support_sizes else 0),
        letters.count("X") / float(denom),
        letters.count("Y") / float(denom),
        letters.count("Z") / float(denom),
    ]


def _supernet_features(payload: Mapping[str, Any]) -> list[float]:
    single_layers = payload.get("single_qubit_layers", ()) or ()
    two_layers = payload.get("two_qubit_layers", ()) or ()
    singles = [str(gate).lower() for layer in single_layers for gate in layer]
    twos = [str(gate).lower() for layer in two_layers for gate in layer]
    n_single = len([gate for gate in singles if gate != "none"])
    n_two = len([gate for gate in twos if gate != "none"])
    denom_single = max(1, len(singles))
    denom_two = max(1, len(twos))
    return [
        float(len(single_layers)),
        float(n_single),
        float(n_two),
        singles.count("rx") / float(denom_single),
        singles.count("ry") / float(denom_single),
        singles.count("rz") / float(denom_single),
        twos.count("cx") / float(denom_two),
        twos.count("rzz") / float(denom_two),
    ]


def encode_row_features(row: Mapping[str, Any]) -> list[float]:
    payload = dict(decoded_ansatz_gene_payload(row) or {})
    kind = str(payload.get("kind", "")).lower()
    base = [
        1.0,
        float(row.get("n_qubits") or payload.get("n_qubits") or 0),
        hamiltonian_ansatz_overlap(row),
        ansatz_pauli_support_coverage(row),
        gradient_sensitivity_proxy(row),
    ]
    if kind == "operator_sequence":
        return base + [1.0, 0.0] + _operator_features(payload) + [0.0] * 8
    if kind == "supernet_native":
        return base + [0.0, 1.0] + [0.0] * 6 + _supernet_features(payload)
    return base + [0.0, 0.0] + [0.0] * 14


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    aug = [list(matrix[i]) + [float(vector[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            continue
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if abs(factor) < 1e-12:
                continue
            aug[row] = [aug[row][i] - factor * aug[col][i] for i in range(n + 1)]
    return [aug[i][-1] for i in range(n)]


class GraphEnergyPredictor:
    def __init__(self, alpha: float = 1.0, min_labels: int = 2):
        self.alpha = float(alpha)
        self.min_labels = int(min_labels)
        self.weights: list[float] = []
        self.train_features: list[list[float]] = []
        self.residual_scale = 1.0

    @property
    def is_fitted(self) -> bool:
        return bool(self.weights)

    def fit(self, labeled_rows: Sequence[Mapping[str, Any]]) -> None:
        rows = [row for row in labeled_rows if _as_float(row.get("fair_best_energy")) is not None]
        if len(rows) < self.min_labels:
            self.weights = []
            self.train_features = []
            return
        x_rows = [encode_row_features(row) for row in rows]
        y = [float(row["fair_best_energy"]) for row in rows]
        n_features = len(x_rows[0])
        xtx = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
        xty = [0.0 for _ in range(n_features)]
        for features, target in zip(x_rows, y):
            for i in range(n_features):
                xty[i] += features[i] * target
                for j in range(n_features):
                    xtx[i][j] += features[i] * features[j]
        for i in range(n_features):
            xtx[i][i] += self.alpha
        self.weights = _solve_linear(xtx, xty)
        self.train_features = x_rows
        residuals = [abs(_dot(self.weights, features) - target) for features, target in zip(x_rows, y)]
        self.residual_scale = max(1e-9, sum(residuals) / float(len(residuals)))

    def predict_row(self, row: Mapping[str, Any]) -> GraphPrediction:
        if not self.is_fitted:
            return GraphPrediction(None, 0.0, "not_fitted")
        features = encode_row_features(row)
        prediction = _dot(self.weights, features)
        nearest = min(
            math.sqrt(sum((a - b) ** 2 for a, b in zip(features, train)))
            for train in self.train_features
        )
        confidence = 1.0 / (1.0 + nearest + self.residual_scale)
        return GraphPrediction(float(prediction), float(max(0.0, min(1.0, confidence))), "ok")


def build_graph_predictor_evaluator(labeled_rows: Sequence[Mapping[str, Any]], *, alpha: float = 1.0):
    predictor = GraphEnergyPredictor(alpha=alpha)
    predictor.fit(labeled_rows)

    def evaluate(row: Mapping[str, Any]) -> dict[str, Any]:
        prediction = predictor.predict_row(row)
        if prediction.prediction is None:
            return {"GNN_PROXY": "", "predictor_confidence": "0.000000000000"}
        return {
            "GNN_PROXY": f"{float(prediction.prediction):.12f}",
            "predictor_confidence": f"{float(prediction.confidence):.12f}",
        }

    return evaluate


__all__ = [
    "GraphEnergyPredictor",
    "GraphPrediction",
    "build_graph_predictor_evaluator",
    "encode_row_features",
]
