"""Reward composition for QAS.

This module intentionally does not import metric functions. It only combines
already computed objective-group scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from ._types import ArchitectureScore, MetricGroupScore


@dataclass
class RewardWeights:
    """Weights for the four orthogonal QAS objective groups."""

    expressibility: float = 0.25
    trainability: float = 0.25
    noise_robustness: float = 0.25
    hardware_efficiency: float = 0.25
    normalize: bool = True

    def __post_init__(self) -> None:
        values = [self.expressibility, self.trainability, self.noise_robustness, self.hardware_efficiency]
        if any(value < 0 for value in values):
            raise ValueError("Weights must be non-negative")
        if self.normalize:
            total = sum(values)
            if total > 0:
                self.expressibility /= total
                self.trainability /= total
                self.noise_robustness /= total
                self.hardware_efficiency /= total

    def to_dict(self) -> Dict[str, float]:
        return {
            "expressibility": self.expressibility,
            "trainability": self.trainability,
            "noise_robustness": self.noise_robustness,
            "hardware_efficiency": self.hardware_efficiency,
        }


class RewardComposer:
    """Compose a weighted score from precomputed metric groups."""

    def __init__(self, weights: Optional[RewardWeights] = None):
        self.weights = weights or RewardWeights()

    def compose_groups(self, groups: Mapping[str, MetricGroupScore]) -> float:
        weights = self.weights.to_dict()
        return float(sum(weights[group_name] * groups[group_name].score for group_name in weights))

    def compose_architecture_score(self, score: ArchitectureScore) -> float:
        return self.compose_groups(score.groups())


__all__ = ["RewardWeights", "RewardComposer"]
