"""Small resource-allocation QUBO benchmark for QAS validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .base import ProblemInstance


def _qubo_values(linear_rewards: Sequence[float], conflict_penalty: np.ndarray) -> np.ndarray:
    rewards = np.asarray(linear_rewards, dtype=float).reshape(-1)
    penalties = np.asarray(conflict_penalty, dtype=float)
    if penalties.shape != (rewards.size, rewards.size):
        raise ValueError("conflict_penalty must be a square matrix matching linear_rewards")
    values = np.zeros(1 << rewards.size, dtype=float)
    for state_index in range(1 << rewards.size):
        bits = np.asarray([(state_index >> (rewards.size - qubit - 1)) & 1 for qubit in range(rewards.size)])
        reward = float(np.dot(rewards, bits))
        conflict = 0.0
        for i in range(rewards.size):
            for j in range(i + 1, rewards.size):
                conflict += float(penalties[i, j]) * bits[i] * bits[j]
        values[state_index] = reward - conflict
    return values


@dataclass
class ResourceAllocationInstance(ProblemInstance):
    """Binary resource-selection objective with pairwise conflict penalties."""

    linear_rewards: np.ndarray = None
    conflict_penalty: np.ndarray = None

    @classmethod
    def from_qubo(
        cls,
        name: str,
        linear_rewards: Sequence[float],
        conflict_penalty: Sequence[Sequence[float]],
    ) -> "ResourceAllocationInstance":
        rewards = np.asarray(linear_rewards, dtype=float).reshape(-1)
        penalties = np.asarray(conflict_penalty, dtype=float)
        values = _qubo_values(rewards, penalties)
        return cls(
            name=name,
            n_qubits=int(rewards.size),
            objective_values=values,
            classical_optimum=float(values.max()),
            maximize=True,
            metadata={"family": "resource_allocation"},
            linear_rewards=rewards,
            conflict_penalty=penalties,
        )


def small_resource_allocation() -> ResourceAllocationInstance:
    rewards = [1.2, 1.0, 0.9, 1.1]
    penalties = [
        [0.0, 1.4, 0.0, 0.4],
        [1.4, 0.0, 0.7, 0.0],
        [0.0, 0.7, 0.0, 1.2],
        [0.4, 0.0, 1.2, 0.0],
    ]
    return ResourceAllocationInstance.from_qubo("resource_allocation_4", rewards, penalties)
