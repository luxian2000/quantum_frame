"""Distributed simulation result."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch

from .state import DistState


@dataclass(frozen=True)
class DistResult:
    """Result metadata and explicit distributed materialization helpers."""

    state: DistState | None
    local_probabilities: object | None
    expectations: Mapping[str, float | complex]
    counts: Mapping[str, int] | None
    rank: int
    world_size: int

    def __post_init__(self):
        object.__setattr__(
            self,
            "expectations",
            MappingProxyType(dict(self.expectations)),
        )
        if self.counts is not None:
            object.__setattr__(
                self,
                "counts",
                MappingProxyType(dict(self.counts)),
            )

    @property
    def is_root(self) -> bool:
        return int(self.rank) == 0

    def gather_probabilities(self, *, root: int = 0):
        if self.local_probabilities is None:
            return None
        if self.state is None:
            raise ValueError(
                "gather_probabilities 需要 result.state 的布局元数据"
            )
        shards = self.state.backend.communicator.gather_to_root(
            self.local_probabilities.reshape(-1),
            root=root,
        )
        if self.rank != int(root):
            return None
        storage = torch.cat(shards).detach().cpu().numpy()
        axes = self.state.layout.logical_to_storage
        if axes != tuple(range(self.state.n_qubits)):
            storage = (
                storage.reshape([2] * self.state.n_qubits)
                .transpose(axes)
                .reshape(-1)
            )
        return np.asarray(storage, dtype=np.float64)
