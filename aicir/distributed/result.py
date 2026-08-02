"""Distributed simulation result."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    expectations: Mapping[str, torch.Tensor | float | complex]
    counts: Mapping[str, int] | None
    rank: int
    world_size: int
    _probability_state: DistState | None = field(
        default=None,
        repr=False,
        compare=False,
    )

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

    @property
    def is_differentiable(self) -> bool:
        """Whether this result retains a paired-real native autograd graph.

        ``gather_probabilities`` and state materialization remain explicit
        detach boundaries; losses must use ``local_probabilities`` or
        ``expectations`` while this property is true.
        """

        if isinstance(self.local_probabilities, torch.Tensor) and (
            self.local_probabilities.requires_grad
            or self.local_probabilities.grad_fn is not None
        ):
            return True
        if any(
            isinstance(value, torch.Tensor)
            and (value.requires_grad or value.grad_fn is not None)
            for value in self.expectations.values()
        ):
            return True
        state = self.state if self.state is not None else self._probability_state
        pair = None if state is None else getattr(state, "_pair", None)
        return bool(
            pair is not None
            and any(
                tensor.requires_grad or tensor.grad_fn is not None
                for tensor in (pair.real, pair.imag)
            )
        )

    def gather_probabilities(self, *, root: int = 0):
        if self.local_probabilities is None:
            return None
        metadata_state = (
            self.state
            if self.state is not None
            else self._probability_state
        )
        if metadata_state is None:
            raise ValueError(
                "gather_probabilities 需要 result.state 的布局元数据"
            )
        shards = metadata_state.backend.communicator.gather_to_root(
            self.local_probabilities.detach().reshape(-1),
            root=root,
        )
        if self.rank != int(root):
            return None
        storage = torch.cat(shards).detach().cpu().numpy()
        axes = metadata_state.layout.logical_to_storage
        if axes != tuple(range(metadata_state.n_qubits)):
            storage = (
                storage.reshape([2] * metadata_state.n_qubits)
                .transpose(axes)
                .reshape(-1)
            )
        return np.asarray(storage, dtype=np.float64)
