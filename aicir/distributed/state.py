"""Distributed state container."""

from __future__ import annotations

import numpy as np
import torch

from ..backends.numpy_backend import NumpyBackend
from ..core.state import State
from ._contracts import reject_requires_grad
from .layout import _Layout, _ShardSpec


class DistState:
    """One rank-local shard of a distributed quantum state."""

    def __init__(
        self,
        local_data,
        *,
        spec: _ShardSpec,
        backend,
        bit_order: str = "msb",
    ):
        reject_requires_grad(local_data)
        casted = backend.cast(local_data)
        if tuple(int(axis) for axis in casted.shape) != spec.local_shape:
            raise ValueError(
                f"local_data shape={tuple(casted.shape)} 与 "
                f"local_shape={spec.local_shape} 不一致"
            )
        if casted.dtype != torch.complex64:
            raise ValueError("DistState 首期仅支持 torch.complex64")
        if bit_order not in {"msb", "lsb"}:
            raise ValueError("bit_order 必须是 'msb' 或 'lsb'")
        if spec.rank != backend.rank or spec.world_size != backend.world_size:
            raise ValueError("spec 的 rank/world_size 与 backend 不一致")

        self._local_data = casted
        self._spec = spec
        self._backend = backend
        self._bit_order = bit_order

    @classmethod
    def from_local(
        cls,
        local_data,
        *,
        spec: _ShardSpec,
        backend,
        bit_order: str = "msb",
    ) -> "DistState":
        return cls(
            local_data,
            spec=spec,
            backend=backend,
            bit_order=bit_order,
        )

    @classmethod
    def zero(
        cls,
        n_qubits: int,
        *,
        backend,
        layout: _Layout,
        bit_order: str = "msb",
    ) -> "DistState":
        spec = _ShardSpec.build(
            n_qubits=n_qubits,
            world_size=backend.world_size,
            rank=backend.rank,
            kind="vector",
            layout=layout,
        )
        local = np.zeros(spec.local_shape, dtype=np.complex64)
        if backend.rank == 0:
            local[0, 0] = 1.0 + 0.0j
        return cls(
            backend.cast(local),
            spec=spec,
            backend=backend,
            bit_order=bit_order,
        )

    @property
    def local_data(self):
        return self._local_data

    @property
    def local_shape(self) -> tuple[int, int]:
        return self._spec.local_shape

    @property
    def global_shape(self) -> tuple[int, int]:
        return self._spec.global_shape

    @property
    def n_qubits(self) -> int:
        return self._spec.n_qubits

    @property
    def kind(self) -> str:
        return self._spec.kind

    @property
    def is_density(self) -> bool:
        return self.kind == "matrix"

    @property
    def bit_order(self) -> str:
        return self._bit_order

    @property
    def rank(self) -> int:
        return self._spec.rank

    @property
    def world_size(self) -> int:
        return self._spec.world_size

    @property
    def layout(self) -> _Layout:
        return self._spec.layout

    @property
    def spec(self) -> _ShardSpec:
        return self._spec

    @property
    def backend(self):
        return self._backend

    def local_probabilities(self):
        if self.kind == "vector":
            probabilities = self._backend.abs_sq(
                self._local_data.reshape(-1)
            )
        else:
            local_rows = self.local_shape[0]
            rows = torch.arange(
                local_rows,
                dtype=torch.long,
                device=self._local_data.device,
            )
            columns = rows + self._spec.global_start
            probabilities = torch.real(self._local_data)[rows, columns]
            probabilities = torch.clamp(probabilities, min=0)

        total = self._backend.communicator.all_reduce_sum(
            probabilities.sum().reshape(())
        )
        if float(total.detach().cpu()) <= 0:
            raise ValueError("分布式状态的全局概率和必须大于 0")
        return probabilities / total

    def _restore_logical_order(self, array: np.ndarray) -> np.ndarray:
        axes = self._spec.layout.logical_to_storage
        if axes == tuple(range(self.n_qubits)):
            return array
        if self.kind == "vector":
            tensor = array.reshape([2] * self.n_qubits)
            return tensor.transpose(axes).reshape(-1, 1)
        tensor = array.reshape([2] * (2 * self.n_qubits))
        permutation = tuple(axes) + tuple(
            self.n_qubits + axis for axis in axes
        )
        return tensor.transpose(permutation).reshape(self.global_shape)

    def gather(self, *, root: int = 0):
        shards = self._backend.communicator.gather_to_root(
            self._local_data,
            root=root,
        )
        if self.rank != int(root):
            return None
        storage_order = torch.cat(shards, dim=0).detach().cpu().numpy()
        logical_order = self._restore_logical_order(storage_order)
        return State(
            logical_order,
            self.n_qubits,
            NumpyBackend(),
            bit_order=self.bit_order,
        )

    def to_numpy(self, *, root: int = 0):
        gathered = self.gather(root=root)
        return None if gathered is None else gathered.to_numpy()

    def __repr__(self) -> str:
        return (
            f"DistState(n_qubits={self.n_qubits}, kind={self.kind!r}, "
            f"rank={self.rank}/{self.world_size}, "
            f"local_shape={self.local_shape})"
        )
