"""Distributed NPU backend."""

from __future__ import annotations

import torch

from ..backends.npu_backend import (
    NPUBackend,
    npu_runtime_context_from_env,
)
from .communication import _Communicator


class DistNPUBackend(NPUBackend):
    """NPU backend reserved for one-state sharding."""

    def __init__(
        self,
        dtype=None,
        device=None,
        fallback_to_cpu: bool = False,
    ):
        if dtype is not None and dtype != torch.complex64:
            raise ValueError("DistNPUBackend 首期仅支持 torch.complex64")
        super().__init__(
            dtype=torch.complex64,
            device=device,
            fallback_to_cpu=fallback_to_cpu,
        )
        self._dist_context = None
        self._communicator = None

    @classmethod
    def from_env(
        cls,
        *,
        fallback_to_cpu: bool = False,
        init_process_group: bool = True,
        process_group_backend: str | None = None,
    ) -> "DistNPUBackend":
        context = npu_runtime_context_from_env()
        if context.world_size & (context.world_size - 1):
            raise ValueError("分布式状态分片要求 world_size 是 2 的幂")

        backend = super().from_distributed_env(
            dtype=torch.complex64,
            fallback_to_cpu=fallback_to_cpu,
            init_process_group=init_process_group,
            process_group_backend=process_group_backend,
        )
        backend._dist_context = backend._runtime_context
        backend._communicator = _Communicator(
            rank=context.rank,
            world_size=context.world_size,
            device=backend._device,
            supports_complex=getattr(backend._device, "type", None) != "npu",
        )
        return backend

    @property
    def world_size(self) -> int:
        context = self._dist_context
        return 1 if context is None else int(context.world_size)

    @property
    def rank(self) -> int:
        context = self._dist_context
        return 0 if context is None else int(context.rank)

    @property
    def local_rank(self) -> int:
        context = self._dist_context
        return 0 if context is None else int(context.local_rank)

    @property
    def communicator(self) -> _Communicator:
        if self._communicator is None:
            self._communicator = _Communicator(
                rank=self.rank,
                world_size=self.world_size,
                device=self._device,
            )
        return self._communicator

    def should_run_batch_index(self, index: int) -> bool:
        raise RuntimeError(
            "DistNPUBackend 使用单状态分片，不能同时启用批次任务并行"
        )

    def gather_indexed_results(self, indexed_results):
        raise RuntimeError(
            "DistNPUBackend 使用单状态分片，不能同时启用批次任务并行"
        )
