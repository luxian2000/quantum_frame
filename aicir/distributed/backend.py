"""Distributed NPU backend."""

from ..backends.npu_backend import NPUBackend


class DistNPUBackend(NPUBackend):
    """NPU backend reserved for one-state sharding."""
