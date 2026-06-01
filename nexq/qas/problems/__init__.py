"""Problem instances for task-level QAS validation."""

from .base import ProblemInstance
from .maxcut import MaxCutInstance, maxcut_line, maxcut_ring
from .resource_allocation import ResourceAllocationInstance, small_resource_allocation

__all__ = [
    "MaxCutInstance",
    "ProblemInstance",
    "ResourceAllocationInstance",
    "maxcut_line",
    "maxcut_ring",
    "small_resource_allocation",
]
