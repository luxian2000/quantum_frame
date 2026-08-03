"""Deterministic checkpoint planning for the private paired-real engine."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from numbers import Integral

import torch


_POLICY_ERROR = "grad_checkpoint 必须是 'none'、'auto' 或正整数"
_INTERVAL_MISMATCH_ERROR = "各 rank 的梯度检查点间隔不一致"
_MEMORY_SOURCE_MISMATCH_ERROR = "各 rank 的梯度检查点内存来源不一致"
_MEMORY_SOURCE_CODES = {"provided": 0, "host": 1, "cuda": 2, "npu": 3, "conservative": 4}


@dataclass(frozen=True)
class _CheckpointPolicy:
    """A validated public policy with no execution-side effects."""

    value: str | int

    @classmethod
    def parse(cls, value) -> "_CheckpointPolicy":
        if type(value) is str and value in {"none", "auto"}:
            return cls(value)
        if isinstance(value, Integral) and not isinstance(value, bool) and int(value) > 0:
            return cls(int(value))
        raise ValueError(_POLICY_ERROR)


def _available_memory_bytes(device) -> tuple[int | None, str]:
    """Return currently available memory without requiring an accelerator runtime.

    CPU tests use ``sysconf``.  CUDA/NPU support is deliberately best-effort:
    an unavailable runtime falls back to the host value rather than probing or
    allocating on a device that is not present.
    """

    try:
        device_type = torch.device(device).type
        resolved_device = torch.device(device)
    except Exception:
        device_type = str(device).split(":", 1)[0].lower()
        resolved_device = device
    try:
        if device_type == "cuda" and torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info(resolved_device)
            return max(1, int(free)), "cuda"
        if device_type == "npu":
            npu = getattr(torch, "npu", None)
            query = getattr(npu, "mem_get_info", None)
            if callable(query):
                free, _ = query(resolved_device)
                return max(1, int(free)), "npu"
            return None, "conservative"
    except Exception:  # runtime capability discovery must never break CPU tests
        if device_type == "npu":
            return None, "conservative"
    try:
        return max(1, int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))), "host"
    except (AttributeError, OSError, ValueError):
        return None, "conservative"


class _CheckpointPlanner:
    """Choose a deterministic interval from local paired-real memory costs."""

    def __init__(self, spec, circuit_depth: int, available_bytes: int):
        if isinstance(circuit_depth, bool) or not isinstance(circuit_depth, Integral) or int(circuit_depth) < 0:
            raise ValueError("circuit_depth 必须是非负整数")
        if isinstance(available_bytes, bool) or not isinstance(available_bytes, Integral) or int(available_bytes) <= 0:
            raise ValueError("available_bytes 必须是正整数")
        shape = tuple(int(axis) for axis in spec.local_shape)
        if not shape or any(axis <= 0 for axis in shape):
            raise ValueError("spec.local_shape 必须是正整数形状")
        self.spec = spec
        self.circuit_depth = int(circuit_depth)
        self.available_bytes = int(available_bytes)
        # paired float32 = real plus imaginary; two full paired working
        # buffers cover local gate output and communication/reduction staging.
        self.state_bytes = math.prod(shape) * 2 * torch.tensor([], dtype=torch.float32).element_size()
        self.temporary_bytes = 2 * self.state_bytes
        self.budget_bytes = int(self.available_bytes * 0.80)

    def estimated_bytes(self, interval: int) -> int:
        interval = int(interval)
        if interval <= 0:
            raise ValueError("checkpoint interval 必须是正整数")
        boundaries = 1 if self.circuit_depth == 0 else math.ceil(self.circuit_depth / interval) + 1
        return math.ceil(1.20 * (boundaries * self.state_bytes + self.temporary_bytes))

    def interval(self) -> int:
        if self.circuit_depth <= 1:
            return 1
        for interval in range(1, self.circuit_depth + 1):
            if self.estimated_bytes(interval) <= self.budget_bytes:
                return interval
        # The largest interval is still the least-memory valid policy.  The
        # caller does not silently disable checkpointing under memory pressure.
        return self.circuit_depth


@dataclass
class _CheckpointMetrics:
    policy: str | int
    interval: int
    saved_state_count: int
    recomputed_gate_count: int = 0
    peak_allocation_bytes: int | None = None
    peak_allocation_status: str = "UNMEASURED"
    memory_source: str = "unknown"


def _recompute_segment(start_state, plans, start: int, stop: int, engine):
    """Replay an exact half-open planned segment through the supplied engine.

    The plan objects are never rebuilt here: their logical/storage axes,
    partner masks, operation indices, and therefore P2P tags are retained.
    """

    start, stop = int(start), int(stop)
    if not 0 <= start <= stop <= len(plans):
        raise ValueError("checkpoint segment 超出线路范围")
    state = start_state
    for operation_index in range(start, stop):
        state = engine.apply(state, plans[operation_index], operation_index=operation_index)
    return state


def _agree_interval(interval: int, communicator) -> int:
    """Reject rank-divergent selected intervals before state collectives."""

    # Keep this control-plane collective in the same float32-only subset used
    # by the paired-real transport path; intervals are bounded by circuit
    # depth, far below float32's exact integer range.
    value = torch.tensor([float(interval)], dtype=torch.float32, device=communicator.device)
    values = communicator.all_gather(value)
    resolved = tuple(int(item.detach().cpu().reshape(-1).item()) for item in values)
    if any(item != resolved[0] for item in resolved[1:]):
        raise ValueError(_INTERVAL_MISMATCH_ERROR)
    return resolved[0]


def _agree_checkpoint_selection(interval: int, source: str, communicator) -> tuple[int, str]:
    """Agree interval and memory source in one float32 control collective."""

    code = _MEMORY_SOURCE_CODES.get(source)
    if code is None:
        raise ValueError("checkpoint memory source 无效")
    control = torch.tensor([float(interval), float(code)], dtype=torch.float32, device=communicator.device)
    values = communicator.all_gather(control)
    decoded = [tuple(int(value.detach().cpu()[index].item()) for index in range(2)) for value in values]
    if any(value[0] != decoded[0][0] for value in decoded[1:]):
        raise ValueError(_INTERVAL_MISMATCH_ERROR)
    if any(value[1] != decoded[0][1] for value in decoded[1:]):
        raise ValueError(_MEMORY_SOURCE_MISMATCH_ERROR)
    return decoded[0][0], source


__all__ = [
    "_CheckpointMetrics",
    "_CheckpointPlanner",
    "_CheckpointPolicy",
    "_POLICY_ERROR",
    "_agree_interval",
    "_agree_checkpoint_selection",
    "_available_memory_bytes",
    "_recompute_segment",
]
