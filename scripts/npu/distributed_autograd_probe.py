#!/usr/bin/env python3
"""Strict multi-NPU acceptance probe for the paired-real statevector kernel.

Run this probe with ``torchrun`` on Ascend hardware.  It has no CPU fallback,
does not claim that forward-only :class:`aicir.distributed.DistSimulator` is
differentiable, and records no live-NPU result until the command is run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import uuid

import torch
import numpy as np

from aicir import Circuit, Hamiltonian, PauliString
from aicir.core.circuit import crx, cry, crz, rx, rxx, ry, rz, rzz, u2, u3
from aicir.ir import Observable
from aicir.distributed import (
    DistNPUBackend,
    DistSimulator,
    DistState,
    DensityParam,
    PureStateParam,
    finite_difference_gradient,
    parameter_shift_gradient,
)
from aicir.distributed.autograd._collectives import _exchange_pair, _replicated_all_reduce, _scatter_root_pair
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.autograd._parameters import _bucket_parameters
from aicir.distributed.autograd._density import _PairMatrixKernel
from aicir.distributed.autograd._channels import _stinespring_kraus
from aicir.distributed.autograd._parameters import StinespringParam
from aicir.distributed.autograd._reducers import _PairReducer
from aicir.distributed.autograd._vector import _PairVectorKernel
from aicir.distributed.gates import _AutogradExecutionContext, _GatePlanner
from aicir.distributed.layout import _Layout, _ShardSpec
from aicir.qml.deriv import psr4
from aicir.noise import (
    AmplitudeDampingChannel,
    BitFlipChannel,
    DepolarizingChannel,
    NoiseModel,
    PhaseFlipChannel,
)


SECTIONS = (
    "environment",
    "statevector",
    "density",
    "gates",
    "probability",
    "observable",
    "noise",
    "stinespring",
    "communication",
    "optimizer",
    "performance",
    "memory",
    "contract",
)

_FAILURE_TYPE_BYTES = 128
_FAILURE_MESSAGE_BYTES = 512
_FAILURE_PAYLOAD_BYTES = 5 + _FAILURE_TYPE_BYTES + _FAILURE_MESSAGE_BYTES
_RAW_SHA256_PLACEHOLDER = "0" * 64


def _require_hccl_backend(name: str) -> None:
    """Keep the strict launch backend as an independently testable contract."""

    if str(name).lower() != "hccl":
        raise ValueError("严格 distributed autograd 探针要求 HCCL process group")


def _require_supported_channel(channel) -> None:
    """Reject a non-native channel before it can enter paired-real replay."""

    supported = (
        AmplitudeDampingChannel,
        BitFlipChannel,
        DepolarizingChannel,
        PhaseFlipChannel,
        StinespringParam,
    )
    if not isinstance(channel, supported):
        raise ValueError(
            "自动微分模式不支持噪声通道 "
            f"{type(channel).__name__}"
        )


def _validate_tag_phases(forward_tags, backward_tags) -> None:
    """Reject injected forward/backward P2P tag overlap deterministically."""

    if set(int(tag) for tag in forward_tags).intersection(
        int(tag) for tag in backward_tags
    ):
        raise ValueError("forward/backward P2P tag 不匹配")


def _strict_backend(*, fallback_to_cpu: bool = False) -> DistNPUBackend:
    """Build the probe backend, rejecting every possible CPU fallback path."""

    if fallback_to_cpu:
        raise ValueError("严格 distributed autograd 探针不允许 fallback_to_cpu=True")
    try:
        npu_available = torch.npu.is_available()
    except AttributeError as error:
        raise RuntimeError("严格 distributed autograd 探针要求 torch.npu") from error
    if not npu_available:
        raise RuntimeError("严格 distributed autograd 探针要求 torch.npu.is_available()")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size not in {2, 4, 8}:
        raise ValueError("distributed autograd 探针只接受 world_size=2、4 或 8")

    device = f"npu:{local_rank}"
    torch.npu.set_device(device)
    backend = DistNPUBackend.from_env(
        fallback_to_cpu=False,
        process_group_backend="hccl",
    )
    try:
        _require_hccl_backend(torch.distributed.get_backend())
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    if backend._device.type != "npu" or backend._device.index != local_rank:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} 必须绑定 npu:{local_rank}，实际为 {backend._device}"
        )
    return backend


def _torch_npu_version() -> str | None:
    try:
        import torch_npu  # type: ignore
    except Exception:  # noqa: BLE001 - probe records unavailable optional runtime
        return None
    return str(getattr(torch_npu, "__version__", "unknown"))


def _cann_identity() -> str:
    """Return the runtime CANN identity when exposed, otherwise ``unknown``."""

    candidates = (
        getattr(torch.version, "cann", None),
        os.environ.get("CANN_VERSION"),
        os.environ.get("ASCEND_VERSION"),
    )
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return "unknown"


def _environment_section(backend: DistNPUBackend) -> dict[str, object]:
    """Exercise paired-real kernels and report the strict runtime identity."""

    device = backend._device
    real = torch.ones((2, 2), dtype=torch.float32, device=device)
    imag = torch.zeros((2, 2), dtype=torch.float32, device=device)
    left = _Pair(real, imag)
    right = _Pair(real * 2.0, imag)
    index = torch.tensor([1, 0], dtype=torch.long, device=device)
    operations = {
        "add": left.add(right),
        "mul": left.mul(right),
        "div_real": left.div_real(torch.tensor(2.0, dtype=torch.float32, device=device)),
        "matmul": left.matmul(right),
        "dagger": left.dagger(),
        "index_select": left.index_select(0, index),
    }
    operations["abs_sq"] = left.abs_sq()
    paired_real_on_npu = {
        name: (
            value.real.device == device
            and value.imag.device == device
            and value.real.dtype == torch.float32
            and value.imag.dtype == torch.float32
        )
        if isinstance(value, _Pair)
        else value.device == device and value.dtype == torch.float32
        for name, value in operations.items()
    }
    passed = all(paired_real_on_npu.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "device_mapping": {
            "rank": backend.rank,
            "local_rank": backend.local_rank,
            "device": str(device),
        },
        "backend": torch.distributed.get_backend(),
        "dtype_capabilities": {
            "paired_real": "float32",
            "backend_state": str(backend._dtype),
            "complex_collectives": bool(getattr(backend.communicator, "supports_complex", False)),
        },
        "versions": {
            "torch": str(torch.__version__),
            "torch_npu": _torch_npu_version(),
            "cann": _cann_identity(),
        },
        "paired_real_on_npu": paired_real_on_npu,
    }


def _communication_section(backend: DistNPUBackend) -> dict[str, object]:
    """Exercise each distributed axis with paired-real forward/backward P2P.

    ``exchange_real`` records only after its asynchronous P2P work handles
    have completed, so the returned evidence is also a teardown-safety check.
    """

    communicator = backend.communicator
    communicator.clear_communication_records()
    axes = tuple(range(backend.world_size.bit_length() - 1))
    real = torch.tensor(
        [float(backend.rank + 1)],
        dtype=torch.float32,
        device=backend._device,
        requires_grad=True,
    )
    imag = torch.tensor(
        [-float(backend.rank + 1)],
        dtype=torch.float32,
        device=backend._device,
        requires_grad=True,
    )
    pair = _Pair(real, imag)
    for axis in axes:
        pair = _exchange_pair(
            pair,
            communicator=communicator,
            peer=backend.rank ^ (1 << axis),
            operation_index=axis,
            phase="forward",
        )

    before_local_gate = len(communicator.communication_records)
    local_gate_pair = pair.mul(pair)
    local_gate_p2p_delta = len(communicator.communication_records) - before_local_gate
    local_gate_pair.abs_sq().sum().backward()
    records = list(communicator.communication_records)
    exchange_records = [record for record in records if record["kind"] == "exchange"]
    forward_tags = sorted(
        record["tag"] for record in exchange_records if record["tag"] % 8 < 4
    )
    backward_tags = sorted(
        record["tag"] for record in exchange_records if record["tag"] % 8 >= 4
    )
    payload_dtypes = sorted({record["dtype"] for record in records})
    peers = sorted({record["peer"] for record in exchange_records})
    expected_per_phase = 2 * len(axes)
    forward_p2p = len(forward_tags)
    backward_p2p = len(backward_tags)
    try:
        _validate_tag_phases(forward_tags, backward_tags)
        tags_valid = True
    except ValueError:
        tags_valid = False
    passed = (
        local_gate_p2p_delta == 0
        and forward_p2p == expected_per_phase
        and backward_p2p == expected_per_phase
        and payload_dtypes == ["torch.float32"]
        and all(peer is not None and peer != backend.rank and 0 <= peer < backend.world_size for peer in peers)
        and tags_valid
        and all(record["bytes"] > 0 for record in records)
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "distributed_axes": list(axes),
        "local_gate_p2p_delta": local_gate_p2p_delta,
        "forward_p2p": forward_p2p,
        "backward_p2p": backward_p2p,
        "payload_dtypes": payload_dtypes,
        "peers": peers,
        "forward_tags": forward_tags,
        "backward_tags": backward_tags,
        "transport_bytes": sum(record["bytes"] for record in records),
        "all_handles_complete": True,
    }


def _synchronize_npu(backend) -> None:
    """Make one timing interval device-complete without a CPU fallback."""

    torch.npu.synchronize(backend._device)


def _rank_disagreement_float32(values, backend) -> float:
    """Measure the maximum all-rank disagreement through fixed float32 payloads."""

    payload = torch.as_tensor(
        values,
        dtype=torch.float32,
        device=backend._device,
    ).reshape(-1).contiguous()
    gathered = backend.communicator.all_gather_real(payload)
    expected_world_size = int(getattr(backend, "world_size", len(gathered)))
    if len(gathered) != expected_world_size:
        raise RuntimeError("rank disagreement exchange did not return every rank")
    reference = gathered[0].detach().to(dtype=torch.float32).reshape(-1)
    if any(
        item.dtype != torch.float32
        or tuple(item.shape) != tuple(reference.shape)
        for item in gathered
    ):
        raise RuntimeError("rank disagreement exchange requires fixed float32 payloads")
    disagreement = max(
        (
            float(
                (item.detach().reshape(-1) - reference)
                .abs()
                .max()
                .cpu()
            )
            for item in gathered[1:]
        ),
        default=0.0,
    )
    if not math.isfinite(disagreement):
        raise RuntimeError("rank disagreement measurement must be finite")
    return disagreement


_BENCHMARK_PATH_METHODS = {
    # Statevector has two deliberately distinct workloads: independent RY
    # gate angles for shift-rule checks, and paired-real raw amplitudes for
    # native/finite-difference checks.
    "statevector": {"native", "parameter_shift", "finite_difference"},
    # Every path has a native paired-real VJP.  FD is the independent oracle
    # for raw density/Stinespring factors and the explicitly named channel
    # logit used by the noise workload.
    "density": {"native", "finite_difference"},
    "noise": {"native", "finite_difference"},
    "stinespring": {"native", "finite_difference"},
}


_BENCHMARK_PARAMETER_FAMILIES = {
    ("statevector", "native"): {"gate_angle", "raw_state"},
    ("statevector", "parameter_shift"): {"gate_angle"},
    ("statevector", "finite_difference"): {"raw_state"},
    ("density", "native"): {"density_factor"},
    ("density", "finite_difference"): {"density_factor"},
    ("noise", "native"): {"channel_logit"},
    ("noise", "finite_difference"): {"channel_logit"},
    ("stinespring", "native"): {"stinespring_factor"},
    ("stinespring", "finite_difference"): {"stinespring_factor"},
}


def _resolve_benchmark_parameter_family(path, gradient_method, parameter_family=None) -> str:
    """Resolve the physical leaf family without silently relabelling an oracle."""

    allowed = _BENCHMARK_PARAMETER_FAMILIES.get((path, gradient_method))
    if allowed is None:
        raise ValueError(f"{gradient_method} is not implemented for {path} benchmark workload")
    if parameter_family is None:
        # Keep the existing statevector/native CLI behavior as a gate-angle
        # workload.  Raw state native is selected explicitly by the probe.
        return "gate_angle" if (path, gradient_method) == ("statevector", "native") else next(iter(allowed))
    if parameter_family not in allowed:
        names = ", ".join(sorted(allowed))
        raise ValueError(f"{path}/{gradient_method} requires parameter_family in {{{names}}}")
    return str(parameter_family)


def _validate_benchmark_workload_config(*, path, gradient_method, n_qubits, depth, parameters, world_size, parameter_family=None):
    """Reject a configuration unless it has a real paired-real implementation.

    One extra local qubit is required in addition to the shard selector axes.
    Thus every workload includes a gate/channel on a distributed axis instead
    of turning a requested multi-rank benchmark into local arithmetic.
    """

    if path not in _BENCHMARK_PATH_METHODS or gradient_method not in _BENCHMARK_PATH_METHODS[path]:
        raise ValueError(f"{gradient_method} is not implemented for {path} benchmark workload")
    family = _resolve_benchmark_parameter_family(path, gradient_method, parameter_family)
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in (n_qubits, depth, parameters, world_size)):
        raise ValueError("benchmark n_qubits/depth/parameters/world_size must be positive integers")
    distributed_axes = int(math.log2(world_size))
    if (1 << distributed_axes) != world_size:
        raise ValueError("benchmark world_size must be a power of two")
    if distributed_axes == 0:
        raise ValueError("benchmark requires a multi-rank P2P workload")
    if n_qubits < distributed_axes + 1:
        raise ValueError("benchmark needs one local qubit plus every distributed shard axis")
    return family


def _benchmark_layout_and_spec(backend, n_qubits):
    distributed_axes = int(math.log2(backend.world_size))
    layout = _Layout.explicit(tuple(range(n_qubits)), n_qubits=n_qubits, distributed_axes=distributed_axes)
    return layout, _ShardSpec.build(n_qubits, backend.world_size, backend.rank, "vector", layout)


def _benchmark_observable(n_qubits):
    return PauliString("Z" + "I" * (n_qubits - 1), n_qubits=n_qubits)


def _benchmark_leaves(backend, parameters, *, requires_grad):
    return tuple(
        torch.tensor(0.19 + 0.03 * index, dtype=torch.float32, device=backend._device, requires_grad=requires_grad)
        for index in range(parameters)
    )


def _benchmark_raw_state_pair(backend, spec, values) -> _Pair:
    """Build one globally normalized raw paired-real state from local factors."""

    indices = torch.arange(spec.global_start, spec.global_stop, dtype=torch.float32, device=backend._device).reshape(-1, 1)
    real = 0.31 + 0.07 * (indices + 1.0)
    imag = -0.13 + 0.05 * (indices + 1.0)
    for index, value in enumerate(values):
        frequency = float(index + 1)
        real = real + value * torch.sin(frequency * (indices + 1.0))
        imag = imag + value * torch.cos(frequency * (indices + 1.0))
    # PureStateParam is the public raw-amplitude container.  Its local
    # normalization is intentionally not used here: a distributed initial
    # state must have one global paired-real norm across all shards.
    raw = PureStateParam(real, imag)._raw_pair()
    local_norm_sq = raw.abs_sq().sum()
    global_norm_sq = _replicated_all_reduce(local_norm_sq.reshape(()), communicator=backend.communicator) * backend.world_size
    return raw.div_real(torch.sqrt(global_norm_sq))


def _benchmark_density_factor_state(backend, vector_spec, values) -> DistState:
    """Scatter one root-owned trace-one DensityParam factorization by row."""

    dimension = 1 << vector_spec.n_qubits
    matrix_spec = _ShardSpec.build(vector_spec.n_qubits, backend.world_size, backend.rank, "matrix", vector_spec.layout)
    root_pair = None
    if backend.rank == 0:
        entries = torch.arange(dimension * dimension, dtype=torch.float32, device=backend._device).reshape(dimension, dimension)
        real = 0.08 + entries / float(dimension * dimension + 3)
        imag = -0.04 + (entries.remainder(dimension) / float(dimension + 5))
        for index, value in enumerate(values):
            real = real + value * torch.sin((index + 1.0) * (entries + 1.0))
            imag = imag + value * torch.cos((index + 1.0) * (entries + 1.0))
        density = DensityParam(real, imag).density_pair()
        root_pair = _Pair(
            density.real.reshape((backend.world_size,) + matrix_spec.local_shape),
            density.imag.reshape((backend.world_size,) + matrix_spec.local_shape),
        )
    return DistState.from_pair(
        _scatter_root_pair(
            root_pair,
            communicator=backend.communicator,
            root=0,
            local_shape=matrix_spec.local_shape,
        ),
        spec=matrix_spec,
        backend=backend,
    )


def _benchmark_unitary_workload(backend, *, path, parameter_family, values, n_qubits, depth):
    """Run the genuine vector/density kernels; every depth and leaf is consumed."""

    layout, vector_spec = _benchmark_layout_and_spec(backend, n_qubits)
    state_pair = (
        _benchmark_raw_state_pair(backend, vector_spec, values)
        if parameter_family == "raw_state"
        else _local_initial_pair(backend, vector_spec)
    )
    context = _AutogradExecutionContext()
    planner = _GatePlanner(backend, layout, n_qubits, execution_context=context)
    distributed_axes = layout.distributed_axes
    if path == "statevector":
        kernel, state = _PairVectorKernel(backend), state_pair
        apply = lambda current, plan, index: kernel.apply(current, plan, operation_index=index)
    else:
        kernel = _PairMatrixKernel(backend)
        state = _benchmark_density_factor_state(backend, vector_spec, values)
        apply = lambda current, plan, index: kernel.apply_unitary(current, plan, operation_index=index)
    operation_index = 40_000
    # Fixed depth layers are deliberately non-trainable, so parameter-shift is
    # valid for each independent RY leaf below rather than for an alias reused
    # at several circuit locations.
    for layer in range(depth):
        axis = layer % distributed_axes
        plan = planner.plan(ry(0.17 + 0.01 * layer, axis), operation_index)
        state = apply(state, plan, operation_index)
        operation_index += 1
    if parameter_family == "gate_angle":
        for parameter_index, value in enumerate(values):
            axis = parameter_index % distributed_axes
            plan = planner.plan(ry(value, axis), operation_index)
            state = apply(state, plan, operation_index)
            operation_index += 1
    if path == "statevector":
        return _PairReducer(backend).expectation(state, vector_spec, _benchmark_observable(n_qubits)), state, vector_spec
    matrix_spec = _ShardSpec.build(n_qubits, backend.world_size, backend.rank, "matrix", layout)
    return _PairReducer(backend).expectation(state._pair, matrix_spec, _benchmark_observable(n_qubits)), state._pair, matrix_spec


def _benchmark_channel_workload(backend, *, path, values, n_qubits, depth):
    """Run real built-in or Stinespring channels through the matrix kernel."""

    layout, vector_spec = _benchmark_layout_and_spec(backend, n_qubits)
    state = DistState.from_pair(_local_initial_pair(backend, vector_spec), spec=vector_spec, backend=backend)
    kernel = _PairMatrixKernel(backend)
    distributed_axes = layout.distributed_axes
    # ``_PairChannelKernel`` reserves a 256-wide subrange per channel and
    # matrix exchange expands it by world size; stay inside the collective
    # descriptor's bounded operation-index domain for every accepted config.
    operation_index = 1_000

    def channel_for(value, axis, *, trainable, parameter_index=0):
        if path == "noise":
            probability = torch.sigmoid(value) if trainable else float(value)
            return DepolarizingChannel(axis, probability)
        # Each leaf perturbs one entry of a non-collinear raw Stinespring
        # factor.  A global scale would be cancelled by Householder
        # normalization and therefore is not a genuine FD workload.
        entries = torch.arange(16, dtype=torch.float32, device=backend._device).reshape(4, 4)
        raw = 0.11 + 0.03 * entries
        imag = -0.07 + 0.02 * entries
        selected = (entries == float(parameter_index % 16)).to(dtype=torch.float32)
        raw = raw + value * selected
        imag = imag + 0.37 * value * selected
        return StinespringParam(2, 2, 2, raw, imag, target_qubits=(axis,))

    fixed = torch.tensor(0.23, dtype=torch.float32, device=backend._device)
    for layer in range(depth):
        state = kernel.apply_channel(state, channel_for(fixed, layer % distributed_axes, trainable=False, parameter_index=layer), instruction_index=operation_index)
        operation_index += 1
    for parameter_index, value in enumerate(values):
        state = kernel.apply_channel(state, channel_for(value, parameter_index % distributed_axes, trainable=True, parameter_index=parameter_index), instruction_index=operation_index)
        operation_index += 1
    matrix_spec = _ShardSpec.build(n_qubits, backend.world_size, backend.rank, "matrix", layout)
    return _PairReducer(backend).expectation(state._pair, matrix_spec, _benchmark_observable(n_qubits)), state._pair, matrix_spec


def _benchmark_workload_value(backend, *, path, parameter_family, values, n_qubits, depth):
    if path in {"statevector", "density"}:
        return _benchmark_unitary_workload(backend, path=path, parameter_family=parameter_family, values=values, n_qubits=n_qubits, depth=depth)
    return _benchmark_channel_workload(backend, path=path, values=values, n_qubits=n_qubits, depth=depth)


def _timed_benchmark_iteration(backend, forward, gradient):
    """Time one workload with exactly four device-completion boundaries."""

    _synchronize_npu(backend)
    started = time.perf_counter()
    forward_result = forward()
    _synchronize_npu(backend)
    forward_ms = (time.perf_counter() - started) * 1000.0
    _synchronize_npu(backend)
    started = time.perf_counter()
    gradient_result = gradient()
    _synchronize_npu(backend)
    backward_ms = (time.perf_counter() - started) * 1000.0
    return forward_result, gradient_result, forward_ms, backward_ms


def _benchmark_state_error(backend, actual, reference):
    local = torch.maximum((actual.real - reference.real).abs().max(), (actual.imag - reference.imag).abs().max())
    total = _replicated_all_reduce(local.reshape(()), communicator=backend.communicator)
    return float((total * backend.world_size).detach().cpu())


def run_benchmark_workload(backend, *, communication_mode, path, gradient_method, n_qubits, depth, parameters, warmups=5, runs=30, parameter_family=None):
    """Measure one shared, real paired-real workload for CLI and probe.

    No field is decorative: depth creates fixed kernel operations, parameters
    create independent trainable leaves, and n_qubits determines the sharded
    state shape.  Numerical methods rerun this exact dispatcher.
    """

    parameter_family = _validate_benchmark_workload_config(path=path, gradient_method=gradient_method, parameter_family=parameter_family, n_qubits=n_qubits, depth=depth, parameters=parameters, world_size=backend.world_size)
    if communication_mode not in {"baseline", "reuse", "overlap"}:
        raise ValueError("invalid paired-real communication mode")
    if warmups <= 0 or runs <= 0:
        raise ValueError("benchmark warmups and runs must be positive")
    communicator = backend.communicator
    communicator.set_autograd_communication_mode(communication_mode)
    if hasattr(communicator, "_autograd_pair_buffer_pool"):
        del communicator._autograd_pair_buffer_pool

    def evaluate(values, *, requires_grad=False):
        leaf_requires_grad = requires_grad and (parameter_family != "density_factor" or backend.rank == 0)
        leaves = _benchmark_leaves(backend, parameters, requires_grad=leaf_requires_grad) if values is None else tuple(
            torch.tensor(float(value), dtype=torch.float32, device=backend._device, requires_grad=leaf_requires_grad) for value in values
        )
        # The globally normalized raw state needs its sharded physical
        # contributions summed.  DensityParam uses a root-owned scatter whose
        # VJP gathers directly to root; channel and gate VJPs already cross
        # shard contributions and must not be bucketed again.
        workload_values = (
            _bucket_parameters(leaves, communicator=backend.communicator)
            if requires_grad and parameter_family == "raw_state"
            else leaves
        )
        value, state, spec = _benchmark_workload_value(backend, path=path, parameter_family=parameter_family, values=workload_values, n_qubits=n_qubits, depth=depth)
        return value, state, spec, leaves

    point = np.asarray([0.19 + 0.03 * index for index in range(parameters)], dtype=np.float64)

    def numerical_objective(points):
        return float(evaluate(points)[0].detach().cpu())

    def one_iteration():
        forward = lambda: evaluate(point, requires_grad=(gradient_method == "native"))
        captured = {}
        def gradient():
            value, _, _, leaves = captured["forward"]
            if gradient_method == "native":
                value.backward()
                return np.asarray([
                    float(leaf.grad.detach().cpu()) if leaf.grad is not None else 0.0
                    for leaf in leaves
                ])
            if gradient_method == "parameter_shift":
                return parameter_shift_gradient(numerical_objective, point)
            return finite_difference_gradient(numerical_objective, point, epsilon=1e-3)
        def captured_forward():
            captured["forward"] = forward()
            return captured["forward"]
        result, gradient, forward_ms, backward_ms = _timed_benchmark_iteration(backend, captured_forward, gradient)
        return result, np.asarray(gradient, dtype=np.float64), forward_ms, backward_ms

    for _ in range(warmups):
        one_iteration()
    communicator.clear_communication_records()
    pool = getattr(communicator, "_autograd_pair_buffer_pool", None)
    if pool is not None:
        pool.reuse_count = 0
    samples = [one_iteration() for _ in range(runs)]
    result, measured_gradient, _, _ = samples[-1]
    # Snapshot selected-mode transport before the baseline parity replay.  The
    # replay is deliberately excluded from timings/bytes yet compares the
    # same global state and gradient graph across communication modes.
    counters = dict(communicator.communication_counters)
    reuse_count = int(getattr(pool, "reuse_count", 0))
    communicator.set_autograd_communication_mode("baseline")
    native_value, native_state, _, native_leaves = evaluate(point, requires_grad=True)
    native_value.backward()
    native_gradient = np.asarray([
        float(leaf.grad.detach().cpu()) if leaf.grad is not None else 0.0
        for leaf in native_leaves
    ])
    reference_value, reference_state, _, _ = evaluate(point)
    state_error = _benchmark_state_error(backend, result[1], reference_state)
    local_gradient_error = float(np.max(np.abs(measured_gradient - native_gradient)))
    if parameter_family == "density_factor":
        # DensityParam leaves are root-owned by construction.  The root
        # gradient is the sole physical parameter VJP; other ranks consume the
        # scattered state but intentionally own no duplicate leaf.
        local_gradient_error = local_gradient_error if backend.rank == 0 else 0.0
        gradient_error = float((_replicated_all_reduce(
            torch.tensor(local_gradient_error, dtype=torch.float32, device=backend._device),
            communicator=backend.communicator,
        ) * backend.world_size).detach().cpu())
    else:
        gradient_error = local_gradient_error
    communicator.set_autograd_communication_mode(communication_mode)
    state_value = float(reference_value.detach().cpu())
    rank_disagreement = _rank_disagreement_float32([state_value], backend)
    return {
        "parameter_family": parameter_family,
        "forward_ms_median": float(np.median([sample[2] for sample in samples])),
        "backward_ms_median": float(np.median([sample[3] for sample in samples])),
        "gradient_ms_median": float(np.median([sample[2] + sample[3] for sample in samples])),
        "gradient_ms_p95": float(np.percentile([sample[2] + sample[3] for sample in samples], 95)),
        "p2p_bytes": int(counters["bytes"]),
        "wait_ms": float(counters["p2p_wait_ms"]),
        "buffer_reuse_count": reuse_count,
        "state_max_abs_error": state_error,
        "gradient_max_abs_error": gradient_error,
        "all_handles_complete": bool(communicator.work_handle_status["all_handles_complete"]),
        "fallback_to_cpu": False,
        "state_value": state_value,
        "rank_disagreement": rank_disagreement,
    }


def _performance_section(backend: DistNPUBackend) -> dict[str, object]:
    """Measure every native-oracle pair under baseline, reuse, and overlap."""

    n_qubits = int(math.log2(backend.world_size)) + 1
    workloads = (
        ("statevector", "gate_angle", "native"),
        ("statevector", "gate_angle", "parameter_shift"),
        ("statevector", "raw_state", "native"),
        ("statevector", "raw_state", "finite_difference"),
        ("density", "density_factor", "native"),
        ("density", "density_factor", "finite_difference"),
        ("noise", "channel_logit", "native"),
        ("noise", "channel_logit", "finite_difference"),
        ("stinespring", "stinespring_factor", "native"),
        ("stinespring", "stinespring_factor", "finite_difference"),
    )
    records = []
    for path, parameter_family, gradient_method in workloads:
        modes = {
            mode: run_benchmark_workload(
                backend,
                communication_mode=mode,
                path=path,
                parameter_family=parameter_family,
                gradient_method=gradient_method,
                n_qubits=n_qubits,
                depth=1,
                parameters=1,
            )
            for mode in ("baseline", "reuse", "overlap")
        }
        records.append({
            "path": path,
            "parameter_family": parameter_family,
            "gradient_method": gradient_method,
            "modes": modes,
        })
    backend.communicator.set_autograd_communication_mode("baseline")
    passed = all(
        metrics["parameter_family"] == record["parameter_family"]
        and 0.0 <= metrics["state_max_abs_error"] <= 1e-6
        and 0.0 <= metrics["gradient_max_abs_error"] <= 1e-4
        and math.isfinite(metrics["gradient_ms_median"])
        and metrics["gradient_ms_median"] > 0.0
        and math.isfinite(metrics["rank_disagreement"])
        and 0.0 <= metrics["rank_disagreement"] <= 1e-6
        and metrics["p2p_bytes"] > 0
        and metrics["all_handles_complete"]
        and not metrics.get("fallback_to_cpu", False)
        for record in records
        for metrics in record["modes"].values()
    ) and all(
        record["modes"]["baseline"]["buffer_reuse_count"] == 0
        and record["modes"]["reuse"]["buffer_reuse_count"] > 0
        and record["modes"]["overlap"]["buffer_reuse_count"] > 0
        for record in records
    ) and all(
        native["modes"][mode]["gradient_ms_median"]
        < oracle["modes"][mode]["gradient_ms_median"]
        for native, oracle in zip(records[::2], records[1::2])
        for mode in ("baseline", "reuse", "overlap")
    )
    return {"status": "PASS" if passed else "FAIL", "passed": passed, "warmups": 5, "runs": 30, "workloads": records, "fallback_to_cpu": False}


def _layout_and_spec(backend):
    n_qubits = backend.world_size.bit_length()
    layout = _Layout.explicit(
        tuple(range(n_qubits)), n_qubits=n_qubits, distributed_axes=n_qubits - 1
    )
    return layout, _ShardSpec.build(
        n_qubits, backend.world_size, backend.rank, "vector", layout
    )


def _local_initial_pair(backend, spec, *, requires_grad=False):
    """Build exactly this rank's normalized shard from global indices."""

    indices = torch.arange(
        spec.global_start, spec.global_stop, dtype=torch.float32, device=backend._device
    )
    dimension = 1 << spec.n_qubits
    norm_sq = dimension * (dimension + 1) * (2 * dimension + 1) / 6.0
    real = ((indices + 1.0) / math.sqrt(norm_sq)).reshape(-1, 1)
    if requires_grad:
        real = real.detach().requires_grad_(True)
    return _Pair(real, torch.zeros_like(real, requires_grad=requires_grad))


def _native_pair_value(
    backend,
    theta,
    axis,
    *,
    probability=False,
    observable=None,
    gate_factory=ry,
    trainable_state=False,
):
    """One paired-real evaluation with a fresh explicit planning context."""

    layout, spec = _layout_and_spec(backend)
    pair = _local_initial_pair(backend, spec, requires_grad=trainable_state)
    context = _AutogradExecutionContext()
    planner = _GatePlanner(
        backend, layout, spec.n_qubits, execution_context=context
    )
    parameters = theta if isinstance(theta, tuple) else (theta,)
    plan = planner.plan(gate_factory(*parameters, axis), axis)
    evolved = _PairVectorKernel(backend).apply(pair, plan, operation_index=axis)
    reducer = _PairReducer(backend)
    value = reducer.probabilities(evolved, spec) if probability else reducer.expectation(
        evolved,
        spec,
        observable or PauliString("Z" + "I" * (spec.n_qubits - 1), n_qubits=spec.n_qubits),
    )
    return value, pair, spec


def _custom_pair_unitary(theta, axis):
    """A trainable RX matrix supplied through the public custom-_Pair route."""

    zero = torch.zeros((), dtype=torch.float32, device=theta.device)
    c, s = torch.cos(theta / 2.0), torch.sin(theta / 2.0)
    return {
        "type": "unitary",
        "parameter": _Pair(
            torch.stack((torch.stack((c, zero)), torch.stack((zero, c)))),
            torch.stack((torch.stack((zero, -s)), torch.stack((-s, zero)))),
        ),
        "n_qubits": 1,
        "qubits": (axis,),
    }


def _gradient_section(backend: DistNPUBackend, *, gate_factory=ry, values=(0.31,), observable=None, four_term=False):
    errors = []
    for axis in range(backend.world_size.bit_length() - 1):
        leaves = tuple(
            torch.tensor(value, dtype=torch.float32, device=backend._device, requires_grad=True)
            for value in values
        )
        value, _, _ = _native_pair_value(
            backend, leaves, axis, gate_factory=gate_factory, observable=observable
        )
        value.backward()
        native = np.array([float(leaf.grad.detach().cpu()) for leaf in leaves])
        shift = psr4 if four_term else parameter_shift_gradient
        shifted = shift(
            lambda point: float(_native_pair_value(
                backend,
                tuple(torch.tensor(float(item), dtype=torch.float32, device=backend._device) for item in point),
                axis,
                gate_factory=gate_factory,
                observable=observable,
            )[0].detach().cpu()),
            np.asarray(values, dtype=np.float64),
        )
        errors.append(float(np.max(np.abs(native - shifted))))
    maximum = max(errors, default=0.0)
    return {"status": "PASS" if maximum <= 1e-4 else "FAIL", "passed": maximum <= 1e-4, "max_abs_error": maximum, "distributed_axes": list(range(backend.world_size.bit_length() - 1))}


def _statevector_section(backend):
    errors, gradients = [], []
    n_qubits = backend.world_size.bit_length()
    observable = PauliString("Z" + "I" * (n_qubits - 1), n_qubits=n_qubits)
    for axis in range(n_qubits - 1):
        theta = torch.tensor(0.31, dtype=torch.float32, device=backend._device, requires_grad=True)
        value, pair, _ = _native_pair_value(
            backend, theta, axis, observable=observable, trainable_state=True
        )
        value.backward()
        gradients.extend((pair.real.grad, pair.imag.grad))
        errors.append(_gradient_section(backend, observable=observable)["max_abs_error"])
    maximum = max(errors, default=0.0)
    state_finite = all(bool(torch.isfinite(item).all().detach().cpu()) for item in gradients)
    simulator = DistSimulator(backend)
    layout = _Layout.explicit(
        tuple(reversed(range(n_qubits))),
        n_qubits=n_qubits,
        distributed_axes=n_qubits - 1,
    )
    spec = _ShardSpec.build(
        n_qubits, backend.world_size, backend.rank, "vector", layout
    )
    dimension = 1 << n_qubits
    root_real = (
        torch.arange(
            1, dimension + 1, dtype=torch.float32, device=backend._device,
        ).requires_grad_()
        if backend.rank == 0
        else None
    )
    root_imag = (
        torch.arange(
            dimension, dtype=torch.float32, device=backend._device,
        ).requires_grad_()
        if backend.rank == 0
        else None
    )
    root_state = simulator._prepare_initial_state(
        n_qubits=n_qubits,
        layout=layout,
        initial_state=(PureStateParam(root_real, root_imag) if backend.rank == 0 else None),
        initial_density_matrix=None,
    )
    root_loss = _PairReducer(backend).expectation(
        root_state._pair,
        spec,
        PauliString("Z" + "I" * (n_qubits - 1), n_qubits=n_qubits),
    )
    root_loss.backward()
    root_materialized = root_state.to_numpy(root=0)
    root_real_reference = np.arange(1, dimension + 1, dtype=np.float64)
    root_imag_reference = np.arange(dimension, dtype=np.float64)
    root_complex_reference = root_real_reference + 1j * root_imag_reference
    logical_signs = np.concatenate(
        (np.ones(dimension // 2), -np.ones(dimension // 2))
    )
    root_norm_sq = float(np.sum(np.abs(root_complex_reference) ** 2))
    root_value_reference = float(
        np.sum(logical_signs * np.abs(root_complex_reference) ** 2)
        / root_norm_sq
    )
    root_real_gradient_reference = (
        2.0
        * (logical_signs - root_value_reference)
        * root_real_reference
        / root_norm_sq
    )
    root_imag_gradient_reference = (
        2.0
        * (logical_signs - root_value_reference)
        * root_imag_reference
        / root_norm_sq
    )
    if backend.rank == 0:
        root_owned_layout_amplitude_error = float(
            np.max(
                np.abs(
                    root_materialized.reshape(-1)
                    - root_complex_reference / math.sqrt(root_norm_sq)
                )
            )
        )
        root_owned_layout_value_error = abs(
            float(root_loss.detach().cpu()) - root_value_reference
        )
        root_owned_layout_gradient_error = max(
            float(
                np.max(
                    np.abs(
                        root_real.grad.detach().cpu().numpy()
                        - root_real_gradient_reference
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        root_imag.grad.detach().cpu().numpy()
                        - root_imag_gradient_reference
                    )
                )
            ),
        )
    else:
        root_owned_layout_amplitude_error = 0.0
        root_owned_layout_value_error = 0.0
        root_owned_layout_gradient_error = 0.0
    root_owned_gradient_finite = (
        True
        if backend.rank != 0
        else all(
            gradient is not None and bool(torch.isfinite(gradient).all().detach().cpu())
            for gradient in (root_real.grad, root_imag.grad)
        )
    )

    shard_real = torch.full(
        spec.local_shape,
        float(backend.rank + 1),
        dtype=torch.float32,
        device=backend._device,
        requires_grad=True,
    )
    shard_imag = torch.full(
        spec.local_shape,
        float(-backend.rank),
        dtype=torch.float32,
        device=backend._device,
        requires_grad=True,
    )
    sharded_state = simulator._prepare_initial_state(
        n_qubits=n_qubits,
        layout=layout,
        initial_state=DistState.from_pair(
            _Pair(shard_real, shard_imag), spec=spec, backend=backend
        ),
        initial_density_matrix=None,
    )
    sharded_value = _PairReducer(backend).expectation(
        sharded_state._pair,
        spec,
        PauliString("Z" + "I" * (n_qubits - 1), n_qubits=n_qubits),
    )
    sharded_value.backward()
    storage_real_reference = np.concatenate(
        [
            np.full(spec.local_shape[0], float(rank + 1))
            for rank in range(backend.world_size)
        ]
    )
    storage_imag_reference = np.concatenate(
        [
            np.full(spec.local_shape[0], -float(rank))
            for rank in range(backend.world_size)
        ]
    )
    storage_reference = storage_real_reference + 1j * storage_imag_reference
    logical_reference = (
        storage_reference.reshape([2] * n_qubits)
        .transpose(layout.logical_to_storage)
        .reshape(-1)
    )
    sharded_value_reference = float(
        np.sum(logical_signs * np.abs(logical_reference) ** 2)
    )
    logical_real_gradient = 2.0 * logical_signs * logical_reference.real
    logical_imag_gradient = 2.0 * logical_signs * logical_reference.imag
    storage_real_gradient = (
        logical_real_gradient.reshape([2] * n_qubits)
        .transpose(layout.storage_to_logical)
        .reshape(-1)
    )
    storage_imag_gradient = (
        logical_imag_gradient.reshape([2] * n_qubits)
        .transpose(layout.storage_to_logical)
        .reshape(-1)
    )
    local_slice = slice(spec.global_start, spec.global_stop)
    sharded_layout_value_error = abs(
        float(sharded_value.detach().cpu()) - sharded_value_reference
    )
    sharded_layout_gradient_error = max(
        float(
            np.max(
                np.abs(
                    shard_real.grad.detach().cpu().numpy().reshape(-1)
                    - storage_real_gradient[local_slice]
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    shard_imag.grad.detach().cpu().numpy().reshape(-1)
                    - storage_imag_gradient[local_slice]
                )
            )
        ),
    )
    sharded_gradient_finite = all(
        gradient is not None and bool(torch.isfinite(gradient).all().detach().cpu())
        for gradient in (shard_real.grad, shard_imag.grad)
    )
    passed = (
        state_finite
        and root_owned_gradient_finite
        and sharded_gradient_finite
        and maximum <= 1e-4
        and root_owned_layout_amplitude_error <= 1e-4
        and root_owned_layout_value_error <= 1e-4
        and root_owned_layout_gradient_error <= 1e-4
        and sharded_layout_value_error <= 1e-4
        and sharded_layout_gradient_error <= 1e-4
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "distributed_axes": list(range(n_qubits - 1)),
        "initial_state_gradient_finite": state_finite,
        "root_owned_initial_state_gradient_finite": root_owned_gradient_finite,
        "sharded_initial_state_gradient_finite": sharded_gradient_finite,
        "layout": list(layout.logical_to_storage),
        "root_owned_layout_amplitude_error": root_owned_layout_amplitude_error,
        "root_owned_layout_value_error": root_owned_layout_value_error,
        "root_owned_layout_gradient_error": root_owned_layout_gradient_error,
        "sharded_layout_value_error": sharded_layout_value_error,
        "sharded_layout_gradient_error": sharded_layout_gradient_error,
        "max_abs_error": maximum,
    }


def _cpu_density_objective(state, theta, *, logical_axis, layout):
    """Independent complex128 ``<Z_0>`` oracle in the layout's storage basis."""

    state = np.asarray(state, dtype=np.complex128).reshape(-1)
    gate = np.array(
        [[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]],
        dtype=np.complex128,
    )
    unitary = np.array([[1.0]], dtype=np.complex128)
    observable = np.array([[1.0]], dtype=np.complex128)
    for storage_axis in range(layout.n_qubits):
        unitary = np.kron(
            unitary,
            gate if storage_axis == layout.logical_to_storage[logical_axis] else np.eye(2),
        )
        observable = np.kron(
            observable,
            np.diag([1.0, -1.0]) if storage_axis == layout.logical_to_storage[0] else np.eye(2),
        )
    evolved = unitary @ state
    return float(np.vdot(evolved, observable @ evolved).real)


def _density_section(backend):
    """Validate paired-real density values and gradients without a CPU fallback.

    CPU float64 and parameter shift are correctness oracles only; the native
    execution below remains entirely in paired float32 buffers on HCCL.
    """

    layout, vector_spec = _layout_and_spec(backend)
    matrix_spec = _ShardSpec.build(
        vector_spec.n_qubits, backend.world_size, backend.rank, "matrix", layout
    )
    n_qubits, dimension = vector_spec.n_qubits, 1 << vector_spec.n_qubits
    raw = np.arange(1, dimension + 1, dtype=np.float64)
    raw /= np.linalg.norm(raw)
    value_errors, gradient_errors, physical_errors = [], [], []
    logical_axes = tuple(
        logical for logical, storage in enumerate(layout.logical_to_storage)
        if storage < layout.distributed_axes
    )
    for axis in logical_axes:
        theta = torch.tensor(0.31, dtype=torch.float32, device=backend._device, requires_grad=True)
        vector = DistState.from_pair(
            _local_initial_pair(backend, vector_spec), spec=vector_spec, backend=backend
        )
        kernel = _PairMatrixKernel(backend)
        density = kernel.promote_vector(vector)
        plan = _GatePlanner(backend, layout, n_qubits, execution_context=_AutogradExecutionContext()).plan(ry(theta, axis), axis)
        evolved = kernel.apply_unitary(density, plan, operation_index=axis)
        value = _PairReducer(backend).expectation(
            evolved._pair, matrix_spec, PauliString("Z" + "I" * (n_qubits - 1), n_qubits=n_qubits)
        )
        value.backward()
        reference = _cpu_density_objective(raw, 0.31, logical_axis=axis, layout=layout)
        value_errors.append(abs(float(value.detach().cpu()) - reference))
        shifted = parameter_shift_gradient(
            lambda point: _cpu_density_objective(raw, float(point[0]), logical_axis=axis, layout=layout),
            np.array([0.31], dtype=np.float64),
        )[0]
        gradient_errors.append(abs(float(theta.grad.detach().cpu()) - float(shifted)))
        local = evolved._pair
        rows = torch.arange(local.real.shape[0], dtype=torch.long, device=backend._device)
        columns = rows + matrix_spec.global_start
        trace = float(backend.world_size) * _replicated_all_reduce(local.real[rows, columns].sum().reshape(()), communicator=backend.communicator)
        physical_errors.append(abs(float(trace.detach().cpu()) - 1.0))

    # A direct DensityParam factor remains a paired-real finite-difference
    # check.  It is intentionally local here: root/sharded ownership wiring is
    # covered by the initial-state contract and no density is gathered.
    factor_real = torch.tensor([[1.0, -0.2], [0.3, 0.5]], dtype=torch.float32, device=backend._device, requires_grad=True)
    factor_imag = torch.tensor([[0.1, 0.4], [-0.2, 0.6]], dtype=torch.float32, device=backend._device, requires_grad=True)
    factor = DensityParam(factor_real, factor_imag).density_pair()
    factor_value = factor.real[0, 0] - factor.real[1, 1]
    factor_value.backward()
    epsilon = 1e-4
    def _factor_oracle(entry):
        real = factor_real.detach().cpu().numpy().astype(np.float64).copy(); real[0, 0] = entry
        imag = factor_imag.detach().cpu().numpy().astype(np.float64); value = real + 1j * imag
        rho = value @ value.conj().T; rho /= np.trace(rho)
        return float((rho[0, 0] - rho[1, 1]).real)
    factor_error = abs(float(factor_real.grad[0, 0].detach().cpu()) - (_factor_oracle(float(factor_real.detach()[0, 0]) + epsilon) - _factor_oracle(float(factor_real.detach()[0, 0]) - epsilon)) / (2 * epsilon))
    maximum = max((*value_errors, *gradient_errors, *physical_errors, factor_error), default=0.0)
    passed = maximum <= 1e-4
    return {"status": "PASS" if passed else "FAIL", "passed": passed, "distributed_axes": list(logical_axes), "storage_axes": [layout.logical_to_storage[axis] for axis in logical_axes], "value_error": max(value_errors, default=0.0), "gradient_error": max(gradient_errors, default=0.0), "trace_error": max(physical_errors, default=0.0), "density_factor_finite_difference_error": factor_error, "max_abs_error": maximum}


def _memory_growth_percent(measurements) -> float:
    """Return non-negative growth from at least two same-policy measurements."""

    values = [float(value) for value in measurements]
    if len(values) < 2 or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("memory growth requires at least two finite non-negative measurements")
    baseline = values[0]
    growth = max(0.0, (values[-1] - baseline) / max(baseline, 1.0) * 100.0)
    if not math.isfinite(growth):
        raise ValueError("memory growth measurement must be finite")
    return growth


def _memory_section(backend):
    """Record measured checkpoint accounting through the private native hook.

    This is a preliminary measurement section, not a public-autograd claim:
    it deliberately calls ``_run_paired_real`` while ``DistSimulator.run``
    keeps its forward-only release gate closed.
    """

    n_qubits = backend.world_size.bit_length()
    layout = _Layout.explicit(
        tuple(reversed(range(n_qubits))),
        n_qubits=n_qubits,
        distributed_axes=n_qubits - 1,
    )
    spec = _ShardSpec.build(n_qubits, backend.world_size, backend.rank, "vector", layout)
    observable = PauliString("Z" + "I" * (n_qubits - 1), n_qubits=n_qubits)
    reports = {}
    reference_gradient = None
    for policy in ("none", "auto", 16):
        measurements = []
        gradients = []
        policy_metrics = []
        for _ in range(2):
            theta = torch.tensor(0.31, dtype=torch.float32, device=backend._device, requires_grad=True)
            circuit = Circuit(n_qubits=n_qubits)
            for index in range(16):
                circuit.append(ry(theta, index % n_qubits))
            state = DistState.from_pair(_local_initial_pair(backend, spec), spec=spec, backend=backend)
            evolved, metrics = DistSimulator(backend)._measure_paired_real(
                circuit,
                initial_state=state,
                layout=layout,
                grad_checkpoint=policy,
            )
            value = _PairReducer(backend).expectation(evolved._pair, evolved.spec, observable)
            value.backward()
            gradient = float(theta.grad.detach().cpu())
            if reference_gradient is None:
                reference_gradient = gradient
            gradients.append(gradient)
            policy_metrics.append(metrics)
            measurements.append(
                None
                if metrics.peak_allocation_bytes is None
                else int(metrics.peak_allocation_bytes)
            )
        try:
            growth = _memory_growth_percent(measurements)
            measurement_failure = None
        except (TypeError, ValueError):
            growth = None
            measurement_failure = (
                "allocator peak measurement unavailable or non-finite"
            )
        reports[str(policy)] = {
            "saved_state_count": [int(item.saved_state_count) for item in policy_metrics],
            "recomputed_gate_count": [int(item.recomputed_gate_count) for item in policy_metrics],
            "chosen_interval": [int(item.interval) for item in policy_metrics],
            "peak_allocation_bytes": measurements,
            "peak_allocation_status": [item.peak_allocation_status for item in policy_metrics],
            "memory_source": [item.memory_source for item in policy_metrics],
            "gradient_error": max(abs(gradient - reference_gradient) for gradient in gradients),
            "memory_growth_percent": growth,
            "repeated_measurements": len(measurements),
            "measurement_failure": measurement_failure,
        }
    measured_growth = [
        item["memory_growth_percent"]
        for item in reports.values()
        if item["memory_growth_percent"] is not None
    ]
    maximum_growth = (
        max(measured_growth)
        if len(measured_growth) == len(reports)
        else None
    )
    passed = all(
        item["gradient_error"] <= 1e-4
        and item["repeated_measurements"] >= 2
        and item["memory_growth_percent"] is not None
        and math.isfinite(item["memory_growth_percent"])
        and item["memory_growth_percent"] <= 1.0
        for item in reports.values()
    )
    failed_invariants = []
    if any(item["measurement_failure"] is not None for item in reports.values()):
        failed_invariants.append("memory allocator measurements unavailable")
    if any(item["gradient_error"] > 1e-4 for item in reports.values()):
        failed_invariants.append("checkpoint gradients disagree")
    if any(
        item["memory_growth_percent"] is not None
        and item["memory_growth_percent"] > 1.0
        for item in reports.values()
    ):
        failed_invariants.append("memory growth exceeds 1%")
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "policies": reports,
        "repeated_policy": "each",
        "repeated_measurements": 2,
        "memory_growth_percent": maximum_growth,
        "failed_invariants": failed_invariants,
    }


def _channel_probe_pair(backend, spec, *, storage_axis: int, plus: bool) -> _Pair:
    """Return a normalized basis/plus state without a full-state materialization."""

    indices = torch.arange(spec.global_start, spec.global_stop, dtype=torch.long, device=backend._device)
    bit = 1 << (spec.n_qubits - 1 - storage_axis)
    if plus:
        real = ((indices == 0) | (indices == bit)).to(torch.float32).reshape(-1, 1) / math.sqrt(2.0)
    else:
        real = (indices == bit).to(torch.float32).reshape(-1, 1)
    return _Pair(real.detach().requires_grad_(True), torch.zeros_like(real, requires_grad=True))


def _noise_section(backend):
    """Probe analytic built-in channels on every distributed storage axis.

    Values use simple independently-derived one-qubit formulas; execution and
    transport remain entirely paired-real on the strict HCCL backend.
    """

    layout, vector_spec = _layout_and_spec(backend)
    matrix_spec = _ShardSpec.build(vector_spec.n_qubits, backend.world_size, backend.rank, "matrix", layout)
    logical_axes = tuple(logical for logical, storage in enumerate(layout.logical_to_storage) if storage < layout.distributed_axes)
    cases = (
        ("bit_flip", BitFlipChannel, False, "Z", lambda p: -1.0 + 2.0 * p, 2.0),
        ("phase_flip", PhaseFlipChannel, True, "X", lambda p: 1.0 - 2.0 * p, -2.0),
        ("depolarizing", DepolarizingChannel, False, "Z", lambda p: -1.0 + 4.0 * p / 3.0, 4.0 / 3.0),
        ("amplitude_damping", AmplitudeDampingChannel, False, "Z", lambda p: -1.0 + 2.0 * p, 2.0),
    )
    errors, transport, probability_gradients, probability_errors = {}, {}, {}, {}
    for logical_axis in logical_axes:
        storage_axis = layout.logical_to_storage[logical_axis]
        for name, factory, plus, word, reference, derivative in cases:
            probability = torch.tensor(0.23, dtype=torch.float32, device=backend._device, requires_grad=True)
            state = DistState.from_pair(_channel_probe_pair(backend, vector_spec, storage_axis=storage_axis, plus=plus), spec=vector_spec, backend=backend)
            backend.communicator.clear_communication_records()
            evolved = _PairMatrixKernel(backend).apply_channel(state, factory(logical_axis, probability), instruction_index=700 + logical_axis * 8 + len(errors))
            probabilities = _PairReducer(backend).probabilities(evolved._pair, matrix_spec)
            # Logical Pauli words use logical index order, so construct it explicitly.
            labels = ["I"] * vector_spec.n_qubits; labels[logical_axis] = word
            observable = PauliString("".join(labels), n_qubits=vector_spec.n_qubits)
            value = _PairReducer(backend).expectation(evolved._pair, matrix_spec, observable)
            value.backward()
            key = f"{name}:axis{storage_axis}"
            dimension = 1 << vector_spec.n_qubits
            bit = 1 << (vector_spec.n_qubits - 1 - storage_axis)
            expected_probabilities = np.zeros(dimension, dtype=np.float64)
            if plus:
                expected_probabilities[0], expected_probabilities[bit] = 0.5, 0.5
            elif name == "depolarizing":
                expected_probabilities[0], expected_probabilities[bit] = 2.0 * 0.23 / 3.0, 1.0 - 2.0 * 0.23 / 3.0
            else:
                expected_probabilities[0], expected_probabilities[bit] = 0.23, 0.77
            actual_probabilities = np.concatenate([
                part.detach().cpu().numpy()
                for part in backend.communicator.all_gather_real(probabilities.detach())
            ])
            probability_errors[key] = float(np.max(np.abs(actual_probabilities - expected_probabilities)))
            errors[key] = max(abs(float(value.detach().cpu()) - reference(0.23)), abs(float(probability.grad.detach().cpu()) - derivative), probability_errors[key])
            probability_gradients[key] = float(probability.grad.detach().cpu())
            records = [record for record in backend.communicator.communication_records if record["kind"] == "exchange"]
            transport[key] = {0, 1, 4, 5}.issubset({record["tag"] % 8 for record in records}) and all(record["dtype"] == "torch.float32" and record["bytes"] > 0 for record in records)
    maximum = max((*errors.values(), *probability_errors.values()), default=0.0)
    passed = maximum <= 1e-4 and all(transport.values())
    return {"status": "PASS" if passed else "FAIL", "passed": passed, "distributed_axes": list(range(layout.distributed_axes)), "channel_errors": errors, "probability_errors": probability_errors, "probability_pauli_gradients": probability_gradients, "forward_backward_p2p": transport, "max_abs_error": maximum}


def _stinespring_section(backend):
    """Probe fixed Householder Kraus order and physical channel invariants."""

    layout, vector_spec = _layout_and_spec(backend)
    dimension = 1 << vector_spec.n_qubits
    raw_size = 4 * dimension * dimension
    real_leaf = torch.linspace(-0.7, 0.9, raw_size, dtype=torch.float32, device=backend._device, requires_grad=True)
    imag_leaf = torch.linspace(0.8, -0.6, raw_size, dtype=torch.float32, device=backend._device, requires_grad=True)
    real, imag = real_leaf.reshape(2 * dimension, 2 * dimension), imag_leaf.reshape(2 * dimension, 2 * dimension)
    parameter = StinespringParam(dimension, dimension, 2, real, imag)
    kraus = _stinespring_kraus(parameter)
    completeness = _Pair(torch.zeros((dimension, dimension), dtype=torch.float32, device=backend._device), torch.zeros((dimension, dimension), dtype=torch.float32, device=backend._device))
    for matrix in kraus:
        completeness = completeness.add(matrix.dagger().matmul(matrix))
    identity = torch.eye(dimension, dtype=torch.float32, device=backend._device)
    completeness_error = float(torch.maximum((completeness.real - identity).abs().max(), completeness.imag.abs().max()).detach().cpu())
    state = DistState.from_pair(_local_initial_pair(backend, vector_spec, requires_grad=True), spec=vector_spec, backend=backend)
    backend.communicator.clear_communication_records()
    evolved = _PairMatrixKernel(backend).apply_channel(state, parameter, instruction_index=800)
    matrix_spec = _ShardSpec.build(vector_spec.n_qubits, backend.world_size, backend.rank, "matrix", layout)
    value = _PairReducer(backend).expectation(evolved._pair, matrix_spec, PauliString("Z" + "I" * (vector_spec.n_qubits - 1), n_qubits=vector_spec.n_qubits))
    value.backward()
    rows = torch.arange(evolved._pair.real.shape[0], dtype=torch.long, device=backend._device)
    columns = rows + matrix_spec.global_start
    trace = backend.world_size * _replicated_all_reduce(evolved._pair.real[rows, columns].sum().reshape(()), communicator=backend.communicator)
    records = [record for record in backend.communicator.communication_records if record["kind"] == "exchange"]
    transport = {0, 1, 4, 5}.issubset({record["tag"] % 8 for record in records}) and all(record["dtype"] == "torch.float32" and record["bytes"] > 0 for record in records)
    raw_gradient = max(float(real_leaf.grad.detach().abs().max().cpu()), float(imag_leaf.grad.detach().abs().max().cpu()))
    trace_error = abs(float(trace.detach().cpu()) - 1.0)
    actual = np.concatenate([part.detach().cpu().numpy() for part in backend.communicator.all_gather_real(evolved._pair.real.detach())], axis=0) + 1j * np.concatenate([part.detach().cpu().numpy() for part in backend.communicator.all_gather_real(evolved._pair.imag.detach())], axis=0)
    hermiticity_error = float(np.max(np.abs(actual - actual.conj().T)))
    positivity_error = max(0.0, -float(np.linalg.eigvalsh(actual).min()))
    passed = completeness_error <= 1e-5 and trace_error <= 1e-5 and hermiticity_error <= 1e-5 and positivity_error <= 1e-5 and transport and math.isfinite(raw_gradient)
    return {"status": "PASS" if passed else "FAIL", "passed": passed, "distributed_axes": list(range(layout.distributed_axes)), "target_qubits": list(parameter.target_qubits), "nonzero_targets": [axis for axis in parameter.target_qubits if axis != 0], "kraus_order": list(range(parameter.environment_dim)), "term_count": len(kraus), "completeness_error": completeness_error, "trace_error": trace_error, "hermiticity_error": hermiticity_error, "positivity_error": positivity_error, "max_raw_parameter_gradient": raw_gradient, "forward_backward_p2p": transport}


def _optimizer_digest_tensor(parameter, optimizer, backend):
    """Encode parameter and optimizer state equality as a float32 control value."""

    digest = hashlib.sha256()
    digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    for key in sorted(optimizer.state):
        for name, value in sorted(optimizer.state[key].items()):
            digest.update(str(name).encode("ascii"))
            if isinstance(value, torch.Tensor):
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                digest.update(repr(value).encode("ascii"))
    return torch.tensor(
        list(digest.digest()), dtype=torch.float32, device=backend._device
    )


def _optimizer_digest_tensors(parameters, optimizer, backend):
    """Encode a complete replicated optimizer state for control-plane equality."""

    digest = hashlib.sha256()
    for parameter in parameters:
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    for parameter_id, state in sorted(optimizer.state_dict()["state"].items()):
        digest.update(str(parameter_id).encode("ascii"))
        for name, value in sorted(state.items()):
            digest.update(name.encode("ascii"))
            digest.update(
                value.detach().cpu().contiguous().numpy().tobytes()
                if isinstance(value, torch.Tensor)
                else repr(value).encode("ascii")
            )
    return torch.tensor(
        list(digest.digest()), dtype=torch.float32, device=backend._device
    )


def _observed_handle_metrics(communicator):
    """Expose actual communicator work-handle state in probe reports."""

    status = communicator.work_handle_status
    return {
        "unfinished_work_handles": int(status["unfinished_work_handles"]),
        "all_handles_complete": bool(status["all_handles_complete"]),
    }


def _all_ranks_equal_real(payload: torch.Tensor, backend) -> bool:
    """Compare a contiguous fixed-shape float32 payload on every rank."""

    if payload.dtype != torch.float32 or torch.is_complex(payload):
        raise TypeError("optimizer agreement payload 必须是实数 torch.float32")
    payload = payload.detach().reshape(-1).contiguous()
    gathered = backend.communicator.all_gather_real(payload)
    return all(torch.equal(item, gathered[0]) for item in gathered[1:])


def _integrated_private_path_optimizer_case(backend):
    """Run the private engine with typed gate, built-in noise and Stinespring.

    This is intentionally a small acceptance subcase rather than a replacement
    for the required 100-step synthetic 32/128 optimizer matrix below.  It
    proves that the same bucket binds aliases from every replicated parameter
    family in one real circuit execution.
    """

    n_qubits = backend.world_size.bit_length() - 1
    layout = _Layout.explicit(
        tuple(reversed(range(n_qubits))),
        n_qubits=n_qubits,
        distributed_axes=n_qubits,
    )
    spec = _ShardSpec.build(n_qubits, backend.world_size, backend.rank, "vector", layout)
    theta = torch.nn.Parameter(torch.tensor(0.37, dtype=torch.float32, device=backend._device))
    damping = torch.nn.Parameter(torch.tensor(0.23, dtype=torch.float32, device=backend._device))
    missing = torch.nn.Parameter(torch.tensor(0.11, dtype=torch.float32, device=backend._device))
    raw_real = torch.nn.Parameter(
        torch.tensor([[0.4, -0.3], [0.2, 0.7]], dtype=torch.float32, device=backend._device)
    )
    raw_imag = torch.nn.Parameter(
        torch.tensor([[0.1, 0.6], [-0.5, 0.2]], dtype=torch.float32, device=backend._device)
    )
    stinespring = StinespringParam(2, 2, 1, raw_real, raw_imag, target_qubits=(0,))
    circuit = Circuit(ry(theta, 0), ry(theta, 0), n_qubits=n_qubits)
    circuit.noise_model = (
        NoiseModel()
        .add_channel(AmplitudeDampingChannel(0, damping))
        .add_channel(stinespring)
        # The unmatched rule intentionally verifies missing-gradient zero fill
        # in the same replicated bucket as the real trainable objects.
        .add_channel(BitFlipChannel(0, missing), after_gates=("never",))
    )
    leaves = (theta, damping, missing, raw_real, raw_imag)
    optimizer = torch.optim.SGD(leaves, lr=0.005, momentum=0.9)
    reports = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        backend.communicator.clear_communication_records()
        indices = torch.arange(
            spec.global_start, spec.global_stop, dtype=torch.long, device=backend._device
        )
        local_state = DistState.from_pair(
            _Pair(
                (indices == 0).to(torch.float32).reshape(-1, 1),
                torch.zeros(spec.local_shape, dtype=torch.float32, device=backend._device),
            ),
            spec=spec,
            backend=backend,
        )
        evolved, _ = DistSimulator(backend)._run_paired_real(
            circuit,
            initial_state=local_state,
            layout=layout,
            grad_checkpoint="none",
        )
        before_backward = len(backend.communicator.communication_records)
        value = _PairReducer(backend).expectation(
            evolved._pair,
            evolved.spec,
            PauliString("Z" + "I" * (n_qubits - 1), n_qubits=n_qubits),
        )
        value.backward()
        expected_bucket_bytes = sum(parameter.numel() for parameter in leaves) * 4
        bucket_records = [
            record
            for record in backend.communicator.communication_records[before_backward:]
            if record["kind"] == "all_reduce" and record["bytes"] == expected_bucket_bytes
        ]
        local_gradients = torch.cat(
            [parameter.grad.detach().reshape(-1) for parameter in leaves]
        ).contiguous()
        reports.append(
            {
                "gradient_all_reduce_count": len(bucket_records),
                "gradient_agreement": _all_ranks_equal_real(local_gradients, backend),
                "missing_gradient_zero": bool(torch.equal(missing.grad, torch.zeros_like(missing))),
                "caller_kraus_cache_empty": circuit.noise_model._kraus_cache == {},
                **_observed_handle_metrics(backend.communicator),
            }
        )
        optimizer.step()
    digest_agreement = _all_ranks_equal_real(
        _optimizer_digest_tensors(leaves, optimizer, backend), backend
    )
    passed = (
        digest_agreement
        and all(
            report["gradient_all_reduce_count"] == 1
            and report["gradient_agreement"]
            and report["missing_gradient_zero"]
            and report["caller_kraus_cache_empty"]
            and report["unfinished_work_handles"] == 0
            and report["all_handles_complete"]
            for report in reports
        )
    )
    return {
        "steps": 2,
        "gradient_all_reduce_count": [report["gradient_all_reduce_count"] for report in reports],
        "gradient_agreement": [report["gradient_agreement"] for report in reports],
        "parameter_and_optimizer_state_agree": digest_agreement,
        "missing_gradient_zero": [report["missing_gradient_zero"] for report in reports],
        "caller_kraus_cache_empty": [report["caller_kraus_cache_empty"] for report in reports],
        "unfinished_work_handles": [report["unfinished_work_handles"] for report in reports],
        "all_handles_complete": [report["all_handles_complete"] for report in reports],
        "passed": passed,
    }


def _optimizer_section(backend):
    """Measure the actual one-bucket SGD/Adam synchronization contract."""

    cases = {}
    for count in (32, 128):
        for name, factory in (
            ("sgd", lambda parameter: torch.optim.SGD([parameter], lr=0.01, momentum=0.9)),
            ("adam", lambda parameter: torch.optim.Adam([parameter], lr=0.01)),
        ):
            parameter = torch.nn.Parameter(
                torch.linspace(-0.4, 0.5, count, dtype=torch.float32, device=backend._device)
            )
            optimizer = factory(parameter)
            backend.communicator.clear_communication_records()
            agreement = True
            for _ in range(100):
                optimizer.zero_grad(set_to_none=True)
                (alias,) = _bucket_parameters((parameter,), communicator=backend.communicator)
                (alias * float(backend.rank + 1)).sum().backward()
                optimizer.step()
                agreement = agreement and _all_ranks_equal_real(
                    _optimizer_digest_tensor(parameter, optimizer, backend), backend
                )
            records = backend.communicator.communication_records
            all_reduce_records = [record for record in records if record["kind"] == "all_reduce"]
            handles = _observed_handle_metrics(backend.communicator)
            cases[f"{name}-{count}"] = {
                "gradient_all_reduce_count": len(all_reduce_records),
                "parameter_and_optimizer_state_agree": agreement,
                "all_float32": all(record["dtype"] == "torch.float32" for record in all_reduce_records),
                **handles,
            }
    integrated_private_path = _integrated_private_path_optimizer_case(backend)
    passed = integrated_private_path["passed"] and all(
        metrics["gradient_all_reduce_count"] == 100
        and metrics["parameter_and_optimizer_state_agree"]
        and metrics["all_float32"]
        and metrics["unfinished_work_handles"] == 0
        and metrics["all_handles_complete"]
        for metrics in cases.values()
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "steps": 100,
        "cases": cases,
        "integrated_private_path": integrated_private_path,
        "all_handles_complete": (
            all(metrics["all_handles_complete"] for metrics in cases.values())
            and all(integrated_private_path["all_handles_complete"])
        ),
    }


def _contract_section(backend):
    """Run the exact public native-autograd contract matrix on every rank."""

    n_qubits = backend.world_size.bit_length()
    layout = _Layout.explicit(
        tuple(reversed(range(n_qubits))),
        n_qubits=n_qubits,
        distributed_axes=n_qubits - 1,
    )
    spec = _ShardSpec.build(
        n_qubits, backend.world_size, backend.rank, "vector", layout
    )
    simulator = DistSimulator(backend)
    public_routing_enabled = False
    try:
        theta = torch.tensor(
            0.31,
            dtype=torch.float32,
            device=backend._device,
            requires_grad=True,
        )
        result = simulator.run(
            Circuit(ry(theta, 0), n_qubits=n_qubits),
            observables={"z": PauliString("Z" + "I" * (n_qubits - 1), n_qubits=n_qubits)},
        )
        result.expectations["z"].backward()
        public_routing_enabled = (
            result.state is not None
            and result.state._pair is not None
            and theta.grad is not None
        )
    except Exception:  # noqa: BLE001 - report the contract as a failed invariant
        public_routing_enabled = False

    def trainable_circuit(parameter=None):
        parameter = (
            torch.tensor(
                0.1,
                dtype=torch.float32,
                device=backend._device,
                requires_grad=True,
            )
            if parameter is None
            else parameter
        )
        return Circuit(ry(parameter, 0), n_qubits=n_qubits)

    def run_kwargs(**kwargs):
        return lambda: simulator.run(trainable_circuit(), **kwargs)

    def direct_complex():
        return simulator.run(
            Circuit(n_qubits=n_qubits),
            initial_state=(
                torch.zeros(
                    1 << n_qubits,
                    dtype=torch.complex64,
                    device=backend._device,
                    requires_grad=True,
                )
                if backend.rank == 0
                else None
            ),
        )

    def parameter_schema():
        parameter = torch.full(
            () if backend.rank == 0 else (1,),
            0.1,
            dtype=torch.float32,
            device=backend._device,
            requires_grad=True,
        )
        return simulator.run(trainable_circuit(parameter))

    def ownership():
        owner = 1 if backend.world_size > 1 else 0
        raw = (
            PureStateParam(
                torch.ones(
                    1 << n_qubits,
                    dtype=torch.float32,
                    device=backend._device,
                    requires_grad=True,
                ),
                torch.zeros(
                    1 << n_qubits,
                    dtype=torch.float32,
                    device=backend._device,
                    requires_grad=True,
                ),
            )
            if backend.rank == owner
            else None
        )
        return simulator.run(trainable_circuit(), initial_state=raw)

    def paired_state_mismatch(*, dtype=False):
        pair = _Pair(
            torch.ones(
                spec.local_shape,
                dtype=torch.float32,
                device=backend._device,
                requires_grad=True,
            ),
            torch.zeros(
                spec.local_shape,
                dtype=torch.float32,
                device=backend._device,
                requires_grad=True,
            ),
        )
        state = DistState.from_pair(pair, spec=spec, backend=backend)
        if backend.rank == 0:
            object.__setattr__(
                state._pair,
                "real",
                torch.ones(
                    (spec.local_shape[0] + 1, *spec.local_shape[1:])
                    if not dtype
                    else spec.local_shape,
                    dtype=torch.float64 if dtype else torch.float32,
                    device=backend._device,
                ),
            )
        return simulator.run(trainable_circuit(), initial_state=state)

    def unsupported_gate():
        parameter = torch.tensor(
            0.1,
            dtype=torch.float32,
            device=backend._device,
            requires_grad=True,
        )
        return simulator.run(
            Circuit(
                {
                    "type": "unsupported",
                    "target_qubit": 0,
                    "parameter": parameter,
                },
                n_qubits=n_qubits,
            )
        )

    def unsupported_channel():
        circuit = trainable_circuit()
        circuit.noise_model = NoiseModel().add_channel(
            object(), after_gates=("ry",)
        )
        return simulator.run(circuit)

    def rank_route_mismatch():
        parameter = torch.tensor(
            0.1,
            dtype=torch.float32,
            device=backend._device,
            requires_grad=backend.rank == 0,
        )
        return simulator.run(trainable_circuit(parameter))

    def tag_mismatch_injection():
        real = torch.tensor(
            [float(backend.rank + 1)],
            dtype=torch.float32,
            device=backend._device,
            requires_grad=True,
        )
        imag = torch.tensor(
            [-float(backend.rank + 1)],
            dtype=torch.float32,
            device=backend._device,
            requires_grad=True,
        )
        backend.communicator.clear_communication_records()
        exchanged = _exchange_pair(
            _Pair(real, imag),
            communicator=backend.communicator,
            peer=backend.rank ^ 1,
            operation_index=990,
            phase="forward",
        )
        exchanged.abs_sq().sum().backward()
        tags = [
            int(record["tag"])
            for record in backend.communicator.communication_records
            if record["kind"] == "exchange"
        ]
        forward = tuple(tag for tag in tags if tag % 8 < 4)
        backward = tuple(tag for tag in tags if tag % 8 >= 4)
        if not forward or not backward:
            raise RuntimeError(
                "tag mismatch injection 要求真实 forward/backward P2P"
            )
        _validate_tag_phases(forward, (forward[0], *backward))

    cases = (
        ("sample", run_kwargs(shots=1), ValueError, "自动微分模式不支持 sample 或 counts"),
        ("counts", run_kwargs(shots=2), ValueError, "自动微分模式不支持 sample 或 counts"),
        ("collapse", run_kwargs(shots=1, collapse=True), ValueError, "自动微分模式不支持 collapse"),
        (
            "direct_complex",
            direct_complex,
            ValueError,
            "原生 distributed autograd 不接受 requires_grad complex initial_state；请使用 PureStateParam(real, imag)",
        ),
        ("parameter_schema", parameter_schema, ValueError, "各 rank 的可训练参数结构不一致"),
        (
            "ownership",
            ownership,
            ValueError,
            "初态必须由所有 rank 提供匹配的 DistState，或仅由 rank 0 提供完整 statevector/density matrix",
        ),
        (
            "shape",
            lambda: paired_state_mismatch(dtype=False),
            ValueError,
            f"pair shape={(spec.local_shape[0] + 1, *spec.local_shape[1:])} 与 local_shape={spec.local_shape} 不一致",
        ),
        (
            "dtype",
            lambda: paired_state_mismatch(dtype=True),
            ValueError,
            "DistState paired-real 初态必须是当前 backend 的实数 torch.float32",
        ),
        (
            "unsupported_gate",
            unsupported_gate,
            ValueError,
            "指令 'unsupported' 没有可用于分布式执行的局部门矩阵",
        ),
        (
            "unsupported_channel",
            unsupported_channel,
            TypeError,
            "自动微分模式不支持噪声通道 object",
        ),
        (
            "unsupported_observable",
            run_kwargs(observables={"bad": object()}),
            TypeError,
            "自动微分模式不支持 observable 'bad'",
        ),
        (
            "non_hccl_strict",
            lambda: _require_hccl_backend("gloo"),
            ValueError,
            "严格 distributed autograd 探针要求 HCCL process group",
        ),
        (
            "cpu_fallback",
            lambda: _strict_backend(fallback_to_cpu=True),
            ValueError,
            "严格 distributed autograd 探针不允许 fallback_to_cpu=True",
        ),
        (
            "checkpoint",
            run_kwargs(grad_checkpoint="invalid"),
            ValueError,
            "grad_checkpoint 必须是 'none'、'auto' 或正整数",
        ),
        (
            "tag_mismatch_injection",
            tag_mismatch_injection,
            ValueError,
            "forward/backward P2P tag 不匹配",
        ),
        (
            "rank_route_mismatch",
            rank_route_mismatch,
            ValueError,
            "各 rank 的自动微分路由不一致",
        ),
    )

    errors, error_contracts = {}, {}
    case_digests = {}
    rank_only_cases = {
        "parameter_schema",
        "ownership",
        "rank_route_mismatch",
    }
    for name, call, expected_type, expected in cases:
        if backend.world_size == 1 and name in rank_only_cases:
            errors[name] = True
            error_contracts[name] = {
                "type": "SKIPPED",
                "message": "requires world_size > 1",
                "expected_type": expected_type.__name__,
                "expected_message": expected,
            }
            case_digests[name] = {
                "sha256": None,
                "unique_digest_count": 1,
            }
            continue
        try:
            call()
        except Exception as error:  # noqa: BLE001 - exact public contract
            actual = {
                "type": type(error).__name__,
                "message": str(error),
            }
        else:
            actual = {"type": "NO_ERROR", "message": "NO_ERROR"}
        errors[name] = actual == {
            "type": expected_type.__name__,
            "message": expected,
        }
        error_contracts[name] = {
            **actual,
            "expected_type": expected_type.__name__,
            "expected_message": expected,
        }
        case_digest = hashlib.sha256(
            json.dumps(
                actual, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).digest()
        gathered = backend.communicator.all_gather_real(
            torch.tensor(
                [float(byte) for byte in case_digest],
                dtype=torch.float32,
                device=backend._device,
            )
        )
        unique = {
            bytes(
                int(value)
                for value in item.detach().cpu().reshape(-1).tolist()
            )
            for item in gathered
        }
        case_digests[name] = {
            "sha256": case_digest.hex(),
            "unique_digest_count": len(unique),
        }
        errors[name] = errors[name] and len(unique) == 1

    # A complete per-rank digest makes exact type/message agreement auditable
    # without object collectives or any state/gradient transport.
    digest_payload = json.dumps(
        error_contracts, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(digest_payload).hexdigest()
    digest_tensor = torch.tensor(
        [float(byte) for byte in bytes.fromhex(digest)],
        dtype=torch.float32,
        device=backend._device,
    )
    digest_values = backend.communicator.all_gather_real(digest_tensor)
    exact_error_digest = len(
        {
            tuple(int(value) for value in item.detach().cpu().tolist())
            for item in digest_values
        }
    ) == 1

    return {
        "status": "PASS"
        if public_routing_enabled
        and all(errors.values())
        and exact_error_digest
        else "FAIL",
        "passed": public_routing_enabled
        and all(errors.values())
        and exact_error_digest,
        "direct_complex_leaf_rejected": errors["direct_complex"],
        "rank_requires_grad_mismatch_rejected": (
            None if backend.world_size == 1 else errors["rank_route_mismatch"]
        ),
        "public_routing_enabled": public_routing_enabled,
        "exact_errors": errors,
        "error_contracts": error_contracts,
        "case_digests": case_digests,
        "error_digest": digest,
        "one_unique_error_digest": exact_error_digest,
    }


def _gates_section(backend):
    n_qubits = backend.world_size.bit_length()
    local_axis = n_qubits - 1
    cases = (
        ("rx", rx, (0.31,), False), ("ry", ry, (-0.47,), False), ("rz", rz, (0.29,), False),
        ("crx", lambda theta, axis: crx(theta, axis, (local_axis,)), (0.23,), True),
        ("cry", lambda theta, axis: cry(theta, axis, (local_axis,)), (-0.41,), True),
        ("crz", lambda theta, axis: crz(theta, axis, (local_axis,)), (0.37,), True),
        ("rzz", lambda theta, axis: rzz(theta, axis, local_axis), (-0.19,), False),
        ("rxx", lambda theta, axis: rxx(theta, axis, local_axis), (0.53,), False),
        ("u2", u2, (0.17, -0.29), False), ("u3", u3, (0.21, -0.33, 0.45), False),
        ("custom_pair_unitary", _custom_pair_unitary, (0.31,), False),
    )
    observable = PauliString("Z" + "I" * (n_qubits - 1), n_qubits=n_qubits)
    errors = {name: _gradient_section(backend, gate_factory=factory, values=values, observable=observable, four_term=four_term)["max_abs_error"] for name, factory, values, four_term in cases}
    maximum = max(errors.values(), default=0.0)
    return {"status": "PASS" if maximum <= 1e-4 else "FAIL", "passed": maximum <= 1e-4, "distributed_axes": list(range(n_qubits - 1)), "gate_errors": errors, "max_abs_error": maximum}


def _probability_section(backend):
    """Check every global probability Jacobian row through VJP bases."""

    _, spec = _layout_and_spec(backend)
    n_qubits = spec.n_qubits
    errors = []
    for axis in range(n_qubits - 1):
        for global_component in range(1 << n_qubits):
            theta = torch.tensor(0.31, dtype=torch.float32, device=backend._device, requires_grad=True)
            probabilities, _, _ = _native_pair_value(backend, theta, axis, probability=True)
            if spec.global_start <= global_component < spec.global_stop:
                loss = probabilities[global_component - spec.global_start]
            else:
                loss = probabilities.sum() * 0.0
            loss.backward()
            shifted = parameter_shift_gradient(
                lambda values: abs(
                    _cpu_ry_amplitude(
                        global_component, n_qubits, float(values[0]), axis
                    )
                ) ** 2,
                np.array([0.31]),
            )[0]
            errors.append(abs(float(theta.grad.detach().cpu()) - float(shifted)))
    maximum = max(errors, default=0.0)
    return {"status": "PASS" if maximum <= 1e-4 else "FAIL", "passed": maximum <= 1e-4, "distributed_axes": list(range(n_qubits - 1)), "jacobian_rows": 1 << n_qubits, "max_abs_error": maximum}


def _cpu_ry_amplitude(index, n_qubits, theta, axis):
    """Independent scalar complex128 oracle with no full-state allocation."""

    dimension = 1 << n_qubits
    norm = math.sqrt(dimension * (dimension + 1) * (2 * dimension + 1) / 6.0)
    bit = 1 << (n_qubits - 1 - axis)
    own = np.complex128((index + 1) / norm)
    partner = np.complex128(((index ^ bit) + 1) / norm)
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return c * own + (s if index & bit else -s) * partner


def _cpu_pauli_expectation(n_qubits, theta, axis, word):
    total = np.complex128(0.0)
    for output in range(1 << n_qubits):
        source, phase = output, np.complex128(1.0)
        for qubit, symbol in enumerate(word):
            bit = 1 << (n_qubits - 1 - qubit)
            if symbol == "X":
                source ^= bit
            elif symbol == "Y":
                source ^= bit
                phase *= 1j if output & bit else -1j
            elif symbol == "Z" and output & bit:
                phase *= -1.0
        total += np.conj(_cpu_ry_amplitude(output, n_qubits, theta, axis)) * phase * _cpu_ry_amplitude(source, n_qubits, theta, axis)
    return float(total.real)


def _cpu_complex128_expectation(n_qubits, theta, axis, kind):
    """Oracle values for Pauli, a multi-term Hamiltonian, and dense local Z."""

    z_word = "Z" + "I" * (n_qubits - 1)
    if kind in {"pauli", "dense"}:
        return _cpu_pauli_expectation(n_qubits, theta, axis, z_word)
    if kind == "hamiltonian":
        return 0.7 * _cpu_pauli_expectation(n_qubits, theta, axis, z_word) - 0.2 * _cpu_pauli_expectation(n_qubits, theta, axis, "X" + "I" * (n_qubits - 1))
    raise AssertionError(kind)


def _observable_section(backend):
    n_qubits = backend.world_size.bit_length()
    z_word = "Z" + "I" * (n_qubits - 1)
    cases = (
        ("pauli", PauliString(z_word, n_qubits=n_qubits)),
        ("hamiltonian", Hamiltonian([(z_word, 0.7), ("X" + "I" * (n_qubits - 1), -0.2)])),
        ("dense", Observable("matrix", np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex64), metadata={"qubits": (0,)})),
    )
    values, gradients = {}, {}
    for kind, observable in cases:
        values[kind] = max(
            abs(float(_native_pair_value(backend, torch.tensor(0.31, dtype=torch.float32, device=backend._device), axis, observable=observable)[0].detach().cpu()) - _cpu_complex128_expectation(n_qubits, 0.31, axis, kind))
            for axis in range(n_qubits - 1)
        )
        gradients[kind] = _gradient_section(backend, observable=observable)["max_abs_error"]
    maximum = max((*values.values(), *gradients.values()), default=0.0)
    return {"status": "PASS" if maximum <= 1e-4 else "FAIL", "passed": maximum <= 1e-4, "distributed_axes": list(range(n_qubits - 1)), "value_errors": values, "gradient_errors": gradients, "max_abs_error": maximum}
def _bounded_exception_message(error: Exception) -> str:
    try:
        message = str(error)
    except Exception:  # noqa: BLE001 - probes must preserve collective order
        message = "<unprintable exception>"
    return message.replace("\n", " ").replace("\r", " ")


def _encode_failure_payload(backend, error: Exception | None) -> torch.Tensor:
    """Encode a bounded local error record for device-side collective transport."""

    payload = bytearray(_FAILURE_PAYLOAD_BYTES)
    if error is not None:
        type_bytes = type(error).__name__.encode("utf-8")[:_FAILURE_TYPE_BYTES]
        message_bytes = _bounded_exception_message(error).encode("utf-8")[:_FAILURE_MESSAGE_BYTES]
        payload[0] = 1
        payload[1:3] = len(type_bytes).to_bytes(2, byteorder="big")
        payload[3:5] = len(message_bytes).to_bytes(2, byteorder="big")
        type_end = 5 + len(type_bytes)
        payload[5:type_end] = type_bytes
        payload[type_end : type_end + len(message_bytes)] = message_bytes
    return torch.tensor(
        [float(byte) for byte in payload],
        dtype=torch.float32,
        device=backend._device,
    )


def _decode_failure_payload(payload: torch.Tensor) -> dict[str, str] | None:
    raw = bytes(
        int(value)
        for value in payload.detach().cpu().reshape(-1).tolist()
    )
    if raw[0] == 0:
        return None
    type_length = int.from_bytes(raw[1:3], byteorder="big")
    message_length = int.from_bytes(raw[3:5], byteorder="big")
    type_end = 5 + min(type_length, _FAILURE_TYPE_BYTES)
    message_end = type_end + min(message_length, _FAILURE_MESSAGE_BYTES)
    return {
        "type": raw[5:type_end].decode("utf-8", errors="replace"),
        "message": raw[type_end:message_end].decode("utf-8", errors="replace"),
    }


def _synchronize_section_failure(backend, error: Exception | None):
    """Return the canonical first-rank failure after every rank receives it."""

    gathered = backend.communicator.all_gather_real(
        _encode_failure_payload(backend, error)
    )
    failures = [
        (rank, decoded)
        for rank, payload in enumerate(gathered)
        if (decoded := _decode_failure_payload(payload)) is not None
    ]
    if not failures:
        return None
    rank, payload = failures[0]
    torch.distributed.barrier()
    return {"rank": rank, **payload}


def _run_section_collectively(
    backend: DistNPUBackend,
    name: str,
    *,
    runner=None,
) -> dict[str, object]:
    """Run a section and synchronize one bounded failure before teardown."""

    error = None
    try:
        result = _blocked_section(name) if runner is None else runner(backend)
    except Exception as caught:  # noqa: BLE001 - preserve collective order
        error = caught
        result = None

    synchronized_error = _synchronize_section_failure(backend, error)
    if synchronized_error is not None:
        return {
            "status": "FAIL",
            "passed": False,
            "error": synchronized_error,
        }

    local_failed = torch.tensor(
        [float(not result["passed"])],
        dtype=torch.float32,
        device=backend._device,
    )
    failed_ranks = backend.communicator.all_reduce_sum_real(local_failed)
    failed_rank_count = int(failed_ranks[0].detach().cpu())
    if failed_rank_count not in {0, backend.world_size}:
        return {
            "status": "FAIL",
            "passed": False,
            "failed_ranks": failed_rank_count,
        }
    return result


SECTION_RUNNERS = {
    "environment": _environment_section,
    "statevector": _statevector_section,
    "density": _density_section,
    "gates": _gates_section,
    "probability": _probability_section,
    "observable": _observable_section,
    "noise": _noise_section,
    "stinespring": _stinespring_section,
    "communication": _communication_section,
    "optimizer": _optimizer_section,
    "performance": _performance_section,
    "memory": _memory_section,
    "contract": _contract_section,
}


def _git_commit() -> str:
    """Return the exact tested source revision for a hardware evidence file."""

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("distributed autograd 探针无法确定完整 git commit")
    return commit


def _require_clean_git_source() -> str:
    """Return HEAD only when tracked, staged, and untracked source is clean."""

    commit = _git_commit()
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.stdout:
        raise RuntimeError(
            "release evidence refuses a dirty source tree; commit or remove all changes first"
        )
    return commit


def _probe_command(
    world_size: int,
    output_json: Path,
    *,
    section: str,
) -> str:
    return shlex.join(
        (
            "torchrun",
            f"--nproc-per-node={int(world_size)}",
            "scripts/npu/distributed_autograd_probe.py",
            "--section",
            section,
            "--output-json",
            str(output_json),
        )
    )


def _canonical_probe_command(world_size: int, output_json: Path) -> str:
    """Reconstruct the frozen release command in one deterministic form."""

    return _probe_command(world_size, output_json, section="all")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _rank_devices(backend) -> list[str]:
    """Collect rank/local-rank bindings without object collectives."""

    local = torch.tensor(
        [float(backend.rank), float(backend.local_rank)],
        dtype=torch.float32,
        device=backend._device,
    )
    gathered = backend.communicator.all_gather_real(local)
    if len(gathered) != backend.world_size:
        raise RuntimeError("device provenance did not include every rank")
    devices = []
    for expected_rank, payload in enumerate(gathered):
        values = payload.detach().cpu().reshape(-1).tolist()
        if (
            payload.dtype != torch.float32
            or len(values) != 2
            or int(values[0]) != expected_rank
            or values[0] != float(int(values[0]))
            or values[1] != float(int(values[1]))
            or int(values[1]) != expected_rank
        ):
            raise RuntimeError("rank device provenance is malformed")
        devices.append(f"npu:{int(values[1])}")
    return devices


def _report_contract(
    *,
    commit: str,
    command: str,
    exit_code: int,
    world_size: int,
    rank_devices,
    torch_version: str,
    torch_npu_version: str,
    cann_version: str,
    run_id: str,
    started_at: str,
    finished_at: str,
    source_clean: bool,
    sections,
) -> dict[str, object]:
    """Normalize every completed probe section into the release JSON schema."""

    normalized = {}
    failed_invariants = []
    for name, section in sections.items():
        passed = bool(section.get("passed", False))
        metrics = {
            key: value
            for key, value in section.items()
            if key not in {"status", "passed", "failed_invariants"}
        }
        section_failures = list(section.get("failed_invariants", ()))
        if not passed and not section_failures:
            section_failures.append(name)
        normalized[name] = {
            "status": "PASS" if passed else "FAIL",
            "passed": passed,
            "metrics": metrics,
            "failed_invariants": section_failures,
        }
        failed_invariants.extend(section_failures)
    passed = not failed_invariants and all(
        section["passed"] for section in normalized.values()
    )
    return {
        "commit": str(commit),
        "command": str(command),
        "exit_code": int(exit_code),
        "world_size": int(world_size),
        "rank_devices": list(rank_devices),
        "torch_version": str(torch_version),
        "torch_npu_version": str(torch_npu_version),
        "cann_version": str(cann_version),
        "backend": "hccl",
        "fallback_to_cpu": False,
        "run_id": str(run_id),
        "started_at": str(started_at),
        "finished_at": str(finished_at),
        "source_clean": bool(source_clean),
        "passed": passed,
        "failed_invariants": failed_invariants,
        "sections": normalized,
        "raw_sha256": _RAW_SHA256_PLACEHOLDER,
    }


def _report_bytes(report: dict[str, object]) -> bytes:
    """Serialize and bind the digest to the exact producer bytes."""

    value = dict(report)
    value["raw_sha256"] = _RAW_SHA256_PLACEHOLDER
    raw = (
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    marker = (
        b'"raw_sha256":"' + _RAW_SHA256_PLACEHOLDER.encode("ascii") + b'"'
    )
    if raw.count(marker) != 1:
        raise ValueError("report must contain exactly one raw_sha256 field")
    digest = hashlib.sha256(raw).hexdigest().encode("ascii")
    return raw.replace(
        marker,
        b'"raw_sha256":"' + digest + b'"',
        1,
    )


def _write_report(path: Path, report: dict[str, object]) -> None:
    """Atomically write one exact-byte rank-0 evidence report."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _report_bytes(report)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(raw)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _selected_sections(selected: str) -> tuple[str, ...]:
    return SECTIONS if selected == "all" else (selected,)


def _run_probe(selected: str, output_json: Path) -> bool:
    started_at = _utc_timestamp()
    commit = _require_clean_git_source()
    run_id = str(uuid.uuid4())
    backend = _strict_backend(fallback_to_cpu=False)
    sections = {
        name: _run_section_collectively(
            backend,
            name,
            runner=SECTION_RUNNERS[name],
        )
        for name in _selected_sections(selected)
    }
    local_passed = torch.tensor(
        [int(all(section["passed"] for section in sections.values()))],
        dtype=torch.long,
        device=backend._device,
    )
    passed_ranks = backend.communicator.all_reduce_sum(local_passed)
    passed = int(passed_ranks[0].detach().cpu()) == backend.world_size
    rank_devices = _rank_devices(backend)
    finished_at = _utc_timestamp()

    if backend.rank == 0:
        report = _report_contract(
            commit=commit,
            command=_probe_command(
                backend.world_size,
                output_json,
                section=selected,
            ),
            exit_code=0 if passed else 1,
            world_size=backend.world_size,
            rank_devices=rank_devices,
            torch_version=str(torch.__version__),
            torch_npu_version=str(_torch_npu_version() or "unknown"),
            cann_version=_cann_identity(),
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            source_clean=True,
            sections=sections,
        )
        if bool(report["passed"]) != passed:
            raise RuntimeError("distributed autograd 探针 rank 间通过状态不一致")
        _write_report(output_json, report)
    return passed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=("all", *SECTIONS), default="all")
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        passed = _run_probe(args.section, args.output_json)
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
