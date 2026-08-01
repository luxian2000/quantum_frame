#!/usr/bin/env python3
"""Strict multi-NPU acceptance probe for the paired-real statevector kernel.

Run this probe with ``torchrun`` on Ascend hardware.  It has no CPU fallback,
does not claim that forward-only :class:`aicir.distributed.DistSimulator` is
differentiable, and records no live-NPU result until the command is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

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
from aicir.distributed._contracts import AUTOGRAD_ERROR
from aicir.distributed.autograd._collectives import _exchange_pair, _replicated_all_reduce
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

BLOCKED_BY_TASK = {
    "statevector": 2,
    "density": 3,
    "gates": 4,
    "probability": 5,
    "observable": 6,
    "noise": 7,
    "stinespring": 8,
    "communication": 9,
    "optimizer": 10,
    "performance": 11,
    "memory": 11,
    "contract": 11,
}

_FAILURE_TYPE_BYTES = 128
_FAILURE_MESSAGE_BYTES = 512
_FAILURE_PAYLOAD_BYTES = 5 + _FAILURE_TYPE_BYTES + _FAILURE_MESSAGE_BYTES


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
    if torch.distributed.get_backend() != "hccl":
        raise RuntimeError("严格 distributed autograd 探针要求 HCCL process group")
    if backend._device.type != "npu" or backend._device.index != local_rank:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} 必须绑定 npu:{local_rank}，实际为 {backend._device}"
        )
    return backend


def _blocked_section(name: str) -> dict[str, object]:
    return {
        "status": "BLOCKED",
        "passed": False,
        "blocked_by_task": BLOCKED_BY_TASK[name],
    }


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
    passed = (
        local_gate_p2p_delta == 0
        and forward_p2p == expected_per_phase
        and backward_p2p == expected_per_phase
        and payload_dtypes == ["torch.float32"]
        and all(peer is not None and peer != backend.rank and 0 <= peer < backend.world_size for peer in peers)
        and set(forward_tags).isdisjoint(backward_tags)
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


def _performance_exchange_case(backend, mode: str, *, warmups=5, runs=30):
    """Measure an actual paired-real forward/backward exchange mode."""

    communicator = backend.communicator
    communicator.set_autograd_communication_mode(mode)
    if hasattr(communicator, "_autograd_pair_buffer_pool"):
        del communicator._autograd_pair_buffer_pool

    def one_iteration():
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
        _synchronize_npu(backend)
        started = time.perf_counter()
        exchanged = _exchange_pair(
            _Pair(real, imag),
            communicator=communicator,
            peer=backend.rank ^ 1,
            operation_index=101,
            phase="forward",
        )
        _synchronize_npu(backend)
        forward_ms = (time.perf_counter() - started) * 1000.0
        # Keep the backward interval independently delimited even though the
        # preceding forward endpoint is already synchronized.
        _synchronize_npu(backend)
        started = time.perf_counter()
        exchanged.abs_sq().sum().backward()
        _synchronize_npu(backend)
        backward_ms = (time.perf_counter() - started) * 1000.0
        expected = 2.0 * torch.cat((real.detach(), imag.detach()))
        actual = torch.cat((real.grad, imag.grad))
        error = float(torch.max(torch.abs(actual - expected)).detach().cpu())
        return forward_ms, backward_ms, error

    for _ in range(warmups):
        one_iteration()
    communicator.clear_communication_records()
    pool = getattr(communicator, "_autograd_pair_buffer_pool", None)
    if pool is not None:
        pool.reuse_count = 0

    samples = [one_iteration() for _ in range(runs)]
    counters = communicator.communication_counters
    return {
        "forward_ms_median": float(np.median([sample[0] for sample in samples])),
        "backward_ms_median": float(np.median([sample[1] for sample in samples])),
        "gradient_ms_median": float(np.median([sample[0] + sample[1] for sample in samples])),
        "gradient_ms_p95": float(np.percentile([sample[0] + sample[1] for sample in samples], 95)),
        "p2p_bytes": int(counters["bytes"]),
        "wait_ms": float(counters["p2p_wait_ms"]),
        "buffer_reuse_count": int(getattr(pool, "reuse_count", 0)),
        "gradient_max_abs_error": max(sample[2] for sample in samples),
        "all_handles_complete": bool(communicator.work_handle_status["all_handles_complete"]),
    }


def _performance_gradient_oracles(backend):
    """Compare native paired gradients against shift and finite difference."""

    axis = 0
    theta = torch.tensor(0.31, dtype=torch.float32, device=backend._device, requires_grad=True)
    value, _, _ = _native_pair_value(backend, (theta,), axis)
    value.backward()
    native = float(theta.grad.detach().cpu())

    def objective(point):
        result, _, _ = _native_pair_value(
            backend,
            (torch.tensor(float(point[0]), dtype=torch.float32, device=backend._device),),
            axis,
        )
        return float(result.detach().cpu())

    point = np.array([0.31], dtype=np.float64)
    shifted = float(parameter_shift_gradient(objective, point)[0])
    finite = float(finite_difference_gradient(objective, point, epsilon=1e-3)[0])
    return {
        "native_vs_parameter_shift": abs(native - shifted),
        "native_vs_finite_difference": abs(native - finite),
    }


def _performance_section(backend: DistNPUBackend) -> dict[str, object]:
    """Report measured baseline/reuse/overlap P2P evidence on strict NPU."""

    modes = {
        mode: _performance_exchange_case(backend, mode)
        for mode in ("baseline", "reuse", "overlap")
    }
    oracle_errors = _performance_gradient_oracles(backend)
    backend.communicator.set_autograd_communication_mode("baseline")
    passed = (
        all(metrics["gradient_max_abs_error"] <= 1e-4 for metrics in modes.values())
        and all(metrics["all_handles_complete"] for metrics in modes.values())
        and all(value <= 1e-4 for value in oracle_errors.values())
        and modes["baseline"]["buffer_reuse_count"] == 0
        and modes["reuse"]["buffer_reuse_count"] > 0
        and modes["overlap"]["buffer_reuse_count"] > 0
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "warmups": 5,
        "runs": 30,
        "modes": modes,
        "gradient_oracle_max_abs_error": oracle_errors,
        "fallback_to_cpu": False,
    }


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
        reports[str(policy)] = {
            "saved_state_count": int(metrics.saved_state_count),
            "recomputed_gate_count": int(metrics.recomputed_gate_count),
            "chosen_interval": int(metrics.interval),
            "peak_allocation_bytes": metrics.peak_allocation_bytes,
            "peak_allocation_status": metrics.peak_allocation_status,
            "memory_source": metrics.memory_source,
            "gradient_error": abs(gradient - reference_gradient),
        }
    passed = all(item["gradient_error"] <= 1e-4 for item in reports.values())
    return {"status": "PASS" if passed else "FAIL", "passed": passed, "policies": reports}


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
    """Check the private initial-state contracts without exposing ``run`` autograd."""

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
    direct_complex_message = (
        "原生 distributed autograd 不接受 requires_grad complex initial_state；"
        "请使用 PureStateParam(real, imag)"
    )
    try:
        simulator._prepare_initial_state(
            n_qubits=n_qubits,
            layout=layout,
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
            initial_density_matrix=None,
        )
    except ValueError as error:
        direct_complex_leaf_rejected = str(error) == direct_complex_message
    else:
        direct_complex_leaf_rejected = False
    if backend.world_size > 1:
        torch.distributed.barrier()

    if backend.world_size == 1:
        rank_requires_grad_mismatch_rejected = None
    else:
        mismatch = DistState.from_pair(
            _Pair(
                torch.ones(
                    spec.local_shape,
                    dtype=torch.float32,
                    device=backend._device,
                    requires_grad=backend.rank == 0,
                ),
                torch.zeros(
                    spec.local_shape,
                    dtype=torch.float32,
                    device=backend._device,
                    requires_grad=backend.rank == 0,
                ),
            ),
            spec=spec,
            backend=backend,
        )
        try:
            simulator._prepare_initial_state(
                n_qubits=n_qubits,
                layout=layout,
                initial_state=mismatch,
                initial_density_matrix=None,
            )
        except ValueError as error:
            rank_requires_grad_mismatch_rejected = (
                str(error) == "DistState paired-real requires_grad 在各 rank 间不一致"
            )
        else:
            rank_requires_grad_mismatch_rejected = False
        torch.distributed.barrier()

    public_gate_held = False
    try:
        simulator.run(
            Circuit(n_qubits=n_qubits),
            initial_state=(
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
                if backend.rank == 0
                else None
            ),
        )
    except ValueError as error:
        public_gate_held = str(error) == AUTOGRAD_ERROR
    return {
        "status": "PASS"
        if direct_complex_leaf_rejected
        and rank_requires_grad_mismatch_rejected is not False
        and public_gate_held
        else "FAIL",
        "passed": direct_complex_leaf_rejected
        and rank_requires_grad_mismatch_rejected is not False
        and public_gate_held,
        "direct_complex_leaf_rejected": direct_complex_leaf_rejected,
        "rank_requires_grad_mismatch_rejected": rank_requires_grad_mismatch_rejected,
        "public_forward_only_gate_held": public_gate_held,
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
        list(payload),
        dtype=torch.uint8,
        device=backend._device,
    )


def _decode_failure_payload(payload: torch.Tensor) -> dict[str, str] | None:
    raw = bytes(payload.detach().cpu().tolist())
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

    gathered = backend.communicator.all_gather(
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
        [int(not result["passed"])],
        dtype=torch.long,
        device=backend._device,
    )
    failed_ranks = backend.communicator.all_reduce_sum(local_failed)
    failed_rank_count = int(failed_ranks[0].detach().cpu())
    if failed_rank_count not in {0, backend.world_size}:
        return {
            "status": "FAIL",
            "passed": False,
            "failed_ranks": failed_rank_count,
        }
    return result


def _selected_sections(selected: str) -> tuple[str, ...]:
    return SECTIONS if selected == "all" else (selected,)


def _run_probe(selected: str, output_json: Path) -> bool:
    backend = _strict_backend(fallback_to_cpu=False)
    sections = {
        name: _run_section_collectively(
            backend,
            name,
            runner={
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
            }.get(name),
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

    if backend.rank == 0:
        report = {
            "passed": passed,
            "world_size": backend.world_size,
            "fallback_to_cpu": False,
            "process_group_backend": "hccl",
            "sections": sections,
            "failed_sections": [
                name for name, section in sections.items() if not section["passed"]
            ],
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
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
