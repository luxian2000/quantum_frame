#!/usr/bin/env python3
"""Strict multi-NPU acceptance scaffold for distributed gradient validation.

Run this probe with ``torchrun``.  It has no CPU fallback and does not make the
forward-only :class:`aicir.distributed.DistSimulator` differentiable.  Until
the later implementation tasks land, every section is reported as blocked with
the task number that owns it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import numpy as np

from aicir import Hamiltonian, PauliString
from aicir.core.circuit import ry
from aicir.distributed import DistNPUBackend, parameter_shift_gradient
from aicir.distributed.autograd._collectives import _exchange_pair
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.autograd._reducers import _PairReducer
from aicir.distributed.autograd._vector import _PairVectorKernel
from aicir.distributed.gates import _GatePlanner
from aicir.distributed.layout import _Layout, _ShardSpec


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


def _native_pair_value(backend, theta, axis, *, probability=False, observable=None):
    """One paired-real circuit evaluation used by strict gradient sections."""

    n_qubits = backend.world_size.bit_length()
    layout = _Layout.explicit(tuple(range(n_qubits)), n_qubits=n_qubits, distributed_axes=n_qubits - 1)
    spec = _ShardSpec.build(n_qubits, backend.world_size, backend.rank, "vector", layout)
    full = torch.arange(1, (1 << n_qubits) + 1, dtype=torch.float32, device=backend._device)
    full = full / torch.sqrt(full.square().sum())
    local = full[spec.global_start : spec.global_stop].reshape(-1, 1)
    pair = _Pair(local, torch.zeros_like(local))
    plan = _GatePlanner(backend, layout, n_qubits).plan(ry(theta, axis), axis)
    evolved = _PairVectorKernel(backend).apply(pair, plan, operation_index=axis)
    reducer = _PairReducer(backend)
    if probability:
        return reducer.probabilities(evolved, spec)
    return reducer.expectation(
        evolved,
        spec,
        observable or PauliString("Z" + "I" * (n_qubits - 1), n_qubits=n_qubits),
    )


def _gradient_section(backend: DistNPUBackend, *, probability=False, observable=None):
    errors = []
    for axis in range(backend.world_size.bit_length() - 1):
        theta = torch.tensor(0.31, dtype=torch.float32, device=backend._device, requires_grad=True)
        value = _native_pair_value(backend, theta, axis, probability=probability, observable=observable)
        loss = value.sum() if probability else value
        loss.backward()
        shifted = parameter_shift_gradient(
            lambda values: float(_native_pair_value(backend, torch.tensor(float(values[0]), dtype=torch.float32, device=backend._device), axis, probability=probability, observable=observable).sum().detach().cpu()),
            np.array([0.31]),
        )[0]
        errors.append(abs(float(theta.grad.detach().cpu()) - float(shifted)))
    maximum = max(errors, default=0.0)
    return {"status": "PASS" if maximum <= 1e-4 else "FAIL", "passed": maximum <= 1e-4, "max_abs_error": maximum, "distributed_axes": list(range(backend.world_size.bit_length() - 1))}


def _statevector_section(backend):
    return _gradient_section(backend)


def _gates_section(backend):
    return _gradient_section(backend)


def _probability_section(backend):
    return _gradient_section(backend, probability=True)


def _observable_section(backend):
    n_qubits = backend.world_size.bit_length()
    observable = Hamiltonian([("Z" + "I" * (n_qubits - 1), 0.7)])
    return _gradient_section(backend, observable=observable)
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
                "gates": _gates_section,
                "probability": _probability_section,
                "observable": _observable_section,
                "communication": _communication_section,
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
