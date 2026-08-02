#!/usr/bin/env python3
"""Strict paired-real distributed-autograd benchmark contract."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from scripts.npu.distributed_autograd_probe import (
    _strict_backend,
    _resolve_benchmark_parameter_family,
    run_benchmark_workload,
)


_FIELDS = {
    "communication_mode", "gradient_method", "path", "world_size", "n_qubits", "depth",
    "parameters", "warmups", "runs", "forward_ms_median", "backward_ms_median",
    "gradient_ms_median", "gradient_ms_p95", "peak_memory_bytes", "p2p_bytes", "wait_ms",
    "buffer_reuse_count", "fallback_to_cpu",
}

_COLLECTIVE_METRIC_NAMES = (
    "forward_ms_median", "backward_ms_median", "gradient_ms_median",
    "gradient_ms_p95", "p2p_bytes", "wait_ms", "buffer_reuse_count",
    "peak_memory_bytes",
)


def _collective_benchmark_metrics(backend, metrics, *, peak_memory_bytes):
    """Validate every rank and return rank-maximum authoritative metrics.

    The control payload is a fixed-width contiguous float32 tensor.  It keeps
    the strict paired-real collective contract even though the values are
    benchmark metadata rather than state amplitudes.
    """

    state_error = float(metrics.get("state_max_abs_error", float("inf")))
    gradient_error = float(metrics.get("gradient_max_abs_error", float("inf")))
    handles_complete = bool(metrics.get("all_handles_complete", False))
    fallback = bool(metrics.get("fallback_to_cpu", False))
    local_passed = (
        state_error <= 1e-6
        and gradient_error <= 1e-4
        and handles_complete
        and not fallback
    )
    values = (
        float(local_passed),
        state_error,
        gradient_error,
        float(handles_complete),
        float(fallback),
        *(float(metrics[name]) for name in _COLLECTIVE_METRIC_NAMES[:-1]),
        float(peak_memory_bytes),
    )
    payload = torch.tensor(values, dtype=torch.float32, device=backend._device).contiguous()
    gathered = (
        backend.communicator.all_gather_real(payload)
        if backend.world_size > 1
        else [payload]
    )
    if any(float(candidate[0].detach().cpu()) < 0.5 for candidate in gathered):
        raise RuntimeError("benchmark runner failed on this or another rank")
    maxima = torch.stack(tuple(candidate.contiguous() for candidate in gathered)).amax(dim=0)
    # Fields 1--4 are validity diagnostics.  Measurement fields begin at 5.
    return {
        name: float(maxima[5 + index].detach().cpu())
        for index, name in enumerate(_COLLECTIVE_METRIC_NAMES)
    }


def _validate_benchmark_report(report):
    if set(report) != _FIELDS:
        raise ValueError("benchmark JSON 字段必须完整且无额外字段")
    if report["communication_mode"] not in {"baseline", "reuse", "overlap"}:
        raise ValueError("communication_mode 无效")
    if report["gradient_method"] not in {"native", "parameter_shift", "finite_difference"}:
        raise ValueError("gradient_method 无效")
    if report["path"] not in {"statevector", "density", "noise", "stinespring"}:
        raise ValueError("path 无效")
    if report["gradient_method"] == "parameter_shift" and report["path"] != "statevector":
        raise ValueError("parameter_shift 仅适用于 shift-rule statevector workload")
    if report["gradient_method"] == "finite_difference" and report["path"] == "statevector":
        raise ValueError("statevector shift-rule workload 不使用 finite_difference")
    if any(report[name] <= 0 for name in ("world_size", "n_qubits", "depth", "parameters", "warmups", "runs")):
        raise ValueError("benchmark 计数必须为正")
    if any(report[name] < 0 for name in ("forward_ms_median", "backward_ms_median", "gradient_ms_median", "gradient_ms_p95", "wait_ms", "peak_memory_bytes", "p2p_bytes", "buffer_reuse_count")):
        raise ValueError("benchmark 指标不能为负")
    if report["fallback_to_cpu"]:
        raise ValueError("strict benchmark 不允许 CPU fallback")


def _validate_workload(path: str, gradient_method: str) -> None:
    """Reject combinations that the current strict native workload lacks."""

    _resolve_benchmark_parameter_family(path, gradient_method)


def _run_benchmark(args) -> dict[str, object]:
    """Run one real HCCL paired-P2P measurement; never fabricate CPU timings."""

    _validate_workload(args.path, args.gradient_method)
    backend = _strict_backend(fallback_to_cpu=False)
    try:
        torch.npu.reset_peak_memory_stats(backend._device)
        metrics = run_benchmark_workload(
            backend,
            communication_mode=args.communication_mode,
            path=args.path,
            gradient_method=args.gradient_method,
            n_qubits=args.n_qubits,
            depth=args.depth,
            parameters=args.parameters,
            warmups=args.warmups,
            runs=args.runs,
        )
        collective = _collective_benchmark_metrics(
            backend,
            metrics,
            peak_memory_bytes=int(torch.npu.max_memory_allocated(backend._device)),
        )
        report = {
            "communication_mode": args.communication_mode,
            "gradient_method": args.gradient_method,
            "path": args.path,
            "world_size": backend.world_size,
            "n_qubits": args.n_qubits,
            "depth": args.depth,
            "parameters": args.parameters,
            "warmups": args.warmups,
            "runs": args.runs,
            "forward_ms_median": collective["forward_ms_median"],
            "backward_ms_median": collective["backward_ms_median"],
            "gradient_ms_median": collective["gradient_ms_median"],
            "gradient_ms_p95": collective["gradient_ms_p95"],
            "peak_memory_bytes": int(collective["peak_memory_bytes"]),
            "p2p_bytes": int(collective["p2p_bytes"]),
            "wait_ms": collective["wait_ms"],
            "buffer_reuse_count": int(collective["buffer_reuse_count"]),
            "fallback_to_cpu": False,
        }
        _validate_benchmark_report(report)
        return report
    finally:
        backend.communicator.set_autograd_communication_mode("baseline")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--communication-mode", choices=("baseline", "reuse", "overlap"), required=True)
    parser.add_argument("--gradient-method", choices=("native", "parameter_shift", "finite_difference"), required=True)
    parser.add_argument("--path", choices=("statevector", "density", "noise", "stinespring"), required=True)
    for name in ("n-qubits", "depth", "parameters"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if any(getattr(args, name.replace("-", "_")) <= 0 for name in ("n-qubits", "depth", "parameters", "warmups", "runs")):
        parser.error("n-qubits、depth、parameters、warmups 和 runs 必须为正")
    report = _run_benchmark(args)
    if int(os.environ.get("RANK", "0")) == 0:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_json.with_name(f".{args.output_json.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
        temporary.replace(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
