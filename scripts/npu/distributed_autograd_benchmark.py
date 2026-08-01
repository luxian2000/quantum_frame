#!/usr/bin/env python3
"""Strict paired-real distributed-autograd benchmark contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.npu.distributed_autograd_probe import (
    _performance_exchange_case,
    _performance_gradient_oracles,
    _strict_backend,
)


_FIELDS = {
    "communication_mode", "gradient_method", "path", "world_size", "n_qubits", "depth",
    "parameters", "warmups", "runs", "forward_ms_median", "backward_ms_median",
    "gradient_ms_median", "gradient_ms_p95", "peak_memory_bytes", "p2p_bytes", "wait_ms",
    "buffer_reuse_count", "fallback_to_cpu",
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

    if gradient_method == "parameter_shift" and path != "statevector":
        raise ValueError("parameter_shift 仅适用于 statevector workload")
    if gradient_method == "finite_difference" and path == "statevector":
        raise ValueError("statevector workload 使用 parameter_shift 或 native")


def _run_benchmark(args) -> dict[str, object]:
    """Run one real HCCL paired-P2P measurement; never fabricate CPU timings."""

    _validate_workload(args.path, args.gradient_method)
    backend = _strict_backend(fallback_to_cpu=False)
    try:
        torch.npu.reset_peak_memory_stats(backend._device)
        metrics = _performance_exchange_case(
            backend,
            args.communication_mode,
            warmups=args.warmups,
            runs=args.runs,
        )
        if args.gradient_method != "native":
            oracle_errors = _performance_gradient_oracles(backend)
            selected_error = oracle_errors[
                "native_vs_parameter_shift"
                if args.gradient_method == "parameter_shift"
                else "native_vs_finite_difference"
            ]
            if selected_error > 1e-4:
                raise RuntimeError(
                    f"{args.gradient_method} 与 native paired gradient 不一致: {selected_error}"
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
            "forward_ms_median": metrics["forward_ms_median"],
            "backward_ms_median": metrics["backward_ms_median"],
            "gradient_ms_median": metrics["gradient_ms_median"],
            "gradient_ms_p95": metrics["gradient_ms_p95"],
            "peak_memory_bytes": int(torch.npu.max_memory_allocated(backend._device)),
            "p2p_bytes": metrics["p2p_bytes"],
            "wait_ms": metrics["wait_ms"],
            "buffer_reuse_count": metrics["buffer_reuse_count"],
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
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
