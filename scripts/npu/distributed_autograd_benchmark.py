#!/usr/bin/env python3
"""Strict paired-real distributed-autograd benchmark contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--communication-mode", choices=("baseline", "reuse", "overlap"), required=True)
    parser.add_argument("--gradient-method", choices=("native", "parameter_shift", "finite_difference"), required=True)
    parser.add_argument("--path", choices=("statevector", "density", "noise", "stinespring"), required=True)
    for name in ("n-qubits", "depth", "parameters", "warmups", "runs"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    # Timing is deliberately strict-hardware only; this CLI never fabricates a
    # CPU result.  The NPU probe supplies measurements in release environments.
    raise RuntimeError("distributed_autograd_benchmark 需要严格 NPU/HCCL 运行环境")


if __name__ == "__main__":
    main()
