"""Strict full-API acceptance probe for 2 or 4 Ascend NPUs.

Run from repository root:

    source /usr/local/Ascend/cann/set_env.sh
    PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=2 scripts/npu/distributed_api_probe.py --section all
    PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=4 scripts/npu/distributed_api_probe.py --section all
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import torch

from aicir.distributed import DistSimulator


STATE_ATOL = 1e-6
REDUCTION_ATOL = 1e-5
EXPECTED_SECTIONS = (
    "state",
    "layout",
    "continuation",
    "noise",
    "observable",
    "measure",
    "result",
    "communication",
    "contract",
)


def _section(passed, *, metrics=None, status=None):
    return {
        "passed": bool(passed),
        "status": status or ("PASS" if passed else "FAIL"),
        "metrics": dict(metrics or {}),
    }


def _validate_runtime(backend):
    device = backend._device
    if backend.world_size not in {2, 4}:
        raise ValueError("完整 API 探针只接受 world_size=2 或 world_size=4")
    if device.type != "npu":
        raise RuntimeError(f"严格探针要求 NPU，实际设备为 {device}")
    if backend.rank != backend.local_rank or device.index != backend.local_rank:
        raise RuntimeError("rank、local_rank 与 NPU device 不一致")


def _pending_section(_simulator):
    return _section(False, status="NOT_IMPLEMENTED")


SECTION_RUNNERS = {
    name: _pending_section
    for name in EXPECTED_SECTIONS
}


def _run_selected(simulator, selected):
    names = EXPECTED_SECTIONS if selected == "all" else (selected,)
    sections = {}
    for name in names:
        sections[name] = SECTION_RUNNERS[name](simulator)
    failed = [
        name for name, result in sections.items()
        if not result["passed"]
    ]
    return sections, failed


def _run_probe(selected):
    simulator = DistSimulator.from_env(fallback_to_cpu=False)
    backend = simulator.backend
    _validate_runtime(backend)
    sections, failed = _run_selected(simulator, selected)

    passed_count = torch.tensor(
        [int(not failed)],
        dtype=torch.long,
        device=backend._device,
    )
    passed_count = backend.communicator.all_reduce_sum(passed_count)
    passed = int(passed_count[0].detach().cpu()) == backend.world_size

    report = None
    if backend.rank == 0:
        report = {
            "passed": passed,
            "world_size": backend.world_size,
            "fallback_to_cpu": False,
            "sections": sections,
            "failed_invariants": failed,
        }
        print(json.dumps(report, sort_keys=True))
    return passed


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section",
        choices=("all", *EXPECTED_SECTIONS),
        default="all",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    try:
        ok = _run_probe(args.section)
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
