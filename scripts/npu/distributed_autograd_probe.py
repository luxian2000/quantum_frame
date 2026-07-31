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

from aicir.distributed import DistNPUBackend


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
    "environment": 2,
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


def _run_section_collectively(backend: DistNPUBackend, name: str) -> dict[str, object]:
    """Run one placeholder on every rank and share its failure before teardown."""

    result = _blocked_section(name)
    local_failed = torch.tensor(
        [int(not result["passed"])],
        dtype=torch.long,
        device=backend._device,
    )
    failed_ranks = backend.communicator.all_reduce_sum(local_failed)
    if int(failed_ranks[0].detach().cpu()) != backend.world_size:
        return {
            "status": "FAIL",
            "passed": False,
            "failed_ranks": int(failed_ranks[0].detach().cpu()),
        }
    return result


def _selected_sections(selected: str) -> tuple[str, ...]:
    return SECTIONS if selected == "all" else (selected,)


def _run_probe(selected: str, output_json: Path) -> bool:
    backend = _strict_backend(fallback_to_cpu=False)
    sections = {
        name: _run_section_collectively(backend, name)
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
