"""Static contracts for the strict distributed-native-autograd probe."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import socket
import time

import pytest
import torch
import torch.multiprocessing as mp

from aicir.distributed import DistNPUBackend


_PROBE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "npu"
    / "distributed_autograd_probe.py"
)


def _probe_module():
    spec = importlib.util.spec_from_file_location("distributed_autograd_probe", _PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _join(context, *, timeout=30):
    deadline = time.monotonic() + timeout
    try:
        while not context.join(timeout=max(0.0, deadline - time.monotonic())):
            assert time.monotonic() < deadline, "probe contract worker timed out"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
    assert all(process.exitcode == 0 for process in context.processes)


def _runtime_contract_worker(rank, world_size, port, output_path):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world_size),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
    )
    backend = DistNPUBackend.from_env(
        fallback_to_cpu=True,
        process_group_backend="gloo",
    )
    try:
        result = _probe_module()._contract_section(backend)
        if rank == 0:
            Path(output_path).write_text(
                json.dumps(result, sort_keys=True),
                encoding="utf-8",
            )
    finally:
        torch.distributed.destroy_process_group()


def test_probe_declares_every_section_complete_and_routable():
    probe = _probe_module()

    assert len(probe.SECTIONS) == 13
    assert set(probe.SECTIONS) == set(probe.SECTION_RUNNERS)
    assert "BLOCKED" not in _PROBE_PATH.read_text(encoding="utf-8")


def test_probe_report_has_release_gate_top_level_contract():
    probe = _probe_module()
    report = probe._report_contract(
        commit="a" * 40,
        command=probe._canonical_probe_command(2, Path("world2.json")),
        exit_code=0,
        world_size=2,
        rank_devices=["npu:0", "npu:1"],
        torch_version="2.6.0",
        torch_npu_version="2.6.0",
        cann_version="unknown",
        run_id="00000000-0000-4000-8000-000000000002",
        started_at="2026-08-02T00:00:02.000000Z",
        finished_at="2026-08-02T00:00:03.000000Z",
        source_clean=True,
        sections={
            name: {
                "status": "PASS",
                "passed": True,
                "failed_invariants": [],
            }
            for name in probe.SECTIONS
        },
    )

    assert report == {
        "commit": "a" * 40,
        "command": (
            "torchrun --nproc-per-node=2 "
            "scripts/npu/distributed_autograd_probe.py "
            "--section all --output-json world2.json"
        ),
        "exit_code": 0,
        "world_size": 2,
        "rank_devices": ["npu:0", "npu:1"],
        "torch_version": "2.6.0",
        "torch_npu_version": "2.6.0",
        "cann_version": "unknown",
        "backend": "hccl",
        "fallback_to_cpu": False,
        "run_id": "00000000-0000-4000-8000-000000000002",
        "started_at": "2026-08-02T00:00:02.000000Z",
        "finished_at": "2026-08-02T00:00:03.000000Z",
        "source_clean": True,
        "passed": True,
        "failed_invariants": [],
        "sections": report["sections"],
        "raw_sha256": "0" * 64,
    }
    assert all(
        set(section) >= {"status", "passed", "metrics", "failed_invariants"}
        for section in report["sections"].values()
    )


@pytest.mark.parametrize(
    ("call", "message"),
    (
        (lambda probe: probe._require_hccl_backend("gloo"), "严格 distributed autograd 探针要求 HCCL process group"),
        (lambda probe: probe._require_supported_channel(object()), "自动微分模式不支持噪声通道 object"),
        (lambda probe: probe._validate_tag_phases((7,), (7,)), "forward/backward P2P tag 不匹配"),
    ),
)
def test_probe_explicit_error_contracts(call, message):
    probe = _probe_module()

    with pytest.raises(ValueError, match=f"^{message}$"):
        call(probe)


def test_probe_contract_matrix_executes_on_two_processes_with_exact_digests(
    tmp_path,
):
    output = tmp_path / "contract.json"
    context = mp.spawn(
        _runtime_contract_worker,
        args=(2, _free_port(), str(output)),
        nprocs=2,
        join=False,
    )
    _join(context)
    result = json.loads(output.read_text(encoding="utf-8"))

    assert result["passed"]
    assert result["public_routing_enabled"]
    assert set(result["exact_errors"]) == {
        "sample",
        "counts",
        "collapse",
        "direct_complex",
        "parameter_schema",
        "ownership",
        "shape",
        "dtype",
        "unsupported_gate",
        "unsupported_channel",
        "unsupported_observable",
        "non_hccl_strict",
        "cpu_fallback",
        "checkpoint",
        "tag_mismatch_injection",
        "rank_route_mismatch",
    }
    assert all(result["exact_errors"].values())
    assert all(
        item["unique_digest_count"] == 1
        for item in result["case_digests"].values()
    )
