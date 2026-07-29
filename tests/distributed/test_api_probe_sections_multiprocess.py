import json
import os
from pathlib import Path
import socket

import torch
import torch.multiprocessing as mp

from aicir.distributed import DistSimulator
from scripts.npu import distributed_api_probe as probe


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _raise_root_evaluation_error():
    raise RuntimeError("post-collective root evaluation failed")


def _section_worker(rank, world_size, port, output_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    simulator = DistSimulator.from_env(
        fallback_to_cpu=True,
        process_group_backend="gloo",
    )
    try:
        evaluation_failure = probe._evaluate_root_section(
            simulator.backend,
            _raise_root_evaluation_error,
        )
        sections = {
            name: probe.SECTION_RUNNERS[name](simulator)
            for name in ("state", "layout", "continuation")
        }
        payload = {
            "evaluation_failure": evaluation_failure,
            "sections": sections,
        }
        Path(output_dir, f"rank-{rank}.json").write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_api_probe_sections_are_collective_safe(tmp_path):
    mp.spawn(
        _section_worker,
        args=(2, _free_port(), str(tmp_path)),
        nprocs=2,
        join=True,
    )

    root = json.loads((tmp_path / "rank-0.json").read_text())
    nonroot = json.loads((tmp_path / "rank-1.json").read_text())

    assert not root["evaluation_failure"]["passed"]
    assert root["evaluation_failure"]["metrics"][
        "root_evaluation_error"
    ] == {
        "type": "RuntimeError",
        "message": "post-collective root evaluation failed",
    }
    assert nonroot["evaluation_failure"]["passed"] is False
    assert nonroot["evaluation_failure"]["metrics"] == {}

    assert all(
        section["passed"]
        for section in root["sections"].values()
    )
    assert all(
        section["passed"]
        and section["metrics"] == {}
        for section in nonroot["sections"].values()
    )

    state_metrics = root["sections"]["state"]["metrics"]
    assert state_metrics["zero_state_max_error"] <= probe.STATE_ATOL
    assert state_metrics["zero_state_norm_error"] <= probe.REDUCTION_ATOL
    assert state_metrics["zero_probability_max_error"] <= probe.STATE_ATOL
    assert all(
        sizes["zero_vector"] == 2
        for sizes in state_metrics["local_tensor_sizes"]
    )

    layout_metrics = root["sections"]["layout"]["metrics"]
    assert (
        layout_metrics["auto_layout"]
        != layout_metrics["logical_to_storage"]
    )
    for mode in ("auto", "explicit"):
        assert (
            layout_metrics[f"{mode}_reference_statevector_error"]
            <= probe.STATE_ATOL
        )
        assert (
            layout_metrics[f"{mode}_reference_probability_error"]
            <= probe.STATE_ATOL
        )
        assert (
            layout_metrics[f"{mode}_reference_expectation_error"]
            <= probe.REDUCTION_ATOL
        )

    assert set(root["sections"]["continuation"]["metrics"]) == {
        "continuation_vector_error",
        "continuation_density_error",
        "logical_to_storage",
        "local_tensor_sizes",
    }
