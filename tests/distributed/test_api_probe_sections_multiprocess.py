import json
import os
from pathlib import Path
import socket

import pytest
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
            for name in (
                "state",
                "layout",
                "continuation",
                "noise",
                "observable",
                "measure",
                "result",
            )
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


@pytest.mark.parametrize("world_size", [2, 4])
def test_api_probe_sections_are_collective_safe(world_size, tmp_path):
    mp.spawn(
        _section_worker,
        args=(world_size, _free_port(), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    payloads = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text())
        for rank in range(world_size)
    ]
    root = payloads[0]
    nonroots = payloads[1:]

    assert not root["evaluation_failure"]["passed"]
    assert root["evaluation_failure"]["metrics"][
        "root_evaluation_error"
    ] == {
        "type": "RuntimeError",
        "message": "post-collective root evaluation failed",
    }
    assert all(
        payload["evaluation_failure"]["passed"] is False
        and payload["evaluation_failure"]["metrics"] == {}
        for payload in nonroots
    )

    assert all(
        section["passed"]
        for section in root["sections"].values()
    )
    assert all(
        section["passed"] and section["metrics"] == {}
        for payload in nonroots
        for section in payload["sections"].values()
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

    noise_metrics = root["sections"]["noise"]["metrics"]
    expected_distributed_axes = list(
        range(world_size.bit_length() - 1)
    )
    expected_layout = list(
        range(world_size.bit_length())
    )
    assert noise_metrics["targeted_distributed_axes"] == (
        expected_distributed_axes
    )
    assert noise_metrics["logical_to_storage"] == expected_layout
    assert noise_metrics["channel_targets"] == {
        "amplitude_damping": expected_distributed_axes[0],
        "bit_flip": expected_distributed_axes[
            1 % len(expected_distributed_axes)
        ],
        "phase_flip": expected_distributed_axes[
            2 % len(expected_distributed_axes)
        ],
        "depolarizing": expected_distributed_axes[
            3 % len(expected_distributed_axes)
        ],
    }
    for metric in (
        "amplitude_damping_error",
        "bit_flip_error",
        "phase_flip_error",
        "depolarizing_error",
        "noise_sequence_error",
        "rule_selection_error",
    ):
        assert noise_metrics[metric] <= probe.STATE_ATOL
    assert noise_metrics["noise_density_error"] <= probe.STATE_ATOL
    assert noise_metrics["noise_trace_error"] <= probe.REDUCTION_ATOL
    assert noise_metrics["noise_probability_error"] <= probe.REDUCTION_ATOL
    assert noise_metrics["channel_count"] == 4
    assert "matched_gate_name" not in noise_metrics

    observable_metrics = root["sections"]["observable"]["metrics"]
    assert observable_metrics["targeted_distributed_axes"] == (
        expected_distributed_axes
    )
    assert observable_metrics["logical_to_storage"] == expected_layout
    assert observable_metrics["local_dense_target"] == (
        expected_distributed_axes[1]
        if len(expected_distributed_axes) > 1
        else expected_distributed_axes[0]
    )
    assert observable_metrics["pauli_error"] <= probe.REDUCTION_ATOL
    assert observable_metrics["hamiltonian_error"] <= probe.REDUCTION_ATOL
    assert observable_metrics["local_dense_error"] <= probe.REDUCTION_ATOL

    measure_metrics = root["sections"]["measure"]["metrics"]
    assert measure_metrics["logical_to_storage"] != list(
        range(world_size.bit_length())
    )
    assert measure_metrics["measure_qubits"] == [0]
    assert measure_metrics["full_register_shots"] == 128
    assert measure_metrics["subset_shots"] == 128
    assert measure_metrics["collapse_shots"] == 1
    assert all(
        len(key) == world_size.bit_length()
        for key in measure_metrics["full_register_support"]
    )
    assert all(
        len(key) == 1
        for key in measure_metrics["subset_support"]
    )
    assert measure_metrics["collapsed_support_error"] <= probe.STATE_ATOL
    assert measure_metrics["collapsed_norm_error"] <= probe.REDUCTION_ATOL
    assert measure_metrics["collapse_count_state_consistent"]

    result_metrics = root["sections"]["result"]["metrics"]
    assert result_metrics["four_return_combinations"]
    assert result_metrics["implicit_gather_delta"] == 0
    assert result_metrics["explicit_gather_delta"] > 0
    assert len(result_metrics["return_combinations"]) == 4
    assert {
        (
            combination["return_state"],
            combination["return_probabilities"],
        ): (
            combination["state_present"],
            combination["local_probabilities_present"],
            combination["state_materialized"],
            combination["probabilities_materialized"],
        )
        for combination in result_metrics["return_combinations"]
    } == {
        (False, False): (False, False, False, False),
        (False, True): (False, True, False, True),
        (True, False): (True, False, True, False),
        (True, True): (True, True, True, True),
    }
