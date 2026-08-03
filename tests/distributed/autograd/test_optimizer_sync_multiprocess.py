"""Replicated optimizer agreement is driven solely by GradientBucketFn."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import time

import pytest
import torch
import torch.multiprocessing as mp

from aicir import Circuit, PauliString
from aicir.core.circuit import ry
from aicir.distributed import DistNPUBackend, DistSimulator, DistState, PureStateParam
from aicir.distributed._contracts import PARAMETER_STRUCTURE_ERROR
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.autograd._collectives import _replicated_all_reduce
from aicir.distributed.autograd._parameters import (
    _bind_replicated_gradient_bucket,
    _bucket_parameters,
    StinespringParam,
)
from aicir.distributed.autograd._reducers import _PairReducer
from aicir.distributed.gates import _GatePlanner
from aicir.distributed.layout import _Layout, _ShardSpec
from aicir.ir import instruction_parameter
from aicir.noise import AmplitudeDampingChannel, BitFlipChannel, NoiseModel
from scripts.npu import distributed_autograd_probe as autograd_probe


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _join(context, *, timeout=60):
    try:
        deadline = time.monotonic() + timeout
        while not context.join(timeout=max(0.0, deadline - time.monotonic())):
            assert time.monotonic() < deadline, "gradient-bucket optimizer test timed out"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
        for process in context.processes:
            if process.is_alive():
                process.kill()
        for process in context.processes:
            process.join(timeout=5)
    assert all(process.exitcode == 0 for process in context.processes)


def _digest(parameter, optimizer):
    state = optimizer.state_dict()
    payload = parameter.detach().cpu().numpy().tobytes()
    for key in sorted(state["state"]):
        for field, value in sorted(state["state"][key].items()):
            payload += field.encode("ascii")
            if isinstance(value, torch.Tensor):
                payload += value.detach().cpu().numpy().tobytes()
            else:
                payload += repr(value).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _digest_tensor(digest: str, *, device="cpu"):
    return torch.tensor(
        list(bytes.fromhex(digest)), dtype=torch.float32, device=device
    ).contiguous()


def _all_ranks_equal_real(backend, payload):
    payload = payload.detach().to(dtype=torch.float32).reshape(-1).contiguous()
    gathered = backend.communicator.all_gather_real(payload)
    return all(torch.equal(item, gathered[0]) for item in gathered[1:])


def _worker(rank, world_size, port, output_path):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size),
        RANK=str(rank), LOCAL_RANK=str(rank),
    )
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    try:
        results = {}
        for count in (32, 128):
            for optimizer_name, factory in (
                ("sgd", lambda values: torch.optim.SGD([values], lr=0.01, momentum=0.9)),
                ("adam", lambda values: torch.optim.Adam([values], lr=0.01)),
            ):
                parameter = torch.nn.Parameter(torch.linspace(-0.4, 0.5, count, dtype=torch.float32))
                optimizer = factory(parameter)
                before = len(backend.communicator.communication_records)
                for _ in range(100):
                    optimizer.zero_grad(set_to_none=True)
                    (alias,) = _bucket_parameters((parameter,), communicator=backend.communicator)
                    # Every rank supplies a distinct local VJP; the one bucket
                    # makes the optimizer see the exact global gradient.
                    (alias * float(rank + 1)).sum().backward()
                    optimizer.step()
                    assert _all_ranks_equal_real(
                        backend,
                        _digest_tensor(_digest(parameter, optimizer)),
                    )
                records = backend.communicator.communication_records[before:]
                results[f"{optimizer_name}-{count}"] = {
                    "all_reduce_count": sum(record["kind"] == "all_reduce" for record in records),
                    "all_float32": all(record["dtype"] == "torch.float32" for record in records),
                    "digest": _digest(parameter, optimizer),
                }
        # Initial-state leaves deliberately bypass the replicated bucket, but
        # must still execute their real distributed ownership paths.  Root
        # leaves use the normalize -> paired scatter -> paired reducer graph;
        # shard leaves retain separate optimizers and are normalized only at
        # the probability/reducer boundary.
        n_qubits = world_size.bit_length() - 1
        layout = _Layout.explicit(
            tuple(reversed(range(n_qubits))),
            n_qubits=n_qubits,
            distributed_axes=n_qubits,
        )
        spec = _ShardSpec.build(n_qubits, world_size, rank, "vector", layout)
        simulator = DistSimulator(backend)
        local_weights = torch.arange(
            spec.global_start + 1,
            spec.global_stop + 1,
            dtype=torch.float32,
        )
        dimension = 1 << n_qubits
        root_real = (
            torch.nn.Parameter(torch.linspace(0.3, 1.1, dimension, dtype=torch.float32))
            if rank == 0 else None
        )
        root_imag = (
            torch.nn.Parameter(torch.linspace(-0.4, 0.5, dimension, dtype=torch.float32))
            if rank == 0 else None
        )
        root_optimizer = torch.optim.Adam([root_real, root_imag], lr=0.01) if rank == 0 else None
        sharded_real = torch.nn.Parameter(
            torch.linspace(float(rank + 1), float(rank + 1.5), spec.local_shape[0], dtype=torch.float32)
        )
        sharded_imag = torch.nn.Parameter(
            torch.linspace(-float(rank + 0.5), -float(rank), spec.local_shape[0], dtype=torch.float32)
        )
        sharded_optimizer = torch.optim.SGD([sharded_real, sharded_imag], lr=0.01, momentum=0.9)
        root_probability_sum_is_one = True
        root_state_norm_is_one = True
        root_gradients_nonzero = True
        sharded_probability_sum_is_one = True
        sharded_probabilities_finite = True
        sharded_gradients_nonzero = True
        for _ in range(100):
            if root_optimizer is not None:
                root_optimizer.zero_grad(set_to_none=True)
            root_state = simulator._prepare_initial_state(
                n_qubits=n_qubits,
                layout=layout,
                initial_state=PureStateParam(root_real, root_imag) if rank == 0 else None,
                initial_density_matrix=None,
            )
            root_probabilities = _PairReducer(backend).probabilities(root_state._pair, spec)
            root_mean = _replicated_all_reduce(
                (root_probabilities * local_weights).sum().reshape(()),
                communicator=backend.communicator,
            )
            root_loss = (
                float(world_size) * root_mean.detach() + root_mean - root_mean.detach()
            )
            root_loss.backward()
            if rank == 0:
                root_gradients_nonzero = root_gradients_nonzero and all(
                    leaf.grad is not None and bool(leaf.grad.detach().abs().max() > 0)
                    for leaf in (root_real, root_imag)
                )
                root_optimizer.step()
            reconstructed_root = simulator._prepare_initial_state(
                n_qubits=n_qubits,
                layout=layout,
                initial_state=PureStateParam(root_real, root_imag) if rank == 0 else None,
                initial_density_matrix=None,
            )
            reconstructed_probabilities = _PairReducer(backend).probabilities(
                reconstructed_root._pair, spec
            )
            root_global_probability = backend.communicator.all_reduce_sum_real(
                reconstructed_probabilities.detach().sum().reshape(())
            )
            root_global_norm = backend.communicator.all_reduce_sum_real(
                reconstructed_root._pair.abs_sq().detach().sum().reshape(())
            )
            root_probability_sum_is_one = root_probability_sum_is_one and bool(
                torch.isclose(root_global_probability, torch.ones_like(root_global_probability), atol=1e-5)
            )
            root_state_norm_is_one = root_state_norm_is_one and bool(
                torch.isclose(torch.sqrt(root_global_norm), torch.ones_like(root_global_norm), atol=1e-5)
            )
            sharded_optimizer.zero_grad(set_to_none=True)
            sharded_state = DistState.from_pair(
                _Pair(sharded_real.reshape(spec.local_shape), sharded_imag.reshape(spec.local_shape)),
                spec=spec,
                backend=backend,
            )
            sharded_probabilities = _PairReducer(backend).probabilities(
                sharded_state._pair, spec
            )
            sharded_mean = _replicated_all_reduce(
                (sharded_probabilities * local_weights).sum().reshape(()),
                communicator=backend.communicator,
            )
            sharded_loss = (
                float(world_size) * sharded_mean.detach()
                + sharded_mean
                - sharded_mean.detach()
            )
            sharded_loss.backward()
            sharded_gradients_nonzero = sharded_gradients_nonzero and all(
                leaf.grad is not None and bool(leaf.grad.detach().abs().max() > 0)
                for leaf in (sharded_real, sharded_imag)
            )
            sharded_optimizer.step()
            reconstructed_sharded = DistState.from_pair(
                _Pair(sharded_real.reshape(spec.local_shape), sharded_imag.reshape(spec.local_shape)),
                spec=spec,
                backend=backend,
            )
            reconstructed_sharded_probabilities = _PairReducer(backend).probabilities(
                reconstructed_sharded._pair, spec
            )
            sharded_global_probability = backend.communicator.all_reduce_sum_real(
                reconstructed_sharded_probabilities.detach().sum().reshape(())
            )
            sharded_probability_sum_is_one = sharded_probability_sum_is_one and bool(
                torch.isclose(sharded_global_probability, torch.ones_like(sharded_global_probability), atol=1e-5)
            )
            sharded_probabilities_finite = sharded_probabilities_finite and bool(
                torch.isfinite(reconstructed_sharded_probabilities).all()
            )
        ownership_payload = torch.tensor(
            [
                float(root_optimizer is not None and bool(root_optimizer.state)),
                float(sharded_optimizer.state != {}),
                float(rank != 0 and root_real is None and root_imag is None and root_optimizer is None),
                float(root_gradients_nonzero if rank == 0 else True),
                float(root_probability_sum_is_one),
                float(root_state_norm_is_one),
                float(sharded_gradients_nonzero),
                float(sharded_probability_sum_is_one),
                float(sharded_probabilities_finite),
            ],
            dtype=torch.float32,
        )
        ownership_rows = backend.communicator.all_gather_real(ownership_payload.contiguous())
        ownership = [
            {
                "root_has_state": bool(row[0]),
                "sharded_has_state": bool(row[1]),
                "root_nonowner_has_no_leaf_or_state": bool(row[2]),
                "root_gradients_nonzero": bool(row[3]),
                "root_global_probability_sum_is_one": bool(row[4]),
                "root_state_norm_is_one": bool(row[5]),
                "sharded_gradients_nonzero": bool(row[6]),
                "sharded_global_probability_sum_is_one": bool(row[7]),
                "sharded_probabilities_finite": bool(row[8]),
                "sharded_normalized": bool(row[7] and row[8]),
            }
            for row in ownership_rows
        ]
        if rank == 0:
            Path(output_path).write_text(json.dumps({"results": results, "ownership": ownership}), encoding="utf-8")
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", (2, 4))
def test_replicated_sgd_and_adam_agree_for_100_steps_on_two_and_four_ranks(tmp_path, world_size):
    output = tmp_path / f"optimizer-{world_size}.json"
    context = mp.spawn(_worker, args=(world_size, _free_port(), str(output)), nprocs=world_size, join=False)
    _join(context)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["ownership"][0]["root_has_state"]
    assert all(not item["root_has_state"] for item in result["ownership"][1:])
    assert all(item["sharded_has_state"] and item["sharded_normalized"] for item in result["ownership"])
    assert result["ownership"][0]["root_gradients_nonzero"]
    assert all(item["root_nonowner_has_no_leaf_or_state"] for item in result["ownership"][1:])
    assert all(item["root_global_probability_sum_is_one"] for item in result["ownership"])
    assert all(item["root_state_norm_is_one"] for item in result["ownership"])
    assert all(item["sharded_gradients_nonzero"] for item in result["ownership"])
    assert all(item["sharded_global_probability_sum_is_one"] for item in result["ownership"])
    assert all(item["sharded_probabilities_finite"] for item in result["ownership"])
    assert set(result["results"]) == {"sgd-32", "sgd-128", "adam-32", "adam-128"}
    for metrics in result["results"].values():
        assert metrics["all_reduce_count"] == 100
        assert metrics["all_float32"]


def _schema_mismatch_worker(rank, world_size, port, output_path):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size),
        RANK=str(rank), LOCAL_RANK=str(rank),
    )
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    try:
        theta = torch.tensor(0.2, dtype=torch.float32, requires_grad=rank == 0)
        circuit = Circuit(ry(theta, 0), n_qubits=1)
        backend.communicator.clear_communication_records()
        try:
            _bind_replicated_gradient_bucket(circuit, communicator=backend.communicator)
        except ValueError as error:
            message = str(error)
        else:
            message = "NO_ERROR"
        records = backend.communicator.communication_records
        observed = backend.communicator.all_gather_real(
            torch.tensor(
                [
                    float(message == PARAMETER_STRUCTURE_ERROR),
                    float(sum(record["kind"] == "all_reduce" for record in records)),
                ],
                dtype=torch.float32,
            )
        )
        if rank == 0:
            Path(output_path).write_text(
                json.dumps(
                    [
                        {
                            "message": PARAMETER_STRUCTURE_ERROR if int(item[0]) else message,
                            "gradient_all_reduces": int(item[1]),
                        }
                        for item in observed
                    ]
                ),
                encoding="utf-8",
            )
    finally:
        torch.distributed.destroy_process_group()


def test_requires_grad_schema_mismatch_is_collective_safe_before_bucket_allreduce(tmp_path):
    output = tmp_path / "schema-mismatch.json"
    context = mp.spawn(_schema_mismatch_worker, args=(2, _free_port(), str(output)), nprocs=2, join=False)
    _join(context)

    observed = json.loads(output.read_text(encoding="utf-8"))
    assert observed == [
        {"message": PARAMETER_STRUCTURE_ERROR, "gradient_all_reduces": 0},
        {"message": PARAMETER_STRUCTURE_ERROR, "gradient_all_reduces": 0},
    ]


def _integrated_state(backend, spec):
    """Return the local shard of ``|0...0>`` without a root-owned leaf."""

    indices = torch.arange(
        spec.global_start,
        spec.global_stop,
        dtype=torch.long,
        device=backend._device,
    )
    real = (indices == 0).to(torch.float32).reshape(-1, 1)
    return DistState.from_pair(
        _Pair(real, torch.zeros_like(real)),
        spec=spec,
        backend=backend,
    )


def _integrated_digest(leaves, optimizer):
    """Hash every caller-owned replicated leaf and its optimizer state."""

    digest = hashlib.sha256()
    for leaf in leaves:
        digest.update(leaf.detach().cpu().contiguous().numpy().tobytes())
    for parameter_id, state in sorted(optimizer.state_dict()["state"].items()):
        digest.update(str(parameter_id).encode("ascii"))
        for key, value in sorted(state.items()):
            digest.update(key.encode("ascii"))
            digest.update(
                value.detach().cpu().contiguous().numpy().tobytes()
                if isinstance(value, torch.Tensor)
                else repr(value).encode("ascii")
            )
    return digest.hexdigest()


def _integrated_private_path_worker(rank, world_size, port, output_path):
    """Exercise the private engine with every replicated parameter family."""

    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size),
        RANK=str(rank), LOCAL_RANK=str(rank),
    )
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    try:
        n_qubits = world_size.bit_length() - 1
        layout = _Layout.explicit(
            tuple(reversed(range(n_qubits))),
            n_qubits=n_qubits,
            distributed_axes=n_qubits,
        )
        spec = _ShardSpec.build(n_qubits, world_size, rank, "vector", layout)
        theta = torch.nn.Parameter(torch.tensor(0.37, dtype=torch.float32))
        damping = torch.nn.Parameter(torch.tensor(0.23, dtype=torch.float32))
        missing = torch.nn.Parameter(torch.tensor(0.11, dtype=torch.float32))
        raw_real = torch.nn.Parameter(
            torch.tensor([[0.4, -0.3], [0.2, 0.7]], dtype=torch.float32)
        )
        raw_imag = torch.nn.Parameter(
            torch.tensor([[0.1, 0.6], [-0.5, 0.2]], dtype=torch.float32)
        )
        stinespring = StinespringParam(2, 2, 1, raw_real, raw_imag, target_qubits=(0,))
        circuit = Circuit(ry(theta, 0), ry(theta, 0), n_qubits=n_qubits)
        circuit.noise_model = (
            NoiseModel()
            .add_channel(AmplitudeDampingChannel(0, damping))
            .add_channel(stinespring)
            # This unmatched rule keeps a genuine missing-gradient leaf in the
            # same circuit/noise/Stinespring bucket.
            .add_channel(BitFlipChannel(0, missing), after_gates=("never",))
        )
        leaves = (theta, damping, missing, raw_real, raw_imag)
        optimizer = torch.optim.SGD(leaves, lr=0.005, momentum=0.9)

        import aicir.distributed.simulator as simulator_module

        captured = {"bound": [], "planned_aliases": []}
        original_bind = simulator_module._bind_replicated_gradient_bucket
        original_plan = _GatePlanner.plan

        def record_bind(candidate, *, communicator):
            rebound = original_bind(candidate, communicator=communicator)
            captured["bound"].append(rebound)
            return rebound

        def record_plan(self, gate, instruction_index):
            parameter = instruction_parameter(gate)
            values = parameter if isinstance(parameter, (tuple, list)) else (parameter,)
            captured["planned_aliases"].extend(
                value
                for value in values
                if getattr(value, "_aicir_gradient_bucket_alias", False)
            )
            return original_plan(self, gate, instruction_index)

        simulator_module._bind_replicated_gradient_bucket = record_bind
        _GatePlanner.plan = record_plan
        try:
            step_reports = []
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                backend.communicator.clear_communication_records()
                evolved, _ = DistSimulator(backend)._run_paired_real(
                    circuit,
                    initial_state=_integrated_state(backend, spec),
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
                records = backend.communicator.communication_records[before_backward:]
                expected_bucket_bytes = sum(leaf.numel() for leaf in leaves) * 4
                bucket_records = [
                    record
                    for record in records
                    if record["kind"] == "all_reduce" and record["bytes"] == expected_bucket_bytes
                ]
                aliases = captured["bound"][-1]
                aliases_planned = len(captured["planned_aliases"]) >= 2
                aliases_bound = (
                    instruction_parameter(aliases.operations[0]) is not theta
                    and getattr(instruction_parameter(aliases.operations[0]), "_aicir_gradient_bucket_alias", False)
                    and aliases.noise_model.rules[0].channel.gamma is not damping
                    and getattr(aliases.noise_model.rules[0].channel.gamma, "_aicir_gradient_bucket_alias", False)
                    and aliases.noise_model.rules[1].channel.real is not raw_real
                    and aliases.noise_model.rules[1].channel.imag is not raw_imag
                )
                gradients = torch.cat(
                    [leaf.grad.detach().reshape(-1) for leaf in leaves]
                ).contiguous()
                step_reports.append(
                    {
                        "bucket_all_reduce_count": len(bucket_records),
                        "aliases_bound": aliases_bound,
                        "aliases_planned": aliases_planned,
                        "gradient_agreement": _all_ranks_equal_real(backend, gradients),
                        "nonzero_gradients": [
                            bool(leaf.grad is not None and leaf.grad.detach().abs().max() > 0)
                            for leaf in (theta, damping, raw_real, raw_imag)
                        ],
                        "missing_gradient_zero": bool(torch.equal(missing.grad, torch.zeros_like(missing))),
                        "caller_cache_unchanged": circuit.noise_model._kraus_cache == {},
                        "rebound_cache_empty": aliases.noise_model._kraus_cache == {},
                        "caller_hooks_empty": all(not getattr(leaf, "_backward_hooks", None) for leaf in leaves),
                        "handles": backend.communicator.work_handle_status,
                    }
                )
                optimizer.step()
        finally:
            simulator_module._bind_replicated_gradient_bucket = original_bind
            _GatePlanner.plan = original_plan

        digest_agreement = _all_ranks_equal_real(
            backend,
            _digest_tensor(_integrated_digest(leaves, optimizer)),
        )
        if rank == 0:
            Path(output_path).write_text(
                json.dumps({"steps": step_reports, "digest_agreement": digest_agreement}),
                encoding="utf-8",
            )
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", (2, 4))
def test_private_paired_real_buckets_real_circuit_noise_and_stinespring_for_two_sgd_steps(
    tmp_path, world_size
):
    """One private execution bucket covers real typed objects on W2 and W4."""

    output = tmp_path / f"private-integrated-{world_size}.json"
    context = mp.spawn(
        _integrated_private_path_worker,
        args=(world_size, _free_port(), str(output)),
        nprocs=world_size,
        join=False,
    )
    _join(context, timeout=120)

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["digest_agreement"]
    assert len(result["steps"]) == 2
    for step in result["steps"]:
        assert step["bucket_all_reduce_count"] == 1
        assert step["aliases_bound"] and step["aliases_planned"]
        assert step["gradient_agreement"]
        assert all(step["nonzero_gradients"])
        assert step["missing_gradient_zero"]
        assert step["caller_cache_unchanged"] and step["rebound_cache_empty"]
        assert step["caller_hooks_empty"]
        assert step["handles"] == {
            "outstanding_work_handles": 0,
            "unfinished_work_handles": 0,
            "all_handles_complete": True,
        }


def _probe_integrated_private_path_worker(rank, world_size, port, output_path):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size),
        RANK=str(rank), LOCAL_RANK=str(rank),
    )
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    try:
        result = autograd_probe._integrated_private_path_optimizer_case(backend)
        if rank == 0:
            Path(output_path).write_text(json.dumps(result), encoding="utf-8")
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


def test_optimizer_probe_integrated_private_path_reports_real_bucket_metrics(tmp_path):
    """The probe subcase executes, rather than merely declaring its schema."""

    output = tmp_path / "probe-integrated-private-path.json"
    context = mp.spawn(
        _probe_integrated_private_path_worker,
        args=(2, _free_port(), str(output)),
        nprocs=2,
        join=False,
    )
    _join(context, timeout=120)

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["passed"]
    assert result["steps"] == 2
    assert result["gradient_all_reduce_count"] == [1, 1]
    assert result["gradient_agreement"] == [True, True]
    assert result["parameter_and_optimizer_state_agree"]
    assert result["missing_gradient_zero"] == [True, True]
    assert result["caller_kraus_cache_empty"] == [True, True]
    assert result["unfinished_work_handles"] == [0, 0]
    assert result["all_handles_complete"] == [True, True]


class _NeverCompleteWork:
    def is_completed(self):
        return False


def _strict_optimizer_probe_worker(
    rank,
    world_size,
    port,
    output_path,
    forbid_object_collective,
    inject_unfinished_handle,
):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size),
        RANK=str(rank), LOCAL_RANK=str(rank),
    )
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    original_all_gather_object = torch.distributed.all_gather_object
    try:
        if forbid_object_collective:
            def forbidden(*_args, **_kwargs):
                raise AssertionError("strict optimizer path must not use all_gather_object")

            torch.distributed.all_gather_object = forbidden
        if inject_unfinished_handle:
            backend.communicator.register_work_handle(_NeverCompleteWork())
        result = autograd_probe._optimizer_section(backend)
        records = backend.communicator.communication_records
        if rank == 0:
            Path(output_path).write_text(
                json.dumps(
                    {
                        "result": result,
                        "all_gather_dtypes": [
                            record["dtype"] for record in records if record["kind"] == "all_gather"
                        ],
                    }
                ),
                encoding="utf-8",
            )
        torch.distributed.barrier()
    finally:
        torch.distributed.all_gather_object = original_all_gather_object
        torch.distributed.destroy_process_group()


def _run_strict_optimizer_probe(tmp_path, *, forbid_object_collective, inject_unfinished_handle):
    output = tmp_path / "strict-optimizer-probe.json"
    context = mp.spawn(
        _strict_optimizer_probe_worker,
        args=(2, _free_port(), str(output), forbid_object_collective, inject_unfinished_handle),
        nprocs=2,
        join=False,
    )
    _join(context, timeout=120)
    return json.loads(output.read_text(encoding="utf-8"))


def test_strict_optimizer_probe_never_uses_object_collectives_and_gathers_float32(tmp_path):
    result = _run_strict_optimizer_probe(
        tmp_path,
        forbid_object_collective=True,
        inject_unfinished_handle=False,
    )

    assert result["result"]["passed"]
    assert result["result"]["all_handles_complete"]
    assert result["all_gather_dtypes"]
    assert set(result["all_gather_dtypes"]) == {"torch.float32"}


def test_strict_optimizer_probe_observes_an_injected_unfinished_handle(tmp_path):
    result = _run_strict_optimizer_probe(
        tmp_path,
        forbid_object_collective=False,
        inject_unfinished_handle=True,
    )

    assert not result["result"]["passed"]
    assert not result["result"]["all_handles_complete"]
    assert result["result"]["cases"]["sgd-32"]["unfinished_work_handles"] > 0
