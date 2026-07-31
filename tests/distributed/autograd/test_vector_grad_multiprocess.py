"""Cross-shard paired-real statevector gradient regression tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from aicir import Hamiltonian, PauliString
from aicir.ir import Observable
from aicir.core.circuit import ry
from aicir.distributed import DistNPUBackend, parameter_shift_gradient
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.autograd._reducers import _PairReducer
from aicir.distributed.autograd._vector import _PairVectorKernel
from aicir.distributed.gates import _GatePlanner
from aicir.distributed.layout import _Layout, _ShardSpec


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _float64_ry_reference(full, theta, axis, n_qubits):
    """Independent CPU float64 full-state reference for one RY and Z_0."""

    real = torch.tensor(full, dtype=torch.float64, requires_grad=True)
    imag = torch.zeros_like(real, requires_grad=True)
    t = torch.tensor(float(theta), dtype=torch.float64)
    gate = torch.stack((torch.stack((torch.cos(t / 2), -torch.sin(t / 2))), torch.stack((torch.sin(t / 2), torch.cos(t / 2)))) )
    unitary = torch.ones((1, 1), dtype=torch.float64)
    identity = torch.eye(2, dtype=torch.float64)
    for qubit in range(n_qubits):
        unitary = torch.kron(unitary, gate if qubit == axis else identity)
    out_r, out_i = unitary @ real, unitary @ imag
    signs = torch.tensor([1.0 if index < (1 << (n_qubits - 1)) else -1.0 for index in range(1 << n_qubits)], dtype=torch.float64).reshape(-1, 1)
    (signs * (out_r.square() + out_i.square())).sum().backward()
    return real.grad.numpy(), imag.grad.numpy()


def _worker(rank, world_size, port, output_path):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world_size),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
    )
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    n_qubits = int(np.log2(world_size)) + 1
    layout = _Layout.explicit(tuple(range(n_qubits)), n_qubits=n_qubits, distributed_axes=int(np.log2(world_size)))
    spec = _ShardSpec.build(n_qubits, world_size, rank, "vector", layout)
    full = np.arange(1, (1 << n_qubits) + 1, dtype=np.float32)
    full = full / np.linalg.norm(full)
    local = full[spec.global_start : spec.global_stop].reshape(-1, 1)
    observables = (
        PauliString("Z" + "I" * (n_qubits - 1), n_qubits=n_qubits),
        Hamiltonian([("Z" + "I" * (n_qubits - 1), 0.7), ("X" + "I" * (n_qubits - 1), -0.2)]),
        Observable("matrix", np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex64), metadata={"qubits": (0,)}),
    )

    results = []
    for axis in range(layout.distributed_axes):
        def objective(value, observable, *, trainable=False, state_trainable=False):
            theta = torch.tensor(float(value), dtype=torch.float32, requires_grad=trainable)
            real = torch.tensor(local, dtype=torch.float32, requires_grad=state_trainable)
            imag = torch.zeros_like(real, requires_grad=state_trainable)
            pair = _Pair(real, imag)
            plan = _GatePlanner(backend, layout, n_qubits).plan(ry(theta, axis), axis)
            backend.communicator.clear_communication_records()
            evolved = _PairVectorKernel(backend).apply(pair, plan, operation_index=axis)
            value = _PairReducer(backend).expectation(evolved, spec, observable)
            return value, theta, real, imag

        # Pauli also carries the independent initial-state float64 reference;
        # Hamiltonian and dense observables reuse the same distributed axis.
        value, theta, real, imag = objective(0.31, observables[0], trainable=True, state_trainable=True)
        value.backward()
        records = backend.communicator.communication_records
        exchanges = [record for record in records if record["kind"] == "exchange"]
        forward = sum(record["tag"] % 8 < 4 for record in exchanges)
        backward = sum(record["tag"] % 8 >= 4 for record in exchanges)
        shifted = parameter_shift_gradient(
            lambda values: float(objective(values[0], observables[0])[0].detach()), np.array([0.31])
        )[0]
        assert abs(float(theta.grad) - float(shifted)) <= 1e-4, (
            f"rank={rank} axis={axis} native={float(theta.grad)} shifted={float(shifted)}"
        )
        assert forward > 0 and backward > 0
        assert real.grad is not None and imag.grad is not None
        gathered_real = [torch.zeros_like(real.grad) for _ in range(world_size)]
        gathered_imag = [torch.zeros_like(imag.grad) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_real, real.grad)
        torch.distributed.all_gather(gathered_imag, imag.grad)
        if rank == 0:
            reference_real, reference_imag = _float64_ry_reference(full.reshape(-1, 1), 0.31, axis, n_qubits)
            np.testing.assert_allclose(
                torch.cat(gathered_real).numpy(), reference_real, atol=2e-4, rtol=2e-4
            )
            np.testing.assert_allclose(
                torch.cat(gathered_imag).numpy(), reference_imag, atol=2e-4, rtol=2e-4
            )
        for observable in observables[1:]:
            current, current_theta, _, _ = objective(0.31, observable, trainable=True)
            current.backward()
            shifted_observable = parameter_shift_gradient(
                lambda values, observable=observable: float(objective(values[0], observable)[0].detach()), np.array([0.31])
            )[0]
            assert abs(float(current_theta.grad) - float(shifted_observable)) <= 1e-4
        results.append({"axis": axis, "gradient": float(theta.grad), "forward": forward, "backward": backward})

    if rank == 0:
        Path(output_path).write_text(json.dumps(results), encoding="utf-8")
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", (2, 4))
def test_cross_shard_parameter_gradients_use_forward_and_backward_p2p(tmp_path, world_size):
    output = tmp_path / f"world-{world_size}.json"
    mp.spawn(_worker, args=(world_size, _free_port(), str(output)), nprocs=world_size, join=True)
    results = json.loads(output.read_text(encoding="utf-8"))
    assert [item["axis"] for item in results] == list(range(int(np.log2(world_size))))
    assert all(item["forward"] > 0 and item["backward"] > 0 for item in results)
