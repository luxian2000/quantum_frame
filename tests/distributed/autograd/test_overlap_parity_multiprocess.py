"""Parity coverage for baseline, reuse, and kernel-boundary overlap P2P."""

import os
import socket
import time

import pytest
import torch
import torch.multiprocessing as mp

from aicir.core.circuit import ry
from aicir.distributed import DistNPUBackend
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.autograd._vector import _PairVectorKernel
from aicir.distributed.gates import _AutogradExecutionContext, _GatePlanner
from aicir.distributed.layout import _Layout


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank))
    backend = DistNPUBackend.from_env(fallback_to_cpu=True, process_group_backend="gloo")
    layout = _Layout.explicit((0, 1), n_qubits=2, distributed_axes=1)
    results = {}
    for mode in ("baseline", "reuse", "overlap"):
        backend.communicator.set_autograd_communication_mode(mode)
        backend.communicator.clear_communication_records()
        for iteration in range(2):
            theta = torch.tensor(0.31, dtype=torch.float32, requires_grad=True)
            pair = _Pair(
                torch.full((2, 1), float(rank + 1), dtype=torch.float32),
                torch.zeros((2, 1), dtype=torch.float32),
            )
            plan = _GatePlanner(backend, layout, 2, execution_context=_AutogradExecutionContext()).plan(ry(theta, 0), 0)
            evolved = _PairVectorKernel(backend).apply(pair, plan, operation_index=3)
            state = torch.cat((evolved.real.detach().reshape(-1), evolved.imag.detach().reshape(-1)))
            evolved.real.sum().backward()
        results[mode] = (state, theta.grad.detach().clone(), backend.communicator.work_handle_status, getattr(getattr(backend.communicator, "_autograd_pair_buffer_pool", None), "reuse_count", 0))
    baseline_state, baseline_grad, _, _ = results["baseline"]
    for mode in ("reuse", "overlap"):
        state, grad, status, reuse_count = results[mode]
        torch.testing.assert_close(state, baseline_state, atol=1e-6, rtol=0)
        torch.testing.assert_close(grad, baseline_grad, atol=1e-4, rtol=0)
        assert status["outstanding_work_handles"] == 0
        assert reuse_count > 0
    torch.distributed.destroy_process_group()


def _join(context):
    deadline = time.monotonic() + 60
    while not context.join(timeout=max(0.0, deadline - time.monotonic())):
        assert time.monotonic() < deadline, "overlap parity worker timed out"


def test_baseline_reuse_overlap_match_for_cross_shard_gate():
    context = mp.spawn(_worker, args=(_free_port(),), nprocs=2, join=False)
    _join(context)
