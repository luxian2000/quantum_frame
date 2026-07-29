import json
import os
from pathlib import Path
import socket

import numpy as np
import torch
import torch.multiprocessing as mp

from aicir import PauliString, cx, hadamard
from aicir.distributed import DistNPUBackend, DistResult, DistState
from aicir.distributed.gates import _GatePlanner, _VectorKernel
from aicir.distributed.layout import _Layout
from aicir.distributed.reducers import _Reducer


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _reducer_worker(rank, world_size, port, output_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    backend = DistNPUBackend.from_env(
        fallback_to_cpu=True,
        process_group_backend="gloo",
    )
    layout = _Layout.explicit(
        (0, 1),
        n_qubits=2,
        distributed_axes=1,
    )
    state = DistState.zero(2, backend=backend, layout=layout)
    planner = _GatePlanner(backend, layout, 2)
    kernel = _VectorKernel(backend)
    for index, gate in enumerate(
        (
            hadamard(0),
            cx(target_qubit=1, control_qubits=(0,)),
        )
    ):
        state = kernel.apply(
            state,
            planner.plan(gate, instruction_index=index),
        )

    reducer = _Reducer(backend)
    local_probabilities = reducer.probabilities(state)
    z0 = reducer.expectation(
        state,
        PauliString("ZI", n_qubits=2),
    )
    zz = reducer.expectation(
        state,
        PauliString("ZZ", n_qubits=2),
    )
    counts, collapsed = reducer.sample_z(
        state,
        shots=2000,
        measure_qubits=(0, 1),
        seed=17,
        collapse=False,
    )
    _, collapsed_once = reducer.sample_z(
        state,
        shots=1,
        measure_qubits=(0, 1),
        seed=23,
        collapse=True,
    )
    collapsed_array = collapsed_once.to_numpy(root=0)
    result = DistResult(
        state=state,
        local_probabilities=local_probabilities,
        expectations={"z0": z0, "zz": zz},
        counts=counts,
        rank=rank,
        world_size=world_size,
    )
    full_probabilities = result.gather_probabilities(root=0)

    output = Path(output_dir)
    if rank == 0:
        np.save(output / "probabilities.npy", full_probabilities)
        np.save(output / "collapsed.npy", collapsed_array)
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "z0": z0,
                    "zz": zz,
                    "counts": counts,
                    "collapsed": collapsed is not None,
                },
                sort_keys=True,
            )
        )
    else:
        assert counts is None
        assert collapsed is None
        assert collapsed_once is not None
        assert collapsed_array is None
        assert full_probabilities is None

    torch.distributed.destroy_process_group()


def test_bell_reductions_and_sampling_without_full_probability_gather(tmp_path):
    mp.spawn(
        _reducer_worker,
        args=(2, _free_port(), str(tmp_path)),
        nprocs=2,
        join=True,
    )

    np.testing.assert_allclose(
        np.load(tmp_path / "probabilities.npy"),
        [0.5, 0.0, 0.0, 0.5],
        atol=1e-6,
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert abs(summary["z0"]) < 1e-6
    assert abs(summary["zz"] - 1.0) < 1e-6
    assert set(summary["counts"]) <= {"00", "11"}
    assert sum(summary["counts"].values()) == 2000
    assert abs(summary["counts"]["00"] - 1000) < 120
    assert not summary["collapsed"]
    collapsed = np.load(tmp_path / "collapsed.npy")
    assert np.isclose(np.linalg.norm(collapsed), 1.0, atol=1e-6)
    assert (
        np.allclose(collapsed, [1.0, 0.0, 0.0, 0.0], atol=1e-6)
        or np.allclose(collapsed, [0.0, 0.0, 0.0, 1.0], atol=1e-6)
    )
