import json
import os
from pathlib import Path
import socket

import numpy as np
import torch
import torch.multiprocessing as mp

from aicir import (
    AmplitudeDampingChannel,
    Circuit,
    NoiseModel,
    PauliString,
    cx,
    hadamard,
    pauli_z,
)
from aicir.distributed import DistSimulator


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _simulator_worker(rank, world_size, port, output_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    simulator = DistSimulator.from_env(
        fallback_to_cpu=True,
        process_group_backend="gloo",
    )

    bell = Circuit(
        hadamard(0),
        cx(target_qubit=1, control_qubits=(0,)),
        n_qubits=2,
    )
    bell_result = simulator.run(
        bell,
        observables={"zz": PauliString("ZZ", n_qubits=2)},
        shots=256,
        seed=5,
    )
    bell_state = bell_result.state.to_numpy(root=0)

    initial = (
        np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex64)
        if rank == 0
        else None
    )
    initial_result = simulator.run(
        Circuit(n_qubits=2),
        initial_state=initial,
    )
    initial_array = initial_result.state.to_numpy(root=0)

    noisy = Circuit(pauli_z(0), n_qubits=1)
    noisy.noise_model = NoiseModel().add_channel(
        AmplitudeDampingChannel(target_qubit=0, gamma=1.0)
    )
    noisy_initial = (
        np.array([0.0, 1.0], dtype=np.complex64)
        if rank == 0
        else None
    )
    noisy_result = simulator.run(noisy, initial_state=noisy_initial)
    noisy_array = noisy_result.state.to_numpy(root=0)

    output = Path(output_dir)
    if rank == 0:
        np.save(output / "bell.npy", bell_state)
        np.save(output / "initial.npy", initial_array)
        np.save(output / "noisy.npy", noisy_array)
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "zz": bell_result.expectations["zz"],
                    "counts": dict(bell_result.counts),
                },
                sort_keys=True,
            )
        )
    else:
        assert bell_result.counts is None
        assert bell_state is None
        assert initial_array is None
        assert noisy_array is None

    torch.distributed.destroy_process_group()


def test_two_rank_end_to_end_simulator(tmp_path):
    mp.spawn(
        _simulator_worker,
        args=(2, _free_port(), str(tmp_path)),
        nprocs=2,
        join=True,
    )

    bell = np.load(tmp_path / "bell.npy")
    expected_bell = np.array(
        [2**-0.5, 0.0, 0.0, 2**-0.5],
        dtype=np.complex64,
    )
    np.testing.assert_allclose(bell, expected_bell, atol=1e-6)
    np.testing.assert_array_equal(
        np.load(tmp_path / "initial.npy"),
        [0.0, 0.0, 1.0, 0.0],
    )
    np.testing.assert_allclose(
        np.load(tmp_path / "noisy.npy"),
        [[1.0, 0.0], [0.0, 0.0]],
        atol=1e-6,
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert abs(summary["zz"] - 1.0) < 1e-6
    assert set(summary["counts"]) <= {"00", "11"}
    assert sum(summary["counts"].values()) == 256
