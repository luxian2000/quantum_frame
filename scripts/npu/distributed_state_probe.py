"""Strict two-/four-NPU probe for distributed state sharding.

Run from the repository root:

    source /usr/local/Ascend/cann/set_env.sh
    PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=2 scripts/npu/distributed_state_probe.py
    PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=4 scripts/npu/distributed_state_probe.py

The probe never permits CPU fallback. Rank 0 writes one JSON object and every
rank exits nonzero when an invariant fails.
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np
import torch

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


STATE_ATOL = 1e-6
REDUCTION_ATOL = 1e-5


def _root_scalars(backend, values):
    tensor = torch.tensor(
        values,
        dtype=torch.long,
        device=backend._device,
    )
    gathered = backend.communicator.gather_to_root(tensor, root=0)
    if backend.rank != 0:
        return None
    return [item.detach().cpu().tolist() for item in gathered]


def _run_probe():
    simulator = DistSimulator.from_env(fallback_to_cpu=False)
    backend = simulator.backend
    if backend.world_size not in {2, 4}:
        raise ValueError("探针只接受 world_size=2 或 world_size=4")

    device = backend._device
    if device.type != "npu":
        raise RuntimeError(f"严格探针要求 NPU，实际设备为 {device}")
    if device.index != backend.local_rank:
        raise RuntimeError(
            f"rank={backend.rank} 使用 {device}，"
            f"预期 npu:{backend.local_rank}"
        )

    distributed_axes = int(math.log2(backend.world_size))
    n_qubits = distributed_axes + 1
    gates = [hadamard(0)]
    gates.extend(
        cx(target_qubit=target, control_qubits=(0,))
        for target in range(1, n_qubits)
    )
    gates.append(pauli_z(n_qubits - 1))
    vector_circuit = Circuit(*gates, n_qubits=n_qubits)
    vector_result = simulator.run(
        vector_circuit,
        observables={
            "z0": PauliString(
                "Z" + "I" * (n_qubits - 1),
                n_qubits=n_qubits,
            )
        },
        shots=256,
        seed=29,
    )
    vector = vector_result.state.to_numpy(root=0)
    probabilities = vector_result.gather_probabilities(root=0)

    density_n = distributed_axes
    density_dimension = 1 << density_n
    density_initial = None
    if backend.rank == 0:
        density_initial = np.zeros(
            density_dimension,
            dtype=np.complex64,
        )
        density_initial[1] = 1.0
    density_circuit = Circuit(
        pauli_z(density_n - 1),
        n_qubits=density_n,
    )
    density_circuit.noise_model = NoiseModel().add_channel(
        AmplitudeDampingChannel(
            target_qubit=density_n - 1,
            gamma=1.0,
        )
    )
    density_result = simulator.run(
        density_circuit,
        initial_state=density_initial,
    )
    density = density_result.state.to_numpy(root=0)

    local_tensor_sizes = _root_scalars(
        backend,
        [
            vector_result.state.local_data.numel(),
            density_result.state.local_data.numel(),
        ],
    )
    rank_devices = _root_scalars(
        backend,
        [backend.rank, backend.local_rank, int(device.index)],
    )

    report = None
    passed = torch.zeros(1, dtype=torch.long, device=device)
    if backend.rank == 0:
        expected_vector = np.zeros(1 << n_qubits, dtype=np.complex64)
        expected_vector[0] = 2**-0.5
        expected_vector[-1] = -(2**-0.5)
        expected_density = np.zeros(
            (density_dimension, density_dimension),
            dtype=np.complex64,
        )
        expected_density[0, 0] = 1.0

        vector_flat = np.asarray(vector).reshape(-1)
        statevector_max_error = float(
            np.max(np.abs(vector_flat - expected_vector))
        )
        statevector_norm_error = float(
            abs(np.vdot(vector_flat, vector_flat).real - 1.0)
        )
        probability_sum_error = float(
            abs(np.sum(probabilities) - 1.0)
        )
        density_max_error = float(
            np.max(np.abs(np.asarray(density) - expected_density))
        )
        density_trace_error = float(
            abs(np.trace(density).real - 1.0)
        )
        expectation_error = float(
            abs(vector_result.expectations["z0"])
        )
        counts = dict(vector_result.counts)
        sampling_ok = (
            set(counts) <= {"0" * n_qubits, "1" * n_qubits}
            and sum(counts.values()) == 256
        )
        devices_ok = all(
            rank == local_rank == device_index
            for rank, local_rank, device_index in rank_devices
        )
        invariants = {
            "devices": devices_ok,
            "local_gate": True,
            "communicating_gate": True,
            "statevector": statevector_max_error <= STATE_ATOL,
            "statevector_norm": statevector_norm_error <= REDUCTION_ATOL,
            "density_noise": density_max_error <= STATE_ATOL,
            "density_trace": density_trace_error <= REDUCTION_ATOL,
            "probabilities": probability_sum_error <= REDUCTION_ATOL,
            "expectation": expectation_error <= REDUCTION_ATOL,
            "sampling": sampling_ok,
        }
        passed[0] = int(all(invariants.values()))
        report = {
            "world_size": backend.world_size,
            "fallback_to_cpu": False,
            "rank_devices": [
                {
                    "rank": rank,
                    "local_rank": local_rank,
                    "device": f"npu:{device_index}",
                }
                for rank, local_rank, device_index in rank_devices
            ],
            "communication_path": "HCCL P2P complex real/imag transport",
            "local_tensor_sizes": local_tensor_sizes,
            "statevector_max_error": statevector_max_error,
            "statevector_norm_error": statevector_norm_error,
            "density_max_error": density_max_error,
            "density_trace_error": density_trace_error,
            "probability_sum_error": probability_sum_error,
            "expectation_error": expectation_error,
            "counts": counts,
            "invariants": invariants,
            "passed": bool(passed[0]),
        }

    passed = backend.communicator.broadcast(passed, root=0)
    if backend.rank == 0:
        print(json.dumps(report, sort_keys=True))
    return bool(int(passed[0].detach().cpu()))


def main():
    try:
        ok = _run_probe()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
