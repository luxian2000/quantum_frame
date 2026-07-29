"""Strict full-API acceptance probe for 2 or 4 Ascend NPUs.

Run from repository root:

    source /usr/local/Ascend/cann/set_env.sh
    PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=2 scripts/npu/distributed_api_probe.py --section all
    PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=4 scripts/npu/distributed_api_probe.py --section all
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import torch

from aicir import (
    AmplitudeDampingChannel,
    BitFlipChannel,
    Circuit,
    DepolarizingChannel,
    Hamiltonian,
    NoiseModel,
    NumpyBackend,
    Observable,
    PauliString,
    PhaseFlipChannel,
    cx,
    hadamard,
    pauli_x,
    pauli_z,
    rz,
)
from aicir.distributed import DistSimulator


STATE_ATOL = 1e-6
REDUCTION_ATOL = 1e-5
EXPECTED_SECTIONS = (
    "state",
    "layout",
    "continuation",
    "noise",
    "observable",
    "measure",
    "result",
    "communication",
    "contract",
)


def _section(passed, *, metrics=None, status=None):
    return {
        "passed": bool(passed),
        "status": status or ("PASS" if passed else "FAIL"),
        "metrics": dict(metrics or {}),
    }


def _root_only(backend, factory):
    return factory() if backend.rank == 0 else None


def _gather_state_array(state):
    return state.to_numpy(root=0)


def _max_error(actual, expected):
    return float(
        np.max(
            np.abs(
                np.asarray(actual) - np.asarray(expected)
            )
        )
    )


def _apply_kraus_reference(rho, channel, n):
    backend = NumpyBackend()
    reference = np.asarray(rho, dtype=np.complex64)
    accumulated = np.zeros_like(reference, dtype=np.complex64)
    for operator in channel.kraus_operators(n, backend):
        kraus = np.asarray(operator, dtype=np.complex64)
        accumulated = (
            accumulated
            + kraus @ reference @ np.conjugate(kraus.T)
        ).astype(np.complex64)
    return accumulated


def _gather_long_evidence(backend, values):
    evidence = torch.tensor(
        tuple(int(value) for value in values),
        dtype=torch.long,
        device=backend._device,
    )
    gathered = backend.communicator.gather_to_root(evidence, root=0)
    if backend.rank != 0:
        return None
    return [
        [int(value) for value in item.detach().cpu().tolist()]
        for item in gathered
    ]


def _sync_section(backend, root_passed, metrics):
    passed = torch.zeros(1, dtype=torch.long, device=backend._device)
    if backend.rank == 0:
        if root_passed is None:
            raise ValueError("rank 0 必须给出 section 判定")
        passed[0] = int(bool(root_passed))
    passed = backend.communicator.broadcast(passed, root=0)
    return _section(
        bool(int(passed[0].detach().cpu())),
        metrics=metrics if backend.rank == 0 else {},
    )


def _evaluate_root_section(backend, evaluator):
    root_passed = False
    root_metrics = {}
    if backend.rank == 0:
        try:
            root_passed, root_metrics = evaluator()
            root_passed = bool(root_passed)
            root_metrics = dict(root_metrics)
        except Exception as error:  # noqa: BLE001
            root_metrics = {
                "root_evaluation_error": {
                    "type": type(error).__name__,
                    "message": str(error),
                }
            }
    return _sync_section(backend, root_passed, root_metrics)


def _normalized_probe_vector(dimension):
    vector = np.zeros(int(dimension), dtype=np.complex64)
    vector[0] = 1.0 + 0.5j
    vector[1] = -0.75 + 0.25j
    vector[-1] = 0.5 - 1.0j
    vector /= np.linalg.norm(vector)
    return vector


def _run_state_section(simulator):
    backend = simulator.backend
    distributed_axes = int(math.log2(backend.world_size))
    n_qubits = distributed_axes + 1
    dimension = 1 << n_qubits
    empty = Circuit(n_qubits=n_qubits)

    zero_result = simulator.run(empty)
    zero_state = _gather_state_array(zero_result.state)
    zero_probabilities = zero_result.gather_probabilities(root=0)

    initial_vector = _root_only(
        backend,
        lambda: _normalized_probe_vector(dimension),
    )
    vector_result = simulator.run(
        empty,
        initial_state=initial_vector,
    )
    vector = _gather_state_array(vector_result.state)
    vector_probabilities = vector_result.gather_probabilities(root=0)

    initial_density_matrix = _root_only(
        backend,
        lambda: np.outer(
            initial_vector,
            np.conjugate(initial_vector),
        ).astype(np.complex64),
    )
    density_result = simulator.run(
        empty,
        initial_density_matrix=initial_density_matrix,
    )
    density = _gather_state_array(density_result.state)
    density_probabilities = density_result.gather_probabilities(root=0)

    evidence = _gather_long_evidence(
        backend,
        (
            backend.rank,
            zero_result.state.local_data.numel(),
            vector_result.state.local_data.numel(),
            density_result.state.local_data.numel(),
            zero_result.state.kind == "vector",
            vector_result.state.kind == "vector",
            density_result.state.kind == "matrix",
        ),
    )

    def evaluate():
        expected_zero_state = np.zeros(dimension, dtype=np.complex64)
        expected_zero_state[0] = 1.0
        expected_zero_probabilities = np.zeros(
            dimension,
            dtype=np.float64,
        )
        expected_zero_probabilities[0] = 1.0
        zero_state_max_error = _max_error(
            zero_state,
            expected_zero_state,
        )
        zero_state_norm_error = float(
            abs(np.vdot(zero_state, zero_state).real - 1.0)
        )
        zero_probability_max_error = _max_error(
            zero_probabilities,
            expected_zero_probabilities,
        )
        zero_probability_sum_error = float(
            abs(np.sum(zero_probabilities) - 1.0)
        )
        statevector_max_error = _max_error(vector, initial_vector)
        density_max_error = _max_error(
            density,
            initial_density_matrix,
        )
        statevector_norm_error = float(
            abs(np.vdot(vector, vector).real - 1.0)
        )
        density_trace_error = float(abs(np.trace(density) - 1.0))
        vector_probability_sum_error = float(
            abs(np.sum(vector_probabilities) - 1.0)
        )
        density_probability_sum_error = float(
            abs(np.sum(density_probabilities) - 1.0)
        )
        expected_vector_size = dimension // backend.world_size
        expected_density_size = dimension * dimension // backend.world_size
        local_sizes_ok = all(
            (
                rank == index
                and zero_size == expected_vector_size
                and vector_size == expected_vector_size
                and density_size == expected_density_size
                and zero_kind == 1
                and vector_kind == 1
                and density_kind == 1
            )
            for index, (
                rank,
                zero_size,
                vector_size,
                density_size,
                zero_kind,
                vector_kind,
                density_kind,
            ) in enumerate(evidence)
        )
        passed = (
            zero_state_max_error <= STATE_ATOL
            and zero_state_norm_error <= REDUCTION_ATOL
            and zero_probability_max_error <= STATE_ATOL
            and zero_probability_sum_error <= REDUCTION_ATOL
            and statevector_max_error <= STATE_ATOL
            and density_max_error <= STATE_ATOL
            and statevector_norm_error <= REDUCTION_ATOL
            and density_trace_error <= REDUCTION_ATOL
            and vector_probability_sum_error <= REDUCTION_ATOL
            and density_probability_sum_error <= REDUCTION_ATOL
            and local_sizes_ok
        )
        metrics = {
            "zero_state_max_error": zero_state_max_error,
            "zero_state_norm_error": zero_state_norm_error,
            "zero_probability_max_error": zero_probability_max_error,
            "zero_probability_sum_error": zero_probability_sum_error,
            "statevector_max_error": statevector_max_error,
            "density_max_error": density_max_error,
            "statevector_norm_error": statevector_norm_error,
            "density_trace_error": density_trace_error,
            "vector_probability_sum_error": vector_probability_sum_error,
            "density_probability_sum_error": density_probability_sum_error,
            "local_tensor_sizes": [
                {
                    "rank": rank,
                    "zero_vector": zero_size,
                    "vector": vector_size,
                    "density": density_size,
                }
                for (
                    rank,
                    zero_size,
                    vector_size,
                    density_size,
                    _zero_kind,
                    _vector_kind,
                    _density_kind,
                ) in evidence
            ],
        }
        return passed, metrics

    return _evaluate_root_section(backend, evaluate)


def _layout_probe_circuit(n_qubits):
    last = n_qubits - 1
    return Circuit(
        hadamard(0),
        cx(target_qubit=1, control_qubits=(0,)),
        hadamard(last),
        cx(target_qubit=0, control_qubits=(last,)),
        n_qubits=n_qubits,
    )


def _numpy_apply_single_qubit(state, matrix, target, n_qubits):
    tensor = np.asarray(state).reshape((2,) * n_qubits)
    moved = np.moveaxis(tensor, target, 0)
    transformed = np.tensordot(matrix, moved, axes=(1, 0))
    return np.moveaxis(transformed, 0, target).reshape(-1)


def _numpy_apply_cx(state, control, target, n_qubits):
    transformed = np.zeros_like(state)
    control_mask = 1 << (n_qubits - 1 - control)
    target_mask = 1 << (n_qubits - 1 - target)
    for index, amplitude in enumerate(state):
        destination = (
            index ^ target_mask
            if index & control_mask
            else index
        )
        transformed[destination] = amplitude
    return transformed


def _numpy_layout_reference(n_qubits):
    dimension = 1 << n_qubits
    state = np.zeros(dimension, dtype=np.complex64)
    state[0] = 1.0
    hadamard_matrix = np.array(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=np.complex64,
    ) / np.sqrt(np.float32(2.0))
    last = n_qubits - 1
    state = _numpy_apply_single_qubit(
        state,
        hadamard_matrix,
        0,
        n_qubits,
    )
    state = _numpy_apply_cx(state, 0, 1, n_qubits)
    state = _numpy_apply_single_qubit(
        state,
        hadamard_matrix,
        last,
        n_qubits,
    )
    state = _numpy_apply_cx(state, last, 0, n_qubits)
    probabilities = np.abs(state) ** 2
    z_signs = np.array(
        [
            -1.0 if index.bit_count() % 2 else 1.0
            for index in range(dimension)
        ],
        dtype=np.float64,
    )
    expectation = float(np.sum(probabilities * z_signs))
    return state, probabilities, expectation


def _run_layout_section(simulator):
    backend = simulator.backend
    distributed_axes = int(math.log2(backend.world_size))
    n_qubits = distributed_axes + 1
    dimension = 1 << n_qubits
    logical_to_storage = tuple(list(range(1, n_qubits)) + [0])
    circuit = _layout_probe_circuit(n_qubits)
    observable = PauliString("Z" * n_qubits, n_qubits=n_qubits)

    auto_result = simulator.run(
        circuit,
        observables={"pauli": observable},
    )
    explicit_result = simulator.run(
        circuit,
        observables={"pauli": observable},
        layout=logical_to_storage,
    )
    auto_state = _gather_state_array(auto_result.state)
    explicit_state = _gather_state_array(explicit_result.state)
    auto_probabilities = auto_result.gather_probabilities(root=0)
    explicit_probabilities = explicit_result.gather_probabilities(root=0)
    evidence = _gather_long_evidence(
        backend,
        (
            backend.rank,
            auto_result.state.local_data.numel(),
            explicit_result.state.local_data.numel(),
            explicit_result.state.kind == "vector",
        ),
    )

    def evaluate():
        expected_state, expected_probabilities, expected_expectation = (
            _numpy_layout_reference(n_qubits)
        )
        auto_layout = auto_result.state.layout.logical_to_storage
        statevector_error = _max_error(auto_state, explicit_state)
        probability_error = _max_error(
            auto_probabilities,
            explicit_probabilities,
        )
        expectation_error = float(
            abs(
                auto_result.expectations["pauli"]
                - explicit_result.expectations["pauli"]
            )
        )
        auto_reference_statevector_error = _max_error(
            auto_state,
            expected_state,
        )
        explicit_reference_statevector_error = _max_error(
            explicit_state,
            expected_state,
        )
        auto_reference_probability_error = _max_error(
            auto_probabilities,
            expected_probabilities,
        )
        explicit_reference_probability_error = _max_error(
            explicit_probabilities,
            expected_probabilities,
        )
        auto_reference_expectation_error = float(
            abs(
                auto_result.expectations["pauli"]
                - expected_expectation
            )
        )
        explicit_reference_expectation_error = float(
            abs(
                explicit_result.expectations["pauli"]
                - expected_expectation
            )
        )
        expected_local_size = dimension // backend.world_size
        local_sizes_ok = all(
            (
                rank == index
                and auto_size == expected_local_size
                and explicit_size == expected_local_size
                and vector_kind == 1
            )
            for index, (
                rank,
                auto_size,
                explicit_size,
                vector_kind,
            ) in enumerate(evidence)
        )
        layout_ok = (
            explicit_result.state.layout.logical_to_storage
            == logical_to_storage
            and auto_layout != logical_to_storage
        )
        passed = (
            layout_ok
            and local_sizes_ok
            and statevector_error <= STATE_ATOL
            and probability_error <= STATE_ATOL
            and expectation_error <= REDUCTION_ATOL
            and auto_reference_statevector_error <= STATE_ATOL
            and explicit_reference_statevector_error <= STATE_ATOL
            and auto_reference_probability_error <= STATE_ATOL
            and explicit_reference_probability_error <= STATE_ATOL
            and auto_reference_expectation_error <= REDUCTION_ATOL
            and explicit_reference_expectation_error <= REDUCTION_ATOL
        )
        metrics = {
            "auto_layout": list(auto_layout),
            "logical_to_storage": list(logical_to_storage),
            "statevector_error": statevector_error,
            "probability_error": probability_error,
            "expectation_error": expectation_error,
            "auto_reference_statevector_error": (
                auto_reference_statevector_error
            ),
            "explicit_reference_statevector_error": (
                explicit_reference_statevector_error
            ),
            "auto_reference_probability_error": (
                auto_reference_probability_error
            ),
            "explicit_reference_probability_error": (
                explicit_reference_probability_error
            ),
            "auto_reference_expectation_error": (
                auto_reference_expectation_error
            ),
            "explicit_reference_expectation_error": (
                explicit_reference_expectation_error
            ),
        }
        return passed, metrics

    return _evaluate_root_section(backend, evaluate)


def _run_continuation_section(simulator):
    backend = simulator.backend
    distributed_axes = int(math.log2(backend.world_size))
    n_qubits = distributed_axes + 1
    dimension = 1 << n_qubits
    logical_to_storage = tuple(list(range(1, n_qubits)) + [0])
    prefix = Circuit(
        hadamard(0),
        cx(target_qubit=1, control_qubits=(0,)),
        n_qubits=n_qubits,
    )
    suffix = Circuit(
        pauli_z(n_qubits - 1),
        pauli_x(0),
        n_qubits=n_qubits,
    )
    combined = Circuit(
        hadamard(0),
        cx(target_qubit=1, control_qubits=(0,)),
        pauli_z(n_qubits - 1),
        pauli_x(0),
        n_qubits=n_qubits,
    )

    vector_prefix = simulator.run(
        prefix,
        layout=logical_to_storage,
    )
    vector_continued = simulator.run(
        suffix,
        initial_state=vector_prefix.state,
        layout=logical_to_storage,
    )
    vector_combined = simulator.run(
        combined,
        layout=logical_to_storage,
    )
    continued_vector = _gather_state_array(vector_continued.state)
    combined_vector = _gather_state_array(vector_combined.state)

    initial_density_matrix = _root_only(
        backend,
        lambda: np.diag(
            np.array(
                [1.0] + [0.0] * (dimension - 1),
                dtype=np.complex64,
            )
        ),
    )
    density_prefix = simulator.run(
        prefix,
        initial_density_matrix=initial_density_matrix,
        layout=logical_to_storage,
    )
    density_continued = simulator.run(
        suffix,
        initial_density_matrix=density_prefix.state,
        layout=logical_to_storage,
    )
    density_combined = simulator.run(
        combined,
        initial_density_matrix=initial_density_matrix,
        layout=logical_to_storage,
    )
    continued_density = _gather_state_array(density_continued.state)
    combined_density = _gather_state_array(density_combined.state)
    evidence = _gather_long_evidence(
        backend,
        (
            backend.rank,
            vector_prefix.state.local_data.numel(),
            vector_continued.state.local_data.numel(),
            vector_combined.state.local_data.numel(),
            density_prefix.state.local_data.numel(),
            density_continued.state.local_data.numel(),
            density_combined.state.local_data.numel(),
        ),
    )

    def evaluate():
        continuation_vector_error = _max_error(
            continued_vector,
            combined_vector,
        )
        continuation_density_error = _max_error(
            continued_density,
            combined_density,
        )
        expected_vector_size = dimension // backend.world_size
        expected_density_size = dimension * dimension // backend.world_size
        local_sizes_ok = all(
            (
                row[0] == index
                and all(
                    size == expected_vector_size
                    for size in row[1:4]
                )
                and all(
                    size == expected_density_size
                    for size in row[4:]
                )
            )
            for index, row in enumerate(evidence)
        )
        layout_ok = all(
            result.state.layout.logical_to_storage == logical_to_storage
            for result in (
                vector_prefix,
                vector_continued,
                vector_combined,
                density_prefix,
                density_continued,
                density_combined,
            )
        )
        passed = (
            layout_ok
            and local_sizes_ok
            and continuation_vector_error <= STATE_ATOL
            and continuation_density_error <= STATE_ATOL
        )
        metrics = {
            "continuation_vector_error": continuation_vector_error,
            "continuation_density_error": continuation_density_error,
            "logical_to_storage": list(logical_to_storage),
            "local_tensor_sizes": [
                {
                    "rank": row[0],
                    "vector_prefix": row[1],
                    "vector_continued": row[2],
                    "vector_combined": row[3],
                    "density_prefix": row[4],
                    "density_continued": row[5],
                    "density_combined": row[6],
                }
                for row in evidence
            ],
        }
        return passed, metrics

    return _evaluate_root_section(backend, evaluate)


def _coherent_density_matrix(n_qubits):
    vector = np.zeros(1 << n_qubits, dtype=np.complex64)
    vector[0] = np.float32(1.0 / np.sqrt(2.0))
    vector[1 << (n_qubits - 1)] = np.float32(1.0 / np.sqrt(2.0))
    return np.outer(vector, np.conjugate(vector)).astype(np.complex64)


def _run_noise_section(simulator):
    backend = simulator.backend
    distributed_axes = int(math.log2(backend.world_size))
    n_qubits = distributed_axes + 1
    channels = (
        AmplitudeDampingChannel(target_qubit=0, gamma=0.1),
        BitFlipChannel(target_qubit=0, p=0.2),
        PhaseFlipChannel(target_qubit=0, p=0.3),
        DepolarizingChannel(target_qubit=0, p=0.15),
    )
    noise_model = NoiseModel()
    for channel in channels:
        noise_model.add_channel(
            channel,
            after_gates=("pauli_x",),
        )
    circuit = Circuit(pauli_x(0), n_qubits=n_qubits)
    circuit.noise_model = noise_model
    initial_density_matrix = _root_only(
        backend,
        lambda: _coherent_density_matrix(n_qubits),
    )

    result = simulator.run(
        circuit,
        initial_density_matrix=initial_density_matrix,
    )
    density = _gather_state_array(result.state)
    probabilities = result.gather_probabilities(root=0)

    def evaluate():
        reference_backend = NumpyBackend()
        unitary = np.asarray(
            circuit.unitary(backend=reference_backend),
            dtype=np.complex64,
        )
        expected = (
            unitary
            @ initial_density_matrix
            @ np.conjugate(unitary.T)
        ).astype(np.complex64)
        for channel in channels:
            expected = _apply_kraus_reference(
                expected,
                channel,
                n_qubits,
            )
        expected_probabilities = np.real(np.diag(expected))
        noise_density_error = _max_error(density, expected)
        noise_trace_error = float(abs(np.trace(density) - 1.0))
        noise_probability_error = _max_error(
            probabilities,
            expected_probabilities,
        )
        metrics = {
            "noise_density_error": noise_density_error,
            "noise_trace_error": noise_trace_error,
            "noise_probability_error": noise_probability_error,
            "channel_count": len(channels),
            "matched_gate_name": "pauli_x",
        }
        passed = (
            noise_density_error <= STATE_ATOL
            and noise_trace_error <= REDUCTION_ATOL
            and noise_probability_error <= REDUCTION_ATOL
            and len(channels) == 4
        )
        return passed, metrics

    return _evaluate_root_section(backend, evaluate)


def _observable_probe_circuit(n_qubits):
    last = n_qubits - 1
    return Circuit(
        hadamard(0),
        rz(0.37, target_qubit=0),
        hadamard(last),
        cx(target_qubit=last, control_qubits=(0,)),
        rz(-0.23, target_qubit=last),
        n_qubits=n_qubits,
    )


def _run_observable_section(simulator):
    backend = simulator.backend
    distributed_axes = int(math.log2(backend.world_size))
    n_qubits = distributed_axes + 1
    last = n_qubits - 1
    circuit = _observable_probe_circuit(n_qubits)
    pauli = PauliString(
        "XZ",
        n_qubits=n_qubits,
        qubits=(0, last),
    )
    hamiltonian = Hamiltonian(
        n_qubits=n_qubits,
        terms=(
            ("Z", 0.45, (0,)),
            ("XX", -0.3, (0, last)),
            ("Z", 0.2, (last,)),
        ),
    )
    local_dense_matrix = np.array(
        [
            [0.2, 0.7 - 0.1j],
            [0.7 + 0.1j, -0.3],
        ],
        dtype=np.complex64,
    )
    local_dense = Observable.matrix(
        local_dense_matrix,
        metadata={"qubits": [last]},
    )
    result = simulator.run(
        circuit,
        observables={
            "pauli": pauli,
            "hamiltonian": hamiltonian,
            "local_dense": local_dense,
        },
    )

    def evaluate():
        reference_backend = NumpyBackend()
        unitary = np.asarray(
            circuit.unitary(backend=reference_backend),
            dtype=np.complex64,
        )
        initial = np.zeros(1 << n_qubits, dtype=np.complex64)
        initial[0] = 1.0
        state = unitary @ initial
        matrices = {
            "pauli": np.asarray(
                pauli.to_matrix(reference_backend),
                dtype=np.complex64,
            ),
            "hamiltonian": np.asarray(
                hamiltonian.to_matrix(reference_backend),
                dtype=np.complex64,
            ),
        }
        local_dense_full = np.eye(1, dtype=np.complex64)
        for qubit in range(n_qubits):
            factor = (
                local_dense_matrix
                if qubit == last
                else np.eye(2, dtype=np.complex64)
            )
            local_dense_full = np.kron(
                local_dense_full,
                factor,
            ).astype(np.complex64)
        matrices["local_dense"] = local_dense_full
        expected = {
            name: np.vdot(state, matrix @ state)
            for name, matrix in matrices.items()
        }
        pauli_error = float(
            abs(result.expectations["pauli"] - expected["pauli"])
        )
        hamiltonian_error = float(
            abs(
                result.expectations["hamiltonian"]
                - expected["hamiltonian"]
            )
        )
        local_dense_error = float(
            abs(
                result.expectations["local_dense"]
                - expected["local_dense"]
            )
        )
        metrics = {
            "pauli_error": pauli_error,
            "hamiltonian_error": hamiltonian_error,
            "local_dense_error": local_dense_error,
        }
        passed = (
            pauli_error <= REDUCTION_ATOL
            and hamiltonian_error <= REDUCTION_ATOL
            and local_dense_error <= REDUCTION_ATOL
        )
        return passed, metrics

    return _evaluate_root_section(backend, evaluate)


def _validate_runtime(backend):
    device = backend._device
    if backend.world_size not in {2, 4}:
        raise ValueError("完整 API 探针只接受 world_size=2 或 world_size=4")
    if device.type != "npu":
        raise RuntimeError(f"严格探针要求 NPU，实际设备为 {device}")
    if backend.rank != backend.local_rank or device.index != backend.local_rank:
        raise RuntimeError("rank、local_rank 与 NPU device 不一致")


def _pending_section(_simulator):
    return _section(False, status="NOT_IMPLEMENTED")


SECTION_RUNNERS = {
    "state": _run_state_section,
    "layout": _run_layout_section,
    "continuation": _run_continuation_section,
    "noise": _run_noise_section,
    "observable": _run_observable_section,
    "measure": _pending_section,
    "result": _pending_section,
    "communication": _pending_section,
    "contract": _pending_section,
}


def _run_selected(simulator, selected):
    names = EXPECTED_SECTIONS if selected == "all" else (selected,)
    sections = {}
    for name in names:
        sections[name] = SECTION_RUNNERS[name](simulator)
    failed = [
        name for name, result in sections.items()
        if not result["passed"]
    ]
    return sections, failed


def _run_probe(selected):
    simulator = DistSimulator.from_env(fallback_to_cpu=False)
    backend = simulator.backend
    _validate_runtime(backend)
    sections, failed = _run_selected(simulator, selected)

    passed_count = torch.tensor(
        [int(not failed)],
        dtype=torch.long,
        device=backend._device,
    )
    passed_count = backend.communicator.all_reduce_sum(passed_count)
    passed = int(passed_count[0].detach().cpu()) == backend.world_size

    report = None
    if backend.rank == 0:
        report = {
            "passed": passed,
            "world_size": backend.world_size,
            "fallback_to_cpu": False,
            "sections": sections,
            "failed_invariants": failed,
        }
        print(json.dumps(report, sort_keys=True))
    return passed


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section",
        choices=("all", *EXPECTED_SECTIONS),
        default="all",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    try:
        ok = _run_probe(args.section)
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
