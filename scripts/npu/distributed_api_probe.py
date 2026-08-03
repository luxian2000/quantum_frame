"""Strict full-API acceptance probe for 2 or 4 Ascend NPUs.

Run from repository root:

    source /usr/local/Ascend/cann/set_env.sh
    PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=2 scripts/npu/distributed_api_probe.py --section all
    PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=4 scripts/npu/distributed_api_probe.py --section all
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import sys

import numpy as np
import torch

from aicir import (
    AmplitudeDampingChannel,
    BitFlipChannel,
    Circuit,
    ClassicalRegister,
    DepolarizingChannel,
    Hamiltonian,
    NoiseModel,
    NumpyBackend,
    Observable,
    PauliString,
    PhaseFlipChannel,
    cx,
    hadamard,
    if_,
    measure,
    pauli_x,
    pauli_z,
    reset,
    rx,
    rz,
    while_,
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


def _embed_single_qubit_reference(operator, target, n):
    identity = np.eye(2, dtype=np.complex64)
    embedded = np.eye(1, dtype=np.complex64)
    for qubit in range(n):
        factor = operator if qubit == target else identity
        embedded = np.kron(embedded, factor).astype(np.complex64)
    return embedded


def _single_qubit_kraus_reference(channel_name, parameter):
    identity = np.eye(2, dtype=np.complex64)
    pauli_x_matrix = np.array(
        [[0.0, 1.0], [1.0, 0.0]],
        dtype=np.complex64,
    )
    pauli_y_matrix = np.array(
        [[0.0, -1.0j], [1.0j, 0.0]],
        dtype=np.complex64,
    )
    pauli_z_matrix = np.array(
        [[1.0, 0.0], [0.0, -1.0]],
        dtype=np.complex64,
    )
    probability = np.float32(parameter)
    if channel_name == "amplitude_damping":
        return (
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, np.sqrt(np.float32(1.0) - probability)],
                ],
                dtype=np.complex64,
            ),
            np.array(
                [
                    [0.0, np.sqrt(probability)],
                    [0.0, 0.0],
                ],
                dtype=np.complex64,
            ),
        )
    if channel_name == "bit_flip":
        return (
            np.sqrt(np.float32(1.0) - probability) * identity,
            np.sqrt(probability) * pauli_x_matrix,
        )
    if channel_name == "phase_flip":
        return (
            np.sqrt(np.float32(1.0) - probability) * identity,
            np.sqrt(probability) * pauli_z_matrix,
        )
    if channel_name == "depolarizing":
        return (
            np.sqrt(np.float32(1.0) - probability) * identity,
            np.sqrt(probability / np.float32(3.0)) * pauli_x_matrix,
            np.sqrt(probability / np.float32(3.0)) * pauli_y_matrix,
            np.sqrt(probability / np.float32(3.0)) * pauli_z_matrix,
        )
    raise ValueError(f"未知 reference channel: {channel_name}")


def _apply_kraus_reference(rho, local_operators, target, n):
    reference = np.asarray(rho, dtype=np.complex64)
    accumulated = np.zeros_like(reference, dtype=np.complex64)
    for local_operator in local_operators:
        kraus = _embed_single_qubit_reference(
            local_operator,
            target,
            n,
        )
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


def _coherent_density_matrix(n_qubits, distributed_axes):
    coherent = np.array(
        [
            np.sqrt(np.float32(0.7)),
            np.sqrt(np.float32(0.3)) * (0.8 + 0.6j),
        ],
        dtype=np.complex64,
    )
    zero = np.array([1.0, 0.0], dtype=np.complex64)
    vector = np.eye(1, dtype=np.complex64).reshape(-1)
    for qubit in range(n_qubits):
        factor = coherent if qubit < distributed_axes else zero
        vector = np.kron(vector, factor).astype(np.complex64)
    return np.outer(vector, np.conjugate(vector)).astype(np.complex64)


def _run_noise_section(simulator):
    backend = simulator.backend
    distributed_axes = int(math.log2(backend.world_size))
    n_qubits = distributed_axes + 1
    logical_to_storage = tuple(range(n_qubits))
    channel_parameters = (
        ("amplitude_damping", 0.1),
        ("bit_flip", 0.2),
        ("phase_flip", 0.3),
        ("depolarizing", 0.15),
    )
    channel_targets = tuple(
        index % distributed_axes
        for index in range(len(channel_parameters))
    )
    channels = (
        AmplitudeDampingChannel(
            target_qubit=channel_targets[0],
            gamma=channel_parameters[0][1],
        ),
        BitFlipChannel(
            target_qubit=channel_targets[1],
            p=channel_parameters[1][1],
        ),
        PhaseFlipChannel(
            target_qubit=channel_targets[2],
            p=channel_parameters[2][1],
        ),
        DepolarizingChannel(
            target_qubit=channel_targets[3],
            p=channel_parameters[3][1],
        ),
    )
    references = tuple(
        _single_qubit_kraus_reference(name, parameter)
        for name, parameter in channel_parameters
    )
    initial_density_matrix = _root_only(
        backend,
        lambda: _coherent_density_matrix(
            n_qubits,
            distributed_axes,
        ),
    )

    single_channel_densities = []
    for channel in channels:
        single_model = NoiseModel().add_channel(
            channel,
            after_gates=("pauli_x",),
        )
        single_circuit = Circuit(pauli_x(0), n_qubits=n_qubits)
        single_circuit.noise_model = single_model
        single_result = simulator.run(
            single_circuit,
            initial_density_matrix=initial_density_matrix,
            layout=logical_to_storage,
        )
        single_channel_densities.append(
            _gather_state_array(single_result.state)
        )

    sequence_model = NoiseModel()
    for channel in channels:
        sequence_model.add_channel(
            channel,
            after_gates=("pauli_x",),
        )
    sequence_circuit = Circuit(pauli_x(0), n_qubits=n_qubits)
    sequence_circuit.noise_model = sequence_model
    sequence_result = simulator.run(
        sequence_circuit,
        initial_density_matrix=initial_density_matrix,
        layout=logical_to_storage,
    )
    sequence_density = _gather_state_array(sequence_result.state)
    sequence_probabilities = sequence_result.gather_probabilities(root=0)

    selection_circuit = Circuit(pauli_z(0), n_qubits=n_qubits)
    selection_circuit.noise_model = NoiseModel().add_channel(
        AmplitudeDampingChannel(target_qubit=0, gamma=0.41),
        after_gates=("pauli_x",),
    )
    selection_result = simulator.run(
        selection_circuit,
        initial_density_matrix=initial_density_matrix,
        layout=logical_to_storage,
    )
    selection_density = _gather_state_array(selection_result.state)

    def evaluate():
        reference_backend = NumpyBackend()
        x_unitary = np.asarray(
            Circuit(
                pauli_x(0),
                n_qubits=n_qubits,
            ).unitary(backend=reference_backend),
            dtype=np.complex64,
        )
        after_x = (
            x_unitary
            @ initial_density_matrix
            @ np.conjugate(x_unitary.T)
        ).astype(np.complex64)
        single_channel_errors = {}
        for (
            (channel_name, _parameter),
            target,
            local_operators,
            actual_density,
        ) in zip(
            channel_parameters,
            channel_targets,
            references,
            single_channel_densities,
        ):
            expected_single = _apply_kraus_reference(
                after_x,
                local_operators,
                target,
                n_qubits,
            )
            single_channel_errors[channel_name] = _max_error(
                actual_density,
                expected_single,
            )

        expected_sequence = after_x
        for local_operators, target in zip(
            references,
            channel_targets,
        ):
            expected_sequence = _apply_kraus_reference(
                expected_sequence,
                local_operators,
                target,
                n_qubits,
            )
        expected_probabilities = np.real(np.diag(expected_sequence))
        noise_sequence_error = _max_error(
            sequence_density,
            expected_sequence,
        )
        noise_trace_error = float(
            abs(np.trace(sequence_density) - 1.0)
        )
        noise_probability_error = _max_error(
            sequence_probabilities,
            expected_probabilities,
        )
        z_unitary = np.asarray(
            Circuit(
                pauli_z(0),
                n_qubits=n_qubits,
            ).unitary(backend=reference_backend),
            dtype=np.complex64,
        )
        expected_selection = (
            z_unitary
            @ initial_density_matrix
            @ np.conjugate(z_unitary.T)
        ).astype(np.complex64)
        rule_selection_error = _max_error(
            selection_density,
            expected_selection,
        )
        metrics = {
            "amplitude_damping_error": single_channel_errors[
                "amplitude_damping"
            ],
            "bit_flip_error": single_channel_errors["bit_flip"],
            "phase_flip_error": single_channel_errors["phase_flip"],
            "depolarizing_error": single_channel_errors[
                "depolarizing"
            ],
            "noise_density_error": noise_sequence_error,
            "noise_sequence_error": noise_sequence_error,
            "noise_trace_error": noise_trace_error,
            "noise_probability_error": noise_probability_error,
            "rule_selection_error": rule_selection_error,
            "channel_count": len(channels),
            "channel_targets": {
                name: target
                for (name, _parameter), target in zip(
                    channel_parameters,
                    channel_targets,
                )
            },
            "targeted_distributed_axes": sorted(set(channel_targets)),
            "logical_to_storage": list(
                sequence_result.state.layout.logical_to_storage
            ),
        }
        passed = (
            all(
                error <= STATE_ATOL
                for error in single_channel_errors.values()
            )
            and noise_sequence_error <= STATE_ATOL
            and noise_trace_error <= REDUCTION_ATOL
            and noise_probability_error <= REDUCTION_ATOL
            and rule_selection_error <= STATE_ATOL
            and len(channels) == 4
            and sorted(set(channel_targets))
            == list(range(distributed_axes))
            and sequence_result.state.layout.logical_to_storage
            == logical_to_storage
        )
        return passed, metrics

    return _evaluate_root_section(backend, evaluate)


def _observable_probe_circuit(n_qubits, distributed_axes):
    last = n_qubits - 1
    gates = [
        hadamard(0),
        rz(0.37, target_qubit=0),
    ]
    for axis in range(1, distributed_axes):
        gates.extend(
            (
                hadamard(axis),
                rz(0.11 * (axis + 1), target_qubit=axis),
                cx(
                    target_qubit=axis,
                    control_qubits=(axis - 1,),
                ),
            )
        )
    gates.extend(
        (
            hadamard(last),
            cx(target_qubit=last, control_qubits=(0,)),
            rz(-0.23, target_qubit=last),
        )
    )
    return Circuit(
        *gates,
        n_qubits=n_qubits,
    )


def _run_observable_section(simulator):
    backend = simulator.backend
    distributed_axes = int(math.log2(backend.world_size))
    n_qubits = distributed_axes + 1
    logical_to_storage = tuple(range(n_qubits))
    targeted_distributed_axes = tuple(range(distributed_axes))
    local_dense_target = (
        targeted_distributed_axes[1]
        if distributed_axes > 1
        else targeted_distributed_axes[0]
    )
    circuit = _observable_probe_circuit(
        n_qubits,
        distributed_axes,
    )
    pauli = PauliString(
        "X",
        n_qubits=n_qubits,
        qubits=(0,),
    )
    hamiltonian = Hamiltonian(
        n_qubits=n_qubits,
        terms=[
            *(
                ("Z", 0.25 + 0.1 * axis, (axis,))
                for axis in targeted_distributed_axes
            ),
            ("X", -0.2, (0,)),
        ],
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
        metadata={"qubits": [local_dense_target]},
    )
    result = simulator.run(
        circuit,
        observables={
            "pauli": pauli,
            "hamiltonian": hamiltonian,
            "local_dense": local_dense,
        },
        layout=logical_to_storage,
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
                if qubit == local_dense_target
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
            "targeted_distributed_axes": list(
                targeted_distributed_axes
            ),
            "local_dense_target": local_dense_target,
            "logical_to_storage": list(
                result.state.layout.logical_to_storage
            ),
        }
        passed = (
            pauli_error <= REDUCTION_ATOL
            and hamiltonian_error <= REDUCTION_ATOL
            and local_dense_error <= REDUCTION_ATOL
            and result.state.layout.logical_to_storage
            == logical_to_storage
        )
        return passed, metrics

    return _evaluate_root_section(backend, evaluate)


def _ghz_probe_circuit(n_qubits):
    gates = [hadamard(0)]
    gates.extend(
        cx(target_qubit=target, control_qubits=(0,))
        for target in range(1, n_qubits)
    )
    return Circuit(*gates, n_qubits=n_qubits)


def _nonidentity_probe_layout(n_qubits):
    return tuple(list(range(1, n_qubits)) + [0])


def _run_measure_section(simulator):
    backend = simulator.backend
    distributed_axes = int(math.log2(backend.world_size))
    n_qubits = distributed_axes + 1
    logical_to_storage = _nonidentity_probe_layout(n_qubits)
    measure_qubits = (0,)
    circuit = _ghz_probe_circuit(n_qubits)

    full_result = simulator.run(
        circuit,
        shots=128,
        seed=2718,
        layout=logical_to_storage,
    )
    subset_result = simulator.run(
        circuit,
        shots=128,
        measure_qubits=measure_qubits,
        seed=3141,
        layout=logical_to_storage,
    )
    collapse_result = simulator.run(
        circuit,
        shots=1,
        measure_qubits=measure_qubits,
        collapse=True,
        seed=1618,
        layout=logical_to_storage,
    )

    # Recompute probabilities from the returned collapsed DistState.  The
    # collapse run's local_probabilities intentionally describe the state
    # before sampling, so they are not valid collapse evidence.
    collapsed_probabilities_result = simulator.run(
        Circuit(n_qubits=n_qubits),
        initial_state=collapse_result.state,
        layout=logical_to_storage,
        return_state=False,
        return_probabilities=True,
    )
    collapsed_probabilities = (
        collapsed_probabilities_result.gather_probabilities(root=0)
    )

    def evaluate():
        full_counts = dict(full_result.counts)
        subset_counts = dict(subset_result.counts)
        collapse_counts = dict(collapse_result.counts)
        expected_full_support = {
            "0" * n_qubits,
            "1" * n_qubits,
        }
        expected_subset_support = {"0", "1"}
        measured_bit = next(iter(collapse_counts))
        mismatched_probability = sum(
            float(probability)
            for index, probability in enumerate(
                collapsed_probabilities
            )
            if format(index, f"0{n_qubits}b")[0] != measured_bit
        )
        collapsed_norm_error = float(
            abs(np.sum(collapsed_probabilities) - 1.0)
        )
        collapse_count_state_consistent = (
            len(collapse_counts) == 1
            and sum(collapse_counts.values()) == 1
            and mismatched_probability <= STATE_ATOL
        )
        full_shots = sum(full_counts.values())
        subset_shots = sum(subset_counts.values())
        collapse_shots = sum(collapse_counts.values())
        passed = (
            full_shots == 128
            and subset_shots == 128
            and collapse_shots == 1
            and set(full_counts) == expected_full_support
            and set(subset_counts) == expected_subset_support
            and mismatched_probability <= STATE_ATOL
            and collapsed_norm_error <= REDUCTION_ATOL
            and collapse_count_state_consistent
            and collapse_result.state is not None
            and collapse_result.state.layout.logical_to_storage
            == logical_to_storage
        )
        metrics = {
            "logical_to_storage": list(logical_to_storage),
            "measure_qubits": list(measure_qubits),
            "full_register_shots": full_shots,
            "subset_shots": subset_shots,
            "collapse_shots": collapse_shots,
            "full_register_support": sorted(full_counts),
            "subset_support": sorted(subset_counts),
            "collapsed_support_error": mismatched_probability,
            "collapsed_norm_error": collapsed_norm_error,
            "collapse_count_state_consistent": (
                collapse_count_state_consistent
            ),
        }
        return passed, metrics

    return _evaluate_root_section(backend, evaluate)


def _run_result_section(simulator):
    backend = simulator.backend
    distributed_axes = int(math.log2(backend.world_size))
    n_qubits = distributed_axes + 1
    logical_to_storage = _nonidentity_probe_layout(n_qubits)
    circuit = _ghz_probe_circuit(n_qubits)
    communicator = backend.communicator
    original_gather_to_root = communicator.gather_to_root
    gather_calls = 0
    return_combinations = []
    implicit_gather_deltas = []
    local_fields_ok = True

    def counted_gather_to_root(tensor, *, root=0):
        nonlocal gather_calls
        gather_calls += 1
        return original_gather_to_root(tensor, root=root)

    communicator.gather_to_root = counted_gather_to_root
    try:
        for return_state in (False, True):
            for return_probabilities in (False, True):
                gather_calls_before_run = gather_calls
                result = simulator.run(
                    circuit,
                    layout=logical_to_storage,
                    return_state=return_state,
                    return_probabilities=return_probabilities,
                )
                combination_implicit_gather_delta = (
                    gather_calls - gather_calls_before_run
                )
                implicit_gather_deltas.append(
                    combination_implicit_gather_delta
                )
                state_present = result.state is not None
                local_probabilities_present = (
                    result.local_probabilities is not None
                )
                gather_calls_before_explicit = gather_calls
                state_array = (
                    result.state.to_numpy(root=0)
                    if return_state
                    else None
                )
                probability_array = (
                    result.gather_probabilities(root=0)
                    if return_probabilities
                    else None
                )
                combination_explicit_gather_delta = (
                    gather_calls - gather_calls_before_explicit
                )
                local_fields_ok = local_fields_ok and (
                    combination_implicit_gather_delta == 0
                    and combination_explicit_gather_delta
                    == int(return_state) + int(return_probabilities)
                    and state_present == return_state
                    and local_probabilities_present
                    == return_probabilities
                    and dict(result.expectations) == {}
                    and result.counts is None
                    and result.rank == backend.rank
                    and result.world_size == backend.world_size
                    and result.is_root == (backend.rank == 0)
                )
                if backend.rank == 0:
                    return_combinations.append(
                        {
                            "return_state": return_state,
                            "return_probabilities": (
                                return_probabilities
                            ),
                            "state_present": state_present,
                            "local_probabilities_present": (
                                local_probabilities_present
                            ),
                            "state_materialized": (
                                state_array is not None
                            ),
                            "probabilities_materialized": (
                                probability_array is not None
                            ),
                            "implicit_gather_delta": (
                                combination_implicit_gather_delta
                            ),
                            "explicit_gather_delta": (
                                combination_explicit_gather_delta
                            ),
                        }
                    )
    finally:
        communicator.gather_to_root = original_gather_to_root

    field_evidence = _gather_long_evidence(
        backend,
        (
            backend.rank,
            local_fields_ok,
        ),
    )

    def evaluate():
        fields_ok = all(
            rank == index and rank_fields_ok == 1
            for index, (rank, rank_fields_ok) in enumerate(
                field_evidence
            )
        )
        expected_combinations = {
            (False, False): (False, False, False, False, 0, 0),
            (False, True): (False, True, False, True, 0, 1),
            (True, False): (True, False, True, False, 0, 1),
            (True, True): (True, True, True, True, 0, 2),
        }
        actual_combinations = {
            (
                row["return_state"],
                row["return_probabilities"],
            ): (
                row["state_present"],
                row["local_probabilities_present"],
                row["state_materialized"],
                row["probabilities_materialized"],
                row["implicit_gather_delta"],
                row["explicit_gather_delta"],
            )
            for row in return_combinations
        }
        four_return_combinations = (
            fields_ok
            and actual_combinations == expected_combinations
        )
        implicit_gather_delta = sum(implicit_gather_deltas)
        explicit_gather_delta = sum(
            row["explicit_gather_delta"]
            for row in return_combinations
        )
        passed = (
            implicit_gather_delta == 0
            and explicit_gather_delta > 0
            and four_return_combinations
        )
        metrics = {
            "four_return_combinations": four_return_combinations,
            "implicit_gather_delta": implicit_gather_delta,
            "explicit_gather_delta": explicit_gather_delta,
            "implicit_gather_deltas": implicit_gather_deltas,
            "return_combinations": return_combinations,
        }
        return passed, metrics

    return _evaluate_root_section(backend, evaluate)


class _ExchangeRecorder:
    """Probe-local wrapper around one communicator's tensor exchange."""

    def __init__(self, communicator):
        self._communicator = communicator
        self._original_exchange_tensor = None
        self._original_supports_complex = None
        self.events = []

    def install(self):
        if self._original_exchange_tensor is not None:
            raise RuntimeError("_ExchangeRecorder 已安装")
        self._original_exchange_tensor = (
            self._communicator._exchange_tensor
        )
        self._original_supports_complex = (
            self._communicator.supports_complex
        )

        def recorded_exchange(tensor, peer, tag):
            self.events.append((int(peer), int(tag)))
            return self._original_exchange_tensor(tensor, peer, tag)

        self._communicator._exchange_tensor = recorded_exchange
        # Exercise the NPU real/imag transport contract on CPU gloo too.
        self._communicator.supports_complex = False

    def restore(self):
        if self._original_exchange_tensor is None:
            return
        self._communicator._exchange_tensor = (
            self._original_exchange_tensor
        )
        self._communicator.supports_complex = (
            self._original_supports_complex
        )
        self._original_exchange_tensor = None
        self._original_supports_complex = None


def _run_communication_section(simulator):
    backend = simulator.backend
    distributed_axes = int(math.log2(backend.world_size))
    n_qubits = distributed_axes + 1
    logical_to_storage = tuple(range(n_qubits))
    local_target = n_qubits - 1
    recorder = _ExchangeRecorder(backend.communicator)
    recorder.install()
    try:
        before_local = len(recorder.events)
        simulator.run(
            Circuit(pauli_x(local_target), n_qubits=n_qubits),
            layout=logical_to_storage,
            return_probabilities=False,
        )
        local_p2p_delta = len(recorder.events) - before_local

        distributed_axis_deltas = []
        for target in range(distributed_axes):
            before_distributed = len(recorder.events)
            simulator.run(
                Circuit(pauli_x(target), n_qubits=n_qubits),
                layout=logical_to_storage,
                return_probabilities=False,
            )
            distributed_axis_deltas.append(
                len(recorder.events) - before_distributed
            )
        distributed_events = recorder.events[
            before_local + local_p2p_delta:
        ]
    finally:
        recorder.restore()

    peer_mask = 0
    peer_tag_counts = defaultdict(lambda: [0, 0])
    valid_peers = True
    for peer, tag in distributed_events:
        peer_mask |= 1 << peer
        valid_peers = (
            valid_peers
            and 0 <= peer < backend.world_size
            and peer != backend.rank
        )
        peer_tag_counts[(peer, tag // 2)][tag % 2] += 1
    even_tag_count = sum(
        counts[0] for counts in peer_tag_counts.values()
    )
    odd_tag_count = sum(
        counts[1] for counts in peer_tag_counts.values()
    )
    paired_transport_tags = (
        bool(peer_tag_counts)
        and all(
            even_count == odd_count and even_count > 0
            for even_count, odd_count in peer_tag_counts.values()
        )
    )
    distributed_p2p_delta = sum(distributed_axis_deltas)
    distinct_peer_count = peer_mask.bit_count()
    evidence = _gather_long_evidence(
        backend,
        (
            backend.rank,
            local_p2p_delta,
            distributed_p2p_delta,
            peer_mask,
            even_tag_count,
            odd_tag_count,
            paired_transport_tags,
            valid_peers,
            distinct_peer_count,
            all(delta > 0 for delta in distributed_axis_deltas),
            *distributed_axis_deltas,
        ),
    )

    def evaluate():
        per_rank_evidence = []
        ranks_ok = True
        for expected_rank, row in enumerate(evidence):
            axis_deltas = row[10:]
            rank_evidence = {
                "rank": row[0],
                "local_p2p_delta": row[1],
                "distributed_p2p_delta": row[2],
                "peer_mask": row[3],
                "even_tag_count": row[4],
                "odd_tag_count": row[5],
                "paired_transport_tags": bool(row[6]),
                "valid_peers": bool(row[7]),
                "distinct_peer_count": row[8],
                "distributed_axis_deltas": axis_deltas,
            }
            per_rank_evidence.append(rank_evidence)
            ranks_ok = ranks_ok and (
                row[0] == expected_rank
                and row[1] == 0
                and row[2] == sum(axis_deltas)
                and row[2] > 0
                and row[4] == row[5]
                and row[4] > 0
                and row[6] == 1
                and row[7] == 1
                and row[8] >= distributed_axes
                and row[9] == 1
                and len(axis_deltas) == distributed_axes
            )
        metrics = {
            "targeted_distributed_axes": list(range(distributed_axes)),
            "logical_to_storage": list(logical_to_storage),
            "local_target": local_target,
            "per_rank_evidence": per_rank_evidence,
        }
        return ranks_ok, metrics

    return _evaluate_root_section(backend, evaluate)


def _expected_error(call, expected_message, expected_type):
    # ``UNSUPPORTED_AS_DESIGNED`` remains a historical evidence status in
    # archived API reports; Task 11 upgrades the eligible trainable paths to
    # automatic routing and records their remaining exact rejections as
    # ``EXPECTED_ERROR`` instead.
    try:
        call()
    except Exception as error:  # noqa: BLE001
        message = str(error)
        return {
            "exact_message": message == expected_message,
            "exact_type": type(error) is expected_type,
            "type": type(error).__name__,
            "message": message,
        }
    return {
        "exact_message": False,
        "exact_type": False,
        "type": None,
        "message": "NO_EXPECTED_ERROR",
    }


def _run_contract_section(simulator):
    backend = simulator.backend
    distributed_axes = int(math.log2(backend.world_size))
    n_qubits = distributed_axes + 1
    dimension = 1 << n_qubits
    identity_layout = tuple(range(n_qubits))
    empty = Circuit(n_qubits=n_qubits)
    classical = ClassicalRegister(1, "contract")
    body = Circuit(pauli_x(0), n_qubits=n_qubits)
    midcircuit_match = (
        "分布式首期不支持中途测量、reset 或经典控制流"
    )
    trainable_dist_state = simulator.run(
        empty,
        layout=identity_layout,
        return_probabilities=False,
    ).state
    trainable_dist_state.local_data.requires_grad_(True)

    def root_value(value):
        return value if backend.rank == 0 else None

    cases = (
        (
            "invalid_explicit_layout",
            "EXPECTED_ERROR",
            lambda: simulator.run(
                empty,
                layout=(0,) * n_qubits,
            ),
            "layout 必须是 range(n_qubits) 的完整双射",
            ValueError,
        ),
        (
            "invalid_root_vector_shape",
            "EXPECTED_ERROR",
            lambda: simulator.run(
                empty,
                initial_state=root_value(
                    np.zeros(dimension - 1, dtype=np.complex64)
                ),
                layout=identity_layout,
            ),
            f"initial_state 必须包含 {dimension} 个振幅",
            ValueError,
        ),
        (
            "invalid_root_density_shape",
            "EXPECTED_ERROR",
            lambda: simulator.run(
                empty,
                initial_density_matrix=root_value(
                    np.zeros(
                        (dimension - 1, dimension - 1),
                        dtype=np.complex64,
                    )
                ),
                layout=identity_layout,
            ),
            (
                "initial_density_matrix 形状必须是 "
                f"({dimension}, {dimension})"
            ),
            ValueError,
        ),
        (
            "dual_root_initial_values",
            "EXPECTED_ERROR",
            lambda: simulator.run(
                empty,
                initial_state=root_value(
                    np.eye(
                        1,
                        dimension,
                        dtype=np.complex64,
                    ).reshape(-1)
                ),
                initial_density_matrix=root_value(
                    np.eye(dimension, dtype=np.complex64)
                ),
                layout=identity_layout,
            ),
            "initial_state 与 initial_density_matrix 不能同时提供",
            ValueError,
        ),
        (
            "inconsistent_rank_input_modes",
            "EXPECTED_ERROR",
            lambda: simulator.run(
                empty,
                initial_state=(
                    np.eye(1, dimension, dtype=np.complex64).reshape(-1)
                    if backend.rank == 0
                    else None
                ),
                initial_density_matrix=(
                    np.eye(dimension, dtype=np.complex64)
                    if backend.rank == 1
                    else None
                ),
                layout=identity_layout,
            ),
            (
                "初态必须由所有 rank 提供匹配的 DistState，或仅由 "
                "rank 0 提供完整 statevector/density matrix"
            ),
            ValueError,
        ),
        (
            "mid_measure",
            "EXPECTED_ERROR",
            lambda: simulator.run(
                Circuit(measure(0), n_qubits=n_qubits),
                layout=identity_layout,
            ),
            midcircuit_match,
            ValueError,
        ),
        (
            "mid_reset",
            "EXPECTED_ERROR",
            lambda: simulator.run(
                Circuit(reset(0), n_qubits=n_qubits),
                layout=identity_layout,
            ),
            midcircuit_match,
            ValueError,
        ),
        (
            "mid_if",
            "EXPECTED_ERROR",
            lambda: simulator.run(
                Circuit(
                    if_(classical[0] == 0, body),
                    n_qubits=n_qubits,
                ),
                layout=identity_layout,
            ),
            midcircuit_match,
            ValueError,
        ),
        (
            "mid_while",
            "EXPECTED_ERROR",
            lambda: simulator.run(
                Circuit(
                    while_(
                        classical[0] == 0,
                        body,
                        max_iterations=1,
                    ),
                    n_qubits=n_qubits,
                ),
                layout=identity_layout,
            ),
            midcircuit_match,
            ValueError,
        ),
        (
            "trainable_root_state",
            "EXPECTED_ERROR",
            lambda: simulator.run(
                empty,
                initial_state=root_value(
                    torch.zeros(
                        dimension,
                        dtype=torch.complex64,
                        requires_grad=True,
                    )
                ),
                layout=identity_layout,
            ),
            "原生 distributed autograd 不接受 requires_grad complex initial_state；请使用 PureStateParam(real, imag)",
            ValueError,
        ),
        (
            "trainable_root_density",
            "EXPECTED_ERROR",
            lambda: simulator.run(
                empty,
                initial_density_matrix=root_value(
                    torch.eye(
                        dimension,
                        dtype=torch.complex64,
                        requires_grad=True,
                    )
                ),
                layout=identity_layout,
            ),
            "原生 distributed autograd 不接受 requires_grad complex initial_state；请使用 PureStateParam(real, imag)",
            ValueError,
        ),
        (
            "trainable_dist_state",
            "EXPECTED_ERROR",
            lambda: simulator.run(
                empty,
                initial_state=trainable_dist_state,
                layout=identity_layout,
            ),
            "自动微分模式的初态必须是 PureStateParam、DensityParam 或 paired-real DistState",
            TypeError,
        ),
        (
            "trainable_custom_unitary",
            "EXPECTED_ERROR",
            lambda: simulator.run(
                Circuit(
                    {
                        "type": "unitary",
                        "parameter": torch.eye(
                            2,
                            dtype=torch.complex64,
                            requires_grad=True,
                        ),
                        "n_qubits": 1,
                        "qubits": [0],
                    },
                    n_qubits=n_qubits,
                ),
                layout=identity_layout,
            ),
            "原生 distributed autograd 不接受 requires_grad complex unitary；请提供 _Pair(real, imag) 或在 CPU 参考路径构造该矩阵",
            TypeError,
        ),
    )

    case_evidence = []
    all_cases_matched = True
    for (
        name,
        expected_status,
        call,
        expected_message,
        expected_type,
    ) in cases:
        local_result = _expected_error(
            call,
            expected_message,
            expected_type,
        )
        digest_payload = (
            f"{local_result['type']}\0{local_result['message']}"
            .encode("utf-8")
        )
        local_digest = hashlib.sha256(digest_payload).digest()
        digest_tensor = torch.tensor(
            list(local_digest),
            dtype=torch.uint8,
            device=backend._device,
        )
        gathered_digests = backend.communicator.all_gather(
            digest_tensor
        )
        gathered = _gather_long_evidence(
            backend,
            (
                backend.rank,
                1,
                local_result["exact_message"],
                local_result["exact_type"],
            ),
        )
        if backend.rank == 0:
            digest_values = [
                bytes(item.detach().cpu().tolist())
                for item in gathered_digests
            ]
            expected_digest = hashlib.sha256(
                (
                    f"{expected_type.__name__}\0{expected_message}"
                ).encode("utf-8")
            ).digest()
            participating_rank_count = sum(row[1] for row in gathered)
            matched_rank_count = sum(
                row[2] and row[3] for row in gathered
            )
            exact_message = all(row[2] == 1 for row in gathered)
            exact_type = all(row[3] == 1 for row in gathered)
            message_digest_agreement = (
                len(set(digest_values)) == 1
                and digest_values[0] == expected_digest
            )
            matched = (
                participating_rank_count == backend.world_size
                and matched_rank_count == backend.world_size
                and exact_message
                and exact_type
                and message_digest_agreement
            )
            actual_status = expected_status if matched else "FAIL"
            case_evidence.append(
                {
                    "name": name,
                    "status": actual_status,
                    "expected_type": expected_type.__name__,
                    "expected_message": expected_message,
                    "actual_type": local_result["type"],
                    "actual_message": local_result["message"],
                    "exact_message": exact_message,
                    "exact_type": exact_type,
                    "message_digest": digest_values[0].hex(),
                    "message_digest_agreement": (
                        message_digest_agreement
                    ),
                    "participating_rank_count": (
                        participating_rank_count
                    ),
                    "matched_rank_count": matched_rank_count,
                }
            )
            all_cases_matched = all_cases_matched and matched

    def evaluate():
        return all_cases_matched, {
            "all_ranks_participated": all(
                row["participating_rank_count"] == backend.world_size
                for row in case_evidence
            ),
            "case_statuses": {
                row["name"]: row["status"]
                for row in case_evidence
            },
            "case_evidence": case_evidence,
        }

    return _evaluate_root_section(backend, evaluate)


def _validate_runtime(backend):
    device = backend._device
    if backend.world_size not in {2, 4}:
        raise ValueError("完整 API 探针只接受 world_size=2 或 world_size=4")
    if device.type != "npu":
        raise RuntimeError(f"严格探针要求 NPU，实际设备为 {device}")
    if backend.rank != backend.local_rank or device.index != backend.local_rank:
        raise RuntimeError("rank、local_rank 与 NPU device 不一致")


SECTION_RUNNERS = {
    "state": _run_state_section,
    "layout": _run_layout_section,
    "continuation": _run_continuation_section,
    "noise": _run_noise_section,
    "observable": _run_observable_section,
    "measure": _run_measure_section,
    "result": _run_result_section,
    "communication": _run_communication_section,
    "contract": _run_contract_section,
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
