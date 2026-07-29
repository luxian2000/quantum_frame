import numpy as np
import pytest
import torch

from aicir import (
    Circuit,
    Hamiltonian,
    Observable,
    PauliString,
    hadamard,
    measure,
    rx,
)
from aicir.distributed import DistNPUBackend, DistSimulator


def _simulator(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    backend = DistNPUBackend.from_env(
        fallback_to_cpu=True,
        init_process_group=False,
    )
    return DistSimulator(backend)


def test_from_env_builds_explicit_backend(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    simulator = DistSimulator.from_env(
        fallback_to_cpu=True,
        init_process_group=False,
    )

    assert isinstance(simulator.backend, DistNPUBackend)


def test_exact_run_returns_state_probabilities_and_expectations(monkeypatch):
    simulator = _simulator(monkeypatch)
    circuit = Circuit(hadamard(0), n_qubits=1)

    result = simulator.run(
        circuit,
        observables={"z": PauliString("Z", n_qubits=1)},
    )

    np.testing.assert_allclose(
        result.gather_probabilities(),
        [0.5, 0.5],
        atol=1e-6,
    )
    assert abs(result.expectations["z"]) < 1e-6
    assert result.counts is None


def test_rejects_multi_shot_collapse(monkeypatch):
    simulator = _simulator(monkeypatch)

    with pytest.raises(ValueError, match="collapse=True"):
        simulator.run(Circuit(n_qubits=1), shots=2, collapse=True)


def test_rejects_midcircuit_measurement(monkeypatch):
    simulator = _simulator(monkeypatch)
    circuit = Circuit(hadamard(0), measure(0), n_qubits=1)

    with pytest.raises(ValueError, match="中途测量"):
        simulator.run(circuit)


def test_rejects_trainable_gate_parameter(monkeypatch):
    simulator = _simulator(monkeypatch)
    theta = torch.tensor(0.2, requires_grad=True)
    circuit = Circuit(rx(theta, 0), n_qubits=1)

    with pytest.raises(ValueError, match="requires_grad"):
        simulator.run(circuit)


def test_root_owned_initial_density_matrix(monkeypatch):
    simulator = _simulator(monkeypatch)
    rho = np.array([[0.25, 0.0], [0.0, 0.75]], dtype=np.complex64)

    result = simulator.run(
        Circuit(n_qubits=1),
        initial_density_matrix=rho,
    )

    np.testing.assert_allclose(result.state.to_numpy(), rho, atol=1e-6)
    assert result.state.is_density


def test_state_and_probability_return_flags_are_independent(monkeypatch):
    simulator = _simulator(monkeypatch)
    circuit = Circuit(hadamard(0), n_qubits=1)

    probabilities_only = simulator.run(circuit, return_state=False)
    assert probabilities_only.state is None
    np.testing.assert_allclose(
        probabilities_only.gather_probabilities(),
        [0.5, 0.5],
        atol=1e-6,
    )

    state_only = simulator.run(
        circuit,
        return_probabilities=False,
    )
    assert state_only.state is not None
    assert state_only.local_probabilities is None
    assert state_only.gather_probabilities() is None


def test_structured_hamiltonian_and_local_dense_expectations(monkeypatch):
    simulator = _simulator(monkeypatch)
    circuit = Circuit(hadamard(0), n_qubits=2)
    hamiltonian = Hamiltonian(
        n_qubits=2,
        terms=[("XI", 0.5), ("ZI", 0.25)],
    )
    local_x = Observable.matrix(
        np.array([[0, 1], [1, 0]], dtype=np.complex64),
        metadata={"qubits": [0]},
    )

    result = simulator.run(
        circuit,
        observables={
            "hamiltonian": hamiltonian,
            "local_x": local_x,
        },
    )

    assert abs(result.expectations["hamiltonian"] - 0.5) < 1e-6
    assert abs(result.expectations["local_x"] - 1.0) < 1e-6
