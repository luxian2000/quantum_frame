import numpy as np
import pytest

pytest.importorskip("torch")

from aicir.qas import config, run
from aicir.qas.algorithms.dqas import DQASConfig, DifferentiableQAS, train_dqas


def test_dqas_samples_independent_categorical_architectures():
    model = DifferentiableQAS(DQASConfig(n_qubits=1, layers=2, batch_size=4, seed=123))

    samples, log_probs = model.sample_architectures(batch_size=4)

    assert samples.shape == (4, 2)
    assert log_probs.shape == (4,)
    assert samples.min() >= 0
    assert samples.max() < len(model.operation_labels)


def test_dqas_score_function_gradient_matches_softmax_identity():
    model = DifferentiableQAS(DQASConfig(n_qubits=1, layers=1, batch_size=2, seed=3))
    samples = model.backend_tensor([[0], [1]], dtype="long")
    losses = model.backend_tensor([2.0, 4.0])

    grad = model.score_function_alpha_gradient(samples, losses, baseline=3.0)

    probs = model.architecture_probabilities()[0]
    expected = np.array(
        [
            (2.0 - 3.0) * (1.0 - probs[0]) + (4.0 - 3.0) * (0.0 - probs[0]),
            (2.0 - 3.0) * (0.0 - probs[1]) + (4.0 - 3.0) * (1.0 - probs[1]),
        ],
        dtype=np.float32,
    ) / 2.0
    np.testing.assert_allclose(grad[0], expected, atol=1e-6)


def test_train_dqas_runs_with_aicir_simulator():
    hamiltonian = np.zeros((2, 2), dtype=np.complex64)
    cfg = DQASConfig(
        n_qubits=1,
        layers=1,
        search_epochs=2,
        theta_steps=1,
        finetune_steps=1,
        batch_size=3,
        seed=11,
        device="cpu",
    )

    result = train_dqas(hamiltonian=hamiltonian, config=cfg)

    assert result.circuit.n_qubits == 1
    assert np.isfinite(result.minimum_energy)
    assert result.minimum_energy == pytest.approx(0.0, abs=1e-6)
    assert result.architecture_probabilities.shape == (1, 2)
    assert len(result.search_log) == 2
    assert result.config is cfg


def test_dqas_gate_pool_expands_requested_gates_and_pairs():
    model = DifferentiableQAS(
        DQASConfig(
            n_qubits=3,
            layers=1,
            gate_pool=("rx", "ry", "cx"),
            two_qubit_pairs=((0, 1), (1, 2)),
        )
    )

    assert model.operation_labels == (
        "rx_0",
        "rx_1",
        "rx_2",
        "ry_0",
        "ry_1",
        "ry_2",
        "cx_0_to_1",
        "cx_1_to_2",
    )
    assert model.theta.shape == (1, 8, 1)


def test_dqas_pool_alias_normalizes_sets_deterministically():
    model = DifferentiableQAS(DQASConfig(n_qubits=2, layers=1, pool={"cx", "identity", "rx"}))

    assert model.operation_labels == (
        "identity",
        "rx_0",
        "rx_1",
        "cx_0_to_1",
        "cx_1_to_0",
    )


def test_dqas_operation_pool_alias_still_works():
    model = DifferentiableQAS(DQASConfig(n_qubits=1, layers=1, operation_pool=("rz", "ry")))

    assert model.operation_labels == ("rz_0", "ry_0")


def test_dqas_excitation_pool_expands_global_operations_and_hf_prep():
    model = DifferentiableQAS(
        DQASConfig(
            n_qubits=4,
            layers=1,
            gate_pool="excitation",
            single_excitations=((0, 1),),
            double_excitations=((0, 1, 2, 3),),
            hf_occupied_qubits=(1, 3),
        )
    )

    assert model.operation_labels == ("identity", "single_0_1", "double_0_1_2_3")
    assert model.theta.shape == (1, 3, 1)

    indices = model.backend_tensor([1], dtype="long")
    circuit, parameters = model._build_circuit(indices)

    assert [gate["type"] for gate in circuit.gates[:2]] == ["pauli_x", "pauli_x"]
    assert circuit.gates[2]["type"] == "single_excitation"
    assert parameters


def test_dqas_excitation_pool_runs_with_aicir_simulator():
    from aicir import Hamiltonian

    hamiltonian = Hamiltonian(n_qubits=2, terms=[("XX", -0.5), ("YY", -0.5)])
    cfg = DQASConfig(
        n_qubits=2,
        layers=1,
        gate_pool="excitation",
        single_excitations=((0, 1),),
        hf_occupied_qubits=(0,),
        search_epochs=20,
        theta_steps=2,
        finetune_steps=30,
        batch_size=4,
        seed=3,
        device="cpu",
    )

    result = train_dqas(hamiltonian=hamiltonian, config=cfg)

    assert "single_excitation" in {gate["type"] for gate in result.circuit.gates}
    assert "pauli_x" in {gate["type"] for gate in result.circuit.gates}
    assert result.architecture_probabilities.shape == (1, 2)


def test_dqas_config_factory_runner_and_strategy_registration():
    cfg = config.create(
        "DQAS",
        n_qubits=1,
        layers=1,
        search_epochs=0,
        theta_steps=0,
        finetune_steps=0,
        batch_size=2,
        seed=5,
    )

    assert isinstance(cfg, DQASConfig)

    hamiltonian = np.zeros((2, 2), dtype=np.complex64)
    result = run("dqas", hamiltonian=hamiltonian, config=cfg)

    # 3b：run() 统一返回 QASResult；DQASResult.minimum_energy -> result.value。
    assert result.circuit.n_qubits == 1
    assert result.value == pytest.approx(0.0, abs=1e-6)
