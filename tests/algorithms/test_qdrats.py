import numpy as np
import pytest

pytest.importorskip("torch")

from aicir.qas import config, run
from aicir.qas.algorithms.qdrats import QDRATSConfig, QuantumDARTS, train_qdrats


def test_candidate_set_expands_cnot_controls_per_target():
    model = QuantumDARTS(
        QDRATSConfig(
            n_qubits=3,
            layers=1,
            search_epochs=0,
            theta_steps=0,
            finetune_steps=0,
            seed=7,
        )
    )

    assert model.candidate_labels_for_target(0) == (
        "rzryrz",
        "identity",
        "cx_1_to_0",
        "cx_2_to_0",
    )
    assert model.candidate_labels_for_target(2) == (
        "rzryrz",
        "identity",
        "cx_0_to_2",
        "cx_1_to_2",
    )


def test_train_qdrats_runs_with_aicir_simulator():
    hamiltonian = np.zeros((2, 2), dtype=np.complex64)
    cfg = QDRATSConfig(
        n_qubits=1,
        layers=1,
        search_epochs=2,
        theta_steps=1,
        finetune_steps=1,
        hidden_dim=2,
        seed=11,
        device="cpu",
    )

    result = train_qdrats(hamiltonian=hamiltonian, config=cfg)

    assert result.circuit.n_qubits == 1
    assert np.isfinite(result.minimum_energy)
    assert result.minimum_energy == pytest.approx(0.0, abs=1e-6)
    assert result.architecture_probabilities.shape == (1, 1, 2)
    assert len(result.search_log) == 2
    assert result.config is cfg


def test_qdrats_config_factory_and_runner_aliases():
    cfg = config.create(
        "QuantumDARTS",
        n_qubits=1,
        layers=1,
        search_epochs=0,
        theta_steps=0,
        finetune_steps=0,
        hidden_dim=2,
        seed=5,
    )

    assert isinstance(cfg, QDRATSConfig)

    hamiltonian = np.zeros((2, 2), dtype=np.complex64)
    result = run("qdrats", hamiltonian=hamiltonian, config=cfg)

    assert result.circuit.n_qubits == 1
    assert result.minimum_energy == pytest.approx(0.0, abs=1e-6)
