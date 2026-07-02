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


def test_excitation_pool_uses_excitation_gates_and_hf_prep():
    from aicir import Hamiltonian

    # -(XX+YY)/2 is a hopping term: ground state lives in the single-excitation
    # subspace {|01>, |10>} at energy -1, reachable from HF |10> by one
    # single_excitation. HF energy is 0, so a working excitation search must dip
    # well below zero.
    hamiltonian = Hamiltonian(n_qubits=2, terms=[("XX", -0.5), ("YY", -0.5)])
    cfg = QDRATSConfig(
        n_qubits=2,
        layers=1,
        gate_pool="excitation",
        single_excitations=((0, 1),),
        double_excitations=(),
        hf_occupied_qubits=(0,),
        search_epochs=40,
        theta_steps=3,
        finetune_steps=80,
        hidden_dim=4,
        seed=3,
        device="cpu",
    )

    result = train_qdrats(hamiltonian=hamiltonian, config=cfg)

    types = {g["type"] for g in result.circuit.gates}
    assert "single_excitation" in types  # excitation pool was searched
    assert "pauli_x" in types  # HF reference prepended
    assert result.minimum_energy < -0.5  # improved well below HF energy (0)
    assert result.architecture_probabilities.shape == (1, 1, 2)  # 1 slot, {exc, id}


def test_excitation_pool_requires_a_pool():
    with pytest.raises(ValueError):
        QuantumDARTS(
            QDRATSConfig(n_qubits=2, layers=1, gate_pool="excitation")
        )


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
