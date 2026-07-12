"""run(method, problem=...) 与旧版 hamiltonian=/target_state=/target_density_matrix=
关键字参数的路由/互斥关系（Phase 3b）。"""

from __future__ import annotations

import numpy as np
import pytest

from aicir.core.operators import Hamiltonian
from aicir.qas import config, run


def _tiny_dqas_config():
    return config.dqas(n_qubits=1, layers=1, search_epochs=0, theta_steps=0, finetune_steps=0, batch_size=2, seed=5)


def test_run_problem_kwarg_routes_hamiltonian_object():
    pytest.importorskip("torch")
    hamiltonian = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])

    result = run("dqas", problem=hamiltonian, config=_tiny_dqas_config())

    assert result.circuit.n_qubits == 1
    assert result.value == pytest.approx(1.0, abs=1e-6)


def test_run_legacy_hamiltonian_kwarg_still_routes():
    pytest.importorskip("torch")
    hamiltonian = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])

    result = run("dqas", hamiltonian=hamiltonian, config=_tiny_dqas_config())

    assert result.circuit.n_qubits == 1
    assert result.value == pytest.approx(1.0, abs=1e-6)


def test_run_problem_and_legacy_hamiltonian_together_raises():
    pytest.importorskip("torch")
    hamiltonian = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])

    with pytest.raises(ValueError, match="problem="):
        run("dqas", problem=hamiltonian, hamiltonian=hamiltonian, config=_tiny_dqas_config())


def test_run_two_legacy_kwargs_together_raises():
    pytest.importorskip("torch")
    hamiltonian = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])

    with pytest.raises(ValueError):
        run(
            "dqas",
            hamiltonian=hamiltonian,
            target_density_matrix=np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex64),
            config=_tiny_dqas_config(),
        )


def test_run_pporb_problem_kwarg_routes_density_matrix_ndarray():
    pytest.importorskip("torch")
    density_matrix = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex64)
    cfg = config.pporb(max_episodes=2, max_steps_per_episode=2, update_timestep=4, epoch_num=1, hidden_dim=8, seed=1)

    result = run("pporb", problem=density_matrix, epsilon=0.99, config=cfg)

    assert result.circuit.n_qubits == 1


def test_run_pporb_legacy_target_density_matrix_kwarg_still_routes():
    pytest.importorskip("torch")
    density_matrix = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex64)
    cfg = config.pporb(max_episodes=2, max_steps_per_episode=2, update_timestep=4, epoch_num=1, hidden_dim=8, seed=1)

    result = run("pporb", target_density_matrix=density_matrix, epsilon=0.99, config=cfg)

    assert result.circuit.n_qubits == 1


def test_run_pprdql_legacy_target_state_kwarg_still_routes():
    pytest.importorskip("torch")
    from aicir import NumpyBackend, State

    target_state = State.from_array(
        np.array([0.0, 1.0], dtype=np.complex64), n_qubits=1, backend=NumpyBackend()
    )
    cfg = config.pprdql(
        max_episodes=1,
        max_steps_per_episode=1,
        batch_size=1,
        replay_capacity=8,
        warmup_transitions=1,
        target_update_interval=1,
        action_gates=[{"type": "pauli_x", "target_qubit": 0}],
        seed=7,
    )

    result = run("pprdql", target_state=target_state, config=cfg)

    assert result.circuit.n_qubits == 1


def test_run_problem_field_accepts_prebuilt_qasproblem():
    pytest.importorskip("torch")
    from aicir.qas.core.problem import normalize_problem

    hamiltonian = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])
    problem = normalize_problem(hamiltonian)

    result = run("dqas", problem=problem, config=_tiny_dqas_config())

    assert result.value == pytest.approx(1.0, abs=1e-6)
