"""run(method, ...) 统一返回 QASResult（Phase 3b）。

对每个方法：能以极小配置便宜跑通的走真实 smoke run；配置起来仍然偏重的
（vqe_loop 需要完整 P0 bootstrap 家族配置）改为对适配器包装逻辑本身做
monkeypatch 桩测试——3b owns 的是“翻译”逻辑，不是底层算法本身的正确性。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from aicir.protocols import AlgorithmResult
from aicir.qas import QASResult, config, run
from aicir.core.operators import Hamiltonian
from aicir.core.circuit import Circuit


def _assert_common_shape(result, *, method: str) -> None:
    assert isinstance(result, QASResult)
    assert isinstance(result, AlgorithmResult)
    assert result.method == method
    assert result.raw is not None


# ──────────────────────────────────────────────────────────────────────────────
# torch-free：mogvqe
# ──────────────────────────────────────────────────────────────────────────────


def test_run_mogvqe_smoke():
    from aicir.qas.algorithms.mogvqe import block_hardware_efficient_ansatz

    initial = block_hardware_efficient_ansatz(2, layers=1, topology="linear")
    hamiltonian = Hamiltonian(n_qubits=2, terms=[("ZZ", -1.0), ("XI", 0.5)])
    cfg = config.mogvqe(population_size=2, generations=1, parameter_generations=1, parameter_population_size=2, seed=1)

    result = run("mogvqe", initial_ansatz=initial, problem=hamiltonian, config=cfg)

    _assert_common_shape(result, method="mogvqe")
    assert isinstance(result.circuit, Circuit)
    assert isinstance(result.value, float)
    assert isinstance(result.history, list) and len(result.history) >= 1


def test_run_mogvqe_requires_initial_ansatz():
    hamiltonian = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])
    with pytest.raises(ValueError, match="initial_ansatz"):
        run("mogvqe", problem=hamiltonian, config=config.mogvqe())


# ──────────────────────────────────────────────────────────────────────────────
# torch-dependent：dqas / qdrats / crlqas / pprdql / pporb / supernet 系列
# ──────────────────────────────────────────────────────────────────────────────


def test_run_dqas_smoke():
    pytest.importorskip("torch")
    hamiltonian = np.zeros((2, 2), dtype=np.complex64)
    cfg = config.dqas(n_qubits=1, layers=1, search_epochs=0, theta_steps=0, finetune_steps=0, batch_size=2, seed=5)

    result = run("dqas", hamiltonian=hamiltonian, config=cfg)

    _assert_common_shape(result, method="dqas")
    assert isinstance(result.circuit, Circuit)
    assert result.value == pytest.approx(0.0, abs=1e-6)
    assert isinstance(result.parameters, dict)


def test_run_qdrats_smoke():
    pytest.importorskip("torch")
    hamiltonian = np.zeros((2, 2), dtype=np.complex64)
    cfg = config.qdrats(n_qubits=1, layers=1, search_epochs=0, theta_steps=0, finetune_steps=0, hidden_dim=2, seed=5)

    result = run("qdrats", hamiltonian=hamiltonian, config=cfg)

    _assert_common_shape(result, method="qdrats")
    assert isinstance(result.circuit, Circuit)
    assert result.value == pytest.approx(0.0, abs=1e-6)


def test_run_crlqas_smoke():
    pytest.importorskip("torch")
    hamiltonian = np.zeros((2, 2), dtype=np.complex64)
    cfg = config.crlqas(
        max_episodes=2,
        n_act=2,
        batch_size=2,
        replay_capacity=32,
        train_interval=1,
        target_update_interval=5,
        log_interval=0,
        seed=7,
    )

    result = run("crlqas", problem=hamiltonian, config=cfg)

    _assert_common_shape(result, method="crlqas")
    assert isinstance(result.circuit, Circuit)
    assert isinstance(result.value, float)
    assert isinstance(result.history, list)


def test_run_pprdql_smoke():
    pytest.importorskip("torch")
    from aicir import NumpyBackend, State

    target_state = State.from_array(
        np.array([0.0, 1.0], dtype=np.complex64), n_qubits=1, backend=NumpyBackend()
    )
    cfg = config.pprdql(
        max_episodes=3,
        max_steps_per_episode=1,
        batch_size=1,
        replay_capacity=8,
        warmup_transitions=1,
        target_update_interval=1,
        fidelity_threshold=0.99,
        gate_penalty=0.0,
        terminal_bonus=1.0,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1.0,
        action_gates=[{"type": "pauli_x", "target_qubit": 0}],
        seed=7,
    )

    result = run("pprdql", problem=target_state, config=cfg)

    _assert_common_shape(result, method="pprdql")
    assert isinstance(result.circuit, Circuit)
    assert result.value >= 0.99


def test_run_pporb_smoke():
    pytest.importorskip("torch")
    density_matrix = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex64)
    cfg = config.pporb(max_episodes=2, max_steps_per_episode=2, update_timestep=4, epoch_num=1, hidden_dim=8, seed=1)

    result = run("pporb", problem=density_matrix, epsilon=0.99, config=cfg)

    # ppo_rb_qas 只返回 (theta, circuit)，没有可用的保真度字段——value 按文档留空。
    _assert_common_shape(result, method="pporb")
    assert isinstance(result.circuit, Circuit)
    assert result.value is None
    assert result.parameters is not None


def test_run_supernet_classification_smoke():
    pytest.importorskip("torch")
    from aicir.qas.algorithms.supernet import SupernetConfig

    cfg = SupernetConfig(
        n_qubits=3,
        layers=1,
        single_qubit_gates=("ry",),
        two_qubit_pairs=((0, 1), (1, 2)),
        supernet_steps=0,
        ranking_num=2,
        finetune_steps=0,
        seed=1,
        device="cpu",
    )

    result = run("supernet_classification", config=cfg)

    _assert_common_shape(result, method="supernet_classification")
    assert isinstance(result.circuit, Circuit)
    assert isinstance(result.value, float)


def test_run_supernet_h2_smoke():
    pytest.importorskip("torch")
    from aicir.qas.algorithms.supernet import SupernetConfig

    cfg = SupernetConfig(
        n_qubits=4,
        layers=1,
        single_qubit_gates=("ry",),
        two_qubit_pairs=((0, 1), (1, 2), (2, 3)),
        supernet_steps=0,
        ranking_num=2,
        finetune_steps=0,
        seed=1,
        device="cpu",
    )

    result = run("supernet_h2", config=cfg)

    _assert_common_shape(result, method="supernet_h2")
    assert isinstance(result.circuit, Circuit)
    assert isinstance(result.value, float)


def test_run_supernet_custom_objective_smoke():
    pytest.importorskip("torch")
    cfg = config.supernet(
        n_qubits=1,
        layers=1,
        single_qubit_gates=("ry",),
        two_qubit_pairs=(),
        supernet_steps=0,
        ranking_num=2,
        finetune_steps=0,
        seed=7,
        device="cpu",
        task="custom",
    )

    result = run("supernet", objective=lambda circuit, **_: 0.0, config=cfg)

    _assert_common_shape(result, method="supernet")
    assert isinstance(result.circuit, Circuit)
    assert result.value == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# vqe_loop：真实调用需要完整 P0 bootstrap 家族配置（chemistry/supernet_native），
# 太重，改为直接对适配器包装逻辑打桩（monkeypatch 底层 run_vqe_qas_closed_loop）。
# ──────────────────────────────────────────────────────────────────────────────


def test_run_vqe_loop_wraps_stubbed_closed_loop_result():
    from aicir.qas.vqe_loop import ClosedLoopResult

    with tempfile.TemporaryDirectory() as temp:
        cfg = config.vqe_loop(
            output_dir=Path(temp) / "qas_loop",
            n_qubits=2,
            hamiltonian_terms=[(-1.0, "ZI")],
            rounds=0,
            initial_labels=0,
        )
        expected = ClosedLoopResult(
            output_dir=cfg.output_dir,
            candidates=cfg.output_dir / "stage0_candidates.csv",
            initial_queue=cfg.output_dir / "stage1_5_initial_label_queue.csv",
            initial_benchmark_table=cfg.output_dir / "benchmark_table_2q_v2.csv",
            final_benchmark_table=cfg.output_dir / "benchmark_table_2q_v2.csv",
            round_summaries=(),
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("aicir.qas.vqe_loop.run_vqe_qas_closed_loop", lambda config: expected)
            result = run("vqe_loop", config=cfg)

    _assert_common_shape(result, method="vqe_loop")
    assert result.raw is expected
    assert result.value is None
    assert result.circuit is None
    assert result.metadata["output_dir"] == str(expected.output_dir)
