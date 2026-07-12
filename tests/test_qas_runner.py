import numpy as np
import pytest

from aicir import NumpyBackend, State
from aicir.protocols import AlgorithmResult
from aicir.qas import (
    QASResult,
    QASRunConfig,
    available_qas_methods,
    config,
    default_qas_config,
    run,
)


def _tiny_vqa_config():
    return config.supernet(
        n_qubits=3,
        layers=3,
        single_qubit_gates=("ry",),
        two_qubit_pairs=((0, 1), (0, 2), (1, 2)),
        supernet_steps=0,
        ranking_num=3,
        finetune_steps=0,
        seed=7,
        device="cpu",
        task="classification",
    )


def _tiny_ppr_config():
    return config.pprdql(
        episode_num=3,
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


def test_available_qas_methods_contains_public_names():
    # 3b：available_qas_methods() 派生自 SearchStrategy 注册表（core.strategies），
    # 按字典序排列，覆盖全部 10 个 core.config._FACTORIES 方法（含新增的 mogvqe）。
    assert available_qas_methods() == (
        "crlqas",
        "dqas",
        "mogvqe",
        "pporb",
        "pprdql",
        "qdrats",
        "supernet",
        "supernet_classification",
        "supernet_h2",
        "vqe_loop",
    )


def test_config_factory_uses_method_names_without_config_class_imports():
    vqa_config = config.supernet(supernet_steps=0)
    ppr_config = config.create("PPRDQL", episode_num=1)
    ppr_legacy_alias_config = config.create("PPR_DQL", episode_num=2)
    crl_config = config.crlqas(adam_spsa={"iterations": 2})

    assert vqa_config.__class__.__name__ == "SupernetConfig"
    assert vqa_config.supernet_steps == 0
    assert ppr_config.__class__.__name__ == "PPRDQLConfig"
    assert ppr_config.max_episodes == 1
    assert ppr_legacy_alias_config.__class__.__name__ == "PPRDQLConfig"
    assert ppr_legacy_alias_config.max_episodes == 2
    assert crl_config.adam_spsa.iterations == 2


def test_default_qas_config_remains_method_name_wrapper():
    assert default_qas_config("classification").__class__.__name__ == "SupernetConfig"
    assert default_qas_config("VQA_QAS", supernet_steps=0).supernet_steps == 0


def test_run_dispatches_vqa_classification():
    result = run("VQA_QAS", config=_tiny_vqa_config())

    # 3b：run() 统一返回 QASResult；SupernetResult 特有字段仍可从 raw/metadata 取。
    assert isinstance(result, QASResult)
    assert isinstance(result, AlgorithmResult)
    assert result.method == "supernet"
    assert result.circuit.n_qubits == 3
    assert result.raw.best_circuit.n_qubits == 3
    assert result.metadata["best_architecture"] is not None
    assert len(result.metadata["ranking_records"]) == 3


def test_run_accepts_custom_vqa_objective_keyword():
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

    result = run("VQA_QAS", objective=lambda circuit, **_: 0.0, config=cfg)

    assert isinstance(result, QASResult)
    assert result.circuit.n_qubits == 1
    assert result.value == 0.0


def test_run_accepts_request_object_for_pprdql():
    target_state = State.from_array(
        np.array([0.0, 1.0], dtype=np.complex64),
        n_qubits=1,
        backend=NumpyBackend(),
    )

    result = run(QASRunConfig(method="ppr", target_state=target_state, config=_tiny_ppr_config()))

    assert isinstance(result, QASResult)
    assert result.circuit.n_qubits == 1
    assert result.value >= 0.99


def test_legacy_runner_alias_is_not_exported():
    import aicir.qas as qas

    legacy_name = "run" + "_qas"
    assert not hasattr(qas, legacy_name)


def test_run_reports_required_inputs():
    with pytest.raises(ValueError, match="pprdql requires target_state"):
        run("pprdql", config=_tiny_ppr_config())

    with pytest.raises(ValueError, match="pprdql requires target_state"):
        run("ppr_dql", config=_tiny_ppr_config())

    with pytest.raises(ValueError, match="Available methods"):
        run("not_a_method")
