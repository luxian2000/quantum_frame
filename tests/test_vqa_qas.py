import math

import numpy as np
import torch

from aicir.channel.backends.torch_backend import TorchBackend
from aicir.qas import (
    Architecture,
    LayerArchitecture,
    VQAQAS,
    VQAQASConfig,
    classification_vqa_qas,
    h2_vqe_qas,
)


def _classification_config(**kwargs):
    values = dict(
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
    values.update(kwargs)
    return VQAQASConfig(**values)


def _h2_config(**kwargs):
    values = dict(
        n_qubits=4,
        layers=3,
        single_qubit_gates=("ry", "rz"),
        two_qubit_pairs=((0, 1), (1, 2), (2, 3)),
        supernet_steps=0,
        ranking_num=3,
        finetune_steps=0,
        seed=7,
        device="cpu",
        task="h2_vqe",
    )
    values.update(kwargs)
    return VQAQASConfig(**values)


def _arch(single=("ry",), mask=(False,), layers=1):
    if len(single) == 1:
        single = tuple(single[0] for _ in range(1))
    layer = LayerArchitecture(tuple(single), tuple(mask))
    return Architecture(tuple(layer for _ in range(layers)))


def _tiny_dataset():
    x = np.array(
        [
            [0.0, 0.1, -0.2],
            [0.7, -0.4, 0.3],
            [-0.8, 0.2, 0.5],
            [1.0, 0.5, -0.6],
        ],
        dtype=np.float32,
    )
    y = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)
    return {
        "train": (x[:2], y[:2]),
        "validation": (x[2:3], y[2:3]),
        "test": (x[3:], y[3:]),
    }


def test_classification_search_space_size_is_8_cubed():
    qas = VQAQAS(_classification_config())

    assert qas.logical_search_space_size() == 8**3


def test_h2_search_space_size_is_128_cubed():
    qas = VQAQAS(_h2_config())

    assert qas.logical_search_space_size() == 128**3


def test_architecture_sampling_returns_valid_architecture():
    qas = VQAQAS(_classification_config())

    architecture = qas.sample_architecture()

    assert isinstance(architecture, Architecture)
    assert len(architecture.layers) == 3
    for layer in architecture.layers:
        assert len(layer.single_qubit_gates) == 3
        assert len(layer.two_qubit_mask) == 3
        assert all(gate == "ry" for gate in layer.single_qubit_gates)


def test_architecture_maps_to_valid_circuit_show_and_unitary_shape():
    qas = VQAQAS(_classification_config(layers=1))
    architecture = Architecture(
        (
            LayerArchitecture(("ry", "ry", "ry"), (True, False, True)),
        )
    )

    circuit, _, _ = qas.build_circuit(architecture, supernet_id=0)
    diagram = circuit.show()
    unitary = circuit.unitary(backend=TorchBackend(device="cpu"))

    assert circuit.n_qubits == 3
    assert "Ry" in diagram
    assert tuple(unitary.shape) == (8, 8)


def test_weight_sharing_reuses_tensor_for_matching_layer_layout():
    qas = VQAQAS(
        VQAQASConfig(
            n_qubits=2,
            layers=1,
            single_qubit_gates=("ry",),
            two_qubit_pairs=((0, 1),),
            device="cpu",
        )
    )
    arch_a = Architecture((LayerArchitecture(("ry", "ry"), (False,)),))
    arch_b = Architecture((LayerArchitecture(("ry", "ry"), (True,)),))

    _, keys_a, tensors_a = qas.build_circuit(arch_a, supernet_id=0)
    _, keys_b, tensors_b = qas.build_circuit(arch_b, supernet_id=0)

    assert keys_a == keys_b
    assert tensors_a[0] is tensors_b[0]
    assert tensors_a[1] is tensors_b[1]


def test_weight_sharing_uses_different_tensors_for_different_layouts():
    qas = VQAQAS(
        VQAQASConfig(
            n_qubits=1,
            layers=1,
            single_qubit_gates=("ry", "rz"),
            two_qubit_pairs=(),
            device="cpu",
        )
    )
    arch_ry = Architecture((LayerArchitecture(("ry",), ()),))
    arch_rz = Architecture((LayerArchitecture(("rz",), ()),))

    _, keys_ry, tensors_ry = qas.build_circuit(arch_ry, supernet_id=0)
    _, keys_rz, tensors_rz = qas.build_circuit(arch_rz, supernet_id=0)

    assert keys_ry != keys_rz
    assert tensors_ry[0] is not tensors_rz[0]


def test_supernet_selection_chooses_minimum_loss_supernet():
    qas = VQAQAS(
        VQAQASConfig(
            n_qubits=1,
            layers=1,
            single_qubit_gates=("ry",),
            two_qubit_pairs=(),
            supernet_num=2,
            device="cpu",
            task="custom",
        )
    )
    architecture = Architecture((LayerArchitecture(("ry",), ()),))

    def objective(supernet_id, **_):
        return torch.tensor(1.0 if supernet_id == 0 else 0.25)

    selected, losses = qas.select_supernet(architecture, objective)

    assert selected == 1
    assert losses == [1.0, 0.25]


def test_one_qubit_ry_parameter_is_differentiable():
    qas = VQAQAS(
        VQAQASConfig(
            n_qubits=1,
            layers=1,
            single_qubit_gates=("ry",),
            two_qubit_pairs=(),
            device="cpu",
            task="custom",
        )
    )
    architecture = Architecture((LayerArchitecture(("ry",), ()),))
    circuit, _, tensors = qas.build_circuit(architecture)
    state = qas.simulate_state(circuit)
    z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex64)
    expectation = qas.backend.expectation_sv(state, z)
    loss = (expectation - 0.5) ** 2

    loss.backward()

    assert tensors[0].grad is not None
    assert torch.isfinite(tensors[0].grad)


def test_ranking_returns_lowest_loss_architecture_among_candidates():
    qas = VQAQAS(
        VQAQASConfig(
            n_qubits=2,
            layers=1,
            single_qubit_gates=("ry",),
            two_qubit_pairs=((0, 1),),
            ranking_num=2,
            device="cpu",
            task="custom",
        )
    )
    no_cnot = Architecture((LayerArchitecture(("ry", "ry"), (False,)),))
    with_cnot = Architecture((LayerArchitecture(("ry", "ry"), (True,)),))

    def objective(architecture, qas, **_):
        return torch.tensor(float(qas.cnot_count(architecture)))

    records = qas.rank_architectures(objective, candidates=[with_cnot, no_cnot])

    assert records[0]["architecture"] == no_cnot
    assert math.isclose(records[0]["score"], 0.0)
    assert records[1]["architecture"] == with_cnot


def test_classification_vqa_qas_smoke_runs_small_config():
    config = _classification_config(supernet_steps=2, ranking_num=3, finetune_steps=2)

    result = classification_vqa_qas(config)

    assert np.isfinite(result.best_score)
    assert result.best_circuit.n_qubits == 3
    assert len(result.supernet_log) == 2
    assert len(result.ranking_records) == 3
    assert "test_accuracy" in result.final_metrics


def test_h2_vqe_qas_smoke_runs_small_config():
    config = _h2_config(supernet_steps=2, ranking_num=3, finetune_steps=2)

    result = h2_vqe_qas(config)

    assert np.isfinite(result.best_score)
    assert result.best_circuit.n_qubits == 4
    assert len(result.supernet_log) == 2
    assert len(result.ranking_records) == 3
    assert "baseline_vqe_energy" in result.final_metrics
