import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aicir.operators import Hamiltonian
from aicir.backends.gpu_backend import TorchBackend
from aicir.qas.algorithms.supernet import h2_hamiltonian, prepare_classification_dataset
from aicir.qas import (
    Architecture,
    LayerArchitecture,
    Supernet,
    SupernetConfig,
    classification_supernet,
    h2_vqe_supernet,
    supernet_qas,
)


def _classification_config(**kwargs):
    values = dict(
        n_qubits=3,
        layers=3,
        single_qubit_gates=("ry",),
        two_qubit_gates=("cx",),
        two_qubit_pairs=((0, 1), (0, 2), (1, 2)),
        supernet_steps=0,
        ranking_num=3,
        finetune_steps=0,
        seed=7,
        device="cpu",
        task="classification",
    )
    values.update(kwargs)
    return SupernetConfig(**values)


def _h2_config(**kwargs):
    values = dict(
        n_qubits=4,
        layers=3,
        single_qubit_gates=("ry", "rz"),
        two_qubit_gates=("cx",),
        two_qubit_pairs=((0, 1), (1, 2), (2, 3)),
        supernet_steps=0,
        ranking_num=3,
        finetune_steps=0,
        seed=7,
        device="cpu",
        task="h2_vqe",
    )
    values.update(kwargs)
    return SupernetConfig(**values)


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
    qas = Supernet(_classification_config())

    assert qas.logical_search_space_size() == 8**3


def test_h2_search_space_size_is_128_cubed():
    qas = Supernet(_h2_config())

    assert qas.logical_search_space_size() == 128**3


def test_default_gate_set_search_space_uses_full_alphabet():
    # Default single set {i, h, rx, ry, rz} (5) over 3 qubits, two-qubit choice
    # alphabet {none, cx, rzz} (3) over 3 pairs.
    qas = Supernet(
        SupernetConfig(
            n_qubits=3,
            layers=1,
            two_qubit_pairs=((0, 1), (0, 2), (1, 2)),
            device="cpu",
            task="custom",
        )
    )

    assert qas.layer_search_space_size() == (5**3) * (3**3)


def test_supernet_qas_accepts_task_override_without_duplicate_config_key(monkeypatch):
    def fake_train(self, hamiltonian=None, dataset=None):
        return self.config.task

    monkeypatch.setattr(Supernet, "train", fake_train)

    task = supernet_qas(
        SimpleNamespace(n_qubits=1),
        layers=1,
        two_qubit_pairs=(),
        supernet_steps=0,
        ranking_num=1,
        finetune_steps=0,
        task="vqe",
    )

    assert task == "vqe"


def test_hamiltonian_expectation_matches_dense_matrix_value_and_gradient():
    qas = Supernet(
        SupernetConfig(
            n_qubits=2,
            layers=1,
            two_qubit_pairs=(),
            device="cpu",
            task="vqe",
        )
    )
    hamiltonian = Hamiltonian(
        n_qubits=2,
        terms=[
            ("ZI", 0.2),
            ("XX", -0.7),
            ("YZ", 0.3),
        ],
    )
    state = torch.tensor(
        [0.3 + 0.1j, -0.2 + 0.4j, 0.5 - 0.1j, -0.6 + 0.2j],
        dtype=torch.complex64,
        requires_grad=True,
    )
    state = state / torch.linalg.vector_norm(state)
    state.retain_grad()

    termwise = qas._hamiltonian_expectation(state, hamiltonian)
    dense = qas.backend.expectation_sv(state, hamiltonian.to_matrix(qas.backend))

    assert torch.allclose(termwise, dense, atol=1e-6)
    assert qas._hamiltonian_cache == {}

    termwise.backward(retain_graph=True)
    termwise_grad = state.grad.detach().clone()
    state.grad.zero_()
    dense.backward()

    assert torch.allclose(termwise_grad, state.grad, atol=1e-6)


def test_identity_single_qubit_gate_emits_no_gate_and_no_parameter():
    qas = Supernet(
        SupernetConfig(
            n_qubits=2,
            layers=1,
            single_qubit_gates=("i", "ry"),
            two_qubit_gates=("cx",),
            two_qubit_pairs=(),
            device="cpu",
            task="custom",
        )
    )
    architecture = Architecture((LayerArchitecture(("i", "i"), ()),))

    circuit, keys, tensors = qas.build_circuit(architecture)

    assert circuit.gates == []
    assert keys == []
    assert tensors == []


def test_hadamard_single_qubit_gate_is_emitted_without_parameters():
    qas = Supernet(
        SupernetConfig(
            n_qubits=2,
            layers=1,
            single_qubit_gates=("h", "ry"),
            two_qubit_gates=("cx",),
            two_qubit_pairs=(),
            device="cpu",
            task="custom",
        )
    )
    architecture = Architecture((LayerArchitecture(("h", "h"), ()),))

    circuit, keys, tensors = qas.build_circuit(architecture)

    # Hadamard is a fixed (zero-parameter) gate: emitted, but no trainable angle.
    assert [gate["type"] for gate in circuit.gates] == ["hadamard", "hadamard"]
    assert keys == []
    assert tensors == []
    assert "H" in circuit.show()


def test_rzz_two_qubit_gate_is_trainable_and_differentiable():
    qas = Supernet(
        SupernetConfig(
            n_qubits=2,
            layers=1,
            single_qubit_gates=("i", "rx"),
            two_qubit_gates=("rzz",),
            two_qubit_pairs=((0, 1),),
            device="cpu",
            task="custom",
        )
    )
    architecture = Architecture((LayerArchitecture(("rx", "rx"), ("rzz",)),))

    circuit, keys, tensors = qas.build_circuit(architecture)

    # The identity placeholder emits nothing; rzz adds a trainable two-qubit angle.
    tq_keys = [key for key in keys if key[0] == "tq"]
    assert len(tq_keys) == 1
    assert tq_keys[0][5] == "rzz"
    assert len(tensors) == 3  # two rx angles + one rzz angle
    assert tensors[-1] is qas.shared_parameters[tq_keys[0]]

    with torch.no_grad():
        for tensor in tensors[:2]:
            tensor.copy_(torch.tensor(0.5, dtype=tensor.dtype))
    state = qas.simulate_state(circuit)
    # X⊗I does not commute with ZZ, so the rzz angle affects the expectation.
    observable = torch.tensor(
        [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
        dtype=torch.complex64,
    )
    loss = (qas.backend.expectation_sv(state, observable) - 0.3) ** 2

    loss.backward()

    assert tensors[-1].grad is not None
    assert torch.isfinite(tensors[-1].grad)


def test_two_qubit_mask_is_backward_compatible_view():
    layer = LayerArchitecture(("ry", "ry"), (True, False))

    assert layer.two_qubit_choices == ("cx", "none")
    assert layer.two_qubit_mask == (True, False)


def test_supplementary_config_fields_are_available():
    config = SupernetConfig()

    assert config.track_best_validation is True
    assert config.ranking_strategy == "random"
    assert config.use_evolutionary_ranking is False
    assert config.noise_mode == "none"


def test_architecture_sampling_returns_valid_architecture():
    qas = Supernet(_classification_config())

    architecture = qas.sample_architecture()

    assert isinstance(architecture, Architecture)
    assert len(architecture.layers) == 3
    for layer in architecture.layers:
        assert len(layer.single_qubit_gates) == 3
        assert len(layer.two_qubit_mask) == 3
        assert all(gate == "ry" for gate in layer.single_qubit_gates)


def test_architecture_maps_to_valid_circuit_show_and_unitary_shape():
    qas = Supernet(_classification_config(layers=1))
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
    qas = Supernet(
        SupernetConfig(
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
    qas = Supernet(
        SupernetConfig(
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
    qas = Supernet(
        SupernetConfig(
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
    qas = Supernet(
        SupernetConfig(
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
    qas = Supernet(
        SupernetConfig(
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


def test_evolutionary_ranking_extension_raises_clear_error():
    qas = Supernet(_h2_config(ranking_strategy="evolutionary"))

    with pytest.raises(NotImplementedError, match="Evolutionary ranking"):
        qas.rank_architectures(lambda **_: 0.0)


def test_supplementary_classification_dataset_has_expected_split_and_labels():
    dataset = prepare_classification_dataset(None, _classification_config(), torch.device("cpu"))

    assert dataset["train"][0].shape == (100, 3)
    assert dataset["validation"][0].shape == (100, 3)
    assert dataset["test"][0].shape == (100, 3)
    for _, labels in dataset.values():
        assert set(labels.cpu().numpy().tolist()).issubset({0.0, 1.0})


def test_h2_hamiltonian_matches_supplementary_eq_19_terms():
    hamiltonian = h2_hamiltonian()
    terms = [(term.coefficient.real, tuple(term.qubit_labels)) for term in hamiltonian.terms]

    assert (-0.042, ("I", "I", "I", "I")) in terms
    assert (0.178, ("Z", "I", "I", "I")) in terms
    assert (0.178, ("I", "Z", "I", "I")) in terms
    assert (-0.243, ("I", "I", "Z", "I")) in terms
    assert (-0.243, ("I", "I", "I", "Z")) in terms
    assert (0.171, ("Z", "Z", "I", "I")) in terms
    assert (0.123, ("Z", "I", "Z", "I")) in terms
    assert (0.168, ("Z", "I", "I", "Z")) in terms
    assert (0.168, ("I", "Z", "Z", "I")) in terms
    assert (0.123, ("I", "Z", "I", "Z")) in terms
    assert (0.176, ("I", "I", "Z", "Z")) in terms
    assert (0.045, ("Y", "X", "X", "Y")) in terms
    assert (-0.045, ("Y", "Y", "X", "X")) in terms
    assert (-0.045, ("X", "X", "Y", "Y")) in terms
    assert (0.045, ("X", "Y", "Y", "X")) in terms


def test_classification_supernet_smoke_runs_small_config():
    config = _classification_config(supernet_steps=2, ranking_num=3, finetune_steps=2)

    result = classification_supernet(config)

    assert np.isfinite(result.best_score)
    assert result.best_circuit.n_qubits == 3
    assert isinstance(result.best_supernet_id, int)
    assert len(result.supernet_log) == 2
    assert len(result.ranking_records) == 3
    assert "test_accuracy" in result.final_metrics
    assert "best_supernet_validation_accuracy" in result.final_metrics


def test_h2_vqe_supernet_smoke_runs_small_config():
    config = _h2_config(supernet_steps=2, ranking_num=3, finetune_steps=2)

    result = h2_vqe_supernet(config)

    assert np.isfinite(result.best_score)
    assert result.best_circuit.n_qubits == 4
    assert isinstance(result.best_supernet_id, int)
    assert len(result.supernet_log) == 2
    assert len(result.ranking_records) == 3
    assert "baseline_vqe_energy" in result.final_metrics


def _parameter_shift_fixture():
    qas = Supernet(
        SupernetConfig(
            n_qubits=2,
            layers=1,
            single_qubit_gates=("ry",),
            two_qubit_pairs=(),
            supernet_steps=0,
            use_parameter_shift=True,
            device="cpu",
            seed=11,
            task="custom",
        )
    )
    architecture = qas.sample_architecture()

    def objective(circuit):
        state = qas.simulate_state(circuit)
        return qas._probability_qubit_zero(state, qubit=0)

    _, _, _, active_tensors = qas._loss(architecture, 0, objective, split="train")
    active = []
    seen = set()
    for tensor in active_tensors:
        if id(tensor) not in seen:
            seen.add(id(tensor))
            active.append(tensor)

    def loss_closure():
        loss, _, _, _ = qas._loss(architecture, 0, objective, split="train")
        return loss

    return qas, active, loss_closure


def test_parameter_shift_update_uses_qml_psr(monkeypatch):
    import aicir.qas.algorithms.supernet as vqa_qas_module

    qas, active, loss_closure = _parameter_shift_fixture()
    captured = {}

    def fake_psr(fn, params):
        captured["fn"] = fn
        captured["params"] = np.asarray(params, dtype=float).copy()
        return np.full_like(np.asarray(params, dtype=float), 0.5)

    monkeypatch.setattr(vqa_qas_module, "psr", fake_psr)
    optimizer = torch.optim.Adam(active, lr=0.0)

    qas._parameter_shift_update(active, optimizer, loss_closure)

    # The gradient rule is delegated to aicir.qml.deriv.psr, called once with
    # the current angle vector; its return value is written onto each .grad.
    assert "fn" in captured
    np.testing.assert_allclose(
        captured["params"], [float(t.detach()) for t in active]
    )
    for tensor in active:
        assert pytest.approx(0.5) == float(tensor.grad)


def test_parameter_shift_update_matches_analytic_shift_rule():
    qas, active, loss_closure = _parameter_shift_fixture()
    base = [float(t.detach()) for t in active]

    optimizer = torch.optim.Adam(active, lr=0.0)
    grad_norm = qas._parameter_shift_update(active, optimizer, loss_closure)
    psr_grads = np.array([float(t.grad) for t in active])

    with torch.no_grad():
        for tensor, value in zip(active, base):
            tensor.copy_(torch.tensor(value, dtype=tensor.dtype))

    analytic = []
    for tensor in active:
        with torch.no_grad():
            tensor.add_(math.pi / 2.0)
            plus = float(loss_closure().detach())
            tensor.add_(-math.pi)
            minus = float(loss_closure().detach())
            tensor.add_(math.pi / 2.0)
        analytic.append((plus - minus) / 2.0)
    analytic = np.array(analytic)

    np.testing.assert_allclose(psr_grads, analytic, atol=1e-5)
    assert grad_norm == pytest.approx(float(np.linalg.norm(analytic)), abs=1e-5)
    # psr leaves the tensors restored to their starting angles.
    np.testing.assert_allclose([float(t.detach()) for t in active], base, atol=1e-6)
