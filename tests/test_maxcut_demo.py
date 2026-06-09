import sys
from types import SimpleNamespace

import networkx as nx
import pytest

from aicir import Circuit
from aicir.channel.backends.numpy_backend import NumpyBackend
from aicir.core.circuit import hadamard
from demos.MaxCut import maxcut
from demos.MaxCut import maxcut_hamiltonian


def test_maxcut_metrics_separate_expected_cut_from_best_readout_cut():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1.0)
    hamiltonian, _ = maxcut.build_ising_hamiltonian(graph)
    circuit = Circuit(hadamard(1), n_qubits=2)

    metrics, partition = maxcut.evaluate_vqe_cut_metrics(
        circuit,
        graph,
        hamiltonian,
        NumpyBackend(),
    )

    assert metrics["expected_cut"] == pytest.approx(0.5)
    assert metrics["achieved_cut"] == pytest.approx(1.0)
    assert metrics["approx_ratio"] == pytest.approx(1.0)
    assert partition[0] != partition[1]


def test_generated_maxcut_circuit_metrics_match_checked_in_artifact():
    graph = nx.Graph()
    graph.add_nodes_from(range(maxcut_hamiltonian.N_NODES))
    graph.add_weighted_edges_from(maxcut_hamiltonian.EDGES)
    hamiltonian, _ = maxcut.build_ising_hamiltonian(graph)

    metrics, _ = maxcut.evaluate_vqe_cut_metrics(
        maxcut_hamiltonian.build_vqe_circuit(),
        graph,
        hamiltonian,
        NumpyBackend(),
    )

    stored_metrics = maxcut_hamiltonian.VQE_METRICS

    for key in (
        "vqe_energy",
        "exact_ground_energy",
        "max_cut",
        "expected_cut",
        "achieved_cut",
        "approx_ratio",
        "expected_approx_ratio",
    ):
        assert metrics[key] == pytest.approx(stored_metrics[key])
    assert metrics["n_gates"] == stored_metrics["n_gates"]
    assert metrics["expected_cut"] <= metrics["max_cut"] + 1e-6
    assert metrics["achieved_cut"] <= metrics["max_cut"] + 1e-6


def test_maxcut_cli_can_disable_rzz(monkeypatch, tmp_path):
    captured = {}

    def fake_supernet_qas(hamiltonian, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(best_circuit=Circuit(hadamard(0), n_qubits=2))

    monkeypatch.setattr(maxcut, "supernet_qas", fake_supernet_qas)
    monkeypatch.setattr(maxcut, "HAMILTONIAN_PY", tmp_path / "maxcut_hamiltonian.py")
    monkeypatch.setattr(maxcut, "CIRCUIT_PNG", tmp_path / "maxcut_hamiltonian.png")
    monkeypatch.setattr(maxcut, "plot_graph_and_circuit", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", [
        "maxcut.py",
        "--n-nodes",
        "2",
        "--edge-prob",
        "1.0",
        "--supernet-steps",
        "0",
        "--finetune-steps",
        "0",
        "--ranking-num",
        "1",
        "--disable-rzz",
    ])

    maxcut.main()

    assert captured["two_qubit_gates"] == ("cx",)
