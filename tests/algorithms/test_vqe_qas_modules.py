"""Regression tests for the modular QAS/VQE building blocks."""

import math
import csv

from aicir.metrics.circuit_structure import entanglement_coverage_score, structural_expressibility_proxy_score
from aicir.qas.library.ansatz import (
    HEAMask,
    LayerwiseAnsatzGene,
    architecture_from_hea_mask,
    architecture_from_layerwise_gene,
    enumerate_hea_masks,
    sample_layerwise_genes,
)
from aicir.qas.core.backend_utils import backend_runtime_metadata, resolve_qas_backend
from aicir.qas.vqe_loop.fair_vqe import evaluate_vqe_energy, optimize_vqe_energy
from aicir.qas.problems.hamiltonians import VQEProblem, tfim_chain_hamiltonian


def test_closed_loop_entry_can_stamp_literal_hamiltonian_terms_into_queue(tmp_path) -> None:
    from aicir.qas.vqe_loop.vqe_qas_loop import stamp_literal_hamiltonian_terms

    queue = tmp_path / "queue.csv"
    queue.write_text(
        "architecture_id,label_status\n"
        "arch_a,pending\n",
        encoding="utf-8",
    )

    terms = [(-1.0, "ZI"), (-0.5, "IX")]
    stamp_literal_hamiltonian_terms(
        queue,
        terms,
        hamiltonian_id="custom2",
        hamiltonian_class="literal",
    )

    rows = list(csv.DictReader(queue.open(newline="", encoding="utf-8-sig")))
    assert rows[0]["hamiltonian_id"] == "custom2"
    assert rows[0]["hamiltonian_class"] == "literal"
    assert rows[0]["hamiltonian_terms"] == '[[-1.0, "ZI"], [-0.5, "IX"]]'


def test_closed_loop_auto_defaults_scale_with_qubits() -> None:
    from aicir.qas.vqe_loop.vqe_qas_loop import (
        default_initial_labels_for_qubits,
        default_max_rounds_for_qubits,
        resolve_closed_loop_defaults,
    )

    assert default_initial_labels_for_qubits(4) == 12
    assert default_initial_labels_for_qubits(8) == 24
    assert default_initial_labels_for_qubits(12) == 36
    assert default_initial_labels_for_qubits(16) == 48

    assert default_max_rounds_for_qubits(4) == 4
    assert default_max_rounds_for_qubits(8) == 4
    assert default_max_rounds_for_qubits(12) == 2
    assert default_max_rounds_for_qubits(16) == 2

    resolved = resolve_closed_loop_defaults(n_qubits=12)
    assert resolved.initial_labels == 36
    assert resolved.rounds == 2
    assert (resolved.local, resolved.boundary, resolved.sparse, resolved.control) == (3, 2, 2, 1)


def test_closed_loop_manual_quota_overrides_auto_defaults() -> None:
    from aicir.qas.vqe_loop.vqe_qas_loop import resolve_closed_loop_defaults

    resolved = resolve_closed_loop_defaults(
        n_qubits=12,
        initial_labels=40,
        rounds=5,
        batch_size=6,
        local=4,
    )

    assert resolved.initial_labels == 40
    assert resolved.rounds == 5
    assert resolved.local == 4
    assert resolved.boundary + resolved.sparse + resolved.control == 2


def test_closed_loop_adaptive_quotas_prefer_exploration_for_weak_lih_oracle() -> None:
    from aicir.qas.vqe_loop.vqe_qas_loop import decide_next_round_quotas

    decision = decide_next_round_quotas(
        n_qubits=12,
        base_quotas=(3, 2, 2, 1),
        calibration={
            "k_min": 3,
            "tr_in_count": 4,
            "tr_in_mae": 0.4578,
            "tr_out_mae": 0.8198,
            "sparse_abstain_rate": 1.0,
            "passes": {"overall": False, "sparse_abstain": True},
        },
        local_improved=False,
    )

    assert decision.mode == "balanced_explore"
    assert (decision.local, decision.boundary, decision.sparse, decision.control) == (1, 2, 3, 2)


def test_closed_loop_adaptive_quotas_keep_more_local_when_oracle_passes() -> None:
    from aicir.qas.vqe_loop.vqe_qas_loop import decide_next_round_quotas

    low_qubit = decide_next_round_quotas(
        n_qubits=4,
        base_quotas=(2, 1, 1, 0),
        calibration={
            "k_min": 3,
            "tr_in_count": 5,
            "tr_in_mae": 0.1,
            "tr_out_mae": 0.4,
            "sparse_abstain_rate": 1.0,
            "passes": {"overall": True, "sparse_abstain": True},
        },
    )
    high_qubit = decide_next_round_quotas(
        n_qubits=12,
        base_quotas=(3, 2, 2, 1),
        calibration={
            "k_min": 3,
            "tr_in_count": 5,
            "tr_in_mae": 0.1,
            "tr_out_mae": 0.4,
            "sparse_abstain_rate": 1.0,
            "passes": {"overall": True, "sparse_abstain": True},
        },
    )

    assert low_qubit.mode == "local"
    assert high_qubit.mode == "local"
    assert low_qubit.local / sum(low_qubit.as_tuple()) > high_qubit.local / sum(high_qubit.as_tuple())


def test_qas_core_modules_expose_mainline_vqe_loop_building_blocks() -> None:
    terms = tfim_chain_hamiltonian(n_qubits=2, J=1.0, h=0.5, periodic=False)
    problem = VQEProblem(name="tfim2", n_qubits=2, hamiltonian=terms, reference_energy=-1.0)
    architecture = architecture_from_hea_mask(HEAMask(n_qubits=2, layers=1, rotation_block="ry"))

    energy = evaluate_vqe_energy(architecture, problem, [math.pi / 2, 0.0, 0.0, 0.0])

    assert isinstance(energy, float)
    assert enumerate_hea_masks(2)
    assert architecture_from_layerwise_gene(sample_layerwise_genes(n_qubits=2, layers=2, count=1, seed=1)[0]).n_qubits == 2


def test_fair_vqe_uses_basic_vqe_and_records_best_parameters() -> None:
    problem = VQEProblem(name="x_only", n_qubits=1, hamiltonian=[(-1.0, "X")], reference_energy=-1.0)
    gene = LayerwiseAnsatzGene(
        n_qubits=1,
        single_blocks=("ry", "none"),
        edge_entanglers=((),),
        entangle_pattern="linear",
    )
    architecture = architecture_from_layerwise_gene(gene)

    result = optimize_vqe_energy(
        architecture,
        problem,
        seed=7,
        n_starts=1,
        max_evaluations=10,
        budget_override=10,
        initial_parameters=[math.pi / 2],
    )

    assert result.metadata["vqe_engine"] == "BasicVQE"
    assert result.best_parameters
    assert result.evaluations > 0


def test_backend_utils_report_resolved_backend_metadata() -> None:
    backend = resolve_qas_backend("numpy", dtype="complex128")
    metadata = backend_runtime_metadata(backend)

    assert metadata["backend_class"] == "NumpyBackend"
    assert "complex128" in metadata["backend_dtype"]


def test_circuit_metrics_expose_stage1_structural_scores() -> None:
    entanglement = entanglement_coverage_score(two_q_count=3, n_qubits=4, layers=1, topology="linear")
    expressibility = structural_expressibility_proxy_score(
        n_params=16,
        n_qubits=4,
        layers=1,
        rotation_block="rx_ry_rz",
        final_rotation="ry_rz",
        entanglement_score=entanglement,
    )

    assert entanglement == 1.0
    assert 0.0 < expressibility <= 1.0
