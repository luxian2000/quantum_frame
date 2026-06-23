from pathlib import Path

from aicir.qas.vqe_loop.vqe_qas_loop import (
    ClosedLoopConfig,
    decide_next_round_quotas,
    run_vqe_qas_closed_loop,
)
from aicir.qas.vqe_loop.geometry import CandidateRecord, DistanceScales


def test_closed_loop_passes_literal_hamiltonian_class_to_stage0_preparation(monkeypatch, tmp_path):
    calls = []

    def fake_run_module(module, args, *, cwd):
        calls.append((module, list(args)))
        if module.endswith(".preparation"):
            output_dir = Path(args[args.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "stage1_5_initial_label_queue.csv").write_text(
                "architecture_id,label_status,hamiltonian_class,hamiltonian_id,hamiltonian_terms\n"
                "arch_a,pending,,,\n",
                encoding="utf-8",
            )
            (output_dir / "stage0_candidates.csv").write_text(
                "architecture_id,hamiltonian_class\narch_a,molecular_lih\n",
                encoding="utf-8",
            )
        elif module.endswith(".labeling"):
            output = Path(args[args.index("--output") + 1])
            output.write_text(
                "architecture_id,label_status,protocol_version,fair_best_energy,source,hamiltonian_class,hamiltonian_id,hamiltonian_terms\n"
                "arch_a,completed,fair_vqe_protocol_v2,-1.0,initial_train,molecular_lih,lih_sto3g_jw_r15,\"[[1.0, \"\"ZI\"\"]]\"\n",
                encoding="utf-8",
            )

    monkeypatch.setattr("aicir.qas.vqe_loop.vqe_qas_loop._run_module", fake_run_module)

    run_vqe_qas_closed_loop(
        ClosedLoopConfig(
            output_dir=tmp_path,
            n_qubits=2,
            hamiltonian_terms=[(1.0, "ZI")],
            hamiltonian_id="lih_sto3g_jw_r15",
            hamiltonian_class="molecular_lih",
            rounds=0,
            initial_labels=1,
            include_layerwise=False,
        )
    )

    prep_call = next(args for module, args in calls if module.endswith(".preparation"))
    assert "--hamiltonian-class" in prep_call
    assert prep_call[prep_call.index("--hamiltonian-class") + 1] == "molecular_lih"


def test_weak_lih_oracle_with_failed_local_prefers_sparse_and_keeps_control_sentinels():
    decision = decide_next_round_quotas(
        n_qubits=12,
        base_quotas=(3, 2, 2, 1),
        calibration={
            "k_min": 3,
            "tr_in_count": 7,
            "tr_in_mae": 0.5658,
            "tr_out_mae": 0.7873,
            "sparse_abstain_rate": 1.0,
            "passes": {"overall": False, "sparse_abstain": True},
        },
        local_improved=False,
    )

    assert decision.mode == "sparse_explore"
    assert (decision.local, decision.boundary, decision.sparse, decision.control) == (1, 1, 4, 2)
    assert decision.sparse > decision.boundary
    assert decision.control == 2


def test_boundary_selection_prefers_neighbors_of_top_sparse_or_initial_labels():
    from aicir.qas.vqe_loop.next_batch import _rank_boundary_records

    top_initial = CandidateRecord(
        architecture_id="top_initial",
        canonical_arch_hash="top_initial",
        family="ry_rzz_ring_L1",
        entangler_type="rzz",
        topology="ring",
        depth_group="L1",
        n_params=36,
        two_q_count=12,
        hamiltonian_class="molecular_lih",
        hamiltonian_coverage=1.3,
        metadata={"source": "initial_train"},
    )
    weak_label = CandidateRecord(
        architecture_id="weak_local",
        canonical_arch_hash="weak_local",
        family="layerwise_gene",
        entangler_type="mixed_edge",
        topology="linear",
        depth_group="L3",
        n_params=96,
        two_q_count=28,
        hamiltonian_class="molecular_lih",
        hamiltonian_coverage=0.5,
        metadata={"source": "trackA_local"},
    )
    near_boundary = CandidateRecord(
        architecture_id="near_boundary",
        canonical_arch_hash="near_boundary",
        family="ry_rzz_ring_L1",
        entangler_type="rzz",
        topology="ring",
        depth_group="L2",
        n_params=58,
        two_q_count=22,
        hamiltonian_class="molecular_lih",
        hamiltonian_coverage=1.3,
    )
    far_boundary = CandidateRecord(
        architecture_id="far_boundary",
        canonical_arch_hash="far_boundary",
        family="layerwise_gene",
        entangler_type="mixed_edge",
        topology="linear",
        depth_group="L3",
        n_params=168,
        two_q_count=36,
        hamiltonian_class="molecular_lih",
        hamiltonian_coverage=0.5,
    )

    ranked = _rank_boundary_records(
        [far_boundary, near_boundary],
        completed=[(weak_label, -6.6), (top_initial, -7.59)],
        scales=DistanceScales(n_params=43, two_q_count=21, hamiltonian_coverage=1.0),
        rows_by_id={},
        count=1,
    )

    assert [record.architecture_id for record in ranked] == ["near_boundary"]
