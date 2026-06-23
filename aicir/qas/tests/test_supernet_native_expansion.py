import csv
import json
import sys
import types

from aicir.qas.vqe_loop.protocol import BENCHMARK_TABLE_FIELDS


def test_supernet_native_gene_round_trips_to_vqe_architecture():
    from aicir.qas.primitives.ansatz import SupernetAnsatzGene, architecture_from_supernet_gene

    gene = SupernetAnsatzGene(
        n_qubits=3,
        single_qubit_layers=(("ry", "rz", "i"), ("rx", "ry", "rz")),
        two_qubit_layers=(("cx", "none"), ("rzz", "cx")),
        two_qubit_pairs=((0, 1), (1, 2)),
    )

    encoded = gene.to_jsonable()
    decoded = SupernetAnsatzGene.from_jsonable(encoded)
    architecture = architecture_from_supernet_gene(decoded)

    assert encoded["kind"] == "supernet_native"
    assert decoded == gene
    assert architecture.n_qubits == 3
    assert architecture.metadata["ansatz_gene"]["kind"] == "supernet_native"
    assert any(gate.get("type") == "ry" for gate in architecture.circuit.gates)
    assert any(gate.get("type") == "rzz" for gate in architecture.circuit.gates)


def test_supernet_native_rows_use_supernet_sampled_ranked_architectures(tmp_path):
    from aicir.qas.vqe_loop.supernet_native import build_supernet_native_rows

    terms = [
        (-1.0, "ZI"),
        (-0.5, "IZ"),
        (0.25, "XX"),
    ]

    rows, summary = build_supernet_native_rows(
        hamiltonian_terms=terms,
        hamiltonian_id="toy_2q",
        hamiltonian_class="pauli_terms",
        count=2,
        layers=1,
        supernet_num=1,
        supernet_steps=1,
        ranking_num=3,
        finetune_steps=1,
        seed=7,
        device="cpu",
        params_dir=tmp_path,
    )

    assert len(rows) == 2
    assert summary["generated_rows"] == 2
    assert summary["ranking_num"] == 3
    for row in rows:
        assert row["architecture_id"].startswith("2q_supernet_native_")
        assert row["family"] == "supernet_native"
        assert row["hamiltonian_id"] == "toy_2q"
        assert row["hamiltonian_class"] == "pauli_terms"
        assert row["supernet_rank_score"] != ""
        assert row["supernet_init_params_ref"] != ""
        params_path = tmp_path / row["supernet_init_params_ref"]
        assert params_path.exists()
        params = json.loads(params_path.read_text(encoding="utf-8"))
        assert isinstance(params, list)
        assert len(params) == int(row["n_params"])
        assert row["screening_energy_is_final_label"] == "false"
        assert row["supernet_warm_start_status"] == "ready"
        payload = json.loads(row["ansatz_gene"])
        assert payload["kind"] == "supernet_native"


def test_supernet_native_defaults_are_local_for_nine_qubits():
    from aicir.qas.vqe_loop.supernet_native import _default_single_qubit_gates, _default_two_qubit_pairs

    assert _default_single_qubit_gates(9) == ("ry", "rz")
    assert _default_two_qubit_pairs(9) == tuple((index, index + 1) for index in range(8))


def test_next_batch_writes_supernet_native_warm_start_refs(tmp_path, monkeypatch):
    from aicir.qas.vqe_loop import next_batch

    candidates_path = tmp_path / "candidates.csv"
    benchmark_path = tmp_path / "benchmark.csv"
    queue_path = tmp_path / "round1_queue.csv"
    summary_path = tmp_path / "round1_summary.json"

    with candidates_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=list(BENCHMARK_TABLE_FIELDS)).writeheader()

    benchmark_row = {field: "" for field in BENCHMARK_TABLE_FIELDS}
    benchmark_row.update(
        {
            "architecture_id": "initial_best",
            "canonical_arch_hash": "initial_best",
            "protocol_version": "fair_vqe_protocol_v2",
            "source": "initial_train",
            "label_status": "completed",
            "n_qubits": "2",
            "hamiltonian_id": "toy_2q",
            "hamiltonian_class": "pauli_terms",
            "family": "layerwise_gene",
            "depth_group": "L1",
            "entangler_type": "cx",
            "topology": "linear",
            "n_params": "1",
            "two_q_count": "1",
            "hamiltonian_coverage": "1.000000",
            "hamiltonian_coverage_features": "1.000000",
            "hamiltonian_terms": json.dumps([[-1.0, "ZI"], [-0.5, "IZ"]]),
            "fair_best_energy": "-1.000000",
        }
    )
    with benchmark_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BENCHMARK_TABLE_FIELDS))
        writer.writeheader()
        writer.writerow(benchmark_row)

    def fake_build_supernet_native_rows(**kwargs):
        assert kwargs["finetune_steps"] == 3
        params_dir = kwargs["params_dir"]
        assert params_dir == queue_path.parent
        (params_dir / "supernet_params.json").write_text("[0.125]\n", encoding="utf-8")
        row = {field: "" for field in BENCHMARK_TABLE_FIELDS}
        row.update(
            {
                "architecture_id": "2q_supernet_native_fake",
                "canonical_arch_hash": "fake_supernet_gene",
                "n_qubits": "2",
                "hamiltonian_id": "toy_2q",
                "hamiltonian_class": "pauli_terms",
                "family": "supernet_native",
                "depth_group": "L1",
                "entangler_type": "mixed_supernet",
                "topology": "supernet_pairs",
                "n_params": "1",
                "two_q_count": "1",
                "hamiltonian_coverage": "1.000000",
                "hamiltonian_coverage_features": "1.000000",
                "zero_cost_status": "pass",
                "ansatz_gene": json.dumps({"kind": "supernet_native"}),
                "supernet_rank_score": "-0.900000",
                "supernet_init_params_ref": "supernet_params.json",
                "screening_energy": "-0.950000",
                "screening_energy_is_final_label": "false",
                "supernet_warm_start_status": "ready",
            }
        )
        return [row], {"enabled": True, "generated_rows": 1, "warm_start_params_written": 1}

    fake_module = types.ModuleType("aicir.qas.vqe_loop.supernet_native")
    fake_module.build_supernet_native_rows = fake_build_supernet_native_rows
    monkeypatch.setitem(sys.modules, "aicir.qas.vqe_loop.supernet_native", fake_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "next_batch",
            "--candidates",
            str(candidates_path),
            "--benchmark-table",
            str(benchmark_path),
            "--output",
            str(queue_path),
            "--summary",
            str(summary_path),
            "--batch-id",
            "round1",
            "--local",
            "0",
            "--boundary",
            "0",
            "--sparse",
            "1",
            "--control",
            "0",
            "--supernet-native-count",
            "1",
            "--supernet-native-finetune-steps",
            "3",
        ],
    )

    next_batch.main()

    with queue_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["source"] == "trackB_supernet"
    assert rows[0]["supernet_init_params_ref"] == "supernet_params.json"
    assert rows[0]["screening_energy_is_final_label"] == "false"
    assert rows[0]["supernet_warm_start_status"] == "ready"
    assert (queue_path.parent / rows[0]["supernet_init_params_ref"]).exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["source_counts"] == {"trackB_supernet": 1}
    assert summary["supernet_native"]["warm_start_params_written"] == 1


def test_supernet_bootstrap_queue_records_top_candidates_for_oracle(tmp_path, monkeypatch):
    from aicir.qas.vqe_loop.vqe_qas_loop import ClosedLoopConfig, write_supernet_bootstrap_queue

    def fake_build_supernet_native_rows(**kwargs):
        assert kwargs["hamiltonian_terms"] == [(-1.0, "ZI"), (-0.5, "IZ")]
        assert kwargs["count"] == 1
        assert kwargs["finetune_steps"] == 2
        params_dir = kwargs["params_dir"]
        (params_dir / "bootstrap_params.json").write_text("[0.25]\n", encoding="utf-8")
        row = {field: "" for field in BENCHMARK_TABLE_FIELDS}
        row.update(
            {
                "architecture_id": "2q_supernet_native_bootstrap",
                "canonical_arch_hash": "bootstrap_gene",
                "n_qubits": "2",
                "hamiltonian_id": "toy_ising",
                "hamiltonian_class": "ising",
                "family": "supernet_native",
                "depth_group": "L1",
                "entangler_type": "mixed_supernet",
                "topology": "supernet_pairs",
                "n_params": "1",
                "two_q_count": "1",
                "hamiltonian_coverage": "1.000000",
                "hamiltonian_coverage_features": "1.000000",
                "zero_cost_status": "pass",
                "ansatz_gene": json.dumps({"kind": "supernet_native"}),
                "supernet_rank_score": "-1.250000",
                "supernet_init_params_ref": "bootstrap_params.json",
                "screening_energy": "-1.300000",
                "screening_energy_is_final_label": "false",
                "supernet_warm_start_status": "ready",
            }
        )
        return [row], {"enabled": True, "generated_rows": 1, "warm_start_params_written": 1}

    fake_module = types.ModuleType("aicir.qas.vqe_loop.supernet_native")
    fake_module.build_supernet_native_rows = fake_build_supernet_native_rows
    monkeypatch.setitem(sys.modules, "aicir.qas.vqe_loop.supernet_native", fake_module)

    config = ClosedLoopConfig(
        output_dir=tmp_path,
        n_qubits=2,
        hamiltonian_terms=[(-1.0, "ZI"), (-0.5, "IZ")],
        hamiltonian_id="toy_ising",
        hamiltonian_class="ising",
        initial_labels=0,
        rounds=0,
        supernet_native_count=1,
        supernet_native_layers=1,
        supernet_native_supernet_num=1,
        supernet_native_steps=2,
        supernet_native_ranking_num=4,
        supernet_native_finetune_steps=2,
        supernet_native_seed=17,
        supernet_native_device="cpu",
    )

    queue_path, oracle_path, summary_path = write_supernet_bootstrap_queue(
        config,
        output_dir=tmp_path,
        protocol_version="fair_vqe_protocol_v2",
    )

    with queue_path.open(newline="", encoding="utf-8") as handle:
        queue_rows = list(csv.DictReader(handle))
    with oracle_path.open(newline="", encoding="utf-8") as handle:
        oracle_rows = list(csv.DictReader(handle))

    assert queue_rows == oracle_rows
    assert queue_rows[0]["source"] == "trackB_supernet"
    assert queue_rows[0]["label_status"] == "pending"
    assert queue_rows[0]["hamiltonian_id"] == "toy_ising"
    assert queue_rows[0]["hamiltonian_class"] == "ising"
    assert json.loads(queue_rows[0]["hamiltonian_terms"]) == [[-1.0, "ZI"], [-0.5, "IZ"]]
    assert queue_rows[0]["supernet_init_params_ref"] == "bootstrap_params.json"
    assert queue_rows[0]["screening_energy_is_final_label"] == "false"
    assert summary_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["mode"] == "supernet_native_bootstrap"


def test_trackb_supernet_completed_label_is_preferred_boundary_anchor():
    from aicir.qas.vqe_loop.geometry import CandidateRecord
    from aicir.qas.vqe_loop.next_batch import _priority_boundary_anchors
    from aicir.qas.vqe_loop.protocol import LabelSource

    def record(architecture_id, source):
        return CandidateRecord(
            architecture_id=architecture_id,
            canonical_arch_hash=architecture_id,
            family="layerwise_gene",
            entangler_type="mixed",
            topology="linear",
            depth_group="L1",
            n_params=2.0,
            two_q_count=1.0,
            hamiltonian_id="toy",
            hamiltonian_class="pauli_terms",
            hamiltonian_coverage=1.0,
            metadata={"source": source, "n_qubits": 2},
        )

    supernet = record("supernet_top", LabelSource.TRACKB_SUPERNET.value)
    initial = record("initial_best", LabelSource.INITIAL_TRAIN.value)
    boundary = record("boundary_best", LabelSource.TRACKB_BOUNDARY.value)

    anchors = _priority_boundary_anchors(
        [
            (boundary, -2.0),
            (initial, -1.0),
            (supernet, -1.5),
        ],
        limit=2,
    )

    assert [anchor.architecture_id for anchor in anchors] == ["supernet_top", "initial_best"]
