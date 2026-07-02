import csv
import json
import sys
import types

from aicir.qas.vqe_loop.benchmark_table import BENCHMARK_TABLE_FIELDS


def test_supernet_construction_does_not_materialize_full_layout_space(monkeypatch):
    import aicir.qas.algorithms.supernet as supernet_module

    def fail_product(*_args, **_kwargs):
        raise AssertionError("supernet must sample layouts lazily, not materialize product()")

    if hasattr(supernet_module, "product"):
        monkeypatch.setattr(supernet_module, "product", fail_product)

    config = supernet_module.SupernetConfig(
        n_qubits=10,
        layers=2,
        supernet_num=1,
        supernet_steps=0,
        ranking_num=1,
        finetune_steps=0,
        device="cpu",
    )
    supernet = supernet_module.Supernet(config)

    architecture = supernet.sample_architecture()

    assert len(architecture.layers) == 2
    assert len(architecture.layers[0].single_qubit_gates) == 10


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
    from aicir.qas.vqe_loop.p0_supernet_native import build_supernet_native_rows

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
        assert row["source"] == "trackB_supernet"
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
        assert row["expressibility_score"] not in {"", None}
        assert row["trainability_score"] not in {"", None}
        assert row["entanglement_score"] not in {"", None}
        assert row["zero_cost_feature_score"] not in {"", None}
        payload = json.loads(row["ansatz_gene"])
        assert payload["kind"] == "supernet_native"


def test_supernet_native_rows_select_by_finetuned_screening_energy(tmp_path, monkeypatch):
    from aicir.qas.primitives.ansatz import architecture_from_supernet_gene
    from aicir.qas.vqe_loop import supernet_native

    class FakeLayer:
        def __init__(self, single_qubit_gates, two_qubit_choices):
            self.single_qubit_gates = tuple(single_qubit_gates)
            self.two_qubit_choices = tuple(two_qubit_choices)

    class FakeArchitecture:
        def __init__(self, label, single_gate):
            self.label = label
            self.layers = (FakeLayer((single_gate, single_gate), ("cx",)),)

    class FakeSupernetConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeSupernet:
        def __init__(self, config):
            self.config = config
            self.architectures = [
                FakeArchitecture("cheap_rank_winner", "ry"),
                FakeArchitecture("finetune_winner", "rz"),
            ]

        def optimize_supernet(self, *_args, **_kwargs):
            return None

        def encode_architecture(self, architecture):
            return ((0 if architecture.label == "cheap_rank_winner" else 1,),)

        def rank_architectures(self, *_args, **_kwargs):
            return [
                {
                    "architecture": self.architectures[0],
                    "architecture_indices": self.encode_architecture(self.architectures[0]),
                    "selected_supernet_id": 0,
                    "score": -0.10,
                    "two_qubit_count": 1,
                },
                {
                    "architecture": self.architectures[1],
                    "architecture_indices": self.encode_architecture(self.architectures[1]),
                    "selected_supernet_id": 0,
                    "score": 0.20,
                    "two_qubit_count": 1,
                },
            ]

        def finetune_architecture(self, architecture, *_args, **_kwargs):
            gene = supernet_native.gene_from_supernet_architecture(
                architecture,
                n_qubits=2,
                two_qubit_pairs=((0, 1),),
            )
            circuit = architecture_from_supernet_gene(gene).circuit
            energy = -0.20 if architecture.label == "cheap_rank_winner" else -1.50
            return circuit, [], {}, energy

    fake_module = types.ModuleType("aicir.qas.algorithms.supernet")
    fake_module.Supernet = FakeSupernet
    fake_module.SupernetConfig = FakeSupernetConfig
    monkeypatch.setitem(sys.modules, "aicir.qas.algorithms.supernet", fake_module)

    rows, summary = supernet_native.build_supernet_native_rows(
        hamiltonian_terms=[(-1.0, "ZI"), (-0.5, "IZ")],
        hamiltonian_id="toy_2q",
        hamiltonian_class="pauli_terms",
        count=1,
        layers=1,
        supernet_num=1,
        supernet_steps=1,
        ranking_num=2,
        finetune_steps=1,
        seed=7,
        device="cpu",
        params_dir=tmp_path,
    )

    selected_gene = json.loads(rows[0]["ansatz_gene"])
    assert selected_gene["single_qubit_layers"] == [["rz", "rz"]]
    assert rows[0]["screening_energy"] == "-1.500000000000"
    assert summary["screened_candidate_count"] == 2
    random_baseline = summary["random_baseline_row"]
    assert random_baseline["screening_energy"] == "-0.200000000000"
    assert random_baseline["supernet_rank_score"] == "-0.100000000000"
    assert random_baseline["supernet_warm_start_status"] == "ready"


def test_supernet_native_defaults_mirror_supernet_for_nine_qubits():
    from aicir.qas.vqe_loop.p0_supernet_native import _default_single_qubit_gates, _default_two_qubit_pairs

    assert _default_single_qubit_gates(9) == ("i", "h", "rx", "ry", "rz")
    assert _default_two_qubit_pairs(9) == tuple(
        (left, right)
        for left in range(9)
        for right in range(left + 1, 9)
        if right - left <= 2
    )


def test_supernet_bootstrap_queue_records_top_candidates_for_oracle(tmp_path, monkeypatch):
    from aicir.qas.vqe_loop.p0_bootstrap_fair import ClosedLoopConfig, write_supernet_bootstrap_queue

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

    fake_module = types.ModuleType("aicir.qas.vqe_loop.p0_supernet_native")
    fake_module.build_supernet_native_rows = fake_build_supernet_native_rows
    monkeypatch.setitem(sys.modules, "aicir.qas.vqe_loop.p0_supernet_native", fake_module)

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


def test_supernet_bootstrap_closed_loop_runs_p0_fair_only(tmp_path, monkeypatch):
    from aicir.qas.vqe_loop.p0_bootstrap_fair import ClosedLoopConfig, run_vqe_qas_closed_loop

    def fake_build_supernet_native_rows(**kwargs):
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
                "zero_cost_status": "pass",
                "ansatz_gene": json.dumps({"kind": "supernet_native"}),
                "supernet_init_params_ref": "bootstrap_params.json",
                "screening_energy_is_final_label": "false",
            }
        )
        return [row], {"enabled": True, "generated_rows": 1, "warm_start_params_written": 1}

    def fake_run_module(module, args, *, cwd):
        assert module == "aicir.qas.vqe_loop.fair_labeling"
        output = Path(args[args.index("--output") + 1])
        output.write_text(
            "architecture_id,label_status,protocol_version,fair_best_energy,source\n"
            "2q_supernet_native_bootstrap,completed,fair_vqe_protocol_v2,-1.0,trackB_supernet\n",
            encoding="utf-8",
        )

    fake_module = types.ModuleType("aicir.qas.vqe_loop.p0_supernet_native")
    fake_module.build_supernet_native_rows = fake_build_supernet_native_rows
    monkeypatch.setitem(sys.modules, "aicir.qas.vqe_loop.p0_supernet_native", fake_module)
    monkeypatch.setattr("aicir.qas.vqe_loop.p0_bootstrap_fair._run_module", fake_run_module)

    result = run_vqe_qas_closed_loop(
        ClosedLoopConfig(
            output_dir=tmp_path,
            n_qubits=2,
            hamiltonian_terms=[(-1.0, "ZI"), (-0.5, "IZ")],
            hamiltonian_id="toy_ising",
            hamiltonian_class="ising",
            initial_labels=0,
            rounds=0,
            supernet_native_count=1,
        )
    )

    assert result.initial_queue.exists()
    assert result.final_benchmark_table.exists()
    summary = json.loads((tmp_path / "closed_loop_summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "p0_bootstrap_fair_only"
    assert summary["final_best_energy"] == -1.0

def test_supernet_bootstrap_count_covers_k_min_when_rounds_need_oracle():
    from aicir.qas.vqe_loop.p0_bootstrap_fair import (
        ClosedLoopConfig,
        ClosedLoopResolvedDefaults,
        effective_supernet_bootstrap_count,
    )

    config = ClosedLoopConfig(
        output_dir="unused",
        n_qubits=9,
        hamiltonian_terms=[(-1.0, "ZI")],
        initial_labels=0,
        rounds=1,
        k_min=3,
        supernet_native_count=1,
    )
    resolved = ClosedLoopResolvedDefaults(
        initial_labels=0,
        rounds=1,
        local=3,
        boundary=2,
        sparse=2,
        control=1,
    )

    assert effective_supernet_bootstrap_count(config, resolved) == 3


def test_closed_loop_uses_k_min_sized_supernet_bootstrap(tmp_path, monkeypatch):
    from aicir.qas.vqe_loop import vqe_qas_loop
    from aicir.qas.vqe_loop.p0_bootstrap_fair import ClosedLoopConfig

    observed_counts = []

    def fake_build_supernet_native_rows(**kwargs):
        observed_counts.append(kwargs["count"])
        rows = []
        params_dir = kwargs["params_dir"]
        for index in range(int(kwargs["count"])):
            params_name = f"bootstrap_params_{index}.json"
            (params_dir / params_name).write_text("[0.25]\n", encoding="utf-8")
            row = {field: "" for field in BENCHMARK_TABLE_FIELDS}
            row.update(
                {
                    "architecture_id": f"2q_supernet_native_bootstrap_{index}",
                    "canonical_arch_hash": f"bootstrap_gene_{index}",
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
                    "supernet_rank_score": f"{index:.6f}",
                    "supernet_init_params_ref": params_name,
                    "screening_energy": f"{index:.6f}",
                    "screening_energy_is_final_label": "false",
                    "supernet_warm_start_status": "ready",
                }
            )
            rows.append(row)
        return rows, {"enabled": True, "generated_rows": len(rows), "warm_start_params_written": len(rows)}

    fake_module = types.ModuleType("aicir.qas.vqe_loop.p0_supernet_native")
    fake_module.build_supernet_native_rows = fake_build_supernet_native_rows
    monkeypatch.setitem(sys.modules, "aicir.qas.vqe_loop.p0_supernet_native", fake_module)

    config = ClosedLoopConfig(
        output_dir=tmp_path,
        n_qubits=2,
        hamiltonian_terms=[(-1.0, "ZI"), (-0.5, "IZ")],
        hamiltonian_id="toy_ising",
        hamiltonian_class="ising",
        initial_labels=0,
        rounds=1,
        k_min=3,
        supernet_native_count=1,
    )
    resolved = vqe_qas_loop.resolve_closed_loop_defaults(
        n_qubits=config.n_qubits,
        initial_labels=config.initial_labels,
        rounds=config.rounds,
    )
    config = vqe_qas_loop.replace(
        config,
        supernet_native_count=vqe_qas_loop.effective_supernet_bootstrap_count(config, resolved),
    )

    vqe_qas_loop.write_supernet_bootstrap_queue(config, output_dir=tmp_path)

    assert observed_counts[0] == 3
    summary = json.loads((tmp_path / "supernet_bootstrap_plan_summary.json").read_text(encoding="utf-8"))
    assert summary["planned_total"] == 3

