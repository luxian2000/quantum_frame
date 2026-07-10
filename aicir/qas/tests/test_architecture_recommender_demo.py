import json
from pathlib import Path


def test_molecular_architecture_recommender_writes_txt_report(tmp_path):
    from aicir.qas.demos.VQE_QAS_demo_architecture_recommender import run_recommender

    request_path = tmp_path / "lih_request.json"
    output_path = tmp_path / "lih_recommendations.txt"
    result_dir = tmp_path / "lih_supernet_bootstrap"
    result_dir.mkdir()
    ansatz_gene = {
        "kind": "supernet_native",
        "n_qubits": 12,
        "single_qubit_layers": [["ry"] * 12, ["ry"] * 12],
        "two_qubit_layers": [["cx", "none", "rzz"], ["none", "cx", "none"]],
        "two_qubit_pairs": [[0, 1], [1, 2], [2, 3]],
    }
    (result_dir / "benchmark_table_12q_v2.csv").write_text(
        "architecture_id,source,label_status,n_params,two_q_count,ansatz_gene,"
        "screening_energy,screening_energy_is_final_label,fair_best_energy,delta_ref,reference_energy\n"
        f"lih_supernet_L2_best,trackB_supernet,completed,27,3,\"{json.dumps(ansatz_gene).replace(chr(34), chr(34) + chr(34))}\","
        "-7.91,false,-7.95,0.002,-7.952\n",
        encoding="utf-8",
    )
    request_path.write_text(
        json.dumps(
            {
                "scenario": "LiH dissociation scan, STO-3G, Jordan-Wigner mapping",
                "problem": "Find a lower-energy VQE/QAS ansatz for molecular ground-state estimation.",
                "molecular_spec": {
                    "molecule": "LiH",
                    "distance": 1.5,
                    "basis": "sto3g",
                    "mapping": "jordan_wigner",
                    "driver": "pyscf",
                    "estimated_n_qubits": 12,
                    "estimated_terms": 631,
                },
                "current_architecture": {
                    "name": "hea_mask_L2_ry_rz_cx_linear_ry",
                    "layers": 2,
                    "rotation_block": "ry_rz",
                    "entangler": "cx",
                    "entangle_pattern": "linear",
                    "n_params": 72,
                    "two_q_count": 22,
                    "fair_best_energy": -7.597667217255,
                },
                "top_k": 2,
                "result_dirs": [str(result_dir)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = run_recommender(request_path, output_path)

    assert output_path.exists()
    report = output_path.read_text(encoding="utf-8")
    assert result["output"] == str(output_path)
    assert len(result["recommendations"]) == 2
    assert result["searched_architectures"][0]["architecture_id"] == "lih_supernet_L2_best"
    assert "LiH dissociation scan" in report
    assert "molecular ground-state" in report
    assert "Problem Summary:" in report
    assert "Exact Energy: -7.952" in report
    assert "Random Architecture:" in report
    assert "Random Architecture Energy: n/a" in report
    assert "Optimized Architecture:" in report
    assert "Optimized Architecture Energy: -7.95" in report
    assert '"kind": "supernet_native"' in report
    assert "Current Architecture Diagnosis" not in report
    assert "Supernet Search Jobs" not in report
    assert "vqe_loop_supernet_bootstrap" not in report
    assert "architecture_source:" not in report
    assert "--initial-labels 0" not in report
    assert "screening_energy_is_final_label=false" not in report
    assert "supernet_bootstrap_queue.csv" not in report
    assert "chemistry_layerwise_l3_ry_rz_rxx_rzz" not in report
    assert "compact_hea_l2_ry_rz_cx_linear" not in report
    assert "architecture_output_files:" not in report
    assert "why:" not in report
    assert len(report.splitlines()) <= 18
    assert "estimated_n_qubits: 12" not in report


def test_recommender_accepts_direct_pauli_terms_without_optional_pyscf(tmp_path):
    from aicir.qas.demos.VQE_QAS_demo_architecture_recommender import run_recommender

    request_path = tmp_path / "h2_terms_request.json"
    output_path = tmp_path / "h2_terms_recommendations.txt"
    request_path.write_text(
        json.dumps(
            {
                "scenario": "Tiny H2 sanity check",
                "problem": "Molecular ground-state VQE benchmark.",
                "hamiltonian": {
                    "kind": "pauli_terms",
                    "terms": [[-0.5, "ZI"], [0.25, "IZ"], [0.1, "XX"]],
                },
                "top_k": 1,
            }
        ),
        encoding="utf-8",
    )

    result = run_recommender(request_path, output_path)

    report = output_path.read_text(encoding="utf-8")
    assert result["hamiltonian"]["n_qubits"] == 2
    assert result["hamiltonian"]["term_count"] == 3
    assert len(result["recommendations"]) == 1
    assert "Problem Summary:" in report
    assert "Exact Energy: n/a" in report
    assert "vqe_loop_supernet_bootstrap" not in report
    assert "architecture_source:" not in report


def test_recommender_runs_vqe_loop_and_prints_random_and_optimized_architectures(tmp_path, monkeypatch):
    import csv

    from aicir.qas.demos import VQE_QAS_demo_architecture_recommender as demo

    terms_path = tmp_path / "terms.json"
    terms_path.write_text(json.dumps([[0.0, "II"], [-1.0, "ZZ"], [-0.5, "XI"]]), encoding="utf-8")
    request_path = tmp_path / "run_request.json"
    output_path = tmp_path / "run_report.txt"
    run_dir = tmp_path / "supernet_run"
    request_path.write_text(
        json.dumps(
            {
                "scenario": "Executable Pauli demo",
                "problem": "Run vqe-loop supernet and print searched architectures.",
                "hamiltonian_file": str(terms_path),
                "run_search": True,
                "output_dir": str(run_dir),
                "top_k": 1,
            }
        ),
        encoding="utf-8",
    )

    observed_layers = []

    def fake_run_vqe_qas_closed_loop(config):
        observed_layers.append(config.supernet_native_layers)
        assert config.supernet_native_single_qubit_gates is None
        assert config.supernet_native_steps == 120
        assert config.supernet_native_ranking_num == 40
        assert config.supernet_native_finetune_steps == 100
        assert config.max_evals == 100
        config.output_dir.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "architecture_id",
            "source",
            "label_status",
            "n_params",
            "two_q_count",
            "ansatz_gene",
            "screening_energy",
            "screening_energy_is_final_label",
            "fair_best_energy",
            "delta_ref",
            "reference_energy",
        ]
        layer_count = int(config.supernet_native_layers)
        queue_gene = json.dumps(
            {
                "kind": "supernet_native",
                "n_qubits": 2,
                "single_qubit_layers": [["ry", "ry"] for _ in range(layer_count)],
                "two_qubit_layers": [["cx"] for _ in range(layer_count)],
                "two_qubit_pairs": [[0, 1]],
            }
        )
        best_gene = json.dumps(
            {
                "kind": "supernet_native",
                "n_qubits": 2,
                "single_qubit_layers": [["rz", "rz"] for _ in range(layer_count)],
                "two_qubit_layers": [["rzz"] for _ in range(layer_count)],
                "two_qubit_pairs": [[0, 1]],
            }
        )
        with (config.output_dir / "supernet_bootstrap_queue.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "architecture_id": "top_queue_arch",
                    "source": "trackB_supernet",
                    "label_status": "pending",
                    "n_params": "3",
                    "two_q_count": "1",
                    "ansatz_gene": queue_gene,
                    "screening_energy": "-0.25",
                    "screening_energy_is_final_label": "false",
                }
            )
        with (config.output_dir / "supernet_random_baseline.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "architecture_id": f"random_arch_L{layer_count}",
                    "source": "trackB_supernet_random",
                    "label_status": "pending",
                    "n_params": "2",
                    "two_q_count": "0",
                    "ansatz_gene": queue_gene,
                    "screening_energy": "-0.10",
                    "screening_energy_is_final_label": "false",
                }
            )
        with (config.output_dir / "benchmark_table_2q_v2.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "architecture_id": f"optimized_arch_L{layer_count}",
                    "source": "trackB_supernet",
                    "label_status": "completed",
                    "n_params": "4",
                    "two_q_count": "1",
                    "ansatz_gene": best_gene,
                    "screening_energy": "-0.25",
                    "screening_energy_is_final_label": "false",
                    "fair_best_energy": str(-1.70 - 0.05 * layer_count),
                    "delta_ref": "0.01",
                    "reference_energy": "-1.76",
                }
            )

    monkeypatch.setattr(demo, "run_vqe_qas_closed_loop", fake_run_vqe_qas_closed_loop)

    class FakeVQEResult:
        energy = -0.55

    def fake_optimize_vqe_energy(*_args, **kwargs):
        assert kwargs["budget_override"] == 100
        assert kwargs["initial_parameters"] is None
        return FakeVQEResult()

    monkeypatch.setattr(demo, "optimize_vqe_energy", fake_optimize_vqe_energy)

    result = demo.run_recommender(request_path, output_path)

    report = output_path.read_text(encoding="utf-8")
    assert result["search_executed"] is True
    assert observed_layers == [1, 2, 3]
    assert "Problem Summary: Executable Pauli demo - Run vqe-loop supernet and print searched architectures." in report
    assert "Exact Energy: -1.76" in report
    assert "Random Architecture: random_arch_L3" in report
    assert "Random Architecture Energy: -0.550000000000" in report
    assert "Random Architecture: top_queue_arch" not in report
    assert "Optimized Architecture: optimized_arch_L3" in report
    assert "Optimized Architecture Energy: -1.85" in report
    assert '"two_qubit_layers": [["rzz"], ["rzz"], ["rzz"]]' in report


def test_demo_repo_root_points_at_package_parent():
    import aicir.qas.demos.VQE_QAS_demo_architecture_recommender as demo

    assert (demo.REPO_ROOT / "aicir").exists()


def test_scenario_markdown_states_architectures_come_from_vqe_loop_supernet():
    path = Path(__file__).resolve().parents[1] / "demos" / "vqe_qas_architecture_recommender_scenarios.md"

    text = path.read_text(encoding="utf-8")

    assert "Ising TFIM Chain" in text
    assert "Hamiltonian Path / Cycle QUBO" in text
    assert "H2 Molecule" in text
    assert "LiH Molecule" in text
    assert "Generic Pauli Hamiltonian" in text
    assert "Do not hand-author the final architecture" in text
    assert "supernet_bootstrap_queue.csv" in text
