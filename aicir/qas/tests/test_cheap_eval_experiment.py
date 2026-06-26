import csv
import json


def test_experiment_schema_includes_required_proxy_fields():
    from aicir.qas.vqe_loop.cheap_eval_experiment import EXPERIMENT_FIELDS

    for field in [
        "problem_id",
        "sampling_mode",
        "depth",
        "architecture_id",
        "ansatz_gene",
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E5_mean",
        "E5_min",
        "E5_std",
        "fair_high",
        "fair_warm",
        "fair_random",
        "hit_rate",
        "exposure_count",
        "evaluation_order_index",
        "error_log",
    ]:
        assert field in EXPERIMENT_FIELDS


def test_write_empty_experiment_csv_creates_header(tmp_path):
    from aicir.qas.vqe_loop.cheap_eval_experiment import EXPERIMENT_FIELDS, write_empty_experiment_csv

    output = tmp_path / "experiment.csv"
    write_empty_experiment_csv(output)

    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    assert header == list(EXPERIMENT_FIELDS)


def test_write_experiment_manifest_records_e1_e5_protocol(tmp_path):
    from aicir.qas.vqe_loop.cheap_eval_experiment import CheapEvalExperimentConfig, write_experiment_manifest

    output = tmp_path / "manifest.json"
    config = CheapEvalExperimentConfig(
        benchmark_set=("h2_4q", "lih_12q", "ising8_lattice"),
        depths=(3, 4),
        n_architectures=40,
        sampling_mode="uniform",
        proxy_fields=("E1", "E2", "E3", "E4", "E5"),
        target_field="fair_high",
    )

    write_experiment_manifest(config, output)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded["benchmark_set"] == ["h2_4q", "lih_12q", "ising8_lattice"]
    assert loaded["depths"] == [3, 4]
    assert loaded["n_architectures"] == 40
    assert loaded["sampling_mode"] == "uniform"
    assert loaded["proxy_fields"] == ["E1", "E2", "E3", "E4", "E5"]
    assert loaded["target_field"] == "fair_high"
    assert loaded["notes"]["E1"] == "random init low-budget light VQE"
    assert loaded["notes"]["E5"] == "current native supernet screening"


def test_experiment_cli_writes_manifest_and_empty_csv(tmp_path):
    from aicir.qas.vqe_loop.cheap_eval_experiment import main

    manifest = tmp_path / "manifest.json"
    csv_path = tmp_path / "rows.csv"

    main(
        [
            "--manifest",
            str(manifest),
            "--output-csv",
            str(csv_path),
            "--benchmarks",
            "h2_4q,lih_12q,ising8_lattice",
            "--depths",
            "3,4",
            "--n-architectures",
            "40",
            "--sampling-mode",
            "uniform",
        ]
    )

    loaded = json.loads(manifest.read_text(encoding="utf-8"))
    assert loaded["proxy_fields"] == ["E1", "E2", "E3", "E4", "E5"]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert "fair_high" in header


def test_run_experiment_writes_rows_from_injected_evaluators(tmp_path):
    from aicir.qas.vqe_loop.cheap_eval_experiment import CheapEvalExperimentConfig, run_experiment

    config = CheapEvalExperimentConfig(
        benchmark_set=("h2_4q",),
        depths=(3,),
        n_architectures=2,
    )

    calls = []

    def sampler(received_config):
        assert received_config is config
        return (
            {
                "problem_id": "h2_4q",
                "hamiltonian_class": "molecular",
                "depth": 3,
                "architecture_id": "arch_a",
                "ansatz_gene": "{}",
                "n_qubits": 4,
                "n_params": 6,
                "two_q_count": 2,
            },
            {
                "problem_id": "h2_4q",
                "hamiltonian_class": "molecular",
                "depth": 3,
                "architecture_id": "arch_b",
                "ansatz_gene": "{}",
                "n_qubits": 4,
                "n_params": 7,
                "two_q_count": 3,
            },
            {
                "problem_id": "h2_4q",
                "hamiltonian_class": "molecular",
                "depth": 3,
                "architecture_id": "arch_c_over_limit",
                "ansatz_gene": "{}",
                "n_qubits": 4,
                "n_params": 8,
                "two_q_count": 4,
            },
        )

    def evaluator(field, value):
        def inner(row):
            calls.append((field, row["architecture_id"]))
            if isinstance(value, dict):
                return dict(value)
            return value

        return inner

    output = tmp_path / "rows.csv"
    run_experiment(
        config,
        output,
        evaluator_registry={
            "E1": evaluator("E1", -1.0),
            "E2": evaluator("E2", {"E2": -2.0, "hit_rate": 0.75}),
            "E3": evaluator("E3", -3.0),
            "E4": evaluator("E4", -4.0),
            "E5": evaluator("E5", {"E5": -5.0, "E5_mean": -4.8, "E5_min": -5.1, "E5_std": 0.2}),
        },
        architecture_sampler=sampler,
        fair_vqe_runner=lambda row: {
            "fair_high": -6.0,
            "fair_warm": -5.8,
            "fair_random": -5.2,
            "walltime_fair_high": 0.01,
        },
    )

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["architecture_id"] for row in rows] == ["arch_a", "arch_b"]
    assert rows[0]["sampling_mode"] == "uniform"
    assert rows[0]["evaluation_order_index"] == "0"
    assert rows[1]["evaluation_order_index"] == "1"
    assert rows[0]["E1"] == "-1.0"
    assert rows[0]["E2"] == "-2.0"
    assert rows[0]["E5_mean"] == "-4.8"
    assert rows[0]["fair_high"] == "-6.0"
    assert rows[0]["hit_rate"] == "0.75"
    assert rows[0]["walltime_E1"] != ""
    assert calls[:5] == [
        ("E1", "arch_a"),
        ("E2", "arch_a"),
        ("E3", "arch_a"),
        ("E4", "arch_a"),
        ("E5", "arch_a"),
    ]


def test_run_experiment_preserves_rows_when_evaluator_or_fair_runner_fails(tmp_path):
    from aicir.qas.vqe_loop.cheap_eval_experiment import CheapEvalExperimentConfig, run_experiment

    config = CheapEvalExperimentConfig(
        benchmark_set=("h2_4q",),
        depths=(3,),
        n_architectures=2,
        proxy_fields=("E1", "E3"),
    )
    flush_calls = []

    def sampler(_config):
        return (
            {"architecture_id": "arch_a", "depth": 3},
            {"architecture_id": "arch_b", "depth": 3},
        )

    def failing_e3(row):
        if row["architecture_id"] == "arch_a":
            raise RuntimeError("missing shared params for architecture arch_a")
        return {"E3": -3.0}

    def fair_runner(row):
        if row["architecture_id"] == "arch_b":
            raise ValueError("COBYLA failed to converge")
        return {"fair_high": -5.0}

    run_experiment(
        config,
        tmp_path / "rows.csv",
        evaluator_registry={
            "E1": lambda _row: -1.0,
            "E3": failing_e3,
        },
        architecture_sampler=sampler,
        fair_vqe_runner=fair_runner,
        flush_callback=lambda: flush_calls.append("flushed"),
    )

    with (tmp_path / "rows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["architecture_id"] for row in rows] == ["arch_a", "arch_b"]
    assert rows[0]["E1"] == "-1.0"
    assert rows[0]["E3"] == ""
    assert rows[0]["fair_high"] == "-5.0"
    assert "E3:RuntimeError" in rows[0]["error_log"]
    assert "missing shared params" in rows[0]["error_log"]
    assert rows[1]["E3"] == "-3.0"
    assert rows[1]["fair_high"] == ""
    assert "fair_high:ValueError" in rows[1]["error_log"]
    assert len(flush_calls) == 2


def test_run_experiment_evaluator_results_do_not_overwrite_sampler_identity(tmp_path):
    from aicir.qas.vqe_loop.cheap_eval_experiment import CheapEvalExperimentConfig, run_experiment

    config = CheapEvalExperimentConfig(
        benchmark_set=("h2_4q",),
        depths=(3,),
        n_architectures=1,
        proxy_fields=("E5",),
    )

    run_experiment(
        config,
        tmp_path / "rows.csv",
        evaluator_registry={
            "E5": lambda _row: {
                "E5": -1.5,
                "architecture_id": "wrong",
                "depth": 99,
                "hit_rate": 0.3,
            }
        },
        architecture_sampler=lambda _config: (
            {"architecture_id": "arch_a", "depth": 3, "hit_rate": 0.8},
        ),
        fair_vqe_runner=lambda _row: {
            "fair_high": -2.0,
            "architecture_id": "also_wrong",
        },
    )

    with (tmp_path / "rows.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert row["architecture_id"] == "arch_a"
    assert row["depth"] == "3"
    assert row["hit_rate"] == "0.8"
    assert row["E5"] == "-1.5"
    assert row["fair_high"] == "-2.0"


def test_run_experiment_requires_configured_proxy_evaluators(tmp_path):
    from aicir.qas.vqe_loop.cheap_eval_experiment import CheapEvalExperimentConfig, run_experiment

    config = CheapEvalExperimentConfig(
        benchmark_set=("h2_4q",),
        depths=(3,),
        n_architectures=1,
        proxy_fields=("E1", "E5"),
    )

    try:
        run_experiment(
            config,
            tmp_path / "rows.csv",
            evaluator_registry={"E1": lambda _row: -1.0},
            architecture_sampler=lambda _config: ({"architecture_id": "arch_a"},),
            fair_vqe_runner=lambda _row: {"fair_high": -2.0},
        )
    except KeyError as exc:
        assert "E5" in str(exc)
    else:
        raise AssertionError("missing configured proxy evaluator should fail before writing partial rows")


def test_light_vqe_registry_runs_e1_e2_and_fair_with_configured_budgets(tmp_path):
    from aicir.qas.primitives.ansatz import LayerwiseAnsatzGene
    from aicir.qas.problems.hamiltonians import VQEProblem
    from aicir.qas.vqe_loop.cheap_eval_experiment import (
        CheapEvalExperimentConfig,
        build_light_vqe_evaluator_registry,
        run_experiment,
    )

    problem = VQEProblem(
        name="two_qubit_z",
        n_qubits=2,
        hamiltonian=((-1.0, "ZI"),),
        reference_energy=-1.0,
    )
    gene = LayerwiseAnsatzGene(
        n_qubits=2,
        single_blocks=("ry", "ry"),
        edge_entanglers=(("cx",),),
        entangle_pattern="linear",
    )
    calls = []

    class FakeResult:
        def __init__(self, energy, budget):
            self.energy = energy
            self.best_parameters = [0.1, 0.2]
            self.evaluations = budget
            self.metadata = {"nfev": budget}

    def fake_optimizer(architecture, received_problem, **kwargs):
        calls.append(
            {
                "architecture": architecture,
                "problem": received_problem,
                "budget": kwargs["budget_override"],
                "seed": kwargs["seed"],
                "n_starts": kwargs["n_starts"],
            }
        )
        energy = {
            20: -1.2,
            250: -1.7,
            1000: -1.9,
        }[kwargs["budget_override"]]
        return FakeResult(energy, kwargs["budget_override"])

    registry, fair_runner = build_light_vqe_evaluator_registry(
        problem=problem,
        e1_max_evals=20,
        e2_max_evals=250,
        fair_max_evals=1000,
        seed=11,
        n_starts=1,
        optimizer=fake_optimizer,
    )
    config = CheapEvalExperimentConfig(
        benchmark_set=("two_qubit_z",),
        depths=(1,),
        n_architectures=1,
        proxy_fields=("E1", "E2"),
    )

    run_experiment(
        config,
        tmp_path / "rows.csv",
        evaluator_registry=registry,
        architecture_sampler=lambda _config: (
            {
                "problem_id": "two_qubit_z",
                "architecture_id": "arch_a",
                "ansatz_gene": json.dumps(gene.to_jsonable()),
                "n_qubits": 2,
            },
        ),
        fair_vqe_runner=fair_runner,
    )

    with (tmp_path / "rows.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert row["E1"] == "-1.2"
    assert row["E2"] == "-1.7"
    assert row["fair_high"] == "-1.9"
    assert row["n_params"] != ""
    assert row["two_q_count"] == "1"
    assert [call["budget"] for call in calls] == [20, 250, 1000]
    assert [call["seed"] for call in calls] == [11, 11, 1011]
    assert all(call["problem"] is problem for call in calls)


def test_light_vqe_registry_uses_same_proxy_seed_set_for_e1_and_e2(tmp_path):
    from aicir.qas.primitives.ansatz import LayerwiseAnsatzGene
    from aicir.qas.problems.hamiltonians import VQEProblem
    from aicir.qas.vqe_loop.cheap_eval_experiment import (
        CheapEvalExperimentConfig,
        build_light_vqe_evaluator_registry,
        run_experiment,
    )

    problem = VQEProblem(
        name="two_qubit_z",
        n_qubits=2,
        hamiltonian=((-1.0, "ZI"),),
        reference_energy=-1.0,
    )
    gene = LayerwiseAnsatzGene(
        n_qubits=2,
        single_blocks=("ry", "ry"),
        edge_entanglers=(("cx",),),
        entangle_pattern="linear",
    )
    calls = []

    class FakeResult:
        def __init__(self, energy):
            self.energy = energy
            self.best_parameters = []
            self.evaluations = 1
            self.metadata = {"nfev": 1}

    def fake_optimizer(_architecture, _problem, **kwargs):
        calls.append((kwargs["budget_override"], kwargs["seed"]))
        return FakeResult(energy=-(kwargs["budget_override"] + kwargs["seed"] / 1000.0))

    registry, fair_runner = build_light_vqe_evaluator_registry(
        problem=problem,
        e1_max_evals=20,
        e2_max_evals=250,
        fair_max_evals=1000,
        seed=100,
        proxy_seed_offsets=(0, 7),
        fair_seed_offsets=(1000,),
        optimizer=fake_optimizer,
    )

    run_experiment(
        CheapEvalExperimentConfig(
            benchmark_set=("two_qubit_z",),
            depths=(1,),
            n_architectures=1,
            proxy_fields=("E1", "E2"),
        ),
        tmp_path / "rows.csv",
        evaluator_registry=registry,
        architecture_sampler=lambda _config: (
            {
                "architecture_id": "arch_a",
                "ansatz_gene": json.dumps(gene.to_jsonable()),
                "n_qubits": 2,
            },
        ),
        fair_vqe_runner=fair_runner,
    )

    with (tmp_path / "rows.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert calls == [(20, 100), (20, 107), (250, 100), (250, 107), (1000, 1100)]
    assert row["E1"] == "-20.107"
    assert row["E2"] == "-250.107"


def test_light_vqe_registry_can_build_problem_from_literal_row_terms(tmp_path):
    from aicir.qas.primitives.ansatz import LayerwiseAnsatzGene
    from aicir.qas.vqe_loop.cheap_eval_experiment import (
        CheapEvalExperimentConfig,
        build_light_vqe_evaluator_registry,
        run_experiment,
    )

    gene = LayerwiseAnsatzGene(
        n_qubits=2,
        single_blocks=("ry", "ry"),
        edge_entanglers=(("none",),),
        entangle_pattern="linear",
    )
    seen_problem_names = []

    class FakeResult:
        energy = -0.5
        best_parameters = []
        evaluations = 3
        metadata = {"nfev": 3}

    def fake_optimizer(_architecture, problem, **_kwargs):
        seen_problem_names.append(problem.name)
        assert problem.n_qubits == 2
        assert tuple(problem.hamiltonian) == ((-1.0, "ZI"),)
        assert problem.reference_energy <= -1.0
        return FakeResult()

    registry, fair_runner = build_light_vqe_evaluator_registry(
        problem=None,
        e1_max_evals=3,
        e2_max_evals=4,
        fair_max_evals=5,
        optimizer=fake_optimizer,
    )

    run_experiment(
        CheapEvalExperimentConfig(
            benchmark_set=("row_problem",),
            depths=(1,),
            n_architectures=1,
            proxy_fields=("E1",),
        ),
        tmp_path / "rows.csv",
        evaluator_registry={"E1": registry["E1"]},
        architecture_sampler=lambda _config: (
            {
                "architecture_id": "arch_row_problem",
                "ansatz_gene": json.dumps(gene.to_jsonable()),
                "n_qubits": 2,
                "hamiltonian_terms": json.dumps([[-1.0, "ZI"]]),
                "hamiltonian_id": "row_problem",
            },
        ),
        fair_vqe_runner=fair_runner,
    )

    assert seen_problem_names == ["row_problem", "row_problem"]


def test_light_vqe_registry_reuses_cached_architecture_and_problem(tmp_path):
    from aicir.qas.primitives.ansatz import LayerwiseAnsatzGene
    from aicir.qas.problems.hamiltonians import VQEProblem
    from aicir.qas.vqe_loop.cheap_eval_experiment import (
        CheapEvalExperimentConfig,
        build_light_vqe_evaluator_registry,
        run_experiment,
    )

    problem = VQEProblem(
        name="two_qubit_z",
        n_qubits=2,
        hamiltonian=((-1.0, "ZI"),),
        reference_energy=-1.0,
    )
    gene = LayerwiseAnsatzGene(
        n_qubits=2,
        single_blocks=("ry", "ry"),
        edge_entanglers=(("cx",),),
        entangle_pattern="linear",
    )
    seen_architecture_ids = []
    seen_problem_ids = []

    class FakeResult:
        energy = -1.0
        best_parameters = []
        evaluations = 1
        metadata = {"nfev": 1}

    def fake_optimizer(architecture, received_problem, **_kwargs):
        seen_architecture_ids.append(id(architecture))
        seen_problem_ids.append(id(received_problem))
        return FakeResult()

    registry, fair_runner = build_light_vqe_evaluator_registry(
        problem=problem,
        e1_max_evals=1,
        e2_max_evals=2,
        fair_max_evals=3,
        optimizer=fake_optimizer,
    )

    run_experiment(
        CheapEvalExperimentConfig(
            benchmark_set=("two_qubit_z",),
            depths=(1,),
            n_architectures=1,
            proxy_fields=("E1", "E2"),
        ),
        tmp_path / "rows.csv",
        evaluator_registry=registry,
        architecture_sampler=lambda _config: (
            {
                "architecture_id": "arch_a",
                "ansatz_gene": json.dumps(gene.to_jsonable()),
                "n_qubits": 2,
            },
        ),
        fair_vqe_runner=fair_runner,
    )

    assert len(set(seen_architecture_ids)) == 1
    assert len(set(seen_problem_ids)) == 1


def test_light_vqe_registry_passes_initial_parameters_for_warm_start(tmp_path):
    from aicir.qas.primitives.ansatz import LayerwiseAnsatzGene
    from aicir.qas.problems.hamiltonians import VQEProblem
    from aicir.qas.vqe_loop.cheap_eval_experiment import (
        CheapEvalExperimentConfig,
        build_light_vqe_evaluator_registry,
        run_experiment,
    )

    problem = VQEProblem(
        name="two_qubit_z",
        n_qubits=2,
        hamiltonian=((-1.0, "ZI"),),
        reference_energy=-1.0,
    )
    gene = LayerwiseAnsatzGene(
        n_qubits=2,
        single_blocks=("ry", "ry"),
        edge_entanglers=(("cx",),),
        entangle_pattern="linear",
    )
    initial_parameters_seen = []

    class FakeResult:
        energy = -1.0
        best_parameters = []
        evaluations = 1
        metadata = {"nfev": 1}

    def fake_optimizer(_architecture, _problem, **kwargs):
        initial_parameters_seen.append(kwargs["initial_parameters"])
        assert kwargs["init_mode"] == "warm_start"
        return FakeResult()

    registry, fair_runner = build_light_vqe_evaluator_registry(
        problem=problem,
        e1_max_evals=1,
        e2_max_evals=2,
        fair_max_evals=3,
        optimizer=fake_optimizer,
        init_mode="warm_start",
        initial_parameters=lambda row: [float(row["warm_theta_0"]), 0.25],
    )

    run_experiment(
        CheapEvalExperimentConfig(
            benchmark_set=("two_qubit_z",),
            depths=(1,),
            n_architectures=1,
            proxy_fields=("E1",),
        ),
        tmp_path / "rows.csv",
        evaluator_registry={"E1": registry["E1"]},
        architecture_sampler=lambda _config: (
            {
                "architecture_id": "arch_a",
                "ansatz_gene": json.dumps(gene.to_jsonable()),
                "n_qubits": 2,
                "warm_theta_0": "0.125",
            },
        ),
        fair_vqe_runner=fair_runner,
    )

    assert initial_parameters_seen == [[0.125, 0.25], [0.125, 0.25]]
