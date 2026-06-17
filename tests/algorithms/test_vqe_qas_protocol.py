import json
from pathlib import Path

from aicir.qas.vqe_qas_protocol import (
    BENCHMARK_TABLE_FIELDS,
    DistanceScales,
    LabelSource,
    LabelStatus,
    ZeroCostStatus,
    CandidateRecord,
    append_benchmark_rows,
    assign_repaired_holdout_sources,
    compute_abstain_rate,
    derive_trust_region_d_max,
    extract_pauli_hamiltonian_features,
    fit_distance_scales,
    hamiltonian_feature_distance,
    next_label_status_after_failure,
    task_aware_composite_distance,
    select_ea_candidates,
    select_latest_ea_candidates,
    select_farthest_first,
    select_initial_label_batch,
    select_stage0_anchors,
    select_target_fewshot_batch,
    trust_region_geometry,
)
from aicir.qas.demos.vqe_qas_prepare_oracle import (
    _apply_zero_cost_stage1b,
    candidate_record_from_layerwise_gene,
    candidate_record_from_mask,
)
from aicir.qas.demos.vqe_qas_plan_target_fewshot import (
    _candidate_record_from_row_for_target,
    _queue_row_for_target,
    _terms_for_candidate_row,
)
from aicir.qas.demos.vqe_qas_plan_next_batch import (
    _supernet_priority,
    _attach_supernet_sidecar,
)
from aicir.qas.demos.vqe_qas_supernet_screen import _screening_sidecar_record
from aicir.qas.demos.vqe_qas_compare_transfer_oracle import compare_transfer_oracles
from aicir.qas.demos.vqe_qas_run_fair_labels import (
    _architecture_from_row,
    _label_row,
    _load_warm_start_vector,
    _load_protocol,
    _mask_from_row,
    _problem_from_row_or_protocol,
    _validate_queue_protocol_versions,
)
from aicir.qas.demos.vqe_qas_labels_to_supernet_sidecar import build_sidecar_records
from aicir.qas.vqe_hea_demo import (
    HEAMask,
    LayerwiseAnsatzGene,
    architecture_from_layerwise_gene,
    sample_layerwise_genes,
)
from aicir.qas.task_evaluation import parameter_count


def _candidate(index: int, family: str, entangler: str, topology: str, depth: str) -> CandidateRecord:
    return CandidateRecord(
        architecture_id=f"cand_{index}",
        canonical_arch_hash=f"hash_{index:03d}",
        family=family,
        entangler_type=entangler,
        topology=topology,
        depth_group=depth,
        n_params=10 + index,
        two_q_count=2 + (index % 5),
        hamiltonian_coverage=float(index % 7),
    )


def _pool() -> list[CandidateRecord]:
    families = ["hea", "realamplitudes", "qaoa_like"]
    entanglers = ["cx", "cz", "rzz"]
    topologies = ["linear", "ring"]
    depths = ["L1", "L2", "L3"]
    records = []
    index = 0
    for family in families:
        for entangler in entanglers:
            for topology in topologies:
                for depth in depths:
                    records.append(_candidate(index, family, entangler, topology, depth))
                    index += 1
    return records


def test_stage0_anchors_cover_nonempty_cells() -> None:
    records = _pool()
    anchors = select_stage0_anchors(records)
    cells = {(record.family, record.entangler_type, record.topology) for record in records}
    assert len(anchors) == len(cells)
    assert {(record.family, record.entangler_type, record.topology) for record in anchors} == cells


def test_initial_label_batch_includes_holdout_sources() -> None:
    labels = select_initial_label_batch(_pool(), total_labels=32, holdout_fraction=0.25)
    assert len(labels) == 32
    assert LabelSource.INITIAL_TRAIN in labels.values()
    assert LabelSource.HOLDOUT_ID in labels.values()
    assert LabelSource.HOLDOUT_BOUNDARY in labels.values()


def test_initial_label_batch_can_group_by_n_qubits() -> None:
    records = [
        _candidate(index, "hea", "cx", "linear", "L1")
        for index in range(18)
    ]
    records.extend(
        CandidateRecord(
            architecture_id=f"cand6_{index}",
            canonical_arch_hash=f"hash6_{index:03d}",
            family="hea",
            entangler_type="cz",
            topology="ring",
            depth_group="L2",
            n_params=20 + index,
            two_q_count=4,
            metadata={"n_qubits": 6},
        )
        for index in range(18)
    )
    labels = select_initial_label_batch(records, total_labels=12, holdout_fraction=0.25, group_key="n_qubits")
    assert len(labels) == 12
    by_prefix = {"cand_": 0, "cand6_": 0}
    for architecture_id in labels:
        if architecture_id.startswith("cand6_"):
            by_prefix["cand6_"] += 1
        else:
            by_prefix["cand_"] += 1
    assert by_prefix == {"cand_": 6, "cand6_": 6}


def test_small_initial_train_batch_covers_entanglers_before_fine_family() -> None:
    labels = select_initial_label_batch(_pool(), total_labels=12, holdout_fraction=0.25)
    train_ids = {
        architecture_id
        for architecture_id, source in labels.items()
        if source == LabelSource.INITIAL_TRAIN
    }
    records_by_id = {record.architecture_id: record for record in _pool()}
    train_entanglers = {records_by_id[architecture_id].entangler_type for architecture_id in train_ids}
    train_depths = {records_by_id[architecture_id].depth_group for architecture_id in train_ids}
    train_topologies = {records_by_id[architecture_id].topology for architecture_id in train_ids}

    assert train_entanglers == {"cx", "cz", "rzz"}
    assert train_depths == {"L1", "L2", "L3"}
    assert train_topologies == {"linear", "ring"}


def test_small_batch_can_lower_k_min_for_id_holdout_geometry() -> None:
    labels = select_initial_label_batch(
        _pool(),
        total_labels=12,
        holdout_fraction=0.25,
        trust_d_max=0.40,
        k_min=3,
    )
    assert LabelSource.HOLDOUT_ID in labels.values()


def test_repaired_holdout_sources_follow_train_geometry() -> None:
    train = [
        CandidateRecord(
            architecture_id=f"train_{index}",
            canonical_arch_hash=f"train_hash_{index}",
            family="hea",
            entangler_type="cx",
            topology="linear",
            depth_group="L1",
            n_params=10.0 + index * 0.1,
            two_q_count=3.0,
            hamiltonian_coverage=1.0,
        )
        for index in range(6)
    ]
    pool = [
        CandidateRecord(
            architecture_id=f"id_{index}",
            canonical_arch_hash=f"id_hash_{index}",
            family="hea",
            entangler_type="cx",
            topology="linear",
            depth_group="L1",
            n_params=10.05 + index * 0.1,
            two_q_count=3.0,
            hamiltonian_coverage=1.0,
        )
        for index in range(2)
    ]
    pool.extend(
        CandidateRecord(
            architecture_id=f"boundary_{index}",
            canonical_arch_hash=f"boundary_hash_{index}",
            family="hea",
            entangler_type="cx",
            topology="linear",
            depth_group="L1",
            n_params=210.0 + index,
            two_q_count=3.0,
            hamiltonian_coverage=1.0,
        )
        for index in range(2)
    )
    pool.extend(
        CandidateRecord(
            architecture_id=f"sparse_{index}",
            canonical_arch_hash=f"sparse_hash_{index}",
            family="qaoa_like",
            entangler_type="rzz",
            topology="ring",
            depth_group="L3",
            n_params=1000.0 + index,
            two_q_count=20.0,
            hamiltonian_coverage=0.0,
        )
        for index in range(2)
    )

    scales = DistanceScales(n_params=100.0, two_q_count=100.0, hamiltonian_coverage=100.0)
    labels = assign_repaired_holdout_sources(
        pool,
        train,
        scales,
        holdout_size=6,
        k_min=3,
        d_max=0.25,
    )

    assert {key for key, source in labels.items() if source == LabelSource.HOLDOUT_ID} == {"id_0", "id_1"}
    assert {key for key, source in labels.items() if source == LabelSource.HOLDOUT_BOUNDARY} == {
        "boundary_0",
        "boundary_1",
    }
    assert {key for key, source in labels.items() if source == LabelSource.HOLDOUT_SPARSE} == {
        "sparse_0",
        "sparse_1",
    }
    for candidate in pool:
        geometry = trust_region_geometry(candidate, train, scales, k_min=3, d_max=0.25)
        if labels[candidate.architecture_id] == LabelSource.HOLDOUT_ID:
            assert geometry["neighbor_count"] >= 3
            assert geometry["in_trust_region"]
        if labels[candidate.architecture_id] == LabelSource.HOLDOUT_SPARSE:
            assert geometry["neighbor_count"] < 3
            assert not geometry["in_trust_region"]


def test_derive_trust_region_d_max_uses_leave_one_out_kth_distance() -> None:
    records = [
        CandidateRecord(
            architecture_id=f"train_{index}",
            canonical_arch_hash=f"train_hash_{index}",
            family="hea",
            entangler_type="cx",
            topology="linear",
            depth_group="L1",
            n_params=float(index),
            two_q_count=3.0,
            hamiltonian_coverage=1.0,
        )
        for index in range(5)
    ]
    scales = DistanceScales(n_params=1.0, two_q_count=1.0, hamiltonian_coverage=1.0)

    assert derive_trust_region_d_max(records, scales, k_min=2, multiplier=1.0) == 0.125


def test_track_b_farthest_first_is_deterministic() -> None:
    records = _pool()
    scales = fit_distance_scales(records)
    labeled = records[:5]
    pool = records[5:]
    first = select_farthest_first(pool, labeled, scales, count=6)
    second = select_farthest_first(pool, labeled, scales, count=6)
    assert [record.architecture_id for record in first] == [record.architecture_id for record in second]
    assert len(first) == 6


def test_ea_candidates_rank_by_fitness_and_exclude_labeled() -> None:
    records = [
        CandidateRecord(
            architecture_id=f"cand_{index}",
            canonical_arch_hash=f"hash_{index}",
            family="hea",
            entangler_type="cx",
            topology="linear",
            depth_group="L1",
            n_params=float(index + 1),
            two_q_count=2.0,
            metadata={"hea_mask": [4, index, "ry", "cx", "ry", "linear"]},
        )
        for index in range(4)
    ]

    selected = select_ea_candidates(
        records,
        seeds=[records[0]],
        excluded_ids={"cand_0"},
        fitness=lambda candidate: float(candidate.metadata["hea_mask"][1]),
        population_size=4,
        generations=2,
        mutation_rate=0.0,
        crossover_rate=0.0,
        limit=2,
        random_seed=7,
    )

    assert [candidate.architecture_id for candidate in selected] == ["cand_3", "cand_2"]


def test_ea_candidates_fall_back_when_gene_metadata_is_missing() -> None:
    records = [_candidate(index, "hea", "cx", "linear", "L1") for index in range(4)]
    selected = select_ea_candidates(
        records,
        seeds=[],
        excluded_ids={"cand_3"},
        fitness=lambda candidate: -float(candidate.n_params),
        population_size=3,
        generations=1,
        limit=3,
        random_seed=5,
    )

    assert [candidate.architecture_id for candidate in selected] == ["cand_0", "cand_1", "cand_2"]


def test_latest_ea_uses_mog_vqe_nsga2_selection() -> None:
    records = [
        CandidateRecord(
            architecture_id=f"mog_{index}",
            canonical_arch_hash=f"mog_hash_{index}",
            family="hea",
            entangler_type="cx",
            topology="linear",
            depth_group="L1",
            n_params=4 + index,
            two_q_count=2,
            metadata={
                "mog_vqe_blocks": [
                    {"control": 0, "target": 1, "block_type": "generalized_cnot"},
                    {"control": 1, "target": 2, "block_type": "generalized_cnot"},
                ]
            },
        )
        for index in range(4)
    ]
    scores = {record.architecture_id: float(index) for index, record in enumerate(records)}

    selected, backend = select_latest_ea_candidates(
        records,
        seeds=[records[0]],
        excluded_ids={"mog_0"},
        fitness=lambda candidate: scores[candidate.architecture_id],
        population_size=4,
        generations=2,
        mutation_rate=0.5,
        crossover_rate=0.75,
        elite_count=1,
        limit=2,
        random_seed=7,
    )

    assert backend == "mog_vqe_nsga2_pool"
    assert [candidate.architecture_id for candidate in selected] == ["mog_3", "mog_2"]


def test_latest_ea_uses_mog_vqe_proxy_when_no_explicit_blocks_are_available() -> None:
    records = [_candidate(index, "hea", "cx", "linear", "L1") for index in range(4)]

    selected, backend = select_latest_ea_candidates(
        records,
        seeds=[],
        excluded_ids={"cand_3"},
        fitness=lambda candidate: -float(candidate.n_params),
        population_size=4,
        generations=2,
        limit=2,
        random_seed=5,
    )

    assert backend == "mog_vqe_nsga2_pool"
    assert [candidate.architecture_id for candidate in selected] == ["cand_0", "cand_1"]


def test_abstain_rate_uses_deduped_track_a_candidates() -> None:
    rows = [
        {"architecture_id": "a", "abstain": True},
        {"architecture_id": "a", "abstain": False},
        {"architecture_id": "b", "abstain": True},
        {"architecture_id": "c", "abstain": False},
    ]
    assert compute_abstain_rate(rows) == 1 / 3


def test_failure_status_respects_retry_limit() -> None:
    assert next_label_status_after_failure(retry_count=0) == LabelStatus.FAILED_RETRYABLE
    assert next_label_status_after_failure(retry_count=2) == LabelStatus.FAILED_NONRETRYABLE


def test_hamiltonian_features_capture_pauli_locality_and_mix() -> None:
    features = extract_pauli_hamiltonian_features([
        (-1.0, "ZZI"),
        (0.5, "XII"),
        (0.25, "IYI"),
    ])

    assert features["n_terms"] == 3
    assert features["n_qubits"] == 3
    assert features["locality_mean"] == 4 / 3
    assert features["one_body_fraction"] == 2 / 3
    assert features["two_body_fraction"] == 1 / 3
    assert features["x_fraction"] == 1 / 4
    assert features["y_fraction"] == 1 / 4
    assert features["z_fraction"] == 2 / 4


def test_hamiltonian_features_capture_coefficient_weighted_pauli_profile() -> None:
    source = extract_pauli_hamiltonian_features([
        (-1.0, "ZZII"),
        (-1.0, "IZZI"),
        (-1.0, "IIZZ"),
        (-0.5, "XIII"),
        (-0.5, "IXII"),
        (-0.5, "IIXI"),
        (-0.5, "IIIX"),
    ])
    target = extract_pauli_hamiltonian_features([
        (-0.8, "ZZII"),
        (-0.8, "IZZI"),
        (-0.8, "IIZZ"),
        (-0.8, "XIII"),
        (-0.8, "IXII"),
        (-0.8, "IIXI"),
        (-0.8, "IIIX"),
    ])

    assert source["x_coeff_l1"] == 2.0
    assert source["z_coeff_l1"] == 6.0
    assert source["x_coeff_fraction"] == 2.0 / 8.0
    assert source["z_coeff_fraction"] == 6.0 / 8.0
    assert source["one_body_x_coeff_l1"] == 2.0
    assert source["two_body_zz_coeff_l1"] == 3.0

    assert abs(target["x_coeff_l1"] - 3.2) < 1e-12
    assert abs(target["z_coeff_l1"] - 4.8) < 1e-12
    assert abs(target["x_coeff_fraction"] - 3.2 / 8.0) < 1e-12
    assert abs(target["z_coeff_fraction"] - 4.8 / 8.0) < 1e-12
    assert abs(target["one_body_x_coeff_l1"] - 3.2) < 1e-12
    assert abs(target["two_body_zz_coeff_l1"] - 2.4) < 1e-12

    assert hamiltonian_feature_distance(source, target) > 0.07


def test_hamiltonian_features_separate_two_body_pair_channels() -> None:
    features = extract_pauli_hamiltonian_features([
        (-1.0, "XXI"),
        (-2.0, "IYY"),
        (-3.0, "ZZI"),
        (-4.0, "XZI"),
        (-5.0, "XYZ"),
    ])

    assert features["two_body_xx_coeff_l1"] == 1.0
    assert features["two_body_yy_coeff_l1"] == 2.0
    assert features["two_body_zz_coeff_l1"] == 3.0
    assert features["two_body_xz_coeff_l1"] == 4.0
    assert features["two_body_mixed_coeff_l1"] == 0.0
    assert features["many_body_coeff_l1"] == 5.0
    assert features["two_body_zz_coeff_fraction"] == 3.0 / 10.0
    assert features["two_body_xz_coeff_fraction"] == 4.0 / 10.0


def test_hamiltonian_feature_distance_is_zero_for_identical_features() -> None:
    features = extract_pauli_hamiltonian_features([(-1.0, "ZZ"), (0.5, "XI")])
    shifted = extract_pauli_hamiltonian_features([(-1.0, "ZZ"), (0.5, "XI")])
    different = extract_pauli_hamiltonian_features([(-1.0, "XX"), (0.5, "YY")])

    assert hamiltonian_feature_distance(features, shifted) == 0.0
    assert hamiltonian_feature_distance(features, different) > 0.0


def test_task_aware_distance_separates_same_architecture_across_hamiltonians() -> None:
    old_features = extract_pauli_hamiltonian_features([(-1.0, "ZZ"), (0.5, "XI")])
    new_features = extract_pauli_hamiltonian_features([(-1.0, "XX"), (0.5, "YY")])
    old = CandidateRecord(
        architecture_id="old",
        canonical_arch_hash="same_arch",
        family="hea",
        entangler_type="cx",
        topology="linear",
        depth_group="L1",
        n_params=8,
        two_q_count=3,
        hamiltonian_id="old_h",
        hamiltonian_class="ising",
        hamiltonian_features=old_features,
    )
    new = CandidateRecord(
        architecture_id="new",
        canonical_arch_hash="same_arch",
        family="hea",
        entangler_type="cx",
        topology="linear",
        depth_group="L1",
        n_params=8,
        two_q_count=3,
        hamiltonian_id="new_h",
        hamiltonian_class="ising",
        hamiltonian_features=new_features,
    )
    scales = fit_distance_scales([old, new])

    assert task_aware_composite_distance(old, old, scales) == 0.0
    assert task_aware_composite_distance(old, new, scales) > 0.0


def test_target_fewshot_batch_selects_only_target_hamiltonian() -> None:
    records = []
    for hamiltonian_id in ["tfim_source", "tfim_target"]:
        for index in range(8):
            records.append(
                CandidateRecord(
                    architecture_id=f"{hamiltonian_id}_{index}",
                    canonical_arch_hash=f"{hamiltonian_id}_hash_{index}",
                    family="hea",
                    entangler_type=["cx", "cz"][index % 2],
                    topology=["linear", "ring"][index % 2],
                    depth_group=["L1", "L2"][index % 2],
                    n_params=10 + index,
                    two_q_count=2 + index % 3,
                    hamiltonian_id=hamiltonian_id,
                    metadata={"n_qubits": 4 if index < 4 else 6},
                )
            )

    selected = select_target_fewshot_batch(
        records,
        target_hamiltonian_id="tfim_target",
        total_labels=6,
        holdout_fraction=0.25,
        group_key="n_qubits",
    )

    assert len(selected) == 6
    assert all(architecture_id.startswith("tfim_target_") for architecture_id in selected)
    assert LabelSource.TARGET_FEWSHOT_TRAIN in selected.values()
    assert any(source.value.startswith("target_holdout_") for source in selected.values())


def test_target_fewshot_batch_forwards_small_batch_geometry_knobs() -> None:
    records = []
    for index, record in enumerate(_pool()):
        records.append(
            CandidateRecord(
                architecture_id=f"source_{index}",
                canonical_arch_hash=f"source_hash_{index}",
                family=record.family,
                entangler_type=record.entangler_type,
                topology=record.topology,
                depth_group=record.depth_group,
                n_params=record.n_params,
                two_q_count=record.two_q_count,
                hamiltonian_id="source_h",
            )
        )
        records.append(
            CandidateRecord(
                architecture_id=f"target_{index}",
                canonical_arch_hash=f"target_hash_{index}",
                family=record.family,
                entangler_type=record.entangler_type,
                topology=record.topology,
                depth_group=record.depth_group,
                n_params=record.n_params,
                two_q_count=record.two_q_count,
                hamiltonian_id="target_h",
            )
        )

    selected = select_target_fewshot_batch(
        records,
        target_hamiltonian_id="target_h",
        total_labels=12,
        holdout_fraction=0.25,
        trust_d_max=0.40,
        k_min=3,
    )

    assert selected
    assert all(architecture_id.startswith("target_") for architecture_id in selected)
    assert LabelSource.TARGET_HOLDOUT_ID in selected.values()


def test_target_fewshot_row_helpers_override_hamiltonian_fields() -> None:
    terms = [(-1.0, "ZZ"), (0.5, "XI")]
    features = extract_pauli_hamiltonian_features(terms)
    row = {
        "architecture_id": "4q_arch",
        "canonical_arch_hash": "hash",
        "family": "hea",
        "entangler_type": "cx",
        "topology": "linear",
        "depth_group": "L1",
        "n_params": "8",
        "two_q_count": "3",
        "n_qubits": "4",
        "hamiltonian_id": "old_h",
        "hamiltonian_class": "tfim",
        "hamiltonian_coverage": "0.6",
        "hea_mask": '[4, 1, "ry", "cx", "ry", "linear"]',
    }

    record = _candidate_record_from_row_for_target(
        row,
        target_hamiltonian_id="new_h",
        target_hamiltonian_class="ising_x",
        hamiltonian_features=features,
    )
    queued = _queue_row_for_target(
        row,
        source=LabelSource.TARGET_FEWSHOT_TRAIN,
        protocol_version="fair_vqe_protocol_v1",
        batch_id="target_batch",
        target_hamiltonian_id="new_h",
        target_hamiltonian_class="ising_x",
        hamiltonian_features=features,
        hamiltonian_terms=terms,
    )

    assert record.hamiltonian_id == "new_h"
    assert record.hamiltonian_class == "ising_x"
    assert record.hamiltonian_features["n_terms"] == 2
    assert queued["hamiltonian_id"] == "new_h"
    assert queued["hamiltonian_class"] == "ising_x"
    assert queued["source"] == LabelSource.TARGET_FEWSHOT_TRAIN.value
    assert "\"n_terms\": 2.0" in queued["hamiltonian_coverage_features"]
    assert queued["hamiltonian_terms"] == json.dumps(terms, ensure_ascii=False)


def test_target_fewshot_planner_can_generate_scaled_tfim_terms_per_row() -> None:
    class Args:
        target_tfim_J = 0.8
        target_tfim_h = 0.3
        target_tfim_boundary = "OBC"

    terms = _terms_for_candidate_row({"n_qubits": "4"}, Args(), [])

    assert len(terms) == 7
    assert all(len(pauli) == 4 for _coeff, pauli in terms)
    assert (-0.8, "ZZII") in terms
    assert (-0.3, "XIII") in terms


def test_benchmark_table_schema_keeps_literal_hamiltonian_terms() -> None:
    assert "hamiltonian_terms" in BENCHMARK_TABLE_FIELDS


def test_fair_label_runner_builds_target_problem_from_row_terms() -> None:
    protocol = {
        "protocol_version": "fair_vqe_protocol_v1",
        "frozen": True,
        "hamiltonian": {"J": 1.0, "h": 0.5, "boundary": "OBC"},
    }
    row = {
        "hamiltonian_id": "shifted_ising_2q",
        "hamiltonian_class": "custom_pauli",
        "hamiltonian_terms": json.dumps([[-0.25, "ZZ"], [-0.75, "XI"]]),
    }

    problem = _problem_from_row_or_protocol(row, n_qubits=2, protocol=protocol)

    assert problem.name == "shifted_ising_2q"
    assert problem.n_qubits == 2
    assert tuple(problem.hamiltonian) == ((-0.25, "ZZ"), (-0.75, "XI"))
    assert problem.reference_energy < 0


def _completed_label_row(
    architecture_id: str,
    *,
    hamiltonian_id: str,
    source: str,
    energy: float,
    family: str = "hea",
    entangler: str = "cx",
    topology: str = "linear",
    depth: str = "L1",
    features: str = "{}",
) -> dict[str, str]:
    return {
        "architecture_id": architecture_id,
        "canonical_arch_hash": architecture_id,
        "protocol_version": "fair_vqe_protocol_v1",
        "source": source,
        "label_status": LabelStatus.COMPLETED.value,
        "n_qubits": "4",
        "hamiltonian_id": hamiltonian_id,
        "hamiltonian_class": "tfim",
        "family": family,
        "depth_group": depth,
        "entangler_type": entangler,
        "topology": topology,
        "n_params": "8",
        "two_q_count": "3",
        "hamiltonian_coverage": "0.5",
        "hamiltonian_coverage_features": features,
        "fair_best_energy": str(energy),
    }


def test_compare_transfer_oracles_splits_source_target_and_holdout() -> None:
    target_features = '{"n_terms": 2, "x_fraction": 0.5, "z_fraction": 0.5}'
    source_features = '{"n_terms": 2, "x_fraction": 0.0, "z_fraction": 1.0}'
    rows = [
        _completed_label_row("source_a", hamiltonian_id="source_h", source=LabelSource.INITIAL_TRAIN.value, energy=-1.0, features=source_features),
        _completed_label_row("source_b", hamiltonian_id="source_h", source=LabelSource.INITIAL_TRAIN.value, energy=-1.2, entangler="cz", features=source_features),
        _completed_label_row("target_train", hamiltonian_id="target_h", source=LabelSource.TARGET_FEWSHOT_TRAIN.value, energy=-2.0, features=target_features),
        _completed_label_row("target_holdout", hamiltonian_id="target_h", source=LabelSource.TARGET_HOLDOUT_ID.value, energy=-2.1, features=target_features),
    ]

    report = compare_transfer_oracles(rows, target_hamiltonian_id="target_h", protocol_version="fair_vqe_protocol_v1", k_min=1)

    assert report["n_source_train"] == 2
    assert report["n_target_train"] == 1
    assert report["n_target_holdout"] == 1
    assert report["scenarios"]["target_only"]["n_train"] == 1
    assert report["scenarios"]["source_plus_target"]["n_train"] == 3
    assert report["scenarios"]["source_only"]["n_train"] == 2
    assert report["scenarios"]["target_only"]["mae"] == 0.10000000000000009


def test_compare_transfer_oracles_allows_same_architecture_across_hamiltonians() -> None:
    rows = [
        _completed_label_row(
            "same_arch",
            hamiltonian_id="source_h",
            source=LabelSource.INITIAL_TRAIN.value,
            energy=-1.0,
        ),
        _completed_label_row(
            "same_arch",
            hamiltonian_id="target_h",
            source=LabelSource.TARGET_HOLDOUT_ID.value,
            energy=-2.0,
        ),
    ]

    report = compare_transfer_oracles(
        rows,
        target_hamiltonian_id="target_h",
        protocol_version="fair_vqe_protocol_v1",
        k_min=1,
        d_max=1.0,
    )

    assert report["scenarios"]["source_only"]["predicted"] == 1
    assert report["scenarios"]["source_only"]["mae"] == 1.0


def test_append_benchmark_rows_replaces_same_task_pending_and_keeps_cross_hamiltonian() -> None:
    base = [
        {
            "architecture_id": "arch_a",
            "hamiltonian_id": "source_h",
            "protocol_version": "fair_vqe_protocol_v2",
            "label_status": LabelStatus.PENDING.value,
            "source": LabelSource.TRACKA_LOCAL.value,
        },
        {
            "architecture_id": "arch_a",
            "hamiltonian_id": "target_h",
            "protocol_version": "fair_vqe_protocol_v2",
            "label_status": LabelStatus.COMPLETED.value,
            "source": LabelSource.TARGET_FEWSHOT_TRAIN.value,
            "fair_best_energy": "-2.0",
        },
    ]
    incoming = [
        {
            "architecture_id": "arch_a",
            "hamiltonian_id": "source_h",
            "protocol_version": "fair_vqe_protocol_v2",
            "label_status": LabelStatus.COMPLETED.value,
            "source": LabelSource.TRACKA_LOCAL.value,
            "fair_best_energy": "-1.1",
        }
    ]

    merged = append_benchmark_rows(base, incoming)

    assert len(merged) == 2
    source_rows = [row for row in merged if row["hamiltonian_id"] == "source_h"]
    target_rows = [row for row in merged if row["hamiltonian_id"] == "target_h"]
    assert source_rows[0]["label_status"] == LabelStatus.COMPLETED.value
    assert source_rows[0]["fair_best_energy"] == "-1.1"
    assert target_rows[0]["fair_best_energy"] == "-2.0"


def test_supernet_screening_sidecar_separates_rank_params_and_energy() -> None:
    record = _screening_sidecar_record(
        rank=1,
        architecture_indices=((0, 1), (2, 3)),
        selected_supernet_id=0,
        shared_weight_score=-1.25,
        screening_energy=-1.5,
        init_params_path="params/best.json",
        cnot_count=2,
        two_qubit_count=3,
    )

    assert record["supernet_rank_score"] == -1.25
    assert record["screening_energy"] == -1.5
    assert record["supernet_init_params_ref"] == "params/best.json"
    assert record["supernet_rank_score"] != record["screening_energy"]


def test_plan_next_batch_attaches_supernet_sidecar_by_architecture_id() -> None:
    row = {"architecture_id": "arch_a", "family": "hea"}
    sidecar = {
        "arch_a": {
            "supernet_rank_score": -2.5,
            "supernet_init_params_ref": "arch_a_params.json",
            "screening_energy": -2.7,
            "screening_energy_is_final_label": False,
        }
    }

    enriched = _attach_supernet_sidecar(row, sidecar)

    assert enriched["supernet_rank_score"] == "-2.5"
    assert enriched["supernet_init_params_ref"] == "arch_a_params.json"
    assert enriched["screening_energy"] == "-2.7"
    assert enriched["screening_energy_is_final_label"] == "false"
    assert _supernet_priority(enriched) == -2.5
    assert _supernet_priority({"architecture_id": "plain"}) == 0.0


def test_load_warm_start_vector_reads_relative_param_reference(tmp_path: Path) -> None:
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"theta0": 0.1, "theta1": -0.2}), encoding="utf-8")
    row = {"supernet_init_params_ref": params_path.name}

    vector, status = _load_warm_start_vector(row, queue_path=tmp_path / "queue.csv", n_params=2)

    assert vector == [0.1, -0.2]
    assert status == "loaded"


def test_load_warm_start_vector_rejects_param_count_mismatch(tmp_path: Path) -> None:
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps([0.1, -0.2, 0.3]), encoding="utf-8")
    row = {"supernet_init_params_ref": params_path.name}

    vector, status = _load_warm_start_vector(row, queue_path=tmp_path / "queue.csv", n_params=2)

    assert vector is None
    assert status == "param_count_mismatch"


def test_layerwise_ansatz_gene_allows_different_single_and_two_qubit_gates() -> None:
    gene = LayerwiseAnsatzGene(
        n_qubits=4,
        single_blocks=("rx", "ry_rz", "rx_ry_rz", "ry"),
        edge_entanglers=(
            ("rxx", "cz", "none"),
            ("rzz", "cx", "rxx"),
            ("none", "rzz", "cz"),
        ),
        entangle_pattern="linear",
    )

    architecture = architecture_from_layerwise_gene(gene)
    gate_types = [gate["type"] for gate in architecture.circuit.gates]

    assert "rx" in gate_types
    assert "ry" in gate_types
    assert "rz" in gate_types
    assert "rxx" in gate_types
    assert "rzz" in gate_types
    assert "cx" in gate_types
    assert "cz" in gate_types
    assert parameter_count(architecture.circuit) > 16


def test_run_fair_labels_prefers_layerwise_gene_over_hea_mask() -> None:
    gene = LayerwiseAnsatzGene(
        n_qubits=4,
        single_blocks=("rx", "ry"),
        edge_entanglers=(("rxx", "none", "cz"),),
        entangle_pattern="linear",
    )
    row = {
        "architecture_id": "layerwise_demo",
        "ansatz_gene": json.dumps(gene.to_jsonable()),
        "hea_mask": json.dumps([4, 1, "ry", "cx", "ry", "linear"]),
    }

    architecture = _architecture_from_row(row)

    gate_types = [gate["type"] for gate in architecture.circuit.gates]
    assert architecture.name.startswith("layerwise_L1")
    assert "rxx" in gate_types
    assert "cx" not in gate_types


def test_run_fair_labels_treats_json_empty_ansatz_gene_as_missing() -> None:
    row = {
        "architecture_id": "hea_demo",
        "ansatz_gene": json.dumps(""),
        "hea_mask": json.dumps([4, 1, "ry", "cx", "ry", "linear"]),
    }

    architecture = _architecture_from_row(row)

    gate_types = [gate["type"] for gate in architecture.circuit.gates]
    assert architecture.name == "hea_mask_L1_ry_cx_linear_ry"
    assert "cx" in gate_types


def test_label_row_records_best_parameters_in_trace(tmp_path: Path) -> None:
    protocol = _load_protocol(Path(__file__).resolve().parents[2] / "aicir" / "qas" / "configs" / "fair_vqe_protocol_v2.json")
    row = {
        "architecture_id": "param_trace_demo",
        "n_qubits": "2",
        "hea_mask": json.dumps([2, 1, "ry", "cx", "ry", "linear"]),
        "hamiltonian_terms": json.dumps([[-1.0, "ZI"], [-0.5, "IZ"]]),
    }

    labeled = _label_row(
        row,
        protocol=protocol,
        seed=123,
        n_seeds=1,
        success_delta_ref=0.1,
        max_evals_override=2,
        backend_kind="numpy",
        dtype="complex128",
        queue_path=tmp_path / "queue.csv",
    )

    trace = json.loads(labeled["best_trace"])
    assert trace[0]["best_parameters"]
    assert len(trace[0]["best_parameters"]) == parameter_count(_architecture_from_row(row).circuit)


def test_labels_to_supernet_sidecar_exports_best_params(tmp_path: Path) -> None:
    rows = [
        {
            "architecture_id": "arch_a",
            "label_status": "completed",
            "fair_best_energy": "-1.25",
            "n_params": "2",
            "best_trace": json.dumps(
                [
                    {"energy": -1.0, "best_parameters": [0.1, 0.2]},
                    {"energy": -1.25, "best_parameters": [0.3, 0.4]},
                ]
            ),
        }
    ]

    sidecar = build_sidecar_records(rows, output_dir=tmp_path, run_id="seeded")

    assert sidecar["records"][0]["architecture_id"] == "arch_a"
    assert sidecar["records"][0]["supernet_rank_score"] == -1.25
    params_ref = sidecar["records"][0]["supernet_init_params_ref"]
    assert json.loads((tmp_path / params_ref).read_text(encoding="utf-8")) == [0.3, 0.4]


def test_sample_layerwise_genes_is_deterministic_and_varies_layers() -> None:
    first = sample_layerwise_genes(n_qubits=4, layers=3, count=4, seed=7)
    second = sample_layerwise_genes(n_qubits=4, layers=3, count=4, seed=7)

    assert [item.key() for item in first] == [item.key() for item in second]
    assert len({item.key() for item in first}) == 4
    assert any(len(set(item.single_blocks)) > 1 for item in first)
    assert any(len(set(gate for layer in item.edge_entanglers for gate in layer)) > 1 for item in first)


def test_prepare_oracle_candidate_record_from_hea_mask() -> None:
    record = candidate_record_from_mask(
        HEAMask(
            n_qubits=4,
            layers=2,
            rotation_block="ry_rz",
            entangler="cz",
            final_rotation="ry",
            entangle_pattern="linear",
        ),
        "tfim",
    )
    assert record.architecture_id == "4q_hea_mask_L2_ry_rz_cz_linear_ry"
    assert record.family == "ry_rz_cz_linear_L2"
    assert record.entangler_type == "cz"
    assert record.topology == "linear"
    assert record.depth_group == "L2"
    assert record.n_params > 0


def test_prepare_oracle_candidate_record_from_layerwise_gene() -> None:
    gene = LayerwiseAnsatzGene(
        n_qubits=4,
        single_blocks=("rx", "ry_rz"),
        edge_entanglers=(("rxx", "cz", "rzz"),),
        entangle_pattern="linear",
    )

    record = candidate_record_from_layerwise_gene(gene, "molecular_h2")

    assert record.family == "layerwise_gene"
    assert record.entangler_type == "mixed_edge"
    assert record.depth_group == "L1"
    assert record.metadata["ansatz_gene"]["single_blocks"] == ["rx", "ry_rz"]
    assert record.n_params > 0
    assert record.two_q_count == 3


def test_fair_label_runner_reads_json_hea_mask() -> None:
    mask = _mask_from_row({"hea_mask": '[4, 2, "ry_rz", "cz", "ry", "linear"]'})
    assert mask == HEAMask(
        n_qubits=4,
        layers=2,
        rotation_block="ry_rz",
        entangler="cz",
        final_rotation="ry",
        entangle_pattern="linear",
    )


def test_fair_label_protocol_v2_is_explicitly_frozen_and_unmeasured() -> None:
    protocol = _load_protocol(Path(__file__).resolve().parents[2] / "aicir" / "qas" / "configs" / "fair_vqe_protocol_v2.json")
    assert protocol["frozen"] is True
    assert protocol["protocol_version"] == "fair_vqe_protocol_v2"
    assert protocol["energy_evaluation"]["state"] == "unmeasured_state"
    assert protocol["energy_evaluation"]["shots"] is None


def test_fair_label_runner_rejects_legacy_v1_protocol_after_energy_fix() -> None:
    try:
        _load_protocol(Path(__file__).resolve().parents[2] / "aicir" / "qas" / "configs" / "fair_vqe_protocol_v1.json")
    except ValueError as exc:
        assert "fair_vqe_protocol_v2" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("legacy v1 labels must not be generated after the energy fix")


def test_fair_label_runner_rejects_blank_queue_protocol_version() -> None:
    try:
        _validate_queue_protocol_versions([{"architecture_id": "a", "protocol_version": ""}], "fair_vqe_protocol_v1")
    except ValueError as exc:
        assert "protocol_version" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("blank protocol_version should fail")


def test_stage1b_zero_cost_filter_keeps_most_structural_candidates() -> None:
    records = [
        candidate_record_from_mask(mask, "tfim")
        for mask in [
            HEAMask(n_qubits=4, layers=1, rotation_block="ry", entangler="cx", final_rotation="ry", entangle_pattern="linear"),
            HEAMask(n_qubits=4, layers=1, rotation_block="ry_rz", entangler="cz", final_rotation="ry", entangle_pattern="ring"),
            HEAMask(n_qubits=4, layers=2, rotation_block="rx_ry_rz", entangler="rzz", final_rotation="ry_rz", entangle_pattern="linear"),
            HEAMask(n_qubits=4, layers=3, rotation_block="rx_ry_rz", entangler="rzz", final_rotation="ry_rz", entangle_pattern="ring"),
        ]
    ]
    annotated = _apply_zero_cost_stage1b(
        records,
        n_samples=4,
        trainability_soft_quantile=0.1,
        expressibility_soft_quantile=0.05,
        trainability_hard_floor=0.01,
        expressibility_hard_floor=0.01,
        entanglement_soft_floor=0.25,
        max_params=None,
        max_two_q=None,
        expressibility_metric="structural_proxy",
        trainability_metric="structure_proxy",
    )
    statuses = [record.metadata["zero_cost_status"] for record in annotated]
    assert ZeroCostStatus.HARD_REJECT.value not in statuses
    assert statuses.count(ZeroCostStatus.PASS.value) >= 2
    assert all(record.metadata["zero_cost_feature_score"] != "" for record in annotated)
    assert all(record.metadata["zero_cost_score_is_ranking_signal"] == "false" for record in annotated)


def test_stage1b_zero_cost_filter_hard_rejects_complexity_cap_only() -> None:
    record = candidate_record_from_mask(
        HEAMask(n_qubits=4, layers=3, rotation_block="rx_ry_rz", entangler="rzz", final_rotation="ry_rz", entangle_pattern="ring"),
        "tfim",
    )
    annotated = _apply_zero_cost_stage1b(
        [record],
        n_samples=4,
        trainability_soft_quantile=0.1,
        expressibility_soft_quantile=0.05,
        trainability_hard_floor=0.01,
        expressibility_hard_floor=0.01,
        entanglement_soft_floor=0.25,
        max_params=1,
        max_two_q=None,
        expressibility_metric="structural_proxy",
        trainability_metric="structure_proxy",
    )
    assert annotated[0].metadata["zero_cost_status"] == ZeroCostStatus.HARD_REJECT.value
    assert "params_over_cap" in annotated[0].metadata["zero_cost_reasons"]
