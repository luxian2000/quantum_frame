"""Native supernet cheap-ranking expansion for VQE-QAS rounds.

This module keeps the supernet algorithm as the source of truth: it asks
``aicir.qas.algorithms.supernet.Supernet`` to train shared weights and rank its
own sampled architecture pool, then converts the top ranked architectures into
vqe_loop queue-compatible rows for fair VQE labeling.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from aicir.metrics.circuit_structure import (
    entanglement_coverage_score,
    parameter_count,
    structural_expressibility_proxy_score,
)
from aicir.metrics.hardware import native_depth_twoq_efficiency
from aicir.metrics.trainability import structure_proxy
from aicir.qas.core.reward import RewardWeights
from aicir.qas.primitives.ansatz import SupernetAnsatzGene, architecture_from_supernet_gene
from aicir.qas.problems.hamiltonians import VQEProblem
from aicir.qas.vqe_loop.benchmark_table import parse_pauli_hamiltonian_terms
from aicir.qas.vqe_loop.benchmark_table import hamiltonian_from_terms
from aicir.qas.vqe_loop.benchmark_table import BENCHMARK_TABLE_FIELDS, ZeroCostStatus

if TYPE_CHECKING:
    from aicir.qas.algorithms.supernet import Architecture


PauliTerm = tuple[float, str]


def _to_hamiltonian(terms: Sequence[PauliTerm]):
    parsed = parse_pauli_hamiltonian_terms(terms)
    if not parsed:
        raise ValueError("supernet native ranking requires non-empty Hamiltonian terms")
    return hamiltonian_from_terms(parsed, n_qubits=len(parsed[0][1]))


def _default_two_qubit_pairs(n_qubits: int) -> tuple[tuple[int, int], ...]:
    from aicir.qas.algorithms.supernet import _default_two_qubit_pairs as supernet_default_two_qubit_pairs

    return tuple((int(left), int(right)) for left, right in supernet_default_two_qubit_pairs(int(n_qubits)))


def _default_single_qubit_gates(n_qubits: int) -> tuple[str, ...]:
    from aicir.qas.algorithms.supernet import SupernetConfig

    return tuple(SupernetConfig(n_qubits=int(n_qubits)).single_qubit_gates)


def gene_from_supernet_architecture(
    architecture: Architecture,
    *,
    n_qubits: int,
    two_qubit_pairs: Sequence[tuple[int, int]],
) -> SupernetAnsatzGene:
    return SupernetAnsatzGene(
        n_qubits=int(n_qubits),
        single_qubit_layers=tuple(tuple(layer.single_qubit_gates) for layer in architecture.layers),
        two_qubit_layers=tuple(tuple(layer.two_qubit_choices) for layer in architecture.layers),
        two_qubit_pairs=tuple((int(left), int(right)) for left, right in two_qubit_pairs),
    )


def _safe_stem(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("._") or "architecture"


def _float_parameter(value: Any) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().cpu().reshape(()))
    return float(value)


def _parameter_vector_from_circuit(circuit: Any) -> list[float]:
    vector: list[float] = []
    for gate in circuit.gates:
        if "parameter" not in gate or gate.get("parameter") is None:
            continue
        parameter = gate["parameter"]
        if isinstance(parameter, (list, tuple)):
            vector.extend(_float_parameter(value) for value in parameter)
        else:
            vector.append(_float_parameter(parameter))
    return vector


def _training_free_scores(circuit: Any, *, n_params: int, n_qubits: int, layers: int, two_q_count: int) -> dict[str, float]:
    entanglement = entanglement_coverage_score(
        two_q_count=float(two_q_count),
        n_qubits=int(n_qubits),
        layers=int(layers),
        topology="supernet_pairs",
    )
    expressibility = structural_expressibility_proxy_score(
        n_params=float(n_params),
        n_qubits=int(n_qubits),
        layers=int(layers),
        rotation_block="mixed_supernet",
        final_rotation="mixed_supernet",
        entanglement_score=float(entanglement),
    )
    trainability = structure_proxy(circuit)
    hardware = native_depth_twoq_efficiency(circuit)
    weights = RewardWeights()
    weighted = (
        weights.expressibility * float(expressibility)
        + weights.trainability * float(trainability)
        + weights.noise_robustness * float(entanglement)
        + weights.hardware_efficiency * float(hardware)
    )
    return {
        "expressibility": float(expressibility),
        "trainability": float(trainability),
        "entanglement": float(entanglement),
        "weighted": float(weighted),
    }


def _row_from_rank_record(
    record: dict[str, Any],
    *,
    n_qubits: int,
    two_qubit_pairs: Sequence[tuple[int, int]],
    hamiltonian_id: str,
    hamiltonian_class: str,
    supernet_init_params_ref: str = "",
    screening_energy: float | None = None,
) -> dict[str, Any]:
    gene = gene_from_supernet_architecture(
        record["architecture"],
        n_qubits=n_qubits,
        two_qubit_pairs=two_qubit_pairs,
    )
    spec = architecture_from_supernet_gene(gene)
    gene_payload = gene.to_jsonable()
    architecture_id = f"{int(n_qubits)}q_{spec.name}"
    n_params = int(parameter_count(spec.circuit))
    two_q_count = int(record.get("two_qubit_count", 0))
    training_free = _training_free_scores(
        spec.circuit,
        n_params=n_params,
        n_qubits=int(n_qubits),
        layers=int(gene.layers),
        two_q_count=two_q_count,
    )
    row = {field: "" for field in BENCHMARK_TABLE_FIELDS}
    row.update(
        {
            "architecture_id": architecture_id,
            "canonical_arch_hash": json.dumps(gene_payload, ensure_ascii=False, sort_keys=True),
            "source": "trackB_supernet",
            "n_qubits": int(n_qubits),
            "hamiltonian_id": str(hamiltonian_id),
            "hamiltonian_class": str(hamiltonian_class),
            "family": "supernet_native",
            "entangler_type": "mixed_supernet",
            "topology": "supernet_pairs",
            "depth_group": f"L{gene.layers}",
            "n_params": n_params,
            "two_q_count": two_q_count,
            "hamiltonian_coverage": "1.000000",
            "hamiltonian_coverage_features": "1.000000",
            "zero_cost_status": ZeroCostStatus.PASS.value,
            "zero_cost_reasons": "",
            "expressibility_score": f"{training_free['expressibility']:.12f}",
            "trainability_score": f"{training_free['trainability']:.12f}",
            "entanglement_score": f"{training_free['entanglement']:.12f}",
            "zero_cost_feature_score": f"{training_free['weighted']:.12f}",
            "zero_cost_score_is_ranking_signal": "false",
            "ansatz_gene": json.dumps(gene_payload, ensure_ascii=False),
            "supernet_rank_score": f"{float(record['score']):.12f}",
            "supernet_init_params_ref": str(supernet_init_params_ref),
            "screening_energy": f"{float(screening_energy if screening_energy is not None else record['score']):.12f}",
            "screening_energy_is_final_label": "false",
            "supernet_warm_start_status": "ready" if supernet_init_params_ref else "missing",
        }
    )
    return row


def _supernet_gene_from_candidate_row(row: Mapping[str, Any]) -> SupernetAnsatzGene:
    cached = row.get("_cached_supernet_gene")
    if isinstance(cached, SupernetAnsatzGene):
        return cached
    raw_gene = row.get("ansatz_gene", "")
    if raw_gene is None or str(raw_gene).strip() in {"", '""', "null"}:
        raise ValueError("E5 native supernet screening requires ansatz_gene")
    parsed = json.loads(raw_gene) if isinstance(raw_gene, str) else raw_gene
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    if not isinstance(parsed, Mapping):
        raise ValueError("ansatz_gene must decode to a JSON object")
    if str(parsed.get("kind", "")).lower() != "supernet_native":
        raise ValueError("E5 native supernet screening requires a supernet_native ansatz_gene")
    gene = SupernetAnsatzGene.from_jsonable(parsed)
    if isinstance(row, dict):
        row["_cached_supernet_gene"] = gene
    return gene


def _supernet_architecture_from_gene(gene: SupernetAnsatzGene, *, native_classes: bool = True) -> Any:
    if native_classes:
        from aicir.qas.algorithms.supernet import Architecture, LayerArchitecture

        layers = tuple(
            LayerArchitecture(single_qubit_gates=single_layer, two_qubit_choices=two_layer)
            for single_layer, two_layer in zip(gene.single_qubit_layers, gene.two_qubit_layers)
        )
        return Architecture(layers=layers)
    layers = tuple(
        SimpleNamespace(single_qubit_gates=single_layer, two_qubit_choices=two_layer)
        for single_layer, two_layer in zip(gene.single_qubit_layers, gene.two_qubit_layers)
    )
    return SimpleNamespace(layers=layers)


def _hamiltonian_terms_from_candidate_row(
    row: Mapping[str, Any],
    default_problem: VQEProblem | None,
) -> tuple[PauliTerm, ...]:
    raw_terms = row.get("hamiltonian_terms", "")
    if raw_terms is not None and str(raw_terms).strip() != "":
        terms = parse_pauli_hamiltonian_terms(json.loads(str(raw_terms)))
    elif default_problem is not None:
        terms = parse_pauli_hamiltonian_terms(default_problem.hamiltonian)
    else:
        raise ValueError("E5 native supernet screening requires hamiltonian_terms when no default problem is provided")
    if not terms:
        raise ValueError("E5 native supernet screening requires non-empty Hamiltonian terms")
    return tuple((float(coeff), str(pauli)) for coeff, pauli in terms)


def _mean_min_std(values: Sequence[float]) -> tuple[float, float, float]:
    if not values:
        raise ValueError("statistics require at least one value")
    mean = sum(values) / float(len(values))
    variance = sum((value - mean) ** 2 for value in values) / float(len(values))
    return float(mean), float(min(values)), float(math.sqrt(variance))


def _supernet_gene_static_counts(gene: SupernetAnsatzGene) -> tuple[int, int]:
    parameterized_single = {"rx", "ry", "rz"}
    parameterized_two = {"rzz"}
    n_params = 0
    two_q_count = 0
    for layer in gene.single_qubit_layers:
        n_params += sum(1 for gate in layer if str(gate).lower() in parameterized_single)
    for layer in gene.two_qubit_layers:
        for gate in layer:
            normalized = str(gate).lower()
            if normalized != "none":
                two_q_count += 1
            if normalized in parameterized_two:
                n_params += 1
    return int(n_params), int(two_q_count)


def build_native_supernet_e5_evaluator(
    *,
    problem: VQEProblem | None,
    supernet_num: int = 5,
    supernet_steps: int = 250,
    finetune_steps: int = 250,
    ranking_num: int = 80,
    seed: int = 2,
    device: str = "cpu",
    learning_rate: float = 0.1,
    finetune_learning_rate: float = 0.05,
    single_qubit_gates: Sequence[str] = ("i", "h", "rx", "ry", "rz"),
    two_qubit_gates: Sequence[str] = ("cx", "rzz"),
    use_parameter_shift: bool = False,
    supernet_factory: Callable[[Any], Any] | None = None,
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """Build the E5 evaluator: native supernet screening for explicit rows."""

    real_supernet_factory = supernet_factory is None
    factory = supernet_factory
    trained_supernets: dict[str, Any] = {}
    hamiltonian_cache: dict[str, Hamiltonian] = {}

    def _rank_row(row: Mapping[str, Any]) -> tuple[SupernetAnsatzGene, Any, Hamiltonian, Any, dict[str, Any], list[float]]:
        gene = _supernet_gene_from_candidate_row(row)
        candidate = _supernet_architecture_from_gene(gene, native_classes=real_supernet_factory)
        terms = _hamiltonian_terms_from_candidate_row(row, problem)
        hamiltonian_key = json.dumps(
            {"n_qubits": int(gene.n_qubits), "terms": [[coeff, pauli] for coeff, pauli in terms]},
            sort_keys=True,
        )
        hamiltonian = hamiltonian_cache.get(hamiltonian_key)
        if hamiltonian is None:
            hamiltonian = _to_hamiltonian(terms)
            hamiltonian_cache[hamiltonian_key] = hamiltonian

        config_payload = {
            "n_qubits": int(gene.n_qubits),
            "layers": int(gene.layers),
            "single_qubit_gates": tuple(str(gate).lower() for gate in single_qubit_gates),
            "two_qubit_gates": tuple(str(gate).lower() for gate in two_qubit_gates),
            "two_qubit_pairs": tuple((int(left), int(right)) for left, right in gene.two_qubit_pairs),
            "supernet_num": int(supernet_num),
            "supernet_steps": int(supernet_steps),
            "ranking_num": int(ranking_num),
            "finetune_steps": int(finetune_steps),
            "learning_rate": float(learning_rate),
            "finetune_learning_rate": float(finetune_learning_rate),
            "seed": int(seed),
            "device": str(device),
            "task": "vqe",
            "use_parameter_shift": bool(use_parameter_shift),
        }
        cache_key = json.dumps({**config_payload, "hamiltonian": [[coeff, pauli] for coeff, pauli in terms]}, sort_keys=True)
        supernet = trained_supernets.get(cache_key)
        if supernet is None:
            if real_supernet_factory:
                from aicir.qas.algorithms.supernet import Supernet, SupernetConfig

                config = SupernetConfig(**config_payload)
                supernet = Supernet(config)
            else:
                config = SimpleNamespace(**config_payload)
                supernet = factory(config)
            supernet.optimize_supernet(objective=None, dataset=None, hamiltonian=hamiltonian)
            trained_supernets[cache_key] = supernet

        records = supernet.rank_architectures(
            objective=None,
            dataset=None,
            hamiltonian=hamiltonian,
            candidates=[candidate],
            split="train",
        )
        if not records:
            raise RuntimeError("native supernet E5 ranking returned no records")
        record = records[0]
        losses = [float(value) for value in record.get("candidate_losses", [])]
        if not losses:
            losses = [float(record["score"])]
        return gene, candidate, hamiltonian, supernet, record, losses

    def warm_start_parameters(row: Mapping[str, Any]) -> list[float]:
        gene, candidate, _hamiltonian, supernet, record, _losses = _rank_row(row)
        selected_supernet_id = int(record.get("selected_supernet_id", 0))
        circuit, _active_keys, _active_tensors = supernet.build_circuit(candidate, supernet_id=selected_supernet_id)
        vector = _parameter_vector_from_circuit(circuit)
        expected, _two_q_count = _supernet_gene_static_counts(gene)
        if len(vector) != expected:
            raise ValueError(f"supernet warm-start parameter count mismatch: expected {expected}, got {len(vector)}")
        return vector

    def evaluator(row: Mapping[str, Any]) -> Mapping[str, Any]:
        gene, candidate, _hamiltonian, supernet, record, losses = _rank_row(row)
        loss_mean, loss_min, loss_std = _mean_min_std(losses)
        selected_supernet_id = int(record.get("selected_supernet_id", 0))
        screening_energy = float(record["score"])
        if int(finetune_steps) > 0:
            _circuit, _params, _log, screening_energy = supernet.finetune_architecture(
                candidate,
                selected_supernet_id,
                objective=None,
                dataset=None,
                hamiltonian=_hamiltonian,
            )
            screening_energy = float(screening_energy)
        n_params, two_q_count = _supernet_gene_static_counts(gene)
        return {
            "E5": screening_energy,
            "E5_mean": loss_mean,
            "E5_min": loss_min,
            "E5_std": loss_std,
            "n_qubits": int(gene.n_qubits),
            "n_params": n_params,
            "two_q_count": int(record.get("two_qubit_count", two_q_count)),
            "exposure_count": len(losses),
        }

    setattr(evaluator, "warm_start_parameters", warm_start_parameters)
    return evaluator

def build_supernet_native_rows(
    *,
    hamiltonian_terms: Sequence[PauliTerm],
    hamiltonian_id: str,
    hamiltonian_class: str,
    count: int,
    layers: int = 3,
    supernet_num: int = 2,
    supernet_steps: int = 20,
    ranking_num: int = 24,
    finetune_steps: int = 0,
    seed: int = 11,
    device: str = "cpu",
    single_qubit_gates: Sequence[str] | None = None,
    excluded_ids: set[str] | None = None,
    params_dir: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if int(count) <= 0:
        return [], {"enabled": False, "generated_rows": 0}
    from aicir.qas.algorithms.supernet import Supernet, SupernetConfig

    hamiltonian = _to_hamiltonian(hamiltonian_terms)
    n_qubits = int(hamiltonian.n_qubits)
    two_qubit_pairs = _default_two_qubit_pairs(n_qubits)
    single_qubit_gates = tuple(single_qubit_gates or _default_single_qubit_gates(n_qubits))
    config = SupernetConfig(
        n_qubits=n_qubits,
        layers=int(layers),
        single_qubit_gates=single_qubit_gates,
        two_qubit_pairs=two_qubit_pairs,
        supernet_num=int(supernet_num),
        supernet_steps=int(supernet_steps),
        ranking_num=max(int(ranking_num), int(count)),
        finetune_steps=max(0, int(finetune_steps)),
        seed=int(seed),
        device=str(device),
        task="vqe",
    )
    supernet = Supernet(config)
    supernet.optimize_supernet(None, hamiltonian=hamiltonian)
    ranking_records = supernet.rank_architectures(
        None,
        hamiltonian=hamiltonian,
        split="train",
    )

    rows: list[dict[str, Any]] = []
    seen = set(excluded_ids or set())
    params_root = Path(params_dir) if params_dir is not None else None
    if params_root is not None:
        params_root.mkdir(parents=True, exist_ok=True)
    screened_records: list[dict[str, Any]] = []
    for record in ranking_records:
        gene = gene_from_supernet_architecture(
            record["architecture"],
            n_qubits=n_qubits,
            two_qubit_pairs=config.two_qubit_pairs,
        )
        preview_spec = architecture_from_supernet_gene(gene)
        architecture_id = f"{n_qubits}q_{preview_spec.name}"
        if architecture_id in seen:
            continue
        screening_energy = float(record["score"])
        preview_param_count = int(parameter_count(preview_spec.circuit))
        vector: list[float] = []
        if params_root is not None:
            if preview_param_count > 0:
                circuit, _params, _log, screening_energy = supernet.finetune_architecture(
                    record["architecture"],
                    int(record["selected_supernet_id"]),
                    None,
                    hamiltonian=hamiltonian,
                )
                vector = _parameter_vector_from_circuit(circuit)
            if len(vector) != preview_param_count:
                raise ValueError(
                    f"supernet warm-start vector length {len(vector)} does not match "
                    f"vqe_loop circuit parameter count {preview_param_count}"
                )
        screened_records.append(
            {
                "record": record,
                "architecture_id": architecture_id,
                "screening_energy": float(screening_energy),
                "parameter_vector": vector,
            }
        )
        seen.add(architecture_id)

    selected_records = sorted(screened_records, key=lambda item: float(item["screening_energy"]))[: int(count)]
    for selected in selected_records:
        record = selected["record"]
        params_ref = ""
        if params_root is not None:
            params_name = f"supernet_native_{len(rows) + 1:03d}_{_safe_stem(selected['architecture_id'])}_params.json"
            (params_root / params_name).write_text(
                json.dumps(selected["parameter_vector"], indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            params_ref = params_name
        row = _row_from_rank_record(
            record,
            n_qubits=n_qubits,
            two_qubit_pairs=config.two_qubit_pairs,
            hamiltonian_id=hamiltonian_id,
            hamiltonian_class=hamiltonian_class,
            supernet_init_params_ref=params_ref,
            screening_energy=float(selected["screening_energy"]),
        )
        rows.append(row)
    selected_ids = {str(row.get("architecture_id", "")) for row in rows}
    random_baseline_row: dict[str, Any] | None = None
    for baseline_item in screened_records:
        candidate_id = str(baseline_item["architecture_id"])
        if candidate_id in selected_ids and len(screened_records) > len(selected_ids):
            continue
        params_ref = ""
        if params_root is not None:
            params_name = f"supernet_native_random_{_safe_stem(candidate_id)}_params.json"
            (params_root / params_name).write_text(
                json.dumps(baseline_item["parameter_vector"], indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            params_ref = params_name
        random_baseline_row = _row_from_rank_record(
            baseline_item["record"],
            n_qubits=n_qubits,
            two_qubit_pairs=config.two_qubit_pairs,
            hamiltonian_id=hamiltonian_id,
            hamiltonian_class=hamiltonian_class,
            supernet_init_params_ref=params_ref,
            screening_energy=float(baseline_item["screening_energy"]),
        )
        random_baseline_row["source"] = "trackB_supernet_random"
        break
    summary = {
        "enabled": True,
        "generated_rows": len(rows),
        "count": int(count),
        "layers": int(layers),
        "supernet_num": int(supernet_num),
        "supernet_steps": int(supernet_steps),
        "ranking_num": max(int(ranking_num), int(count)),
        "finetune_steps": max(0, int(finetune_steps)),
        "single_qubit_gates": list(single_qubit_gates),
        "two_qubit_pair_count": len(two_qubit_pairs),
        "two_qubit_pairs": [[int(left), int(right)] for left, right in two_qubit_pairs],
        "seed": int(seed),
        "device": str(device),
        "hamiltonian_id": str(hamiltonian_id),
        "hamiltonian_class": str(hamiltonian_class),
        "best_rank_score": float(ranking_records[0]["score"]) if ranking_records else None,
        "best_screening_energy": float(rows[0]["screening_energy"]) if rows else None,
        "screened_candidate_count": len(screened_records),
        "warm_start_params_written": sum(1 for row in rows if row.get("supernet_init_params_ref")),
        "random_baseline_row": random_baseline_row,
    }
    return rows, summary


