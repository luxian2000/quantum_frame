"""Native supernet cheap-ranking expansion for VQE-QAS rounds.

This module keeps the supernet algorithm as the source of truth: it asks
``aicir.qas.algorithms.supernet.Supernet`` to train shared weights and rank its
own sampled architecture pool, then converts the top ranked architectures into
vqe_loop queue-compatible rows for fair VQE labeling.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from aicir.metrics.circuit_structure import parameter_count
from aicir.operators import Hamiltonian
from aicir.qas.primitives.ansatz import SupernetAnsatzGene, architecture_from_supernet_gene
from aicir.qas.vqe_loop.geometry import parse_pauli_hamiltonian_terms
from aicir.qas.vqe_loop.protocol import BENCHMARK_TABLE_FIELDS, ZeroCostStatus

if TYPE_CHECKING:
    from aicir.qas.algorithms.supernet import Architecture


PauliTerm = tuple[float, str]


def _to_hamiltonian(terms: Sequence[PauliTerm]) -> Hamiltonian:
    parsed = parse_pauli_hamiltonian_terms(terms)
    if not parsed:
        raise ValueError("supernet native ranking requires non-empty Hamiltonian terms")
    width = len(parsed[0][1])
    return Hamiltonian(n_qubits=width, terms=[(pauli, coeff) for coeff, pauli in parsed])


def _default_two_qubit_pairs(n_qubits: int) -> tuple[tuple[int, int], ...]:
    n = int(n_qubits)
    if n > 4:
        return tuple((index, index + 1) for index in range(n - 1))
    return tuple(
        (left, right)
        for left in range(n)
        for right in range(left + 1, n)
        if right - left <= 2
    )


def _default_single_qubit_gates(n_qubits: int) -> tuple[str, ...]:
    if int(n_qubits) > 4:
        return ("ry", "rz")
    return ("i", "h", "rx", "ry", "rz")


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
    row = {field: "" for field in BENCHMARK_TABLE_FIELDS}
    row.update(
        {
            "architecture_id": architecture_id,
            "canonical_arch_hash": json.dumps(gene_payload, ensure_ascii=False, sort_keys=True),
            "n_qubits": int(n_qubits),
            "hamiltonian_id": str(hamiltonian_id),
            "hamiltonian_class": str(hamiltonian_class),
            "family": "supernet_native",
            "entangler_type": "mixed_supernet",
            "topology": "supernet_pairs",
            "depth_group": f"L{gene.layers}",
            "n_params": int(parameter_count(spec.circuit)),
            "two_q_count": int(record.get("two_qubit_count", 0)),
            "hamiltonian_coverage": "1.000000",
            "hamiltonian_coverage_features": "1.000000",
            "zero_cost_status": ZeroCostStatus.PASS.value,
            "zero_cost_reasons": "",
            "expressibility_score": "",
            "trainability_score": "",
            "entanglement_score": "",
            "zero_cost_feature_score": "",
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
    excluded_ids: set[str] | None = None,
    params_dir: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if int(count) <= 0:
        return [], {"enabled": False, "generated_rows": 0}
    from aicir.qas.algorithms.supernet import Supernet, SupernetConfig

    hamiltonian = _to_hamiltonian(hamiltonian_terms)
    n_qubits = int(hamiltonian.n_qubits)
    two_qubit_pairs = _default_two_qubit_pairs(n_qubits)
    single_qubit_gates = _default_single_qubit_gates(n_qubits)
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
    ranking_records = supernet.rank_architectures(None, hamiltonian=hamiltonian, split="train")

    rows: list[dict[str, Any]] = []
    seen = set(excluded_ids or set())
    params_root = Path(params_dir) if params_dir is not None else None
    if params_root is not None:
        params_root.mkdir(parents=True, exist_ok=True)
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
        params_ref = ""
        screening_energy = float(record["score"])
        preview_param_count = int(parameter_count(preview_spec.circuit))
        if params_root is not None:
            if preview_param_count > 0:
                circuit, _params, _log, screening_energy = supernet.finetune_architecture(
                    record["architecture"],
                    int(record["selected_supernet_id"]),
                    None,
                    hamiltonian=hamiltonian,
                )
                vector = _parameter_vector_from_circuit(circuit)
            else:
                vector = []
            if len(vector) != preview_param_count:
                raise ValueError(
                    f"supernet warm-start vector length {len(vector)} does not match "
                    f"vqe_loop circuit parameter count {preview_param_count}"
                )
            params_name = f"supernet_native_{len(rows) + 1:03d}_{_safe_stem(architecture_id)}_params.json"
            (params_root / params_name).write_text(
                json.dumps(vector, indent=2, ensure_ascii=False) + "\n",
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
            screening_energy=screening_energy,
        )
        rows.append(row)
        seen.add(architecture_id)
        if len(rows) >= int(count):
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
        "warm_start_params_written": sum(1 for row in rows if row.get("supernet_init_params_ref")),
    }
    return rows, summary
