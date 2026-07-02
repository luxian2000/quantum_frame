"""Chemistry excitation ansatz helpers for VQE-QAS queues."""

from __future__ import annotations

import json
import random
from itertools import product
from typing import Any, Sequence

from aicir.qas.library.ansatz import ChemistryExcitationAnsatzGene, architecture_from_chemistry_excitation_gene
from aicir.qas.vqe_loop.benchmark_table import BENCHMARK_TABLE_FIELDS, ZeroCostStatus


def closed_shell_hf_occupied_qubits(
    active_electrons: int,
    active_spatial_orbitals: int,
) -> tuple[int, ...]:
    if int(active_electrons) % 2 != 0:
        raise ValueError("closed-shell HF helpers require an even electron count")
    n_spatial = int(active_spatial_orbitals)
    n_spin_orbitals = 2 * n_spatial
    n_occ_spatial = int(active_electrons) // 2
    if n_occ_spatial > n_spatial:
        raise ValueError("active_electrons exceeds active spin-orbital capacity")
    occupied_spin_orbitals = (
        *range(n_occ_spatial),
        *range(n_spatial, n_spatial + n_occ_spatial),
    )
    return tuple(sorted(n_spin_orbitals - 1 - orbital for orbital in occupied_spin_orbitals))


def closed_shell_excitation_pools(
    active_electrons: int,
    active_spatial_orbitals: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...], tuple[tuple[int, int, int, int], ...]]:
    n_spatial = int(active_spatial_orbitals)
    n_occ = int(active_electrons) // 2
    if n_occ > n_spatial:
        raise ValueError("active_electrons exceeds active spin-orbital capacity")

    def alpha(spatial_orbital: int) -> int:
        return 2 * n_spatial - 1 - int(spatial_orbital)

    def beta(spatial_orbital: int) -> int:
        return 2 * n_spatial - 1 - (n_spatial + int(spatial_orbital))

    occupied = tuple(range(n_occ))
    virtual = tuple(range(n_occ, n_spatial))
    hf = closed_shell_hf_occupied_qubits(active_electrons, n_spatial)
    singles = tuple(
        (virt_q, occ_q)
        for occ, virt in product(occupied, virtual)
        for virt_q, occ_q in ((alpha(virt), alpha(occ)), (beta(virt), beta(occ)))
    )
    paired_doubles = tuple(
        (*sorted((alpha(virt), beta(virt))), *sorted((alpha(occ), beta(occ))))
        for occ, virt in product(occupied, virtual)
    )
    return hf, singles, paired_doubles


def _row_for_gene(
    gene: ChemistryExcitationAnsatzGene,
    *,
    hamiltonian_id: str,
    hamiltonian_class: str,
    source: str,
    screening_energy: float,
) -> dict[str, Any]:
    architecture = architecture_from_chemistry_excitation_gene(gene)
    payload = gene.to_jsonable()
    row = {field: "" for field in BENCHMARK_TABLE_FIELDS}
    row.update(
        {
            "architecture_id": f"{gene.n_qubits}q_{architecture.name}",
            "canonical_arch_hash": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            "source": source,
            "n_qubits": str(gene.n_qubits),
            "hamiltonian_id": str(hamiltonian_id),
            "hamiltonian_class": str(hamiltonian_class),
            "family": "chemistry_excitation",
            "depth_group": f"L{gene.layers}",
            "entangler_type": "single_double_excitation",
            "topology": "chemistry_excitation_pool",
            "n_params": str(int(architecture.parameter_count)),
            "two_q_count": str(int(architecture.two_qubit_gate_count)),
            "hamiltonian_coverage": "1.000000",
            "hamiltonian_coverage_features": "1.000000",
            "zero_cost_status": ZeroCostStatus.PASS.value,
            "zero_cost_score_is_ranking_signal": "false",
            "ansatz_gene": json.dumps(payload, ensure_ascii=False),
            "screening_energy": f"{float(screening_energy):.12f}",
            "screening_energy_is_final_label": "false",
            "supernet_warm_start_status": "not_applicable",
        }
    )
    return row


def build_chemistry_excitation_rows(
    *,
    active_electrons: int,
    active_spatial_orbitals: int,
    hamiltonian_id: str,
    hamiltonian_class: str,
    count: int,
    max_excitations: int = 4,
    seed: int = 0,
    excluded_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if int(count) <= 0:
        return [], {"enabled": False, "generated_rows": 0, "family": "chemistry_excitation"}
    hf, singles, doubles = closed_shell_excitation_pools(active_electrons, active_spatial_orbitals)
    pool: list[dict[str, Any]] = [
        {"type": "single_excitation", "qubits": list(pair)} for pair in singles
    ] + [{"type": "double_excitation", "qubits": list(group)} for group in doubles]
    if not pool:
        raise ValueError("chemistry excitation pool is empty")

    rng = random.Random(int(seed))
    n_qubits = 2 * int(active_spatial_orbitals)
    rows: list[dict[str, Any]] = []
    seen = set(excluded_ids or set())

    seed_sequences: list[list[dict[str, Any]]] = []
    for item in pool:
        seed_sequences.append([dict(item)])
    for index in range(len(pool)):
        seed_sequences.append([dict(pool[index]), dict(pool[(index + 1) % len(pool)])])

    attempts = 0
    while len(rows) < int(count) and attempts < max(100, int(count) * 100):
        attempts += 1
        if seed_sequences:
            excitations = seed_sequences.pop(0)
        else:
            depth = rng.randint(1, max(1, int(max_excitations)))
            excitations = [dict(rng.choice(pool)) for _ in range(depth)]
        gene = ChemistryExcitationAnsatzGene(
            n_qubits=n_qubits,
            hf_occupied_qubits=hf,
            excitations=tuple(excitations),
            active_electrons=int(active_electrons),
            active_spatial_orbitals=int(active_spatial_orbitals),
        )
        row = _row_for_gene(
            gene,
            hamiltonian_id=hamiltonian_id,
            hamiltonian_class=hamiltonian_class,
            source="trackB_chemistry_excitation",
            screening_energy=float(len(rows)),
        )
        if row["architecture_id"] in seen:
            continue
        rows.append(row)
        seen.add(row["architecture_id"])

    summary = {
        "enabled": True,
        "generated_rows": len(rows),
        "family": "chemistry_excitation",
        "active_electrons": int(active_electrons),
        "active_spatial_orbitals": int(active_spatial_orbitals),
        "n_qubits": n_qubits,
        "hf_occupied_qubits": list(hf),
        "single_excitation_count": len(singles),
        "double_excitation_count": len(doubles),
        "count": int(count),
        "max_excitations": int(max_excitations),
        "seed": int(seed),
    }
    return rows, summary
