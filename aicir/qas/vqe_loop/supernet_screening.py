"""Supernet screening sidecar for VQE-QAS expansion candidates.

The output intentionally separates:
- ``supernet_rank_score``: shared-weight rank score for expansion priority.
- ``supernet_init_params_ref``: reference to warm-start parameters.
- ``screening_energy``: optional diagnostic energy, not a final fair label.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aicir.operators import Hamiltonian
from aicir.qas.vqe_loop.geometry import parse_pauli_hamiltonian_terms


def _load_terms(path: Path) -> tuple[tuple[float, str], ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return parse_pauli_hamiltonian_terms(raw)


def _to_hamiltonian(terms: Sequence[tuple[float, str]]) -> Hamiltonian:
    if not terms:
        raise ValueError("Hamiltonian terms cannot be empty")
    width = len(terms[0][1])
    return Hamiltonian(n_qubits=width, terms=[(pauli, coeff) for coeff, pauli in terms])


def _jsonable_key(key: Any) -> str:
    if isinstance(key, tuple):
        return "|".join(str(item) for item in key)
    return str(key)


def _screening_sidecar_record(
    *,
    rank: int,
    architecture_indices: Sequence[Sequence[int]],
    selected_supernet_id: int,
    shared_weight_score: float,
    screening_energy: float | None,
    init_params_path: str,
    cnot_count: int,
    two_qubit_count: int,
) -> dict[str, Any]:
    return {
        "rank": int(rank),
        "architecture_indices": [list(item) for item in architecture_indices],
        "selected_supernet_id": int(selected_supernet_id),
        "supernet_rank_score": float(shared_weight_score),
        "supernet_init_params_ref": str(init_params_path),
        "screening_energy": None if screening_energy is None else float(screening_energy),
        "screening_energy_is_final_label": False,
        "cnot_count": int(cnot_count),
        "two_qubit_count": int(two_qubit_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run supernet screening for a Pauli Hamiltonian")
    parser.add_argument("--target-terms-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="supernet_screen")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--supernet-num", type=int, default=2)
    parser.add_argument("--supernet-steps", type=int, default=20)
    parser.add_argument("--ranking-num", type=int, default=12)
    parser.add_argument("--finetune-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    terms = _load_terms(Path(args.target_terms_file))
    hamiltonian = _to_hamiltonian(terms)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from aicir.qas.algorithms.supernet import supernet_qas

    result = supernet_qas(
        hamiltonian,
        layers=int(args.layers),
        supernet_num=int(args.supernet_num),
        supernet_steps=int(args.supernet_steps),
        finetune_steps=int(args.finetune_steps),
        n_qubits=hamiltonian.n_qubits,
        ranking_num=int(args.ranking_num),
        seed=int(args.seed),
        device=str(args.device),
    )

    params_path = output_dir / f"{args.run_id}_init_params.json"
    params = {
        _jsonable_key(key): float(value)
        for key, value in result.final_metrics.get("fine_tuned_parameters", {}).items()
    }
    params_path.write_text(json.dumps(params, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    best_record = result.ranking_records[0]
    sidecar = {
        "run_id": args.run_id,
        "protocol": "supernet_screening",
        "hamiltonian_n_qubits": int(hamiltonian.n_qubits),
        "terms_count": len(terms),
        "best_score": float(result.best_score),
        "records": [
            _screening_sidecar_record(
                rank=int(record["rank"]),
                architecture_indices=record["architecture_indices"],
                selected_supernet_id=int(record["selected_supernet_id"]),
                shared_weight_score=float(record["score"]),
                screening_energy=float(result.best_score) if record is best_record else None,
                init_params_path=params_path.name if record is best_record else "",
                cnot_count=int(record["cnot_count"]),
                two_qubit_count=int(record["two_qubit_count"]),
            )
            for record in result.ranking_records
        ],
        "note": "supernet_rank_score is for expansion priority; supernet_init_params_ref is for warm-start; screening_energy is diagnostic and not a fair label.",
    }
    sidecar_path = output_dir / f"{args.run_id}_sidecar.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"sidecar": str(sidecar_path), "init_params": str(params_path), "best_score": result.best_score}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
