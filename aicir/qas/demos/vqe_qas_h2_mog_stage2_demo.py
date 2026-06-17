"""Fixed Hamiltonian demos for the MoG-VQE EA Stage-2 loop.

The default Hamiltonian is H2 in the STO-3G basis at bond length 0.735
Angstrom, mapped to four qubits with Jordan-Wigner.  A six-qubit open-boundary
TFIM target is also included for scaling smoke tests.  The demo keeps the
standard fair-label protocol and Stage-2 trust-region loop, but fixes every
queued row to a literal Pauli Hamiltonian so repeated EA rounds stay on the
same task.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aicir.qas.vqe_hea_demo import tfim_chain_hamiltonian
from aicir.qas.vqe_qas_protocol import BENCHMARK_TABLE_FIELDS, extract_pauli_hamiltonian_features


H2_JW_STO3G_R0735_TERMS: tuple[tuple[float, str], ...] = (
    (-0.09706626816762543, "IIII"),
    (0.17141282644776914, "ZIII"),
    (0.17141282644776914, "IZII"),
    (-0.22343153690813447, "ZZII"),
    (0.17141282644776914, "IIZI"),
    (0.16592785033785953, "ZIZI"),
    (0.16592785033785953, "IZZI"),
    (-0.22343153690813447, "IIZZ"),
    (0.17141282644776914, "IIIZ"),
    (0.16592785033785953, "ZIIZ"),
    (0.16592785033785953, "IZIZ"),
    (0.12039548242542646, "XIXI"),
    (0.12039548242542646, "XIXX"),
    (0.12039548242542646, "XXIX"),
    (0.12039548242542646, "XXXX"),
    (0.17391653067620093, "YIYI"),
    (0.17391653067620093, "YIYY"),
    (0.17391653067620093, "YYIY"),
    (-0.17391653067620093, "YYYY"),
)

H2_HAMILTONIAN_ID = "h2_sto3g_jw_4q_r0735"
H2_HAMILTONIAN_CLASS = "molecular_h2"

SIXQ_TFIM_OBC_J1_H05_TERMS: tuple[tuple[float, str], ...] = tfim_chain_hamiltonian(
    n_qubits=6,
    J=1.0,
    h=0.5,
    periodic=False,
)
SIXQ_TFIM_HAMILTONIAN_ID = "tfim_obc_6q_j1_h05"
SIXQ_TFIM_HAMILTONIAN_CLASS = "tfim"


def _target_spec(target: str) -> dict[str, Any]:
    if target == "h2_4q":
        return {
            "n_qubits": 4,
            "hamiltonian_id": H2_HAMILTONIAN_ID,
            "hamiltonian_class": H2_HAMILTONIAN_CLASS,
            "terms": H2_JW_STO3G_R0735_TERMS,
            "initial_batch_id": "h2_initial",
            "round_prefix": "h2_mog_round",
        }
    if target == "tfim_6q":
        return {
            "n_qubits": 6,
            "hamiltonian_id": SIXQ_TFIM_HAMILTONIAN_ID,
            "hamiltonian_class": SIXQ_TFIM_HAMILTONIAN_CLASS,
            "terms": SIXQ_TFIM_OBC_J1_H05_TERMS,
            "initial_batch_id": "tfim6q_initial",
            "round_prefix": "tfim6q_mog_round",
        }
    raise ValueError(f"Unsupported target: {target}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _features_json(terms: tuple[tuple[float, str], ...]) -> str:
    return json.dumps(extract_pauli_hamiltonian_features(terms), sort_keys=True)


def _terms_json(terms: tuple[tuple[float, str], ...]) -> str:
    return json.dumps(list(terms), ensure_ascii=False)


def annotate_target_rows(
    rows: list[dict[str, str]],
    *,
    target: dict[str, Any],
    batch_id: str | None = None,
) -> list[dict[str, Any]]:
    """Attach the selected fixed Hamiltonian fields to candidate or queue rows."""

    features_json = _features_json(target["terms"])
    terms_json = _terms_json(target["terms"])
    annotated: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        updated["n_qubits"] = str(int(target["n_qubits"]))
        updated["hamiltonian_id"] = str(target["hamiltonian_id"])
        updated["hamiltonian_class"] = str(target["hamiltonian_class"])
        updated["hamiltonian_coverage_features"] = features_json
        updated["hamiltonian_terms"] = terms_json
        if batch_id is not None:
            updated["batch_id"] = batch_id
        annotated.append(updated)
    return annotated


def annotate_h2_rows(rows: list[dict[str, str]], *, batch_id: str | None = None) -> list[dict[str, Any]]:
    """Backward-compatible helper for tests and older callers."""

    return annotate_target_rows(rows, target=_target_spec("h2_4q"), batch_id=batch_id)


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=str(cwd), check=True)


def _csv_fieldnames(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    seen = set(preferred)
    extras = sorted({key for row in rows for key in row if key not in seen})
    return preferred + extras


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-H2 MoG-VQE EA Stage-2 demo")
    parser.add_argument("--target", choices=("h2_4q", "tfim_6q"), default="h2_4q")
    parser.add_argument("--output-dir", default="outputs/h2_mog_stage2_demo")
    parser.add_argument("--initial-labels", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--round-local", type=int, default=4)
    parser.add_argument("--round-boundary", type=int, default=2)
    parser.add_argument("--round-sparse", type=int, default=2)
    parser.add_argument("--round-control", type=int, default=0)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--d-max", type=float, default=0.28125)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--max-evals", type=int, default=600)
    parser.add_argument("--label-seed", type=int, default=5200)
    parser.add_argument("--ea-population", type=int, default=32)
    parser.add_argument("--ea-generations", type=int, default=8)
    parser.add_argument("--ea-seed-count", type=int, default=12)
    parser.add_argument("--layerwise-count", type=int, default=18)
    parser.add_argument("--layerwise-layers", type=int, default=3)
    parser.add_argument("--backend", default="numpy")
    parser.add_argument("--dtype", default="complex128")
    args = parser.parse_args()
    target = _target_spec(str(args.target))

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = Path(args.output_dir)
    prep_dir = output_dir / "prep"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_hamiltonian_terms.json").write_text(
        json.dumps(
            {
                "target": args.target,
                "hamiltonian_id": target["hamiltonian_id"],
                "hamiltonian_class": target["hamiltonian_class"],
                "n_qubits": target["n_qubits"],
                "terms": list(target["terms"]),
                "features": extract_pauli_hamiltonian_features(target["terms"]),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_hamiltonian": target["hamiltonian_id"],
                "n_qubits": target["n_qubits"],
                "terms": list(target["terms"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    _run(
        [
            sys.executable,
            str(repo_root / "aicir" / "qas" / "demos" / "vqe_qas_prepare_oracle.py"),
            "--scales",
            str(int(target["n_qubits"])),
            "--hamiltonian-class",
            str(target["hamiltonian_class"]),
            "--batch-id",
            str(target["initial_batch_id"]),
            "--initial-labels",
            str(int(args.initial_labels)),
            "--holdout-fraction",
            "0.25",
            "--k-min",
            str(int(args.k_min)),
            "--zero-cost-samples",
            "0",
            "--include-layerwise",
            "--layerwise-count",
            str(int(args.layerwise_count)),
            "--layerwise-layers",
            str(int(args.layerwise_layers)),
            "--output-dir",
            str(prep_dir),
        ],
        cwd=repo_root,
    )

    file_prefix = str(args.target)
    candidates_path = output_dir / f"{file_prefix}_stage0_candidates.csv"
    initial_queue_path = output_dir / f"{file_prefix}_initial_queue.csv"
    candidate_rows = annotate_target_rows(_read_csv(prep_dir / "stage0_candidates.csv"), target=target)
    initial_queue_rows = annotate_target_rows(
        _read_csv(prep_dir / "stage1_5_initial_label_queue.csv"),
        target=target,
        batch_id=str(target["initial_batch_id"]),
    )
    _write_csv(candidates_path, candidate_rows, _csv_fieldnames(candidate_rows, list(candidate_rows[0].keys())))
    _write_csv(initial_queue_path, initial_queue_rows, list(BENCHMARK_TABLE_FIELDS))

    initial_labels_path = output_dir / f"{file_prefix}_initial_labels.csv"
    _run(
        [
            sys.executable,
            str(repo_root / "aicir" / "qas" / "demos" / "vqe_qas_run_fair_labels.py"),
            "--queue",
            str(initial_queue_path),
            "--output",
            str(initial_labels_path),
            "--protocol",
            str(repo_root / "aicir" / "qas" / "configs" / "fair_vqe_protocol_v2.json"),
            "--seed",
            str(int(args.label_seed)),
            "--n-seeds",
            str(int(args.n_seeds)),
            "--max-evals",
            str(int(args.max_evals)),
            "--backend",
            str(args.backend),
            "--dtype",
            str(args.dtype),
        ],
        cwd=repo_root,
    )

    benchmark_path = initial_labels_path
    round_summaries: list[dict[str, Any]] = []
    for round_index in range(1, max(0, int(args.rounds)) + 1):
        batch_id = f"{target['round_prefix']}{round_index}"
        round_dir = output_dir / batch_id
        _run(
            [
                sys.executable,
                str(repo_root / "aicir" / "qas" / "demos" / "vqe_qas_run_stage2_loop.py"),
                "--candidates",
                str(candidates_path),
                "--benchmark-table",
                str(benchmark_path),
                "--output-dir",
                str(round_dir),
                "--batch-id",
                batch_id,
                "--k-min",
                str(int(args.k_min)),
                "--d-max",
                str(float(args.d_max)),
                "--local",
                str(int(args.round_local)),
                "--boundary",
                str(int(args.round_boundary)),
                "--sparse",
                str(int(args.round_sparse)),
                "--control",
                str(int(args.round_control)),
                "--ea-population",
                str(int(args.ea_population)),
                "--ea-generations",
                str(int(args.ea_generations)),
                "--ea-seed-count",
                str(int(args.ea_seed_count)),
                "--ea-seed",
                str(100 + round_index),
                "--label-seed",
                str(int(args.label_seed) + round_index * 1000),
                "--n-seeds",
                str(int(args.n_seeds)),
                "--max-evals",
                str(int(args.max_evals)),
                "--backend",
                str(args.backend),
                "--dtype",
                str(args.dtype),
            ],
            cwd=repo_root,
        )
        loop_summary_path = round_dir / f"{batch_id}_loop_summary.json"
        if loop_summary_path.exists():
            round_summaries.append(json.loads(loop_summary_path.read_text(encoding="utf-8")))
        benchmark_path = round_dir / f"{batch_id}_benchmark_table.csv"

    summary = {
        "target": args.target,
        "hamiltonian_id": target["hamiltonian_id"],
        "hamiltonian_class": target["hamiltonian_class"],
        "n_qubits": target["n_qubits"],
        "terms": list(target["terms"]),
        "initial_labels": str(initial_labels_path),
        "final_benchmark_table": str(benchmark_path),
        "rounds": round_summaries,
    }
    (output_dir / "h2_mog_stage2_demo_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
