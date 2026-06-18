"""Run one Stage-2 online-search loop for trust-region VQE-QAS.

The loop is intentionally conservative:
1. Plan a next fair-label batch with EA/beam proposals and oracle abstain.
2. Run fair VQE labels for the planned queue.
3. Append completed labels back into a benchmark table.
4. Recalibrate the trust-region oracle on the updated table.

This script runs exactly one round; max-rounds and convergence-stop policies
belong in a thin multi-round controller above this reusable single-round step.
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

from aicir.qas.vqe_loop.protocol import BENCHMARK_TABLE_FIELDS, append_benchmark_rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BENCHMARK_TABLE_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in BENCHMARK_TABLE_FIELDS})


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one VQE-QAS Stage-2 online-search loop")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--benchmark-table", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol", default="aicir/qas/vqe_loop/fair_label_protocol.json")
    parser.add_argument("--protocol-version", default="fair_vqe_protocol_v2")
    parser.add_argument("--batch-id", default="stage2_round1")
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--d-max", type=float, default=0.28125)
    parser.add_argument("--local", type=int, default=2)
    parser.add_argument("--boundary", type=int, default=1)
    parser.add_argument("--sparse", type=int, default=1)
    parser.add_argument("--control", type=int, default=0)
    parser.add_argument("--ea-population", type=int, default=24)
    parser.add_argument("--ea-generations", type=int, default=4)
    parser.add_argument("--ea-seed-count", type=int, default=8)
    parser.add_argument("--ea-seed", type=int, default=17)
    parser.add_argument("--ea-gene-key", default="hea_mask")
    parser.add_argument("--supernet-sidecar", default=None)
    parser.add_argument("--label-seed", type=int, default=3026)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--max-evals", type=int, default=800)
    parser.add_argument("--backend", default="numpy")
    parser.add_argument("--dtype", default="complex128")
    parser.add_argument("--dry-run-labels", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queue_path = output_dir / f"{args.batch_id}_queue.csv"
    plan_summary_path = output_dir / f"{args.batch_id}_plan_summary.json"
    labels_path = output_dir / f"{args.batch_id}_labels.csv"
    appended_table_path = output_dir / f"{args.batch_id}_benchmark_table.csv"
    calibration_json_path = output_dir / f"{args.batch_id}_oracle_calibration.json"
    calibration_md_path = output_dir / f"{args.batch_id}_oracle_calibration.md"

    plan_command = [
        sys.executable,
        "-m",
        "aicir.qas.vqe_loop.next_batch",
        "--candidates",
        str(Path(args.candidates)),
        "--benchmark-table",
        str(Path(args.benchmark_table)),
        "--output",
        str(queue_path),
        "--summary",
        str(plan_summary_path),
        "--protocol-version",
        str(args.protocol_version),
        "--batch-id",
        str(args.batch_id),
        "--k-min",
        str(int(args.k_min)),
        "--d-max",
        str(float(args.d_max)),
        "--local",
        str(int(args.local)),
        "--boundary",
        str(int(args.boundary)),
        "--sparse",
        str(int(args.sparse)),
        "--control",
        str(int(args.control)),
        "--ea-population",
        str(int(args.ea_population)),
        "--ea-generations",
        str(int(args.ea_generations)),
        "--ea-seed-count",
        str(int(args.ea_seed_count)),
        "--ea-seed",
        str(int(args.ea_seed)),
        "--ea-gene-key",
        str(args.ea_gene_key),
    ]
    if args.supernet_sidecar:
        plan_command.extend(["--supernet-sidecar", str(Path(args.supernet_sidecar))])
    _run(plan_command, cwd=repo_root)

    label_command = [
        sys.executable,
        "-m",
        "aicir.qas.vqe_loop.labeling",
        "--queue",
        str(queue_path),
        "--output",
        str(labels_path),
        "--protocol",
        str(Path(args.protocol)),
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
    ]
    if args.dry_run_labels:
        label_command.append("--dry-run")
    _run(label_command, cwd=repo_root)

    benchmark_rows = _read_csv(Path(args.benchmark_table))
    label_rows = _read_csv(labels_path)
    appended_rows = append_benchmark_rows(benchmark_rows, label_rows)
    _write_csv(appended_table_path, appended_rows)

    calibration_command = [
        sys.executable,
        "-m",
        "aicir.qas.vqe_loop.calibration",
        "--benchmark-table",
        str(appended_table_path),
        "--output",
        str(calibration_json_path),
        "--markdown-output",
        str(calibration_md_path),
        "--protocol-version",
        str(args.protocol_version),
        "--k-min",
        str(int(args.k_min)),
        "--d-max",
        str(float(args.d_max)),
    ]
    _run(calibration_command, cwd=repo_root)

    plan_summary = json.loads(plan_summary_path.read_text(encoding="utf-8"))
    calibration_summary = json.loads(calibration_json_path.read_text(encoding="utf-8"))
    completed_labels = sum(1 for row in label_rows if row.get("label_status") == "completed")
    loop_summary = {
        "batch_id": args.batch_id,
        "queue": str(queue_path),
        "labels": str(labels_path),
        "updated_benchmark_table": str(appended_table_path),
        "calibration_json": str(calibration_json_path),
        "calibration_markdown": str(calibration_md_path),
        "planned_total": plan_summary.get("planned_total"),
        "completed_labels": completed_labels,
        "source_counts": plan_summary.get("source_counts", {}),
        "can_use_oracle": plan_summary.get("can_use_oracle"),
        "track_a_abstain_rate": plan_summary.get("ea", {}).get("track_a_abstain_rate"),
        "calibration_passes": calibration_summary.get("passes", {}),
    }
    (output_dir / f"{args.batch_id}_loop_summary.json").write_text(
        json.dumps(loop_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(loop_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
