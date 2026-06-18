"""Build a supernet sidecar from completed fair VQE labels.

This bridge lets Stage-2 fair-label results feed the existing supernet/selection
interface without treating screening energies as final labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _safe_stem(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("._") or "architecture"


def _best_trace_parameters(row: Mapping[str, Any]) -> tuple[list[float] | None, float | None]:
    raw = str(row.get("best_trace", "") or "").strip()
    if not raw:
        return None, None
    trace = json.loads(raw)
    if not isinstance(trace, list):
        return None, None
    best_item: Mapping[str, Any] | None = None
    for item in trace:
        if not isinstance(item, Mapping):
            continue
        params = item.get("best_parameters")
        if not isinstance(params, list) or not params:
            continue
        if best_item is None or float(item.get("energy", "inf")) < float(best_item.get("energy", "inf")):
            best_item = item
    if best_item is None:
        return None, None
    return [float(value) for value in best_item["best_parameters"]], float(best_item.get("energy", "nan"))


def build_sidecar_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    skipped = 0
    completed = [
        row
        for row in rows
        if str(row.get("label_status", "")).strip().lower() == "completed"
        and str(row.get("architecture_id", "")).strip()
    ]
    for rank, row in enumerate(
        sorted(completed, key=lambda item: float(item.get("fair_best_energy", "inf"))),
        start=1,
    ):
        params, trace_energy = _best_trace_parameters(row)
        if params is None:
            skipped += 1
            continue
        architecture_id = str(row["architecture_id"]).strip()
        params_name = f"{run_id}_{rank:03d}_{_safe_stem(architecture_id)}_params.json"
        (output_dir / params_name).write_text(
            json.dumps(params, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        energy = float(row.get("fair_best_energy", trace_energy))
        records.append(
            {
                "rank": len(records) + 1,
                "architecture_id": architecture_id,
                "supernet_rank_score": energy,
                "supernet_init_params_ref": params_name,
                "screening_energy": energy,
                "screening_energy_is_final_label": False,
                "source_label_status": row.get("label_status", ""),
                "source_batch_id": row.get("batch_id", ""),
            }
        )
    return {
        "run_id": run_id,
        "protocol": "fair_label_seeded_supernet_sidecar",
        "records": records,
        "skipped_without_parameters": skipped,
        "note": "supernet_rank_score is only an expansion priority; supernet_init_params_ref is a fair-VQE warm start; screening_energy is diagnostic.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert completed fair VQE labels into a supernet sidecar")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="fair_label_seeded_supernet")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    sidecar = build_sidecar_records(_read_csv(Path(args.labels)), output_dir=output_dir, run_id=str(args.run_id))
    output = Path(args.output) if args.output else output_dir / f"{args.run_id}_sidecar.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"sidecar": str(output), "records": len(sidecar["records"])}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
