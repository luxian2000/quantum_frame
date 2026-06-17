"""Post-process VQE-QAS v3 enumeration CSVs for transfer regret.

This script does not run VQE.  It consumes completed or partial
``phase1_uniform_enum_*q*.csv`` files and asks a small Phase-3 question:
if a cheap predictor ranked candidates before target-scale VQE, would its
top-K contain the true best candidate at that target scale?

Supported predictors:
* family-transfer: rank a target candidate by the mean source-scale delta_ref
  of its structural family.
* exact-name-transfer: rank by the mean source-scale delta_ref of the same
  mask name.
* external-csv: rank by a user-provided score CSV with columns
  n_qubits,name,score, where lower score is better.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ENTANGLERS = {"cx", "cz", "rzz"}
PATTERNS = {"linear", "ring"}


@dataclass(frozen=True)
class EnumRow:
    n_qubits: int
    rank: int
    name: str
    family: str
    energy: float
    delta_ref: float
    n_params: int | None = None


@dataclass(frozen=True)
class RegretRow:
    target_n: int
    predictor: str
    source_scales: tuple[int, ...]
    k: int
    global_best_name: str
    global_best_energy: float
    selected_best_name: str
    selected_best_energy: float
    regret: float
    regret_mha: float
    regret_mha_per_qubit: float
    overlap_at_k: float
    global_best_predicted_rank: int | None


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _family_from_name(name: str) -> str:
    parts = name.split("_")
    if len(parts) < 7 or parts[0] != "hea" or parts[1] != "mask" or not parts[2].startswith("L"):
        return name
    layer = parts[2]
    rest = parts[3:]
    entangler_index = next((index for index, item in enumerate(rest) if item in ENTANGLERS), None)
    if entangler_index is None or entangler_index + 1 >= len(rest):
        return name
    rotation = "_".join(rest[:entangler_index])
    entangler = rest[entangler_index]
    pattern = rest[entangler_index + 1]
    if pattern not in PATTERNS:
        return name
    return f"{rotation}_{entangler}_{pattern}_{layer}"


def _safe_int(raw: str | None) -> int | None:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(float(str(raw)))
    except ValueError:
        return None


def _top_k(candidate_count: int, requested: int | None = None) -> int:
    if candidate_count <= 0:
        return 0
    if requested is not None:
        return min(max(1, int(requested)), candidate_count)
    return min(candidate_count, max(10, int(-(-candidate_count // 10))))


def _load_rows(input_dir: Path, scales: tuple[int, ...]) -> dict[int, list[EnumRow]]:
    data: dict[int, list[EnumRow]] = {}
    for n_qubits in scales:
        canonical = input_dir / f"phase1_uniform_enum_{n_qubits}q.csv"
        paths = [canonical] if canonical.exists() else sorted(input_dir.glob(f"phase1_uniform_enum_{n_qubits}q_shard*of*.csv"))
        if not paths:
            paths = sorted(input_dir.glob(f"phase1_uniform_enum_{n_qubits}q_shard*of*.partial.csv"))
        if not paths:
            raise FileNotFoundError(f"Missing enumeration CSV or shard CSVs for {n_qubits}q in {input_dir}")

        rows: list[EnumRow] = []
        for path in paths:
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for item in reader:
                    name = str(item["name"])
                    family = str(item.get("family") or "").strip()
                    if family in {"", "HEA-mask"}:
                        family = _family_from_name(name)
                    rows.append(
                        EnumRow(
                            n_qubits=n_qubits,
                            rank=int(item.get("rank") or item.get("candidate_index") or len(rows) + 1),
                            name=name,
                            family=family,
                            energy=float(item["energy"]),
                            delta_ref=float(item["delta_ref"]),
                            n_params=_safe_int(item.get("n_params")),
                        )
                    )

        best_by_name = {row.name: row for row in sorted(rows, key=lambda row: (row.energy, row.name))}
        data[n_qubits] = sorted(best_by_name.values(), key=lambda row: (row.energy, row.name))
    return data


def _load_external_scores(path: Path) -> dict[tuple[int, str], float]:
    scores: dict[tuple[int, str], float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"n_qubits", "name", "score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"External score CSV is missing columns: {sorted(missing)}")
        for item in reader:
            scores[(int(item["n_qubits"]), str(item["name"]))] = float(item["score"])
    return scores


def _source_family_scores(data: dict[int, list[EnumRow]], source_scales: tuple[int, ...]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for n_qubits in source_scales:
        for row in data[n_qubits]:
            grouped.setdefault(row.family, []).append(row.delta_ref)
    return {family: mean(values) for family, values in grouped.items()}


def _source_name_scores(data: dict[int, list[EnumRow]], source_scales: tuple[int, ...]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for n_qubits in source_scales:
        for row in data[n_qubits]:
            grouped.setdefault(row.name, []).append(row.delta_ref)
    return {name: mean(values) for name, values in grouped.items()}


def _score_rows(
    data: dict[int, list[EnumRow]],
    target_n: int,
    source_scales: tuple[int, ...],
    predictor: str,
    external_scores: dict[tuple[int, str], float] | None = None,
) -> list[tuple[float, EnumRow]]:
    target_rows = data[target_n]
    if predictor == "family-transfer":
        source_scores = _source_family_scores(data, source_scales)
        return [
            (
                source_scores.get(row.family, float("inf")),
                row,
            )
            for row in target_rows
        ]
    if predictor == "exact-name-transfer":
        source_scores = _source_name_scores(data, source_scales)
        return [(source_scores.get(row.name, float("inf")), row) for row in target_rows]
    if predictor == "external-csv":
        if external_scores is None:
            raise ValueError("external-csv predictor requires --prediction-csv")
        return [(external_scores.get((target_n, row.name), float("inf")), row) for row in target_rows]
    raise ValueError(f"Unknown predictor: {predictor}")


def _fallback_key(scored: tuple[float, EnumRow]) -> tuple[float, int, float, str]:
    score, row = scored
    n_params = row.n_params if row.n_params is not None else 10**9
    return (score, n_params, row.delta_ref, row.name)


def _evaluate_regret(
    data: dict[int, list[EnumRow]],
    target_n: int,
    source_scales: tuple[int, ...],
    predictor: str,
    k: int | None,
    external_scores: dict[tuple[int, str], float] | None,
) -> RegretRow:
    target_rows = data[target_n]
    effective_k = _top_k(len(target_rows), k)
    scored = sorted(_score_rows(data, target_n, source_scales, predictor, external_scores), key=_fallback_key)
    selected = [row for _, row in scored[:effective_k]]
    global_top = target_rows[:effective_k]
    global_best = target_rows[0]
    selected_best = min(selected, key=lambda row: (row.energy, row.name))
    selected_names = {row.name for row in selected}
    global_top_names = {row.name for row in global_top}
    predicted_rank = next((index for index, (_, row) in enumerate(scored, start=1) if row.name == global_best.name), None)
    regret = selected_best.energy - global_best.energy
    return RegretRow(
        target_n=target_n,
        predictor=predictor,
        source_scales=source_scales,
        k=effective_k,
        global_best_name=global_best.name,
        global_best_energy=global_best.energy,
        selected_best_name=selected_best.name,
        selected_best_energy=selected_best.energy,
        regret=regret,
        regret_mha=regret * 1000.0,
        regret_mha_per_qubit=regret * 1000.0 / float(target_n),
        overlap_at_k=len(selected_names & global_top_names) / float(effective_k),
        global_best_predicted_rank=predicted_rank,
    )


def analyze(
    input_dir: Path,
    scales: tuple[int, ...],
    output_dir: Path,
    predictor: str,
    k: int | None,
    prediction_csv: Path | None,
) -> list[RegretRow]:
    data = _load_rows(input_dir, scales)
    external_scores = _load_external_scores(prediction_csv) if prediction_csv else None
    rows: list[RegretRow] = []
    for index, target_n in enumerate(scales):
        source_scales = scales[:index]
        if not source_scales and predictor != "external-csv":
            continue
        rows.append(_evaluate_regret(data, target_n, source_scales, predictor, k, external_scores))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "phase3_transfer_regret.csv", rows)
    _write_json(output_dir / "phase3_transfer_regret.json", rows)
    _write_markdown(output_dir / "phase3_transfer_regret.md", input_dir, rows)
    return rows


def _write_csv(path: Path, rows: list[RegretRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_n",
                "predictor",
                "source_scales",
                "k",
                "global_best_name",
                "global_best_energy",
                "selected_best_name",
                "selected_best_energy",
                "regret",
                "regret_mha",
                "regret_mha_per_qubit",
                "overlap_at_k",
                "global_best_predicted_rank",
            ],
        )
        writer.writeheader()
        for row in rows:
            item = row.__dict__.copy()
            item["source_scales"] = ",".join(str(value) for value in row.source_scales)
            writer.writerow(item)


def _write_json(path: Path, rows: list[RegretRow]) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    **row.__dict__,
                    "source_scales": list(row.source_scales),
                }
                for row in rows
            ],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, input_dir: Path, rows: list[RegretRow]) -> None:
    lines = [
        "VQE-QAS v3 transfer-regret analysis",
        f"input_dir: {input_dir}",
        "",
        "target | predictor | sources | K | regret_mHa | regret_mHa_per_q | overlap@K | best_rank | selected_best | global_best",
    ]
    for row in rows:
        sources = ",".join(str(value) for value in row.source_scales) or "external"
        best_rank = "NA" if row.global_best_predicted_rank is None else str(row.global_best_predicted_rank)
        lines.append(
            f"{row.target_n}q | {row.predictor} | {sources} | {row.k} | "
            f"{row.regret_mha:.3f} | {row.regret_mha_per_qubit:.3f} | "
            f"{row.overlap_at_k:.3f} | {best_rank} | {row.selected_best_name} | {row.global_best_name}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze transfer regret from VQE-QAS v3 enumeration CSVs")
    parser.add_argument("--input-dir", default="outputs/vqe_tfim_v3_scaling_npu_full")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--scales", default="4,6,8")
    parser.add_argument(
        "--predictor",
        choices=("family-transfer", "exact-name-transfer", "external-csv"),
        default="family-transfer",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--prediction-csv", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    rows = analyze(
        input_dir=input_dir,
        scales=_parse_int_list(args.scales),
        output_dir=output_dir,
        predictor=args.predictor,
        k=args.top_k,
        prediction_csv=Path(args.prediction_csv) if args.prediction_csv else None,
    )
    print((output_dir / "phase3_transfer_regret.md").read_text(encoding="utf-8"), end="")
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
