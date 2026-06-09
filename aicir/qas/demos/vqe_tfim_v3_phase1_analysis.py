"""Analyze VQE-QAS v3 uniform-enumeration outputs for cross-scale drift."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


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


def _load_rows(input_dir: Path, scales: tuple[int, ...]) -> dict[int, list[EnumRow]]:
    data: dict[int, list[EnumRow]] = {}
    for n_qubits in scales:
        path = input_dir / f"phase1_uniform_enum_{n_qubits}q.csv"
        paths = [path] if path.exists() else sorted(input_dir.glob(f"phase1_uniform_enum_{n_qubits}q_shard*of*.csv"))
        if not paths:
            paths = sorted(input_dir.glob(f"phase1_uniform_enum_{n_qubits}q_shard*of*.partial.csv"))
        if not paths:
            raise FileNotFoundError(f"Missing enumeration CSV or shard CSVs for {n_qubits}q in {input_dir}")
        rows: list[EnumRow] = []
        for item_path in paths:
            with item_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for item in reader:
                    family = str(item.get("family") or "").strip()
                    if family in {"", "HEA-mask"}:
                        family = _family_from_name(str(item["name"]))
                    rows.append(
                        EnumRow(
                            n_qubits=n_qubits,
                            rank=int(item.get("rank") or item.get("candidate_index") or len(rows) + 1),
                            name=str(item["name"]),
                            family=family,
                            energy=float(item["energy"]),
                            delta_ref=float(item["delta_ref"]),
                        )
                    )
        deduped = {row.name: row for row in sorted(rows, key=lambda row: (row.energy, row.name))}
        data[n_qubits] = sorted(deduped.values(), key=lambda row: (row.energy, row.name))
    return data


def _average_ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = 0.5 * (index + 1 + end)
        for key, _ in ordered[index:end]:
            ranks[key] = rank
        index = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    numerator = sum(a * b for a, b in zip(centered_left, centered_right))
    denom_left = sum(value * value for value in centered_left)
    denom_right = sum(value * value for value in centered_right)
    if denom_left <= 0.0 or denom_right <= 0.0:
        return None
    return numerator / ((denom_left * denom_right) ** 0.5)


def _spearman_by_family(left: list[EnumRow], right: list[EnumRow]) -> float | None:
    left_best = _best_energy_by_family(left)
    right_best = _best_energy_by_family(right)
    common = sorted(set(left_best) & set(right_best))
    if len(common) < 2:
        return None
    left_ranks = _average_ranks({key: left_best[key] for key in common})
    right_ranks = _average_ranks({key: right_best[key] for key in common})
    return _pearson([left_ranks[key] for key in common], [right_ranks[key] for key in common])


def _best_energy_by_family(rows: list[EnumRow]) -> dict[str, float]:
    best: dict[str, float] = {}
    for row in rows:
        best[row.family] = min(best.get(row.family, row.energy), row.energy)
    return best


def _top_k(count: int) -> int:
    if count <= 0:
        return 0
    return min(count, max(10, int(-(-count // 10))))


def _top_family_set(rows: list[EnumRow], k: int | None = None) -> set[str]:
    effective_k = _top_k(len(rows)) if k is None else min(max(0, int(k)), len(rows))
    return {row.family for row in rows[:effective_k]}


def _top_family_overlap(left: list[EnumRow], right: list[EnumRow], k: int | None = None) -> float | None:
    left_families = _top_family_set(left, k=k)
    right_families = _top_family_set(right, k=k)
    denominator = min(len(left_families), len(right_families))
    if denominator <= 0:
        return None
    return len(left_families & right_families) / denominator


def _format_optional(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def analyze(input_dir: Path, scales: tuple[int, ...], output_dir: Path) -> dict[str, object]:
    data = _load_rows(input_dir, scales)
    _write_merged_csvs(output_dir, data)
    pairs = list(zip(scales, scales[1:]))
    migration = []
    for left_n, right_n in pairs:
        left = data[left_n]
        right = data[right_n]
        default_overlap = _top_family_overlap(left, right)
        row = {
            "pair": f"{left_n}->{right_n}",
            "family_spearman": _spearman_by_family(left, right),
            "topK_family_overlap": default_overlap,
            "top5_family_overlap": _top_family_overlap(left, right, k=5),
            "top10_family_overlap": _top_family_overlap(left, right, k=10),
            "top20_family_overlap": _top_family_overlap(left, right, k=20),
        }
        row["allows_weak_extrapolation"] = (
            row["family_spearman"] is not None
            and row["topK_family_overlap"] is not None
            and row["family_spearman"] >= 0.6
            and row["topK_family_overlap"] >= 0.5
        )
        migration.append(row)

    family_tables = {
        str(n_qubits): [
            {"family": family, "best_energy": energy}
            for family, energy in sorted(_best_energy_by_family(rows).items(), key=lambda item: (item[1], item[0]))
        ]
        for n_qubits, rows in data.items()
    }
    report = {
        "input_dir": str(input_dir),
        "scales": list(scales),
        "migration": migration,
        "family_tables": family_tables,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase1_migration_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "VQE-QAS v3 Phase 1 migration analysis",
        f"input_dir: {input_dir}",
        "pair | family_spearman | topK_family_overlap | top5 | top10 | top20 | allows_weak_extrapolation",
    ]
    for row in migration:
        lines.append(
            f"{row['pair']} | {_format_optional(row['family_spearman'])} | "
            f"{_format_optional(row['topK_family_overlap'])} | {_format_optional(row['top5_family_overlap'])} | "
            f"{_format_optional(row['top10_family_overlap'])} | {_format_optional(row['top20_family_overlap'])} | "
            f"{row['allows_weak_extrapolation']}"
        )
    (output_dir / "phase1_migration_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _write_merged_csvs(output_dir: Path, data: dict[int, list[EnumRow]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for n_qubits, rows in data.items():
        path = output_dir / f"phase1_uniform_enum_{n_qubits}q.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["rank", "name", "family", "energy", "delta_ref"],
            )
            writer.writeheader()
            for rank, row in enumerate(rows, start=1):
                writer.writerow(
                    {
                        "rank": rank,
                        "name": row.name,
                        "family": row.family,
                        "energy": f"{row.energy:.12f}",
                        "delta_ref": f"{row.delta_ref:.12f}",
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze VQE-QAS v3 Phase 1 migration from enumeration CSVs")
    parser.add_argument("--input-dir", default="outputs/vqe_tfim_v3_scaling_npu_full")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--scales", default="4,6,8")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    report = analyze(input_dir=input_dir, scales=_parse_int_list(args.scales), output_dir=output_dir)
    print((output_dir / "phase1_migration_analysis.md").read_text(encoding="utf-8"), end="")
    print(f"families_by_scale: {', '.join(f'{scale}q={len(rows)}' for scale, rows in report['family_tables'].items())}")


if __name__ == "__main__":
    main()
