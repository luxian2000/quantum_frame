"""V3 Phase-0/Phase-1 TFIM scaling entry point.

Phase 0 validates dense TFIM references against the open-chain free-fermion
formula. Phase 1 runs uniform-budget HEA full enumeration for 4/6/8q.

NPU full run:
    python aicir/qas/demos/vqe_tfim_v3_scaling.py --mode all --backend npu --no-fallback-to-cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aicir.qas.vqe_hea_demo import (
    THETA_INIT_RANDOM_UNIFORM_PI,
    get_structure_family,
    resolve_qas_backend,
    run_tfim_full_enumeration_baseline,
    validate_tfim_reference_alignment,
)
from aicir.qas.task_evaluation import parameter_count


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_enumeration_csv(path: Path, report: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reference_energy = float(report.metadata.get("reference_energy", 0.0))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "name",
                "family",
                "energy",
                "delta_ref",
                "n_params",
                "nfev",
                "n_starts",
                "theta_init_mode",
                "budget_per_start",
            ],
        )
        writer.writeheader()
        for rank, result in enumerate(report.results, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "name": result.architecture.name,
                    "family": get_structure_family(result.architecture),
                    "energy": f"{result.energy:.12f}",
                    "delta_ref": f"{result.energy - reference_energy:.12f}",
                    "n_params": parameter_count(result.architecture.circuit),
                    "nfev": result.evaluations,
                    "n_starts": result.n_starts,
                    "theta_init_mode": result.metadata.get("theta_init_mode", ""),
                    "budget_per_start": result.metadata.get("budget_per_start", ""),
                }
            )


def _enum_prefix(output_dir: Path, n_qubits: int, shard_index: int, num_shards: int) -> Path:
    base = f"phase1_uniform_enum_{n_qubits}q"
    if int(num_shards) > 1:
        base = f"{base}_shard{int(shard_index)}of{int(num_shards)}"
    return output_dir / base


def main() -> None:
    parser = argparse.ArgumentParser(description="VQE-QAS TFIM v3 reference alignment and uniform enumeration")
    parser.add_argument("--mode", choices=("reference", "enumerate", "all"), default="all")
    parser.add_argument("--scales", default="4,6,8")
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--fair-n-starts", type=int, default=1)
    parser.add_argument("--fair-evals-per-param", type=int, default=200)
    parser.add_argument("--fair-min-evals", type=int, default=1000)
    parser.add_argument("--fair-max-evals", type=int, default=1000000)
    parser.add_argument("--init-mode", default=THETA_INIT_RANDOM_UNIFORM_PI)
    parser.add_argument("--backend", choices=("numpy", "cpu", "npu", "torch"), default="numpy")
    parser.add_argument("--no-fallback-to-cpu", action="store_true")
    parser.add_argument("--dtype", choices=("complex128", "complex64"), default="complex128")
    parser.add_argument("--output-dir", default="outputs/vqe_tfim_v3_scaling")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    scales = _parse_int_list(args.scales)
    output_dir = Path(args.output_dir)

    if args.mode in {"reference", "all"}:
        reference = validate_tfim_reference_alignment(scales=scales, J=args.J, h=args.h, periodic=False)
        lines = reference.summary_lines()
        print("\n".join(lines), flush=True)
        _write_text(output_dir / "phase0_reference_alignment.md", lines)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "phase0_reference_alignment.json").write_text(
            json.dumps(
                {
                    "passed": reference.passed,
                    "tolerance": reference.tolerance,
                    "rows": [row.__dict__ for row in reference.rows],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    if args.mode in {"enumerate", "all"}:
        backend = resolve_qas_backend(args.backend, fallback_to_cpu=not args.no_fallback_to_cpu, dtype=args.dtype)
        for n_qubits in scales:
            prefix = _enum_prefix(output_dir, n_qubits, args.shard_index, args.num_shards)
            checkpoint_path = str(prefix.with_suffix(".partial.csv"))
            print(
                f"\nRunning uniform enumeration: n={n_qubits}, backend={backend.name}, "
                f"shard={args.shard_index}/{args.num_shards}, checkpoint={checkpoint_path}",
                flush=True,
            )
            report = run_tfim_full_enumeration_baseline(
                n_qubits=n_qubits,
                J=args.J,
                h=args.h,
                periodic=False,
                seed=args.seed,
                candidate_limit=args.candidate_limit,
                fair_n_starts=args.fair_n_starts,
                fair_evals_per_param=args.fair_evals_per_param,
                fair_min_evaluations=args.fair_min_evals,
                fair_max_evaluations=args.fair_max_evals,
                adaptive_fair_starts=False,
                init_mode=args.init_mode,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                verbose=args.verbose,
                checkpoint_path=checkpoint_path,
                backend=backend,
            )
            lines = report.summary_lines(top_k=args.top_k)
            print("\n".join(lines), flush=True)
            _write_text(prefix.with_suffix(".md"), lines)
            _write_enumeration_csv(prefix.with_suffix(".csv"), report)
            output_dir.mkdir(parents=True, exist_ok=True)
            prefix.with_suffix(".metadata.json").write_text(
                json.dumps(report.metadata, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
