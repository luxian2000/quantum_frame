"""Run a fair-label queue as independent contiguous shards.

This scheduler wraps ``python -m aicir.qas.vqe_loop.fair_labeling`` for multi-NPU nodes.  It
keeps each VQE task independent, assigns ``LOCAL_RANK`` per shard, preserves
global seed offsets, and merges shard CSVs back into one label table.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aicir.qas.vqe_loop.benchmark_table import BENCHMARK_TABLE_FIELDS
from aicir.qas.vqe_loop.benchmark_table import read_csv_rows, write_csv_rows


def _is_completed_label(row: Mapping[str, Any]) -> bool:
    status = str(row.get("label_status", "") or "").strip().lower()
    energy = str(row.get("fair_best_energy", "") or "").strip().lower()
    return status == "completed" or energy not in {"", "nan", "none", "null"}



def _format_shard_failure(
    *,
    shard_index: int,
    start: int,
    end: int,
    return_code: int | None = None,
    shard_output: Path | None = None,
    missing_output: bool = False,
) -> dict[str, Any]:
    failure: dict[str, Any] = {"shard": shard_index, "start": start, "end": end}
    if return_code is not None:
        failure["return_code"] = return_code
        if return_code < 0:
            signal_number = abs(int(return_code))
            signal_name = "SIGKILL" if signal_number == 9 else f"signal-{signal_number}"
            failure["signal"] = signal_name
            failure["possible_oom"] = signal_number == 9
            if signal_number == 9:
                failure["hint"] = "worker was killed by SIGKILL; on NPU runs this usually means host OOM or cgroup memory limit, not a Python exception"
    if shard_output is not None:
        failure["output"] = str(shard_output)
        if shard_output.exists():
            try:
                rows = read_csv_rows(shard_output)
                counts: dict[str, int] = {}
                for row in rows:
                    status = str(row.get("label_status", "") or "blank").strip() or "blank"
                    counts[status] = counts.get(status, 0) + 1
                failure["shard_output_status"] = ",".join(f"{key}={value}" for key, value in sorted(counts.items()))
            except Exception as exc:  # pragma: no cover - best-effort diagnostic only.
                failure["shard_output_status_error"] = repr(exc)
        elif missing_output:
            failure["missing_output"] = str(shard_output)
    return failure

def _merge_shard_outputs(
    shard_outputs: list[Path],
    output_path: Path,
    *,
    completed_only: bool = False,
) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for shard_output in shard_outputs:
        rows = read_csv_rows(shard_output)
        if completed_only:
            rows = [row for row in rows if _is_completed_label(row)]
        merged.extend(rows)
    write_csv_rows(output_path, merged, fieldnames=BENCHMARK_TABLE_FIELDS)
    return merged


def _contiguous_shards(n_rows: int, num_shards: int) -> list[tuple[int, int]]:
    if int(num_shards) < 1:
        raise ValueError("num_shards must be >= 1")
    if int(n_rows) < 1:
        return []
    effective = min(int(num_shards), int(n_rows))
    base, remainder = divmod(int(n_rows), effective)
    bounds: list[tuple[int, int]] = []
    start = 0
    for shard_index in range(effective):
        size = base + (1 if shard_index < remainder else 0)
        end = start + size
        bounds.append((start, end))
        start = end
    return bounds


def _shard_environment(
    base_env: Mapping[str, str],
    *,
    shard_index: int,
    device_offset: int,
    num_shards: int,
) -> dict[str, str]:
    env = dict(base_env)
    env["LOCAL_RANK"] = str(int(device_offset) + int(shard_index))
    env["AICIR_QAS_SHARD_INDEX"] = str(int(shard_index))
    env["AICIR_QAS_NUM_SHARDS"] = str(int(num_shards))

    # Queue sharding is task parallelism.  Do not inherit distributed variables
    # that would make NPUBackend initialize HCCL for independent VQE jobs.
    for key in ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    return env


def _runner_command(args: argparse.Namespace, shard_queue: Path, shard_output: Path, start_index: int) -> list[str]:
    command = [
        str(args.python),
        "-m",
        "aicir.qas.vqe_loop.fair_labeling",
        "--queue",
        str(shard_queue),
        "--output",
        str(shard_output),
        "--protocol",
        str(args.protocol),
        "--seed",
        str(args.seed),
        "--seed-index-offset",
        str(start_index),
        "--n-seeds",
        str(args.n_seeds),
        "--success-delta-ref",
        str(args.success_delta_ref),
        "--backend",
        str(args.backend),
        "--dtype",
        str(args.dtype),
    ]
    if args.max_evals is not None:
        command.extend(["--max-evals", str(args.max_evals)])
    if args.dry_run:
        command.append("--dry-run")
    return command


def run_sharded_labels(args: argparse.Namespace) -> dict[str, Any]:
    queue_path = Path(args.queue)
    output_path = Path(args.output)
    work_dir = Path(args.work_dir) if args.work_dir else output_path.with_suffix("")
    work_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(queue_path)
    bounds = _contiguous_shards(len(rows), int(args.num_shards))
    processes: list[tuple[int, int, int, Path, Path, subprocess.Popen[str]]] = []
    for shard_index, (start, end) in enumerate(bounds):
        shard_queue = work_dir / f"{output_path.stem}.shard{shard_index:02d}of{len(bounds):02d}.queue.csv"
        shard_output = work_dir / f"{output_path.stem}.shard{shard_index:02d}of{len(bounds):02d}.csv"
        write_csv_rows(shard_queue, rows[start:end], fieldnames=BENCHMARK_TABLE_FIELDS)
        command = _runner_command(args, shard_queue, shard_output, start)
        env = _shard_environment(
            os.environ,
            shard_index=shard_index,
            device_offset=int(args.device_offset),
            num_shards=int(args.num_shards),
        )
        process = subprocess.Popen(command, env=env, text=True)
        processes.append((shard_index, start, end, shard_queue, shard_output, process))

    failures = []
    for shard_index, start, end, _shard_queue, shard_output, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append(
                _format_shard_failure(
                    shard_index=shard_index,
                    start=start,
                    end=end,
                    return_code=return_code,
                    shard_output=shard_output,
                )
            )
        if not shard_output.exists():
            failures.append(
                _format_shard_failure(
                    shard_index=shard_index,
                    start=start,
                    end=end,
                    shard_output=shard_output,
                    missing_output=True,
                )
            )
    if failures:
        raise RuntimeError(f"Fair-label shard failure(s): {failures}")

    _merge_shard_outputs(
        [shard_output for _shard_index, _start, _end, _shard_queue, shard_output, _process in processes],
        output_path,
    )

    summary = {
        "queue": str(queue_path),
        "output": str(output_path),
        "work_dir": str(work_dir),
        "rows": len(rows),
        "num_shards_requested": int(args.num_shards),
        "num_shards_launched": len(bounds),
        "device_offset": int(args.device_offset),
        "backend": str(args.backend),
        "dtype": str(args.dtype),
        "dry_run": bool(args.dry_run),
        "shards": [
            {
                "index": shard_index,
                "start": start,
                "end": end,
                "rows": end - start,
                "queue": str(shard_queue),
                "output": str(shard_output),
            }
            for shard_index, start, end, shard_queue, shard_output, _process in processes
        ],
    }
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".shard_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fair VQE labels as independent queue shards")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--protocol", default="default")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--device-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--success-delta-ref", type=float, default=0.02)
    parser.add_argument("--max-evals", type=int, default=None)
    parser.add_argument("--backend", default="npu")
    parser.add_argument("--dtype", default="complex64")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--merge-existing-only",
        action="store_true",
        help="Do not launch workers; merge already existing shard CSV outputs from --work-dir.",
    )
    parser.add_argument(
        "--completed-only",
        action="store_true",
        help="With --merge-existing-only, merge only rows that already have completed fair labels.",
    )
    args = parser.parse_args()

    if args.merge_existing_only:
        output_path = Path(args.output)
        work_dir = Path(args.work_dir) if args.work_dir else output_path.with_suffix("")
        shard_outputs = sorted(work_dir.glob(f"{output_path.stem}.shard??of??.csv"))
        if not shard_outputs:
            raise FileNotFoundError(f"No shard outputs found in {work_dir}")
        merged = _merge_shard_outputs(shard_outputs, output_path, completed_only=bool(args.completed_only))
        summary = {
            "queue": str(args.queue),
            "output": str(output_path),
            "work_dir": str(work_dir),
            "mode": "merge_existing_only",
            "completed_only": bool(args.completed_only),
            "rows": len(merged),
            "shards": [str(path) for path in shard_outputs],
        }
        summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".shard_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))
        return

    summary = run_sharded_labels(args)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
