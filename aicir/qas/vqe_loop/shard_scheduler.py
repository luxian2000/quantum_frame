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
            failures.append({"shard": shard_index, "start": start, "end": end, "return_code": return_code})
        if not shard_output.exists():
            failures.append({"shard": shard_index, "start": start, "end": end, "missing_output": str(shard_output)})
    if failures:
        raise RuntimeError(f"Fair-label shard failure(s): {failures}")

    merged: list[dict[str, str]] = []
    for _shard_index, _start, _end, _shard_queue, shard_output, _process in processes:
        merged.extend(read_csv_rows(shard_output))
    write_csv_rows(output_path, merged, fieldnames=BENCHMARK_TABLE_FIELDS)

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
    args = parser.parse_args()

    summary = run_sharded_labels(args)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
