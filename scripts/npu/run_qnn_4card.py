#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demos" / "demo_npu_qnn_4card.py"


def quote_cmd(parts: Iterable[str | os.PathLike[str]]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _device_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one device id")
    return devices


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--master_addr",
        args.master_addr,
        "--master_port",
        str(args.master_port),
        str(DEMO),
        "--expected-world-size",
        str(args.nproc_per_node),
        "--samples",
        str(args.samples),
        "--layers",
        str(args.layers),
        "--steps",
        str(args.steps),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--log-every",
        str(args.log_every),
    ]
    if args.allow_cpu_fallback:
        cmd.append("--allow-cpu-fallback")
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch the four-card typed-IR QNN demo with torch.distributed.run."
    )
    parser.add_argument("--nproc-per-node", type=int, default=4, help="Number of local NPU ranks.")
    parser.add_argument(
        "--devices",
        default=None,
        help="Deprecated compatibility option; ignored to match demos/BeH2 LOCAL_RANK -> npu:LOCAL_RANK.",
    )
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29633)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=4)
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Local development only. Real 4-card validation should omit this.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the torchrun command without executing it.")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    if args.nproc_per_node < 2:
        raise SystemExit("--nproc-per-node must be >= 2 for the multi-card QNN demo")
    devices = _device_list(args.devices)
    if devices and len(devices) < int(args.nproc_per_node):
        raise SystemExit(
            f"--devices lists {len(devices)} device(s), but --nproc-per-node={args.nproc_per_node}"
        )
    if args.samples < args.nproc_per_node:
        raise SystemExit("--samples must be >= --nproc-per-node")
    if args.samples % args.nproc_per_node != 0:
        raise SystemExit("--samples must be divisible by --nproc-per-node")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    cmd = build_command(args)
    if args.dry_run:
        print(f"cd {shlex.quote(str(ROOT))}")
        if devices:
            print("# --devices is ignored; using torchrun LOCAL_RANK -> npu:LOCAL_RANK like demos/BeH2")
        print(quote_cmd(cmd))
        return 0

    if devices:
        print("# --devices is ignored; using torchrun LOCAL_RANK -> npu:LOCAL_RANK like demos/BeH2", flush=True)
    print(f"+ {quote_cmd(cmd)}", flush=True)
    completed = subprocess.run(cmd, cwd=ROOT, env=env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
