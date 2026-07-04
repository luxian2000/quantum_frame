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
PROBE = ROOT / "scripts" / "npu" / "multi_card_probe.py"


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
        str(PROBE),
        "--section",
        args.section,
        "--supernet-steps",
        str(args.supernet_steps),
        "--finetune-steps",
        str(args.finetune_steps),
        "--ranking-num",
        str(args.ranking_num),
    ]
    if args.allow_cpu_fallback:
        cmd.append("--allow-cpu-fallback")
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch the strict multi-card NPU validation probe with torch.distributed.run."
    )
    parser.add_argument("--nproc-per-node", type=int, default=2, help="Number of local NPU ranks to launch.")
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated physical NPU ids exported as ASCEND_RT_VISIBLE_DEVICES, e.g. 0,5,6,7.",
    )
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29623)
    parser.add_argument(
        "--section",
        choices=("all", "collectives", "typed-ir", "deriv", "supernet"),
        default="all",
        help="Probe section to run on every rank.",
    )
    parser.add_argument("--supernet-steps", type=int, default=4)
    parser.add_argument("--finetune-steps", type=int, default=3)
    parser.add_argument("--ranking-num", type=int, default=4)
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Local development only. Real multi-card validation should omit this.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the torchrun command without executing it.")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    if args.nproc_per_node < 2:
        raise SystemExit("--nproc-per-node must be >= 2 for multi-card validation")
    devices = _device_list(args.devices)
    if devices and len(devices) < int(args.nproc_per_node):
        raise SystemExit(
            f"--devices lists {len(devices)} device(s), but --nproc-per-node={args.nproc_per_node}"
        )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if devices:
        env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(devices)

    cmd = build_command(args)
    if args.dry_run:
        print(f"cd {shlex.quote(str(ROOT))}")
        if devices:
            print(f"ASCEND_RT_VISIBLE_DEVICES={env['ASCEND_RT_VISIBLE_DEVICES']}")
        print(quote_cmd(cmd))
        return 0

    print(f"+ {quote_cmd(cmd)}", flush=True)
    completed = subprocess.run(cmd, cwd=ROOT, env=env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
