"""Run BeH2 (16-qubit active space) supernet VQE on NPU.

Run from repository root:

    python -m demos.BeH2.BeH2_npu
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch


def _enable_npu() -> object:
    try:
        import torch_npu  # noqa: F401

        return True
    except Exception as exc:  # pragma: no cover
        return exc


def _device_report(device: str) -> list[str]:
    lines = [f"torch version            : {torch.__version__}"]
    npu_mod = getattr(torch, "npu", None)
    if npu_mod is not None:
        try:
            lines.append(f"torch.npu.is_available() : {npu_mod.is_available()}")
            lines.append(f"torch.npu.device_count() : {npu_mod.device_count()}")
        except Exception as exc:  # pragma: no cover
            lines.append(f"torch.npu query failed   : {exc!r}")
    else:
        lines.append("torch.npu                : not present")
    lines.append(f"requested device         : {device}")
    return lines


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BeH2 supernet VQE on NPU.")
    parser.add_argument("--device", default="npu:0", help="torch device, e.g. npu:0 / cpu")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--supernet-num", type=int, default=6)
    parser.add_argument("--supernet-steps", type=int, default=300)
    parser.add_argument("--ranking-num", type=int, default=120)
    parser.add_argument("--finetune-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("BeH2_npu_result.txt")),
        help="text report path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).with_name(args.output)

    handle = output_path.open("w", encoding="utf-8")

    def log(line: str = "") -> None:
        print(line)
        handle.write(line + "\n")
        handle.flush()

    try:
        log("=" * 72)
        log("BeH2 supernet ground-state search on NPU")
        log("=" * 72)
        log(f"timestamp                : {datetime.now().isoformat(timespec='seconds')}")
        log(f"platform                 : {platform.platform()}")

        if "npu" in args.device:
            enabled = _enable_npu()
            if enabled is not True:
                log(f"[warning] import torch_npu failed: {enabled!r}")
        for line in _device_report(args.device):
            log(line)

        from demos.BeH2.BeH2 import beh2_vqe_qas_kwargs, build_beh2_hamiltonian
        from aicir.qas import supernet_qas

        t0 = time.time()
        hamiltonian = build_beh2_hamiltonian()
        t1 = time.time()

        log("")
        log("search configuration:")
        log(f"  qubits                 : {hamiltonian.n_qubits}")
        log(f"  layers (depth)         : {args.layers}")
        log("  single-qubit gates     : (i, h, rx, ry, rz)")
        log("  two-qubit gates        : (cx, rzz)")
        log(f"  supernet_num (W)       : {args.supernet_num}")
        log(f"  supernet_steps         : {args.supernet_steps}")
        log(f"  ranking_num            : {args.ranking_num}")
        log(f"  finetune_steps         : {args.finetune_steps}")
        log(f"  seed                   : {args.seed}")
        log(f"  hamiltonian build time : {t1 - t0:.1f} s")

        kwargs = beh2_vqe_qas_kwargs()
        kwargs.update(
            {
                "layers": args.layers,
                "supernet_num": args.supernet_num,
                "supernet_steps": args.supernet_steps,
                "ranking_num": args.ranking_num,
                "finetune_steps": args.finetune_steps,
                "seed": args.seed,
                "device": args.device,
            }
        )

        log("")
        log("running supernet search ...")
        start = time.time()
        result = supernet_qas(hamiltonian, **kwargs)
        elapsed = time.time() - start

        metrics = result.final_metrics
        qas_energy = float(metrics["fine_tuned_energy"])
        baseline_energy = float(metrics["baseline_vqe_energy"])

        log("")
        log("-" * 72)
        log("results")
        log("-" * 72)
        log(f"QAS fine-tuned energy    : {qas_energy:+.10f} Ha")
        log(f"fixed-ansatz VQE baseline: {baseline_energy:+.10f} Ha")
        log(
            "selected CNOT / two-qubit: "
            f"{metrics['selected_cnot_count']} / {metrics['selected_two_qubit_count']}"
        )
        log(f"wall-clock time          : {elapsed:.1f} s")

        log("")
        log(f"text report saved to     : {output_path}")
    except Exception:
        log("")
        log("[ERROR] the run failed with an exception:")
        log(traceback.format_exc())
        raise
    finally:
        handle.close()


if __name__ == "__main__":
    main()
