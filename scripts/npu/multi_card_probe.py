#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from typed_ir_deriv_probe import DERIV_CASES, TYPED_IR_CASES  # noqa: E402

from aicir import Hamiltonian, NPUBackend  # noqa: E402
from aicir.backends.npu_backend import is_npu_available  # noqa: E402
from aicir.qas.algorithms.supernet import supernet_qas  # noqa: E402
from aicir.qas.core.sharding import (  # noqa: E402
    all_gather,
    all_reduce_mean,
    broadcast_parameters,
    shard_context,
)


Case = tuple[str, Callable[[NPUBackend], None]]


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _log(message: str) -> None:
    print(f"[rank {_rank()}] {message}", flush=True)


def _to_numpy(tensor) -> np.ndarray:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    return np.asarray(tensor)


def _backend(allow_cpu_fallback: bool) -> NPUBackend:
    if not allow_cpu_fallback and not is_npu_available():
        raise SystemExit("NPU is unavailable. Multi-card validation requires a real Ascend NPU runtime.")

    backend = NPUBackend.from_distributed_env(fallback_to_cpu=allow_cpu_fallback)
    ctx = backend.runtime_context
    if ctx is None or not ctx.distributed:
        raise AssertionError("multi-card probe must be launched by torch.distributed.run with WORLD_SIZE > 1")
    if not ctx.process_group_initialized or not dist.is_initialized():
        raise AssertionError("torch.distributed process group was not initialized")

    device_type = getattr(backend._device, "type", None)
    pg_backend = str(dist.get_backend()).lower()
    if not allow_cpu_fallback:
        if device_type != "npu":
            raise AssertionError(f"strict multi-card probe resolved device={backend._device!r}, expected npu")
        if pg_backend != "hccl":
            raise AssertionError(f"strict multi-card probe initialized backend={pg_backend!r}, expected hccl")

    _log(f"backend={backend.name} runtime={backend.runtime_context} dist_backend={pg_backend}")
    return backend


def case_collectives(backend: NPUBackend) -> None:
    ctx = shard_context(backend)
    if not ctx.is_sharded:
        raise AssertionError(f"shard_context did not activate: {ctx}")
    if ctx.rank != dist.get_rank() or ctx.world_size != dist.get_world_size():
        raise AssertionError(f"shard_context mismatch: {ctx}")

    gathered = all_gather({"rank": ctx.rank, "device": str(backend._device)})
    ranks = [item["rank"] for item in gathered]
    if ranks != list(range(ctx.world_size)):
        raise AssertionError(f"all_gather returned ranks={ranks}, expected {list(range(ctx.world_size))}")

    broadcasted = torch.tensor(
        [10.0 + float(ctx.rank), 20.0 + float(ctx.rank)],
        dtype=torch.float32,
        device=backend._device,
    )
    if ctx.rank == 0:
        broadcasted = torch.tensor([3.0, 7.0], dtype=torch.float32, device=backend._device)
    broadcast_parameters({"theta": broadcasted}, src=0)
    if not np.allclose(_to_numpy(broadcasted), np.asarray([3.0, 7.0], dtype=np.float32), atol=1e-6):
        raise AssertionError(f"broadcast_parameters mismatch on rank {ctx.rank}: {broadcasted}")

    reduced = torch.tensor(
        [float(ctx.rank + 1), float((ctx.rank + 1) ** 2)],
        dtype=torch.float32,
        device=backend._device,
    )
    all_reduce_mean([reduced])
    expected = np.asarray(
        [
            (ctx.world_size + 1.0) / 2.0,
            sum(float((rank + 1) ** 2) for rank in range(ctx.world_size)) / float(ctx.world_size),
        ],
        dtype=np.float32,
    )
    if not np.allclose(_to_numpy(reduced), expected, atol=1e-6):
        raise AssertionError(f"all_reduce_mean mismatch on rank {ctx.rank}: actual={_to_numpy(reduced)} expected={expected}")


def case_typed_ir(backend: NPUBackend) -> None:
    for name, fn in TYPED_IR_CASES:
        fn(backend)
        _log(f"[PASS] {name}")


def case_deriv(backend: NPUBackend) -> None:
    for name, fn in DERIV_CASES:
        fn(backend)
        _log(f"[PASS] {name}")


def _run_supernet_mode(
    backend: NPUBackend,
    *,
    mode: str,
    supernet_steps: int,
    finetune_steps: int,
    ranking_num: int,
) -> None:
    ctx = shard_context(backend)
    ham = Hamiltonian(n_qubits=2, terms=[("ZZ", -1.0), ("XI", 0.2), ("IX", 0.2)])
    result = supernet_qas(
        ham,
        layers=1,
        supernet_num=max(2, ctx.world_size),
        supernet_steps=supernet_steps,
        finetune_steps=finetune_steps,
        ranking_num=max(ranking_num, ctx.world_size),
        seed=3,
        device=str(backend._device),
        mode=mode,
    )
    energy = float(result.final_metrics["fine_tuned_energy"])
    if not math.isfinite(energy):
        raise AssertionError(f"supernet {mode} returned non-finite energy: {energy}")

    payloads = all_gather({"rank": ctx.rank, "mode": mode, "energy": energy})
    energies = [float(item["energy"]) for item in payloads]
    if max(energies) - min(energies) > 1e-4:
        raise AssertionError(f"supernet {mode} energies differ across ranks: {energies}")
    if ctx.rank == 0:
        print(f"[supernet:{mode}] energies={energies}", flush=True)


def case_supernet(
    backend: NPUBackend,
    *,
    supernet_steps: int,
    finetune_steps: int,
    ranking_num: int,
) -> None:
    _run_supernet_mode(
        backend,
        mode="safe",
        supernet_steps=supernet_steps,
        finetune_steps=finetune_steps,
        ranking_num=ranking_num,
    )
    _run_supernet_mode(
        backend,
        mode="aggressive",
        supernet_steps=supernet_steps,
        finetune_steps=finetune_steps,
        ranking_num=ranking_num,
    )


def _selected_cases(args: argparse.Namespace) -> list[tuple[str, Callable[[NPUBackend], None]]]:
    cases: list[tuple[str, Callable[[NPUBackend], None]]] = []
    if args.section in {"all", "collectives"}:
        cases.append(("collectives", case_collectives))
    if args.section in {"all", "typed-ir"}:
        cases.append(("typed_ir", case_typed_ir))
    if args.section in {"all", "deriv"}:
        cases.append(("deriv", case_deriv))
    if args.section in {"all", "supernet"}:
        cases.append((
            "supernet",
            lambda backend: case_supernet(
                backend,
                supernet_steps=args.supernet_steps,
                finetune_steps=args.finetune_steps,
                ranking_num=args.ranking_num,
            ),
        ))
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank-local probe for strict multi-card Ascend NPU validation."
    )
    parser.add_argument(
        "--section",
        choices=("all", "collectives", "typed-ir", "deriv", "supernet"),
        default="all",
        help="Subset to run on every distributed rank.",
    )
    parser.add_argument("--supernet-steps", type=int, default=4)
    parser.add_argument("--finetune-steps", type=int, default=3)
    parser.add_argument("--ranking-num", type=int, default=4)
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Local development only; strict multi-card NPU validation should omit this.",
    )
    args = parser.parse_args(argv)

    backend = _backend(args.allow_cpu_fallback)
    failed: list[tuple[str, str]] = []
    try:
        for name, fn in _selected_cases(args):
            try:
                fn(backend)
            except Exception as exc:  # noqa: BLE001
                failed.append((name, f"{type(exc).__name__}: {exc}"))
                _log(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            else:
                _log(f"[PASS] {name}")
        if failed:
            if _rank() == 0:
                print("\nSummary: FAILED", flush=True)
                for name, message in failed:
                    print(f"- {name}: {message}", flush=True)
            return 1
        if _rank() == 0:
            print("\nSummary: PASS", flush=True)
        return 0
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
