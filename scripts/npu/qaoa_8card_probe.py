#!/usr/bin/env python3
"""Eight-card NPU QAOA probe with rank-local circuits and HCCL real collectives."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aicir import Hamiltonian, NPUBackend  # noqa: E402
from aicir.backends.npu_backend import is_npu_available  # noqa: E402
from aicir.qas.core.sharding import all_gather, all_reduce_mean, broadcast_parameters, shard_context  # noqa: E402
from aicir.vqc import BasicQAOA  # noqa: E402


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _log(message: str) -> None:
    print(f"[rank {_rank()}] {message}", flush=True)


def _to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        tensor = torch.real(value) if torch.is_complex(value) else value
        return float(tensor.detach().cpu().reshape(()).item())
    array = np.asarray(value)
    if np.iscomplexobj(array):
        array = array.real
    return float(array.reshape(()))


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(float, copy=True)


def _backend(allow_cpu_fallback: bool) -> NPUBackend:
    if not allow_cpu_fallback and not is_npu_available():
        raise SystemExit("NPU is unavailable. Multi-card QAOA validation requires a real Ascend NPU runtime.")

    backend = NPUBackend.from_distributed_env(fallback_to_cpu=allow_cpu_fallback)
    ctx = backend.runtime_context
    if ctx is None or not ctx.distributed:
        raise AssertionError("QAOA 8-card probe must be launched by torch.distributed.run with WORLD_SIZE > 1")
    if not ctx.process_group_initialized or not dist.is_initialized():
        raise AssertionError("torch.distributed process group was not initialized")

    device_type = getattr(backend._device, "type", None)
    pg_backend = str(dist.get_backend()).lower()
    if not allow_cpu_fallback:
        if device_type != "npu":
            raise AssertionError(f"strict QAOA 8-card probe resolved device={backend._device!r}, expected npu")
        if pg_backend != "hccl":
            raise AssertionError(f"strict QAOA 8-card probe initialized backend={pg_backend!r}, expected hccl")

    _log(f"backend={backend.name} runtime={backend.runtime_context} dist_backend={pg_backend}")
    return backend


def _validate_args(args: argparse.Namespace, world_size: int) -> None:
    if args.expected_world_size is not None and int(args.expected_world_size) != int(world_size):
        raise SystemExit(f"expected WORLD_SIZE={args.expected_world_size}, got {world_size}")
    if args.samples < world_size:
        raise SystemExit("--samples must be >= WORLD_SIZE")
    if args.samples % world_size != 0:
        raise SystemExit("--samples must be divisible by WORLD_SIZE")
    if args.p < 1:
        raise SystemExit("--p must be >= 1")
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if args.lr <= 0:
        raise SystemExit("--lr must be positive")
    if args.eps <= 0:
        raise SystemExit("--eps must be positive")
    if args.shots <= 0:
        raise SystemExit("--shots must be positive")
    if args.log_every < 1:
        raise SystemExit("--log-every must be >= 1")


def _problem_hamiltonian(index: int) -> Hamiltonian:
    scale = 1.0 + 0.03 * float(index % 7)
    return Hamiltonian(
        n_qubits=2,
        terms=[
            ("ZI", 0.35 * scale),
            ("IZ", -0.22 + 0.01 * float(index % 5)),
            ("ZZ", -0.65),
            ("X", [0], 0.08 * float((index % 3) - 1)),
            ("XY", [0, 1], 0.04 * float((index % 4) - 1.5)),
        ],
    )


def _trotter_order(index: int) -> int:
    return 2 if index % 2 else 1


def _energy(backend: NPUBackend, theta: np.ndarray, *, problem_index: int, p: int) -> float:
    qaoa = BasicQAOA(
        problem_hamiltonian=_problem_hamiltonian(problem_index),
        p=p,
        trotter_steps=2,
        trotter_order=_trotter_order(problem_index),
    )
    return float(qaoa.energy(theta, backend=backend))


def _local_loss(backend: NPUBackend, theta: np.ndarray, indices: list[int], *, p: int) -> float:
    values = [_energy(backend, theta, problem_index=index, p=p) for index in indices]
    value = float(np.mean(values))
    if not math.isfinite(value):
        raise FloatingPointError(f"non-finite local QAOA loss: {value!r}")
    return value


def _finite_difference_grad(
    backend: NPUBackend,
    theta: np.ndarray,
    indices: list[int],
    *,
    p: int,
    eps: float,
) -> np.ndarray:
    grad = np.zeros_like(theta, dtype=np.float32)
    for param_index in range(theta.size):
        plus = theta.copy()
        minus = theta.copy()
        plus[param_index] += eps
        minus[param_index] -= eps
        grad[param_index] = (
            _local_loss(backend, plus, indices, p=p) - _local_loss(backend, minus, indices, p=p)
        ) / (2.0 * eps)
    return grad


def _diagonal_sampling_check(backend: NPUBackend, *, shots: int, seed: int) -> None:
    qaoa = BasicQAOA(
        problem_hamiltonian=Hamiltonian(n_qubits=2, terms=[("Z", [0], 0.4), ("ZZ", [0, 1], -0.5)]),
        p=1,
    )
    params = np.array([0.2, -0.1], dtype=float)
    counts = qaoa.sample(params, backend=backend, shots=shots, seed=seed + _rank())
    if sum(counts.values()) != shots:
        raise AssertionError(f"diagonal QAOA counts do not sum to shots: counts={counts!r}, shots={shots}")
    sampled = qaoa.energy(params, backend=backend, shots=shots, seed=seed + _rank())
    if not math.isfinite(float(sampled)):
        raise FloatingPointError(f"non-finite diagonal sampled energy: {sampled!r}")


def run(args: argparse.Namespace) -> int:
    backend = _backend(args.allow_cpu_fallback)
    ctx = shard_context(backend)
    if not ctx.is_sharded:
        raise AssertionError(f"shard_context did not activate: {ctx}")
    _validate_args(args, ctx.world_size)

    rank = int(ctx.rank)
    world_size = int(ctx.world_size)
    device = backend._device
    local_indices = list(range(rank, args.samples, world_size))
    if not local_indices:
        raise RuntimeError(f"rank {rank} received an empty QAOA problem shard")

    rng = np.random.default_rng(args.seed)
    init_np = rng.normal(loc=0.0, scale=0.35, size=2 * int(args.p)).astype(np.float32)
    theta = torch.tensor(init_np, dtype=torch.float32, device=device)
    if rank != 0:
        theta.zero_()
    broadcast_parameters({"theta": theta}, src=0)

    _diagonal_sampling_check(backend, shots=int(args.shots), seed=int(args.seed))

    initial_local = _local_loss(backend, _to_numpy(theta), local_indices, p=int(args.p))
    initial_global = torch.tensor([initial_local], dtype=torch.float32, device=device)
    all_reduce_mean([initial_global])

    last_local = initial_local
    last_global = initial_global
    last_grad_norm = torch.tensor(0.0, dtype=torch.float32, device=device)

    for step in range(1, int(args.steps) + 1):
        theta_np = _to_numpy(theta)
        grad_np = _finite_difference_grad(
            backend,
            theta_np,
            local_indices,
            p=int(args.p),
            eps=float(args.eps),
        )
        grad = torch.tensor(grad_np, dtype=torch.float32, device=device)
        all_reduce_mean([grad])
        last_grad_norm = torch.linalg.vector_norm(grad)
        with torch.no_grad():
            theta -= float(args.lr) * grad

        last_local = _local_loss(backend, _to_numpy(theta), local_indices, p=int(args.p))
        last_global = torch.tensor([last_local], dtype=torch.float32, device=device)
        all_reduce_mean([last_global])
        if rank == 0 and (step == 1 or step == args.steps or step % args.log_every == 0):
            print(
                f"[step {step:03d}] global_loss={_to_float(last_global):.8f} "
                f"grad_norm={_to_float(last_grad_norm):.8f}",
                flush=True,
            )

    payloads = all_gather(
        {
            "rank": rank,
            "device": str(device),
            "local_indices": local_indices,
            "initial_local_loss": initial_local,
            "final_local_loss": last_local,
            "final_global_loss": _to_float(last_global),
            "last_grad_norm": _to_float(last_grad_norm),
        }
    )

    if rank == 0:
        devices = [item["device"] for item in payloads]
        all_indices = sorted(index for item in payloads for index in item["local_indices"])
        if all_indices != list(range(args.samples)):
            raise AssertionError(f"QAOA problem shards do not cover samples exactly: {all_indices!r}")
        print("\nSummary: PASS", flush=True)
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "devices": devices,
                    "samples": args.samples,
                    "p": args.p,
                    "steps": args.steps,
                    "initial_global_loss": _to_float(initial_global),
                    "final_global_loss": _to_float(last_global),
                    "rank_payloads": payloads,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Eight-card NPU QAOA probe with rank-local circuits and HCCL real collectives."
    )
    parser.add_argument("--samples", type=int, default=16, help="Problem instances; must divide WORLD_SIZE.")
    parser.add_argument("--p", type=int, default=1, help="QAOA depth.")
    parser.add_argument("--steps", type=int, default=4, help="Shared-parameter finite-difference update steps.")
    parser.add_argument("--lr", type=float, default=0.15, help="Learning rate.")
    parser.add_argument("--eps", type=float, default=1e-3, help="Finite-difference gradient epsilon.")
    parser.add_argument("--shots", type=int, default=64, help="Shots for the diagonal sampling smoke path.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--expected-world-size",
        type=int,
        default=None,
        help="Optional guard passed by scripts/npu/qaoa_8card.sh.",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Local development only. Real 8-card validation should omit this.",
    )
    args = parser.parse_args(argv)

    try:
        return run(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
