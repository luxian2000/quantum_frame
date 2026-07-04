#!/usr/bin/env python3
"""Four-card NPU QNN demo using typed IR operations and HCCL gradient sync.

Run through the wrapper:

    scripts/npu/qnn_4card.sh --nproc-per-node 4

The demo is data parallel. Each rank keeps a full small statevector on its own
NPU, owns a shard of synthetic samples, computes autodiff gradients locally,
and averages real-valued parameter gradients through torch.distributed/HCCL.
It does not shard one statevector across multiple NPUs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aicir import Hamiltonian, NPUBackend, Operation  # noqa: E402
from aicir.backends.npu_backend import is_npu_available  # noqa: E402
from aicir.core.gates import apply_gate_to_state  # noqa: E402
from aicir.qas.core.sharding import all_gather, all_reduce_mean, broadcast_parameters  # noqa: E402


N_QUBITS = 2
PARAMS_PER_LAYER = 5


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


def _param_count(layers: int) -> int:
    return PARAMS_PER_LAYER * int(layers)


def _make_features(samples: int, device) -> torch.Tensor:
    angles = 2.0 * np.pi * (np.arange(samples, dtype=np.float32) + 0.5) / float(samples)
    features = np.stack(
        [
            np.sin(angles),
            np.cos(2.0 * angles),
        ],
        axis=1,
    ).astype(np.float32)
    return torch.tensor(features, dtype=torch.float32, device=device)


def _qnn_prediction(
    backend: NPUBackend,
    feature: torch.Tensor,
    theta: torch.Tensor,
    observable: torch.Tensor,
    layers: int,
) -> torch.Tensor:
    state = backend.zeros_state(N_QUBITS)
    gates = [
        Operation("ry", qubits=(0,), params=(feature[0],)),
        Operation("ry", qubits=(1,), params=(feature[1],)),
        Operation("rzz", qubits=(0, 1), params=(0.25 * feature[0] - 0.15 * feature[1],)),
    ]

    offset = 0
    for _ in range(layers):
        gates.extend(
            [
                Operation("ry", qubits=(0,), params=(theta[offset],)),
                Operation("rz", qubits=(0,), params=(theta[offset + 1],)),
                Operation("ry", qubits=(1,), params=(theta[offset + 2],)),
                Operation("rz", qubits=(1,), params=(theta[offset + 3],)),
                Operation("cx", qubits=(1,), controls=(0,)),
                Operation("rxx", qubits=(0, 1), params=(theta[offset + 4],)),
            ]
        )
        offset += PARAMS_PER_LAYER

    for gate in gates:
        state = apply_gate_to_state(gate, state, N_QUBITS, backend)
        if state is None:
            raise RuntimeError(f"typed Operation {gate.name!r} did not produce a state update")

    value = backend.expectation_sv(state, observable).reshape(())
    return torch.real(value) if torch.is_complex(value) else value


def _teacher_targets(
    backend: NPUBackend,
    features: torch.Tensor,
    teacher_theta: torch.Tensor,
    observable: torch.Tensor,
    layers: int,
) -> torch.Tensor:
    with torch.no_grad():
        targets = [
            _qnn_prediction(backend, feature, teacher_theta, observable, layers)
            for feature in features
        ]
    return torch.stack(targets).detach()


def _loss_and_mae(
    backend: NPUBackend,
    features: torch.Tensor,
    targets: torch.Tensor,
    theta: torch.Tensor,
    observable: torch.Tensor,
    layers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = [
        _qnn_prediction(backend, feature, theta, observable, layers)
        for feature in features
    ]
    pred = torch.stack(predictions)
    diff = pred - targets
    loss = torch.mean(diff * diff)
    mae = torch.mean(torch.abs(diff))
    return loss, mae


def _backend(allow_cpu_fallback: bool) -> NPUBackend:
    if not allow_cpu_fallback and not is_npu_available():
        raise SystemExit("NPU is unavailable. Omit --allow-cpu-fallback only on a real Ascend NPU host.")

    backend = NPUBackend.from_distributed_env(fallback_to_cpu=allow_cpu_fallback)
    device_type = getattr(backend._device, "type", None)
    if not allow_cpu_fallback and device_type != "npu":
        raise AssertionError(f"strict QNN demo resolved device={backend._device!r}, expected npu")
    if backend.distributed_initialized and not allow_cpu_fallback:
        pg_backend = str(dist.get_backend()).lower()
        if pg_backend != "hccl":
            raise AssertionError(f"strict QNN demo initialized backend={pg_backend!r}, expected hccl")
    _log(f"backend={backend.name} runtime={backend.runtime_context}")
    return backend


def _validate_args(args: argparse.Namespace, world_size: int) -> None:
    if args.layers < 1:
        raise SystemExit("--layers must be >= 1")
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if args.samples < world_size:
        raise SystemExit("--samples must be >= WORLD_SIZE")
    if args.samples % world_size != 0:
        raise SystemExit("--samples must be divisible by WORLD_SIZE so each rank has equal gradient weight")
    if args.expected_world_size is not None and int(args.expected_world_size) != int(world_size):
        raise SystemExit(f"expected WORLD_SIZE={args.expected_world_size}, got {world_size}")


def run(args: argparse.Namespace) -> int:
    backend = _backend(args.allow_cpu_fallback)
    ctx = backend.runtime_context
    world_size = int(ctx.world_size if ctx is not None else 1)
    rank = int(ctx.rank if ctx is not None else 0)
    _validate_args(args, world_size)

    device = backend._device
    features = _make_features(args.samples, device)

    rng = np.random.default_rng(args.seed)
    teacher_np = rng.normal(loc=0.0, scale=0.55, size=_param_count(args.layers)).astype(np.float32)
    init_np = teacher_np + rng.normal(loc=0.0, scale=0.35, size=teacher_np.shape).astype(np.float32)
    teacher_theta = torch.tensor(teacher_np, dtype=torch.float32, device=device)
    theta = torch.nn.Parameter(torch.tensor(init_np, dtype=torch.float32, device=device))
    broadcast_parameters({"teacher": teacher_theta, "theta": theta.data}, src=0)

    observable = Hamiltonian([("ZI", 1.0)]).to_matrix(backend)
    targets = _teacher_targets(backend, features, teacher_theta, observable, args.layers)

    local_features = features[rank::world_size]
    local_targets = targets[rank::world_size]
    if local_features.numel() == 0:
        raise RuntimeError(f"rank {rank} received an empty shard")

    with torch.no_grad():
        initial_loss, initial_mae = _loss_and_mae(
            backend,
            local_features,
            local_targets,
            theta,
            observable,
            args.layers,
        )
    initial_loss_f = _to_float(initial_loss)
    initial_mae_f = _to_float(initial_mae)

    last_loss = initial_loss
    last_mae = initial_mae
    last_grad_norm = torch.tensor(0.0, dtype=torch.float32, device=device)

    for step in range(1, args.steps + 1):
        if theta.grad is not None:
            theta.grad.zero_()
        loss, mae = _loss_and_mae(backend, local_features, local_targets, theta, observable, args.layers)
        if not bool(torch.isfinite(loss.detach()).cpu().reshape(()).item()):
            raise FloatingPointError(f"non-finite loss on rank {rank}: {_to_float(loss)}")
        loss.backward()
        if theta.grad is None:
            raise RuntimeError("theta.grad is None after backward")

        all_reduce_mean([theta.grad])
        last_grad_norm = torch.linalg.vector_norm(theta.grad.detach())
        with torch.no_grad():
            theta -= float(args.lr) * theta.grad
        last_loss, last_mae = loss.detach(), mae.detach()

        if rank == 0 and (step == 1 or step == args.steps or step % args.log_every == 0):
            print(
                f"[step {step:03d}] local_loss={_to_float(last_loss):.8f} "
                f"local_mae={_to_float(last_mae):.8f} grad_norm={_to_float(last_grad_norm):.8f}",
                flush=True,
            )

    with torch.no_grad():
        final_loss, final_mae = _loss_and_mae(
            backend,
            local_features,
            local_targets,
            theta,
            observable,
            args.layers,
        )

    payloads = all_gather(
        {
            "rank": rank,
            "device": str(device),
            "initial_loss": initial_loss_f,
            "initial_mae": initial_mae_f,
            "final_loss": _to_float(final_loss),
            "final_mae": _to_float(final_mae),
            "last_grad_norm": _to_float(last_grad_norm),
            "local_samples": int(local_features.shape[0]),
        }
    )

    if rank == 0:
        mean_initial = sum(item["initial_loss"] for item in payloads) / len(payloads)
        mean_final = sum(item["final_loss"] for item in payloads) / len(payloads)
        mean_mae = sum(item["final_mae"] for item in payloads) / len(payloads)
        devices = [item["device"] for item in payloads]
        print("\nSummary: PASS", flush=True)
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "devices": devices,
                    "samples": args.samples,
                    "layers": args.layers,
                    "steps": args.steps,
                    "initial_loss_mean": mean_initial,
                    "final_loss_mean": mean_final,
                    "final_mae_mean": mean_mae,
                    "rank_payloads": payloads,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        if mean_final > mean_initial:
            print(
                "[warn] final mean loss is higher than initial mean loss; "
                "this demo validates the execution path, not optimizer quality.",
                flush=True,
            )

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Four-card NPU typed-IR QNN demo with autodiff and HCCL gradient averaging."
    )
    parser.add_argument("--samples", type=int, default=32, help="Synthetic samples; must divide WORLD_SIZE.")
    parser.add_argument("--layers", type=int, default=2, help="Variational typed-Operation layers.")
    parser.add_argument("--steps", type=int, default=12, help="Gradient descent steps.")
    parser.add_argument("--lr", type=float, default=0.25, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic teacher/student seed.")
    parser.add_argument("--log-every", type=int, default=4, help="Rank-0 progress interval.")
    parser.add_argument(
        "--expected-world-size",
        type=int,
        default=None,
        help="Optional guard passed by scripts/npu/qnn_4card.sh.",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Local development only. Real 4-card validation should omit this.",
    )
    args = parser.parse_args(argv)
    if args.log_every < 1:
        raise SystemExit("--log-every must be >= 1")

    try:
        return run(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
