"""Numerically verify VQE-QAS backend dtype execution against CPU complex128.

This is the Phase-B0 guardrail for the v3 scaling protocol. It does not trust
backend printouts alone: the same TFIM circuit and the same theta vector are
evaluated on CPU/NumPy complex128 and on the requested backend, then compared.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aicir.qas.task_evaluation import parameter_count
from aicir.qas.vqe_hea_demo import (
    architecture_from_hea_mask,
    backend_runtime_metadata,
    enumerate_hea_masks,
    evaluate_vqe_energy,
    resolve_qas_backend,
    tfim_chain_demo_problem,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check target backend energy against CPU complex128 for one TFIM circuit")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.5)
    parser.add_argument("--candidate-index", type=int, default=0, help="0-based index in enumerate_hea_masks")
    parser.add_argument("--theta-seed", type=int, default=2026)
    parser.add_argument("--target-backend", choices=("npu", "torch", "cpu", "numpy"), default="npu")
    parser.add_argument("--target-dtype", choices=("complex128", "complex64"), default="complex128")
    parser.add_argument("--no-fallback-to-cpu", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--output-dir", default="outputs/vqe_tfim_v3_dtype_guard")
    args = parser.parse_args()

    masks = enumerate_hea_masks(args.n_qubits)
    if args.candidate_index < 0 or args.candidate_index >= len(masks):
        raise ValueError(f"candidate-index must be in [0, {len(masks)}), got {args.candidate_index}")
    architecture = architecture_from_hea_mask(masks[args.candidate_index])
    problem = tfim_chain_demo_problem(n_qubits=args.n_qubits, J=args.J, h=args.h, periodic=False)
    n_params = parameter_count(architecture.circuit)
    rng = np.random.default_rng(int(args.theta_seed))
    theta = rng.uniform(-np.pi, np.pi, size=n_params) if n_params else np.zeros(0, dtype=float)

    cpu_backend = resolve_qas_backend("cpu", dtype="complex128")
    target_backend = resolve_qas_backend(
        args.target_backend,
        dtype=args.target_dtype,
        fallback_to_cpu=not args.no_fallback_to_cpu,
    )

    cpu_energy = evaluate_vqe_energy(architecture, problem, parameters=theta, backend=cpu_backend)
    target_energy = evaluate_vqe_energy(architecture, problem, parameters=theta, backend=target_backend)
    abs_diff = abs(float(cpu_energy) - float(target_energy))
    passed = bool(abs_diff <= float(args.atol))

    report = {
        "passed": passed,
        "atol": float(args.atol),
        "abs_diff": abs_diff,
        "cpu_energy": float(cpu_energy),
        "target_energy": float(target_energy),
        "n_qubits": int(args.n_qubits),
        "candidate_index": int(args.candidate_index),
        "candidate_name": architecture.name,
        "n_params": int(n_params),
        "theta_seed": int(args.theta_seed),
        "cpu_backend": backend_runtime_metadata(cpu_backend),
        "target_backend": backend_runtime_metadata(target_backend),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dtype_guard.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "VQE-QAS v3 dtype guard",
        f"passed: {passed}",
        f"atol: {args.atol:.3e}",
        f"abs_diff: {abs_diff:.3e}",
        f"cpu_energy: {cpu_energy:.12f}",
        f"target_energy: {target_energy:.12f}",
        f"candidate: {architecture.name}",
        f"cpu_backend: {report['cpu_backend']['backend_name']}",
        f"target_backend: {report['target_backend']['backend_name']}",
    ]
    (output_dir / "dtype_guard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
