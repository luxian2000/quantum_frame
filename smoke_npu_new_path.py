"""Smoke tests for nexq new backend-bound circuit path on NPU.

This script focuses on the new path:
- Circuit(..., backend=backend)
- Measure.run / run_density_matrix using circuit-bound backend

Usage:
    python smoke_npu_new_path.py
    python smoke_npu_new_path.py --shots 2048
    python smoke_npu_new_path.py --allow-cpu-fallback
"""

from __future__ import annotations

import argparse
import math
from typing import Callable, List, Tuple

import numpy as np

from nexq import Circuit, Measure, NPUBackend, cnot, crz, hadamard, rx, ry, rz


Case = Tuple[str, Callable[[Measure, NPUBackend, int], None]]


def _assert_probs_normalized(probs: np.ndarray, atol: float = 1e-6) -> None:
    total = float(np.sum(np.asarray(probs, dtype=np.float64)))
    if not np.isclose(total, 1.0, atol=atol):
        raise AssertionError(f"probabilities not normalized, sum={total}")


def case_single_gate(measure: Measure, backend: NPUBackend, shots: int) -> None:
    cir = Circuit(hadamard(0), n_qubits=1, backend=backend)
    res = measure.run(cir, shots=shots)
    _assert_probs_normalized(res.probabilities)
    if not np.isclose(res.probabilities[0], 0.5, atol=2e-2):
        raise AssertionError(f"single-gate check failed, p0={res.probabilities[0]}")


def case_controlled_gate(measure: Measure, backend: NPUBackend, shots: int) -> None:
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2, backend=backend)
    res = measure.run(cir, shots=shots)
    _assert_probs_normalized(res.probabilities)
    if not (np.isclose(res.probabilities[0], 0.5, atol=2e-2) and np.isclose(res.probabilities[3], 0.5, atol=2e-2)):
        raise AssertionError(f"controlled-gate check failed, probs={res.probabilities}")


def case_parametric_gate(measure: Measure, backend: NPUBackend, shots: int) -> None:
    theta = math.pi / 3
    cir = Circuit(
        ry(theta, 0),
        rz(math.pi / 7, 0),
        rx(math.pi / 5, 0),
        crz(math.pi / 4, target_qubit=1, control_qubits=[0]),
        n_qubits=2,
        backend=backend,
    )
    res = measure.run(cir, shots=shots)
    _assert_probs_normalized(res.probabilities)


def case_density_matrix(measure: Measure, backend: NPUBackend, shots: int) -> None:
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2, backend=backend)
    res = measure.run_density_matrix(cir, shots=shots)
    _assert_probs_normalized(res.probabilities)
    if not (np.isclose(res.probabilities[0], 0.5, atol=2e-2) and np.isclose(res.probabilities[3], 0.5, atol=2e-2)):
        raise AssertionError(f"density-matrix check failed, probs={res.probabilities}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test nexq new backend-bound NPU path")
    parser.add_argument("--shots", type=int, default=512, help="Sampling shots for each case")
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Allow CPU fallback when NPU is unavailable",
    )
    args = parser.parse_args()

    backend = NPUBackend.from_distributed_env(fallback_to_cpu=args.allow_cpu_fallback)
    measure = Measure(backend)

    print("=== Smoke NPU New Path ===")
    print(f"backend: {backend.name}")
    print(f"runtime_context: {backend.runtime_context}")

    cases: List[Case] = [
        ("single_gate", case_single_gate),
        ("controlled_gate", case_controlled_gate),
        ("parametric_gate", case_parametric_gate),
        ("density_matrix", case_density_matrix),
    ]

    failed: List[Tuple[str, str]] = []
    for name, fn in cases:
        try:
            fn(measure, backend, args.shots)
            print(f"[PASS] {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")

    if failed:
        print("\nSummary: FAILED")
        for name, msg in failed:
            print(f"- {name}: {msg}")
        raise SystemExit(1)

    print("\nSummary: PASS")


if __name__ == "__main__":
    main()
