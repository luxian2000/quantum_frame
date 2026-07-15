#!/usr/bin/env python3
"""真机 NPU VQE 探针：能量期望 + PSR 梯度的三条设备路径。

只覆盖真正走设备后端的 VQE 变体（VQD/SSVQE 为纯 numpy 稠密路径、
不触设备，故不在此）：
  - vqe_exact：态向量精确期望 + PSR 梯度（expectation_sv + qml.deriv.psr）
  - vqe_shots：shot-based 估计器（设备采样聚合能量）
  - vqe_density：use_density_matrix + 噪声（expectation_dm 路径）

每 case 独立兜底。用法：
    python scripts/npu/vqc_probe.py [--allow-cpu-fallback] [--iters N]
"""
from __future__ import annotations

import argparse
import traceback

import numpy as np

from aicir import Hamiltonian, NPUBackend, PauliString
from aicir.backends.npu_backend import is_npu_available
from aicir.vqc import run_vqe
from aicir.noise import NoiseModel, DepolarizingChannel
from aicir.primitives import ShotEstimator
from aicir.ansatze import hardware_efficient_ansatz


def _backend(allow_cpu_fallback: bool) -> NPUBackend:
    if not allow_cpu_fallback and not is_npu_available():
        raise SystemExit("NPU is unavailable. Use --allow-cpu-fallback only for local script validation.")
    backend = NPUBackend.from_distributed_env(fallback_to_cpu=allow_cpu_fallback)
    if not allow_cpu_fallback and getattr(backend._device, "type", None) != "npu":
        raise AssertionError(f"strict NPU probe resolved device={backend._device!r}, expected npu")
    print(f"[backend] {backend.name}")
    print(f"[runtime] {backend.runtime_context}")
    return backend


def _tfim(n: int) -> Hamiltonian:
    terms = [PauliString("ZZ", n_qubits=n, qubits=[q, q + 1]) for q in range(n - 1)]
    terms += [PauliString("X", n_qubits=n, qubits=[q]) for q in range(n)]
    return Hamiltonian(terms)


def _ref_ground(ham, backend):
    mat = np.asarray(backend.to_numpy(ham.to_matrix(backend)))
    return float(np.min(np.linalg.eigvalsh(mat)))


def case_vqe_exact(backend, n, iters):
    ham = _tfim(n)
    res = run_vqe(ham, n_qubits=n, depth=2, max_iters=iters, lr=0.1, seed=1, backend=backend)
    if not np.isfinite(res.energy):
        raise AssertionError(f"VQE 能量非有限: {res.energy}")
    ref = _ref_ground(ham, backend)
    if res.energy > ref + abs(ref) * 0.5 + 1.0:
        raise AssertionError(f"VQE 能量 {res.energy} 远离基态 {ref}")


def case_vqe_shots(backend, n, iters):
    ham = _tfim(n)
    estimator = ShotEstimator(backend=backend, shots=512)
    ansatz = hardware_efficient_ansatz(n_qubits=n, layers=1)
    res = run_vqe(ham, n_qubits=n, depth=1, max_iters=iters, lr=0.1, seed=2,
                  backend=backend, ansatz=ansatz, energy_estimator=estimator)
    if not np.isfinite(res.energy):
        raise AssertionError(f"shot VQE 能量非有限: {res.energy}")


def case_vqe_density(backend, n, iters):
    ham = _tfim(n)
    model = NoiseModel().add_channel(DepolarizingChannel(target_qubit=0, p=0.02))
    ansatz = hardware_efficient_ansatz(n_qubits=n, layers=1)
    res = run_vqe(ham, n_qubits=n, depth=1, max_iters=iters, lr=0.1, seed=3,
                  backend=backend, ansatz=ansatz, use_density_matrix=True, noise_model=model)
    if not np.isfinite(res.energy):
        raise AssertionError(f"density VQE 能量非有限: {res.energy}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--n-qubits", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    backend = _backend(args.allow_cpu_fallback)
    n = args.n_qubits

    cases = [
        ("vqe_exact", lambda: case_vqe_exact(backend, n, args.iters)),
        ("vqe_shots", lambda: case_vqe_shots(backend, n, args.iters)),
        ("vqe_density", lambda: case_vqe_density(backend, n, args.iters)),
    ]
    passed, failures = 0, []
    for name, fn in cases:
        try:
            fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            first = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
            print(f"[FAIL] {name}: {first}")
            failures.append((name, traceback.format_exc()))
    print(f"vqc_probe: {passed}/{len(cases)} cases passed")
    if failures:
        print("\n===== 失败详情 =====")
        for name, tb in failures:
            print(f"\n--- {name} ---\n{tb}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
