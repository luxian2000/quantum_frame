#!/usr/bin/env python3
"""真机 NPU 验证：2026-07-15 六项内存浪费/设备往返修复的热路径探针。

用法（NPU 机器上，仓库根目录）:
    python scripts/npu/hotpath_probe.py                       # 严格 NPU
    python scripts/npu/hotpath_probe.py --allow-cpu-fallback  # 仅本地开发

覆盖三个 case（各自打印用时，供真机对照基线）:
1. incircuit_joint_pauli：X 基 in-circuit measure、shots>1——对照
   measure_joint_pauli 去重复旋转 + 初始态共享修复；
2. noisy_kraus_cache：depolarizing 噪声密度演化——对照 Kraus 缓存 +
   密度对角线/投影修复；
3. mps_shape_reads：method="mps" GHZ 采样——对照 MPS shape 读取零传输修复。
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from aicir import Circuit, Measure, NPUBackend, cx, hadamard, measure, ry
from aicir.backends.npu_backend import is_npu_available
from aicir.noise import DepolarizingChannel, NoiseModel


def _backend(allow_cpu_fallback: bool) -> NPUBackend:
    if not allow_cpu_fallback and not is_npu_available():
        raise SystemExit("NPU is unavailable. Use --allow-cpu-fallback only for local script validation.")
    backend = NPUBackend.from_distributed_env(fallback_to_cpu=allow_cpu_fallback)
    device_type = getattr(backend._device, "type", None)
    if not allow_cpu_fallback and device_type != "npu":
        raise AssertionError(f"strict NPU probe resolved device={backend._device!r}, expected npu")
    print(f"[backend] {backend.name}")
    print(f"[runtime] {backend.runtime_context}")
    return backend


def case_incircuit_joint_pauli(backend: NPUBackend, n: int, shots: int) -> None:
    # |+> 上的 X 基联合测量恒得 +1；带 in-circuit measure 走每轨迹路径
    gates = [hadamard(q) for q in range(n)] + [measure([0, 1], basis="X")]
    circ = Circuit(*gates, n_qubits=n, backend=backend)
    start = time.perf_counter()
    result = Measure(backend).run(circ, shots=shots, seed=11, return_state=False,
                                  return_probabilities=False)
    elapsed = time.perf_counter() - start
    counts = result.incircuit_counts[n]
    if set(counts) != {1} or counts[1] != shots:
        raise AssertionError(f"|+>^n 的 X⊗X 测量应恒为 +1，得到 {counts}")
    print(f"[incircuit_joint_pauli] n={n} shots={shots} 用时 {elapsed:.2f}s")


def case_noisy_kraus_cache(backend: NPUBackend, n: int, shots: int) -> None:
    gates = [hadamard(0)] + [cx(q, [q - 1]) for q in range(1, n)]
    circ = Circuit(*gates, n_qubits=n, backend=backend)
    circ.noise_model = NoiseModel().add_channel(DepolarizingChannel(target_qubit=0, p=0.05))
    start = time.perf_counter()
    result = Measure(backend).run(circ, shots=shots, seed=7, return_state=False)
    elapsed = time.perf_counter() - start
    total = float(np.sum(np.asarray(result.probabilities, dtype=float)))
    if not np.isclose(total, 1.0, atol=1e-4):
        raise AssertionError(f"噪声路径 probabilities 之和 {total} != 1")
    if sum(result.counts(-1).values()) != shots:
        raise AssertionError("噪声路径计数总和与 shots 不符")
    print(f"[noisy_kraus_cache] n={n} shots={shots} 用时 {elapsed:.2f}s")


def case_mps_shape_reads(backend: NPUBackend, n: int, shots: int) -> None:
    # 末尾两次非相邻 cx(n-1, [0])（自逆，净效果恒等）触发 SWAP 网络 →
    # site_of 非恒等，覆盖 to_statevector 的 flat 位置换 gather 路径
    gates = ([hadamard(0)] + [cx(q, [q - 1]) for q in range(1, n)]
             + [cx(n - 1, [0]), cx(n - 1, [0])])
    circ = Circuit(*gates, n_qubits=n, backend=backend)
    start = time.perf_counter()
    result = Measure(backend).run(circ, shots=shots, seed=5, method="mps",
                                  return_state=False, return_probabilities=False)
    elapsed = time.perf_counter() - start
    counts = result.counts(-1)
    if set(counts) - {"0" * n, "1" * n}:
        raise AssertionError(f"GHZ 计数出现非法比特串 {sorted(counts)}")
    if sum(counts.values()) != shots:
        raise AssertionError("MPS 路径计数总和与 shots 不符")
    print(f"[mps_shape_reads] n={n} shots={shots} 用时 {elapsed:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu-fallback", action="store_true",
                        help="允许 CPU 回退（仅本地开发，不作为真机正确性证据）")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--incircuit-qubits", type=int, default=10)
    parser.add_argument("--noisy-qubits", type=int, default=8)
    parser.add_argument("--mps-qubits", type=int, default=20)
    args = parser.parse_args()

    backend = _backend(args.allow_cpu_fallback)
    cases = [
        ("incircuit_joint_pauli", lambda: case_incircuit_joint_pauli(backend, args.incircuit_qubits, args.shots)),
        ("noisy_kraus_cache", lambda: case_noisy_kraus_cache(backend, args.noisy_qubits, args.shots)),
        ("mps_shape_reads", lambda: case_mps_shape_reads(backend, args.mps_qubits, args.shots)),
    ]
    for name, fn in cases:
        fn()
        print(f"[PASS] {name}")
    print(f"hotpath_probe: {len(cases)}/{len(cases)} cases passed")


if __name__ == "__main__":
    main()
