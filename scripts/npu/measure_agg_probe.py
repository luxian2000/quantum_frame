#!/usr/bin/env python3
"""真机 NPU 验证：Measure.run 多 shot 轻量聚合路径（return_state=False 跳过密度矩阵）。

用法（NPU 机器上，仓库根目录）:
    python scripts/npu/measure_agg_probe.py                       # 严格 NPU
    python scripts/npu/measure_agg_probe.py --allow-cpu-fallback  # 仅本地开发
    python scripts/npu/measure_agg_probe.py --n-qubits 24         # 容量档位

覆盖三个 case:
1. lite 路径确实不构造 (2^n,2^n) 密度矩阵（打桩 _as_density 报错）；
2. 同种子下 lite 与密度路径计数逐项一致，return_state=True 契约不变；
3. 大比特容量档：密度路径不可行的规模下 lite 路径正常出计数。
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from aicir import Circuit, Measure, NPUBackend, cx, hadamard
from aicir.backends.npu_backend import is_npu_available
from aicir.measure import aggregate as _agg_mod


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


def _ghz(n: int, backend: NPUBackend) -> Circuit:
    gates = [hadamard(0)] + [cx(q, [q - 1]) for q in range(1, n)]
    return Circuit(*gates, n_qubits=n, backend=backend)


def _check_ghz_counts(counts: dict, n: int, shots: int, *, label: str) -> None:
    if set(counts) - {"0" * n, "1" * n}:
        raise AssertionError(f"{label}: GHZ 计数出现非法比特串 {sorted(counts)}")
    total = sum(counts.values())
    if total != shots:
        raise AssertionError(f"{label}: 计数总和 {total} != shots {shots}")


def case_lite_skips_density(backend: NPUBackend, n: int, shots: int) -> None:
    original = _agg_mod._as_density

    def _boom(_state):
        raise AssertionError("return_state=False 的多 shot 路径不应构造密度矩阵")

    _agg_mod._as_density = _boom
    try:
        result = Measure(backend).run(_ghz(n, backend), shots=shots, seed=11, return_state=False)
    finally:
        _agg_mod._as_density = original

    if result.state is not None or result.final_state is not None:
        raise AssertionError("return_state=False 时 state/final_state 应为 None")
    total = float(np.sum(np.asarray(result.probabilities, dtype=float)))
    if not np.isclose(total, 1.0, atol=1e-5):
        raise AssertionError(f"probabilities 之和 {total} != 1")
    _check_ghz_counts(result.counts(-1), n, shots, label="lite")


def case_seed_parity_and_contract(backend: NPUBackend, n: int, shots: int) -> None:
    lite = Measure(backend).run(_ghz(n, backend), shots=shots, seed=23, return_state=False)
    dense = Measure(backend).run(_ghz(n, backend), shots=shots, seed=23, return_state=True)

    if lite.counts(-1) != dense.counts(-1):
        raise AssertionError("同种子下 lite 与密度路径计数不一致（采样流被改动）")
    if dense.final_state_kind != "density_matrix":
        raise AssertionError("return_state=True 的 shots>1 聚合态仍应为密度矩阵（契约不变）")
    if not np.allclose(np.asarray(lite.probabilities), np.asarray(dense.probabilities), atol=1e-5):
        raise AssertionError("lite 与密度路径 probabilities 不一致")


def case_capacity(backend: NPUBackend, n: int, shots: int) -> None:
    # 该规模下密度路径需要 (2^n)^2 复数矩阵，不可行；lite 路径应正常完成
    start = time.perf_counter()
    result = Measure(backend).run(_ghz(n, backend), shots=shots, seed=5, return_state=False)
    elapsed = time.perf_counter() - start
    _check_ghz_counts(result.counts(-1), n, shots, label=f"capacity n={n}")
    print(f"[capacity] n={n} shots={shots} 用时 {elapsed:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu-fallback", action="store_true",
                        help="允许 CPU 回退（仅本地开发，不作为真机正确性证据）")
    parser.add_argument("--n-qubits", type=int, default=20,
                        help="容量档 GHZ 比特数（默认 20；密度路径在该规模需 ~2^{2n} 复数存储）")
    parser.add_argument("--shots", type=int, default=200)
    args = parser.parse_args()

    backend = _backend(args.allow_cpu_fallback)
    cases = [
        ("lite_skips_density", lambda: case_lite_skips_density(backend, 6, args.shots)),
        ("seed_parity_and_contract", lambda: case_seed_parity_and_contract(backend, 6, args.shots)),
        ("capacity", lambda: case_capacity(backend, args.n_qubits, args.shots)),
    ]
    for name, fn in cases:
        fn()
        print(f"[PASS] {name}")
    print(f"measure_agg_probe: {len(cases)}/{len(cases)} cases passed")


if __name__ == "__main__":
    main()
