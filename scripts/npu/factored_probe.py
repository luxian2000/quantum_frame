#!/usr/bin/env python3
"""真机 NPU 因子化（乘积态）引擎探针。

因子化引擎把纯态存成若干互不纠缠的因子，门只作用在自己所属的因子上，跨因子时
才 kron 合并。它只依赖两样东西：``_apply_local_matrix_to_state``（稠密路径同一个
门应用入口）与 ``backend.kron``。

**为什么必须上真机**：``NPUBackend`` 里的 real/imag 分解是**按设备门控**的
（``_is_npu_complex`` 要求 ``device.type == "npu"``），CPU 上
``fallback_to_cpu=True`` 会走原生复数分支。因此复数算子的 NPU 安全性
**在 CPU 上根本无法验证**——`tests/simulator/test_factored_npu_safety.py` 只
覆盖了引擎自己控制的部分（不发位运算、不索引复数张量），复数算术这一段只能由
本探针在真机上确认。

覆盖：
  - 正确性：与 CPU 参考逐值比对
  - 因子划分：纯 Python 记账，必须与设备无关
  - 无位运算：位运算在昇腾上静默回落 CPU，是唯一"不崩只变慢"的缺口
  - 规模：全可分线路远超稠密表示的比特数
  - 免稠密化期望：断言 to_statevector 调用数为 0

每个 case 独立兜底，逐条 PASS/FAIL + 首行错误，末尾汇总。
用法（NPU 机器，仓库根目录）:
    python scripts/npu/factored_probe.py                       # 严格 NPU
    python scripts/npu/factored_probe.py --allow-cpu-fallback  # 仅本地开发
    python scripts/npu/factored_probe.py --scale-qubits 28 --output-json out.json
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

import numpy as np
import torch

import aicir as A
from aicir import Circuit, Hamiltonian, NPUBackend, NumpyBackend
from aicir.backends.npu_backend import is_npu_available
from aicir.simulator.factored import factored_expectation, factored_statevector


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


def _sync(backend) -> None:
    """NPU 执行异步：不同步就计时，量到的是 kernel launch 而非完成时间。"""
    device = getattr(backend, "_device", None)
    if getattr(device, "type", None) == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()


def _time(backend, fn, repeats: int = 3) -> float:
    fn()
    _sync(backend)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        _sync(backend)
        samples.append(time.perf_counter() - start)
    return sorted(samples)[len(samples) // 2]


def _mixed_circuit(n: int) -> Circuit:
    """单比特门 + 两对不相交 CNOT：既有因子内门，也触发合并。"""
    gates = [A.ry(0.3, q) for q in range(n)]
    gates += [A.cnot(1, [0]), A.cnot(3, [2])]
    return Circuit(*gates, n_qubits=n)


class _RecordBitwise(torch.utils._python_dispatch.TorchDispatchMode):
    """位运算在昇腾上无内核，会静默回落 CPU——必须一个都不出现。"""

    TAGS = ("bitwise_", "__lshift__", "__rshift__")

    def __init__(self):
        super().__init__()
        self.hits = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if any(tag in name for tag in self.TAGS):
            self.hits.append(name)
        return func(*args, **(kwargs or {}))


def case_correctness_vs_cpu(backend, n, metrics):
    circuit = _mixed_circuit(min(n, 10))
    got = np.asarray(
        factored_statevector(circuit, backend).to_statevector().to_numpy()
    ).reshape(-1)
    want = np.asarray(
        factored_statevector(circuit, NumpyBackend(dtype=np.complex128)).to_statevector().to_numpy()
    ).reshape(-1)
    overlap = abs(complex(np.vdot(want, got)))
    metrics["correctness_overlap"] = overlap
    if abs(overlap - 1.0) > 1e-3:            # complex64 精度
        raise AssertionError(f"NPU 与 CPU 态重叠度 {overlap}，应为 1")


def case_matches_dense_path(backend, n, metrics):
    """与稠密路径对拍——因子化只是换表示，物理态必须一致。"""
    from aicir import Measure

    circuit = _mixed_circuit(min(n, 10))
    factored = np.asarray(
        factored_statevector(circuit, backend).to_statevector().to_numpy()
    ).reshape(-1)
    dense = np.asarray(
        Measure(backend=backend).run(circuit, shots=None).final_state
    ).reshape(-1)
    overlap = abs(complex(np.vdot(dense, factored)))
    metrics["dense_overlap"] = overlap
    if abs(overlap - 1.0) > 1e-3:
        raise AssertionError(f"因子化与稠密路径重叠度 {overlap}，应为 1")


def case_factor_structure(backend, n, metrics):
    """因子划分是纯 Python 记账，必须与设备无关；不一致即说明混入了设备分支。"""
    circuit = _mixed_circuit(min(n, 10))
    npu_part = [qs for qs, _ in factored_statevector(circuit, backend).factors]
    cpu_part = [qs for qs, _ in factored_statevector(circuit, NumpyBackend()).factors]
    metrics["factor_partition"] = [list(qs) for qs in npu_part]
    if npu_part != cpu_part:
        raise AssertionError(f"因子划分不一致: NPU {npu_part} vs CPU {cpu_part}")


def case_no_bitwise_ops(backend, n, metrics):
    circuit = _mixed_circuit(min(n, 8))
    mode = _RecordBitwise()
    with mode:
        factored_statevector(circuit, backend).to_statevector()
    metrics["bitwise_ops"] = sorted(set(mode.hits))
    if mode.hits:
        raise AssertionError(f"发出位运算（昇腾会静默回落 CPU）: {metrics['bitwise_ops']}")


def case_product_state_scale(backend, scale_qubits, metrics):
    """全可分线路：稠密表示需 2^n·8 B，因子化只有 n 个 2 维张量。"""
    circuit = Circuit(*[A.ry(0.2, q) for q in range(scale_qubits)], n_qubits=scale_qubits)
    elapsed = _time(backend, lambda: factored_statevector(circuit, backend))
    state = factored_statevector(circuit, backend)
    dense_gb = (2 ** scale_qubits) * 8 / 1e9
    metrics["scale_qubits"] = scale_qubits
    metrics["scale_seconds"] = elapsed
    metrics["scale_factors"] = state.n_factors
    print(f"        n={scale_qubits} 全可分 {elapsed*1000:.2f} ms, "
          f"factors={state.n_factors} (稠密需 {dense_gb:.2f} GB)")
    if state.max_factor_width != 1:
        raise AssertionError(f"全可分线路却出现宽度 {state.max_factor_width} 的因子")


def case_expectation_without_materialising(backend, scale_qubits, metrics):
    """大比特数求期望且**不得**稠密化——稠密化会立刻 OOM 或极慢。"""
    from aicir.simulator.factored import FactoredState

    circuit = Circuit(*[A.ry(0.2, q) for q in range(scale_qubits)], n_qubits=scale_qubits)
    state = factored_statevector(circuit, backend)

    calls = {"n": 0}
    original = FactoredState.to_statevector

    def _guard(self):
        calls["n"] += 1
        return original(self)

    FactoredState.to_statevector = _guard
    try:
        ham = Hamiltonian(
            n_qubits=scale_qubits,
            terms=[("Z", [0], 1.0), ("Z", [scale_qubits - 1], 1.0)],
        )
        value = factored_expectation(state, ham)
    finally:
        FactoredState.to_statevector = original

    metrics["expectation_value"] = float(value)
    metrics["materialised"] = calls["n"]
    if calls["n"] != 0:
        raise AssertionError("factored_expectation 稠密化了整个态")
    if not np.isfinite(value):
        raise AssertionError(f"期望值非有限: {value}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--n-qubits", type=int, default=10, help="常规 case 的比特数")
    parser.add_argument("--scale-qubits", type=int, default=26, help="全可分规模 case 的比特数")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    backend = _backend(args.allow_cpu_fallback)
    n = args.n_qubits
    metrics: dict = {}

    cases = [
        ("correctness_vs_cpu", lambda: case_correctness_vs_cpu(backend, n, metrics)),
        ("matches_dense_path", lambda: case_matches_dense_path(backend, n, metrics)),
        ("factor_structure", lambda: case_factor_structure(backend, n, metrics)),
        ("no_bitwise_ops", lambda: case_no_bitwise_ops(backend, n, metrics)),
        ("product_state_scale",
         lambda: case_product_state_scale(backend, args.scale_qubits, metrics)),
        ("expectation_without_materialising",
         lambda: case_expectation_without_materialising(backend, args.scale_qubits, metrics)),
    ]

    passed = 0
    failures = []
    for name, fn in cases:
        try:
            fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001 — 探针刻意逐 case 兜底以收集所有缺口
            first = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
            print(f"[FAIL] {name}: {first}")
            failures.append((name, traceback.format_exc()))

    print(f"\nfactored_probe: {passed}/{len(cases)} cases passed")

    if args.output_json:
        payload = {
            "passed": passed,
            "total": len(cases),
            "failed": [name for name, _ in failures],
            "device": str(getattr(backend, "_device", "unknown")),
            "fallback_to_cpu": bool(args.allow_cpu_fallback),
            "n_qubits": n,
            "scale_qubits": args.scale_qubits,
            "metrics": metrics,
        }
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"结果已写入 {args.output_json}")

    if failures:
        print("\n===== 失败详情 =====")
        for name, tb in failures:
            print(f"\n--- {name} ---\n{tb}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
