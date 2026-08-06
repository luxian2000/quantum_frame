#!/usr/bin/env python3
"""真机 NPU 稀疏 Pauli 期望值探针（real/imag 分解路径）。

`Hamiltonian.expectation` 现在走 `_sparse_expectation_torch`：不构造稠密
`2^n × 2^n` 矩阵，而用 `P|b⟩ = i^{n_Y}·(-1)^{popcount(b & z_mask)}·|b ⊕ x_mask⟩`。
复数形式会同时命中昇腾缺失的**三类**内核，故全程拆成实部/虚部：

  - `aclnnIndex`(complex64)      → 两次实数 `index_select`
  - `aclnnAdd`/`aclnnMul`(c64)   → `conj(ψ)·φ = (ac+bd) + i(ad−bc)` 四次实数乘加
  - 复数归约（sum/dot）           → 两次实数归约

CPU 侧已用 `TorchDispatchMode` 复现上述缺口并通过（`tests/core/test_pauli_sparse_torch.py`），
但那只能证明"没有发出被禁算子"，**证明不了真机内核、HBM 容量与实际吞吐**——本探针补这一段。

覆盖：
  - 正确性：与 numpy CPU 参考逐值比对（含 Y 串——两处符号约定唯一会露馅的地方）
  - 密度矩阵路径 Tr(ρH)（符号取在 b，与态矢量分支取在 b⊕x 不同）
  - 设备驻留：中间张量不得回落 CPU
  - 无复数算子：dispatch 拦截在真机上复核
  - 规模：稠密矩阵不可能达到的比特数（n=16 稠密需 34 GB）
  - 稀疏 vs 稠密对拍与加速比（在稠密仍装得下的 n 上）

每个 case 独立兜底，逐条 PASS/FAIL + 首行错误，末尾汇总。
用法（NPU 机器，仓库根目录）:
    python scripts/npu/pauli_sparse_probe.py                       # 严格 NPU
    python scripts/npu/pauli_sparse_probe.py --allow-cpu-fallback  # 仅本地开发
    python scripts/npu/pauli_sparse_probe.py --n-qubits 12 --max-qubits 20
    python scripts/npu/pauli_sparse_probe.py --output-json out.json
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

import numpy as np
import torch

from aicir import Hamiltonian, NPUBackend, NumpyBackend
from aicir.backends.npu_backend import is_npu_available
from aicir.core.state import State


# --------------------------------------------------------------------------
# 环境
# --------------------------------------------------------------------------

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
    """等待设备把队列跑完。

    NPU 执行是异步的：不同步就计时，量到的是 kernel launch 而非完成时间。
    2d21ac3 那轮 n=16 的 171 / 616 / 679 ms（4× 波动）就是这么来的。
    """

    device = getattr(backend, "_device", None)
    if getattr(device, "type", None) == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()


def _time(backend, fn, repeats: int = 3) -> float:
    """同步计时，返回中位数秒数。"""

    fn()
    _sync(backend)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        _sync(backend)
        samples.append(time.perf_counter() - start)
    return sorted(samples)[len(samples) // 2]


def _random_vec(n_qubits: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = 1 << n_qubits
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return vec / np.linalg.norm(vec)


def _tfim(n_qubits: int) -> Hamiltonian:
    """横场 Ising：ZZ 链 + X 场。VQE 里最常见的测试哈密顿量。"""
    return Hamiltonian(
        n_qubits=n_qubits,
        terms=[("ZZ", [q, q + 1], 1.0) for q in range(n_qubits - 1)]
        + [("X", [q], 0.5) for q in range(n_qubits)],
    )


def _cpu_reference(ham: Hamiltonian, vec: np.ndarray) -> float:
    """numpy 稀疏路径作为数值 oracle（已由 tests/core 对稠密矩阵验证过）。"""
    cpu = NumpyBackend(dtype=np.complex128)
    return ham.expectation(State.from_array(vec.astype(np.complex128), backend=cpu), cpu)


class _BanUnsupportedComplexOps(torch.utils._python_dispatch.TorchDispatchMode):
    """在真机上复核：整条路径不得发出昇腾缺失的复数算子。

    真机本应直接崩，但 CANN 版本差异可能让某些算子"恰好能跑"。显式拦截可以
    保证这条路径的 NPU-safe 性质是被**设计**保证的，而非侥幸。
    """

    BANNED = ("add", "mul", "sub", "index", "sum", "dot", "matmul", "mm", "gather")
    ALLOWED = ("view_as_real", "real", "imag", "conj_physical", "_conj", "resolve_conj")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        if not any(tag in name for tag in self.ALLOWED):
            if any(tag in name for tag in self.BANNED):
                for value in list(args) + list(kwargs.values()):
                    if isinstance(value, torch.Tensor) and torch.is_complex(value):
                        raise RuntimeError(f"NPU 不支持的复数算子: {name}")
        return func(*args, **kwargs)


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------

def case_correctness_vs_cpu(backend, n, metrics):
    """与 numpy CPU 参考逐值比对（complex64 精度下 atol 放宽）。"""
    vec = _random_vec(n, seed=n)
    ham = _tfim(n)
    got = ham.expectation(State.from_array(vec, backend=backend), backend)
    want = _cpu_reference(ham, vec)
    metrics["correctness_abs_error"] = abs(got - want)
    if not np.isclose(got, want, atol=1e-3):
        raise AssertionError(f"NPU 稀疏期望 {got} != CPU 参考 {want}（误差 {abs(got-want):.3e}）")


def case_pauli_y_sign_convention(backend, n, metrics):
    """含 Y 的串：两处符号约定唯一会露馅的地方。

    态矢量分支符号取在 b⊕x_mask（算符作用在 ket）。写成 b 时，纯 X / 纯 Z 串
    仍然正确，只有 Y 会错——故必须单独覆盖。
    """
    vec = _random_vec(n, seed=n + 101)
    rng = np.random.default_rng(n)
    worst = 0.0
    for _ in range(8):
        labels = "".join(rng.choice(list("IXYZ"), size=n))
        ham = Hamiltonian([(labels, 1.0)])
        got = ham.expectation(State.from_array(vec, backend=backend), backend)
        want = _cpu_reference(ham, vec)
        worst = max(worst, abs(got - want))
        if not np.isclose(got, want, atol=1e-3):
            raise AssertionError(f"Y 串 {labels}: NPU {got} != CPU {want}")
    metrics["y_string_max_error"] = worst


def case_density_matrix(backend, n, metrics):
    """Tr(ρH)：符号取在 b，与态矢量分支不同。"""
    small = min(n, 8)  # 密度矩阵是 2^n × 2^n，刻意压小
    vec = _random_vec(small, seed=small + 11)
    state = State.from_array(vec, backend=backend).to_density_matrix()
    ham = Hamiltonian(
        n_qubits=small,
        terms=[("XY", [0, small - 1], 0.75), ("Z", [1], -1.0)],
    )
    got = ham.expectation(state, backend)

    cpu = NumpyBackend(dtype=np.complex128)
    cpu_state = State.from_array(vec.astype(np.complex128), backend=cpu).to_density_matrix()
    want = ham.expectation(cpu_state, cpu)
    metrics["density_abs_error"] = abs(got - want)
    if not np.isclose(got, want, atol=1e-3):
        raise AssertionError(f"密度矩阵路径 NPU {got} != CPU {want}")


def case_no_complex_kernels(backend, n, metrics):
    """真机复核：整条路径不发出任何被禁的复数算子。"""
    vec = _random_vec(min(n, 10), seed=7)
    state = State.from_array(vec, backend=backend)
    ham = Hamiltonian(
        n_qubits=min(n, 10),
        terms=[("ZZ", [0, 1], 1.0), ("XY", [2, 3], 0.5), ("Y", [min(n, 10) - 1], -0.25)],
    )
    with _BanUnsupportedComplexOps():
        value = ham.expectation(state, backend)
    metrics["ban_mode_value"] = float(value)
    if not np.isfinite(value):
        raise AssertionError(f"拦截模式下期望值非有限: {value}")


def case_device_residency(backend, n, metrics):
    """设备驻留：不得出现会静默回落 CPU 的算子。

    **这道 case 在 2d21ac3 上是坏的**：当时只记录张量的 device，而 torch_npu 的
    ``npu_cpu_fallback`` 会把结果搬回 NPU，于是记录到的设备恒为 npu——尽管
    ``aten::bitwise_right_shift`` 实际在 CPU 上跑，每个 Pauli 项一次 H2D/D2H 往返。
    一个无法失败的断言什么也没证明。

    现在改为拦截**算子**本身：位运算在昇腾上无内核，出现即判失败。
    """

    bitwise_ops = []
    devices = set()

    class _RecordOps(torch.utils._python_dispatch.TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = str(func)
            if any(tag in name for tag in ("bitwise_", "__lshift__", "__rshift__", "shift")):
                bitwise_ops.append(name)
            out = func(*args, **kwargs)
            for value in (out if isinstance(out, (tuple, list)) else [out]):
                if isinstance(value, torch.Tensor) and value.numel() > 1:
                    devices.add(value.device.type)
            return out

    vec = _random_vec(min(n, 12), seed=21)
    state = State.from_array(vec, backend=backend)
    ham = _tfim(min(n, 12))
    with _RecordOps():
        ham.expectation(state, backend)

    metrics["devices_seen"] = sorted(devices)
    metrics["bitwise_ops"] = sorted(set(bitwise_ops))

    if bitwise_ops:
        raise AssertionError(
            f"期望值路径发出位运算（昇腾无内核，会回落 CPU）: {sorted(set(bitwise_ops))}"
        )
    expected = getattr(backend._device, "type", "cpu")
    stray = devices - {expected}
    if expected == "npu" and stray:
        raise AssertionError(f"期望值路径出现非 NPU 张量: {sorted(stray)}")


def case_sparse_matches_dense(backend, n, metrics):
    """在稠密仍装得下的比特数上做稀疏 vs 稠密对拍，并记录加速比。"""
    small = min(n, 10)  # complex64 稠密 = 2^(2·10)·8B = 8 MB
    vec = _random_vec(small, seed=small + 31)
    state = State.from_array(vec, backend=backend)
    ham = _tfim(small)

    sparse_value = ham.expectation(state, backend)
    sparse_s = _time(backend, lambda: ham.expectation(state, backend), repeats=5)

    dense_matrix = ham.to_matrix(backend)
    dense_value = float(np.real(complex(state.expectation(dense_matrix))))
    dense_s = _time(backend, lambda: state.expectation(ham.to_matrix(backend)), repeats=5)

    metrics["sparse_s"] = sparse_s
    metrics["dense_s"] = dense_s
    metrics["speedup"] = dense_s / sparse_s if sparse_s > 0 else float("inf")
    if not np.isclose(sparse_value, dense_value, atol=1e-3):
        raise AssertionError(f"稀疏 {sparse_value} != 稠密 {dense_value}")


def case_scale_beyond_dense(backend, max_qubits, metrics):
    """稠密矩阵不可能达到的规模。

    complex64 稠密 H 的字节数是 ``2^(2n)·8``：n=14 → 2.1 GB，n=16 → 34 GB，
    n=18 → 550 GB。能跑完本身就是"没走稠密"的证明。
    """
    timings = {}
    for n in range(14, max_qubits + 1, 2):
        vec = _random_vec(n, seed=n)
        state = State.from_array(vec, backend=backend)
        ham = _tfim(n)
        value = ham.expectation(state, backend)  # 预热
        elapsed = _time(backend, lambda: ham.expectation(state, backend))
        dense_gb = (4 ** n) * 8 / 1e9
        timings[f"n{n}"] = {"seconds": elapsed, "value": float(value), "dense_would_need_gb": dense_gb}
        print(f"        n={n:2d}  {elapsed*1000:8.2f} ms   (稠密需 {dense_gb:9.2f} GB)")
        if not np.isfinite(value):
            raise AssertionError(f"n={n} 期望值非有限")
    metrics["scale"] = timings


def case_vqe_energy_loop(backend, n, metrics):
    """端到端：变分循环里的能量求值路径（BasicVQE 默认走的那条）。"""
    from aicir.ansatze import hea
    from aicir.primitives import StatevectorEstimator

    small = min(n, 12)
    circuit = hea(small, layers=2)
    params = np.linspace(0.1, 1.0, len(circuit.parameters))
    bound = circuit.bind_parameters(dict(zip(circuit.parameters, params)))
    ham = _tfim(small)

    estimator = StatevectorEstimator(backend=backend)
    energy = estimator.run(bound, ham).value  # 预热
    elapsed = _time(backend, lambda: estimator.run(bound, ham))

    metrics["vqe_energy"] = float(energy)
    metrics["vqe_seconds"] = elapsed
    print(f"        n={small} 单次能量 {elapsed*1000:.2f} ms, E={energy:.6f}")
    if not np.isfinite(energy):
        raise AssertionError(f"VQE 能量非有限: {energy}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--n-qubits", type=int, default=12, help="常规 case 的比特数")
    parser.add_argument("--max-qubits", type=int, default=18, help="规模 case 的上限（14 起步，步长 2）")
    parser.add_argument("--output-json", default=None, help="把结果写成 JSON（便于归档）")
    args = parser.parse_args()

    backend = _backend(args.allow_cpu_fallback)
    n = args.n_qubits
    metrics: dict = {}

    cases = [
        ("correctness_vs_cpu", lambda: case_correctness_vs_cpu(backend, n, metrics)),
        ("pauli_y_sign_convention", lambda: case_pauli_y_sign_convention(backend, n, metrics)),
        ("density_matrix", lambda: case_density_matrix(backend, n, metrics)),
        ("no_complex_kernels", lambda: case_no_complex_kernels(backend, n, metrics)),
        ("device_residency", lambda: case_device_residency(backend, n, metrics)),
        ("sparse_matches_dense", lambda: case_sparse_matches_dense(backend, n, metrics)),
        ("scale_beyond_dense", lambda: case_scale_beyond_dense(backend, args.max_qubits, metrics)),
        ("vqe_energy_loop", lambda: case_vqe_energy_loop(backend, n, metrics)),
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

    print(f"\npauli_sparse_probe: {passed}/{len(cases)} cases passed")

    if args.output_json:
        payload = {
            "passed": passed,
            "total": len(cases),
            "failed": [name for name, _ in failures],
            "device": str(getattr(backend, "_device", "unknown")),
            "fallback_to_cpu": bool(args.allow_cpu_fallback),
            "n_qubits": n,
            "max_qubits": args.max_qubits,
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
