#!/usr/bin/env python3
"""真机 NPU 编码器路径探针：state-prep + IQP 核（内积）。

覆盖 encoder/ 的设备数值面：
  - AngleEncoder / AmplitudeEncoder / BasisEncoder encode（态制备 + 归一化）
  - IQPEncoder circuit / encode（rz/rzz 对角相位层）
  - IQPEncoder kernel / kernel_matrix（|<Φ(x)|Φ(z)>|^2 内积，走 inner_product）

每 case 独立兜底，逐条 PASS/FAIL + 首行错误，末尾汇总。
用法：python scripts/npu/encoder_probe.py [--allow-cpu-fallback] [--n-qubits N]
"""
from __future__ import annotations

import argparse
import traceback

import numpy as np

from aicir import NPUBackend
from aicir.backends.npu_backend import is_npu_available
from aicir.encoder import AmplitudeEncoder, AngleEncoder, BasisEncoder
from aicir.encoder.iqp import IQPEncoder


def _backend(allow_cpu_fallback: bool) -> NPUBackend:
    if not allow_cpu_fallback and not is_npu_available():
        raise SystemExit("NPU is unavailable. Use --allow-cpu-fallback only for local script validation.")
    backend = NPUBackend.from_distributed_env(fallback_to_cpu=allow_cpu_fallback)
    if not allow_cpu_fallback and getattr(backend._device, "type", None) != "npu":
        raise AssertionError(f"strict NPU probe resolved device={backend._device!r}, expected npu")
    print(f"[backend] {backend.name}")
    print(f"[runtime] {backend.runtime_context}")
    return backend


def _state_norm(encode_result, backend) -> float:
    # encode(...) 返回 (circuit_repr, State)；态在 index [1]
    obj = encode_result[1] if isinstance(encode_result, tuple) else encode_result
    arr = np.asarray(obj.to_numpy()).reshape(-1) if hasattr(obj, "to_numpy") else np.asarray(obj).reshape(-1)
    return float(np.linalg.norm(arr))


def case_angle_encoder(backend, n):
    data = np.linspace(0.1, 1.0, n)
    enc = AngleEncoder(n_qubits=n)
    state = enc.encode(data, backend=backend)
    if not np.isclose(_state_norm(state, backend), 1.0, atol=1e-4):
        raise AssertionError("AngleEncoder 态范数偏离 1")


def case_amplitude_encoder(backend, n):
    data = np.random.default_rng(0).normal(size=1 << n)
    enc = AmplitudeEncoder(n_qubits=n)
    state = enc.encode(data, backend=backend)
    if not np.isclose(_state_norm(state, backend), 1.0, atol=1e-4):
        raise AssertionError("AmplitudeEncoder 态范数偏离 1")


def case_basis_encoder(backend, n):
    bits = [i % 2 for i in range(n)]
    enc = BasisEncoder(n_qubits=n)
    state = enc.encode(bits, backend=backend)
    if not np.isclose(_state_norm(state, backend), 1.0, atol=1e-4):
        raise AssertionError("BasisEncoder 态范数偏离 1")


def case_iqp_encode(backend, n):
    data = np.linspace(0.2, 0.9, n)
    enc = IQPEncoder(n_qubits=n, reps=2)
    state = enc.encode(data, backend=backend)
    if not np.isclose(_state_norm(state, backend), 1.0, atol=1e-4):
        raise AssertionError("IQPEncoder 态范数偏离 1")


def case_iqp_kernel(backend, n):
    # 核对角恒为 1（<Φ(x)|Φ(x)>=1）；内积经 NPU inner_product workaround
    x = np.linspace(0.2, 0.9, n)
    enc = IQPEncoder(n_qubits=n, reps=2)
    k_xx = enc.kernel(x, x, backend=backend)
    if not np.isclose(float(np.real(k_xx)), 1.0, atol=1e-3):
        raise AssertionError(f"IQP kernel 对角 {k_xx} != 1")


def case_iqp_kernel_matrix(backend, n):
    rng = np.random.default_rng(1)
    xs = rng.uniform(0.1, 1.0, size=(3, n))
    enc = IQPEncoder(n_qubits=n, reps=1)
    K = np.asarray(enc.kernel_matrix(xs, backend=backend), dtype=float)
    if K.shape != (3, 3):
        raise AssertionError(f"kernel_matrix 形状 {K.shape} != (3,3)")
    if not np.allclose(np.diag(K), 1.0, atol=1e-3):
        raise AssertionError("kernel_matrix 对角非 1")
    if not np.allclose(K, K.T, atol=1e-3):
        raise AssertionError("kernel_matrix 非对称")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--n-qubits", type=int, default=4)
    args = parser.parse_args()
    backend = _backend(args.allow_cpu_fallback)
    n = args.n_qubits

    cases = [
        ("angle_encoder", lambda: case_angle_encoder(backend, n)),
        ("amplitude_encoder", lambda: case_amplitude_encoder(backend, n)),
        ("basis_encoder", lambda: case_basis_encoder(backend, n)),
        ("iqp_encode", lambda: case_iqp_encode(backend, n)),
        ("iqp_kernel", lambda: case_iqp_kernel(backend, n)),
        ("iqp_kernel_matrix", lambda: case_iqp_kernel_matrix(backend, n)),
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
    print(f"encoder_probe: {passed}/{len(cases)} cases passed")
    if failures:
        print("\n===== 失败详情 =====")
        for name, tb in failures:
            print(f"\n--- {name} ---\n{tb}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
