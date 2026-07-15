#!/usr/bin/env python3
"""真机 NPU 张量网络 / MPS 路径探针：找 complex64 kernel 缺口。

覆盖 hotpath 未触及的 TN/MPS 表面：
  - MPS 期望值 _transfer 收缩（tensordot/conj/transpose 链，Pauli 路径）
  - MPS 稠密回退期望（非 Pauli observable → expectation_sv）
  - MPS bond 截断（_apply_two_site SVD + max_bond_dim，real-embedding SVD）
  - MPS 非相邻双比特门（SWAP 网络 + flat 位置换 gather）
  - TN 全态收缩 tn_statevector
  - TN 期望 tn_expectation
  - TN single_amplitude / partial_amplitude

每个 case 独立兜底，逐条 PASS/FAIL + 首行错误，末尾汇总。
用法（NPU 机器，仓库根目录）:
    python scripts/npu/tn_probe.py                       # 严格 NPU
    python scripts/npu/tn_probe.py --allow-cpu-fallback  # 仅本地开发
    python scripts/npu/tn_probe.py --n-qubits 8
"""
from __future__ import annotations

import argparse
import traceback

import numpy as np

from aicir import (
    Circuit,
    Hamiltonian,
    NPUBackend,
    PauliString,
    cx,
    hadamard,
    rx,
    ry,
    rzz,
)
from aicir.backends.npu_backend import is_npu_available
from aicir.core.state import State
from aicir.simulator import (
    mps_statevector,
    mps_expectation,
    tn_statevector,
    tn_expectation,
    single_amplitude,
)


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


def _entangling_circuit(n: int, backend: NPUBackend) -> Circuit:
    gates = [hadamard(0)]
    for q in range(1, n):
        gates.append(cx(q, [q - 1]))
    gates += [ry(0.4, 0), rx(0.3, n - 1), rzz(0.2, 0, n - 1)]
    return Circuit(*gates, n_qubits=n, backend=backend)


def _ref_state(circ, backend):
    return State.zero_state(circ.n_qubits, backend).evolve(circ.unitary(backend=backend))


# ---------- MPS ----------
def case_mps_pauli_expectation(backend, n):
    circ = _entangling_circuit(n, backend)
    obs = PauliString("ZZ", n_qubits=n, qubits=[0, n - 1])  # ZZ on ends
    val = mps_expectation(circ, obs, backend=backend)   # _transfer 收缩链
    ref_psi = _ref_state(circ, backend)
    zz = np.ones(1 << n)
    for i in range(1 << n):
        b0 = (i >> (n - 1)) & 1
        b1 = i & 1
        zz[i] = (1 - 2 * b0) * (1 - 2 * b1)
    ref = float(np.real(np.vdot(ref_psi.to_numpy(), zz * ref_psi.to_numpy())))
    if not np.isclose(float(np.real(val)), ref, atol=1e-3):
        raise AssertionError(f"MPS Pauli 期望 {val} != 参考 {ref}")


def case_mps_dense_fallback(backend, n):
    # 非 Pauli observable → 稠密回退 expectation_sv
    circ = _entangling_circuit(n, backend)
    dim = 1 << n
    herm = np.diag(np.arange(dim, dtype=float)).astype(complex)  # 对角 Hermitian
    val = mps_expectation(circ, herm, backend=backend)
    ref_psi = _ref_state(circ, backend).to_numpy()
    ref = float(np.real(np.vdot(ref_psi, np.arange(dim) * ref_psi)))
    if not np.isclose(float(np.real(val)), ref, atol=1e-2):
        raise AssertionError(f"MPS 稠密回退期望 {val} != 参考 {ref}")


def case_mps_truncation(backend, n):
    # bond 截断路径：max_bond_dim 强制 _apply_two_site SVD 截断（real-embedding SVD）
    circ = _entangling_circuit(n, backend)
    mps = mps_statevector(circ, max_bond_dim=4, cutoff=1e-8, backend=backend)
    psi = mps.to_statevector().to_numpy().reshape(-1)
    norm = float(np.linalg.norm(psi))
    if not np.isclose(norm, 1.0, atol=1e-2):
        raise AssertionError(f"截断 MPS 末态范数 {norm} 偏离 1")


def case_mps_nonadjacent_gate(backend, n):
    # 非相邻双比特门触发 SWAP 网络 + to_statevector flat 位置换 gather
    gates = [hadamard(0), cx(n - 1, [0]), rzz(0.3, 0, n - 1)]
    circ = Circuit(*gates, n_qubits=n, backend=backend)
    mps = mps_statevector(circ, backend=backend)
    psi = mps.to_statevector().to_numpy().reshape(-1)
    ref = _ref_state(circ, backend).to_numpy().reshape(-1)
    if not np.allclose(psi, ref, atol=1e-3):
        raise AssertionError("非相邻门 MPS 末态与参考不符")


# ---------- 张量网络 ----------
def case_tn_statevector(backend, n):
    circ = _entangling_circuit(n, backend)
    psi = tn_statevector(circ, backend=backend)
    arr = np.asarray(backend.to_numpy(psi)).reshape(-1)
    ref = _ref_state(circ, backend).to_numpy().reshape(-1)
    # TN 与逐门可能差全局相位，比较概率
    if not np.allclose(np.abs(arr) ** 2, np.abs(ref) ** 2, atol=1e-3):
        raise AssertionError("TN 全态概率与参考不符")


def case_tn_expectation(backend, n):
    circ = _entangling_circuit(n, backend)
    ham = Hamiltonian([PauliString("Z", n_qubits=n, qubits=[0]),
                       PauliString("Z", n_qubits=n, qubits=[n - 1])])
    val = tn_expectation(circ, ham, backend=backend)   # expectation_sv on TN 末态
    if not np.isfinite(float(np.real(val))):
        raise AssertionError(f"TN 期望非有限: {val}")


def case_tn_single_amplitude(backend, n):
    circ = _entangling_circuit(n, backend)
    amp = single_amplitude(circ, "0" * n, backend=backend)
    ref = _ref_state(circ, backend).to_numpy().reshape(-1)[0]
    if not np.isclose(abs(complex(amp)), abs(complex(ref)), atol=1e-3):
        raise AssertionError(f"TN single_amplitude |{amp}| != |{ref}|")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--n-qubits", type=int, default=6)
    args = parser.parse_args()

    backend = _backend(args.allow_cpu_fallback)
    n = args.n_qubits

    cases = [
        ("mps_pauli_expectation", lambda: case_mps_pauli_expectation(backend, n)),
        ("mps_dense_fallback", lambda: case_mps_dense_fallback(backend, n)),
        ("mps_truncation", lambda: case_mps_truncation(backend, n)),
        ("mps_nonadjacent_gate", lambda: case_mps_nonadjacent_gate(backend, n)),
        ("tn_statevector", lambda: case_tn_statevector(backend, n)),
        ("tn_expectation", lambda: case_tn_expectation(backend, n)),
        ("tn_single_amplitude", lambda: case_tn_single_amplitude(backend, n)),
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

    print(f"tn_probe: {passed}/{len(cases)} cases passed")
    if failures:
        print("\n===== 失败详情 =====")
        for name, tb in failures:
            print(f"\n--- {name} ---\n{tb}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
