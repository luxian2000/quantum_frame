#!/usr/bin/env python3
"""真机 NPU 化学 + 度量 + primitives 探针。

覆盖三个模块的设备数值面：
  chemistry:  分子 Hamiltonian 期望（get_molecule → Hamiltonian → State.expectation）
  metrics:    expressibility KL_Haar_relative（态保真度 = |<ψ|φ>|^2 内积）
  primitives: StatevectorEstimator / StatevectorSampler / NoisyEstimator

每 case 独立兜底。用法：
    python scripts/npu/chem_metrics_probe.py [--allow-cpu-fallback]
"""
from __future__ import annotations

import argparse
import traceback

import numpy as np

from aicir import Circuit, Hamiltonian, NPUBackend, PauliString, cx, hadamard
from aicir.backends.npu_backend import is_npu_available
from aicir.core.state import State
from aicir.chemistry import molecule_hamiltonian, molecule_matrix
from aicir.metrics.expressibility import KL_Haar_relative
from aicir.primitives import StatevectorEstimator, StatevectorSampler, NoisyEstimator
from aicir.ansatze import hardware_efficient_ansatz
from aicir.noise import NoiseModel, DepolarizingChannel


def _backend(allow_cpu_fallback: bool) -> NPUBackend:
    if not allow_cpu_fallback and not is_npu_available():
        raise SystemExit("NPU is unavailable. Use --allow-cpu-fallback only for local script validation.")
    backend = NPUBackend.from_distributed_env(fallback_to_cpu=allow_cpu_fallback)
    if not allow_cpu_fallback and getattr(backend._device, "type", None) != "npu":
        raise AssertionError(f"strict NPU probe resolved device={backend._device!r}, expected npu")
    print(f"[backend] {backend.name}")
    print(f"[runtime] {backend.runtime_context}")
    return backend


# ---------- chemistry ----------
def case_chem_h2_expectation(backend):
    ham = molecule_hamiltonian("h2")
    n = ham.n_qubits
    # HF |01> 计算基态上的能量期望（走 State.expectation → expectation_sv/dm）
    circ = Circuit(*( [] ), n_qubits=n, backend=backend)
    state = State.zero_state(n, backend).evolve(circ.unitary(backend=backend))
    op = ham.to_matrix(backend)
    val = state.expectation(op)
    ref = float(np.min(np.linalg.eigvalsh(np.asarray(molecule_matrix("h2")))))
    if not np.isfinite(val):
        raise AssertionError(f"H2 能量非有限: {val}")
    if val < ref - 1e-2:
        raise AssertionError(f"H2 |00> 期望 {val} 低于基态 {ref}（不可能）")


def case_chem_lih_hamiltonian(backend):
    ham = molecule_hamiltonian("lih")
    n = ham.n_qubits
    circ = Circuit(hadamard(0), n_qubits=n, backend=backend)
    state = State.zero_state(n, backend).evolve(circ.unitary(backend=backend))
    val = state.expectation(ham.to_matrix(backend))
    if not np.isfinite(val):
        raise AssertionError(f"LiH 期望非有限: {val}")


# ---------- metrics ----------
def case_expressibility(backend, n):
    ansatz = hardware_efficient_ansatz(n_qubits=n, layers=2)
    kl = KL_Haar_relative(ansatz, samples=40, n_bins=25, backend=backend)
    if not (np.isfinite(kl) and kl >= -1e-6):
        raise AssertionError(f"expressibility KL 非法: {kl}")


# ---------- primitives ----------
def case_statevector_estimator(backend, n):
    circ = Circuit(hadamard(0), *[cx(q, [q - 1]) for q in range(1, n)], n_qubits=n, backend=backend)
    ham = Hamiltonian([PauliString("Z", n_qubits=n, qubits=[0]),
                       PauliString("Z", n_qubits=n, qubits=[n - 1])])
    est = StatevectorEstimator(backend=backend)
    res = est.run(circ, ham)
    if not np.isfinite(res.value):
        raise AssertionError(f"StatevectorEstimator 值非有限: {res.value}")


def case_statevector_sampler(backend, n):
    circ = Circuit(hadamard(0), *[cx(q, [q - 1]) for q in range(1, n)], n_qubits=n, backend=backend)
    sampler = StatevectorSampler(backend=backend)
    res = sampler.run(circ)
    total = sum(res.probs.values()) if hasattr(res, "probs") else None
    if total is None or not np.isclose(total, 1.0, atol=1e-4):
        raise AssertionError(f"StatevectorSampler 概率和 {total} != 1")


def case_noisy_estimator(backend, n):
    circ = Circuit(hadamard(0), *[cx(q, [q - 1]) for q in range(1, n)], n_qubits=n, backend=backend)
    ham = Hamiltonian([PauliString("Z", n_qubits=n, qubits=[0])])
    model = NoiseModel().add_channel(DepolarizingChannel(target_qubit=0, p=0.05))
    est = NoisyEstimator(model, backend=backend)
    res = est.run(circ, ham)
    if not np.isfinite(res.value):
        raise AssertionError(f"NoisyEstimator 值非有限: {res.value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--n-qubits", type=int, default=4)
    args = parser.parse_args()
    backend = _backend(args.allow_cpu_fallback)
    n = args.n_qubits

    cases = [
        ("chem_h2_expectation", lambda: case_chem_h2_expectation(backend)),
        ("chem_lih_hamiltonian", lambda: case_chem_lih_hamiltonian(backend)),
        ("expressibility", lambda: case_expressibility(backend, n)),
        ("statevector_estimator", lambda: case_statevector_estimator(backend, n)),
        ("statevector_sampler", lambda: case_statevector_sampler(backend, n)),
        ("noisy_estimator", lambda: case_noisy_estimator(backend, n)),
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
    print(f"chem_metrics_probe: {passed}/{len(cases)} cases passed")
    if failures:
        print("\n===== 失败详情 =====")
        for name, tb in failures:
            print(f"\n--- {name} ---\n{tb}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
