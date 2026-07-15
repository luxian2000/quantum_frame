#!/usr/bin/env python3
"""真机 NPU ansatz 探针：hea / hea_ti / uccsd 的设备门应用。

覆盖 ansatze/ 的设备数值面（不同门族在 NPU 上的应用 + 期望）：
  - hea：局部旋转 + 纠缠拓扑
  - hea_ti：trapped-ion，per-layer RxRyRx + 全局 TFIM 演化（rxx/ms_gate）
  - uccsd：耦合费米子激发 + 非相邻对的 fSWAP 网络

绑定随机参数后在设备上演化，校验末态范数=1 与 TFIM 能量有限。
每 case 独立兜底。用法：
    python scripts/npu/ansatze_probe.py [--allow-cpu-fallback] [--n-qubits N]
"""
from __future__ import annotations

import argparse
import traceback

import numpy as np

from aicir import Hamiltonian, NPUBackend, PauliString
from aicir.backends.npu_backend import is_npu_available
from aicir.core.state import State
from aicir.ansatze import hardware_efficient_ansatz
from aicir.ansatze.hea_ti import hea_ti_ansatz, hea_ti_parameter_count
from aicir.ansatze.uccsd import uccsd, uccsd_parameter_count


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


def _evolve_and_check(ansatz, params, backend, label):
    bound = ansatz.bind_parameters(np.asarray(params, dtype=float))
    state = State.zero_state(ansatz.n_qubits, backend).evolve(bound.unitary(backend=backend))
    norm = float(np.linalg.norm(np.asarray(state.to_numpy()).reshape(-1)))
    if not np.isclose(norm, 1.0, atol=1e-3):
        raise AssertionError(f"{label} 末态范数 {norm} 偏离 1")
    energy = state.expectation(_tfim(ansatz.n_qubits).to_matrix(backend))
    if not np.isfinite(energy):
        raise AssertionError(f"{label} TFIM 能量非有限: {energy}")


def case_hea(backend, n):
    ansatz = hardware_efficient_ansatz(n_qubits=n, layers=2)
    n_params = len(ansatz.parameters) if hasattr(ansatz, "parameters") else None
    rng = np.random.default_rng(0)
    params = rng.uniform(-np.pi, np.pi, size=n_params)
    _evolve_and_check(ansatz, params, backend, "hea")


def case_hea_ti(backend, n):
    ansatz = hea_ti_ansatz(n_qubits=n, layers=2, variant="tfim")
    n_params = hea_ti_parameter_count(n_qubits=n, layers=2, variant="tfim")
    rng = np.random.default_rng(1)
    params = rng.uniform(-np.pi, np.pi, size=n_params)
    _evolve_and_check(ansatz, params, backend, "hea_ti")


def case_uccsd(backend, n):
    # 4-qubit HF |1100> + 单/双激发，触发非相邻 fSWAP 网络
    if n < 4:
        n = 4
    hf = [1, 1] + [0] * (n - 2)
    excitations = [("single", (1, 2)), ("double", (0, 1, 2, 3))]
    ansatz = uccsd(n_qubits=n, hf_occupation=hf, excitations=excitations, reps=1)
    n_params = uccsd_parameter_count(excitations, reps=1)
    rng = np.random.default_rng(2)
    params = rng.uniform(-0.5, 0.5, size=n_params)
    _evolve_and_check(ansatz, params, backend, "uccsd")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--n-qubits", type=int, default=4)
    args = parser.parse_args()
    backend = _backend(args.allow_cpu_fallback)
    n = args.n_qubits

    cases = [
        ("hea", lambda: case_hea(backend, n)),
        ("hea_ti", lambda: case_hea_ti(backend, n)),
        ("uccsd", lambda: case_uccsd(backend, n)),
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
    print(f"ansatze_probe: {passed}/{len(cases)} cases passed")
    if failures:
        print("\n===== 失败详情 =====")
        for name, tb in failures:
            print(f"\n--- {name} ---\n{tb}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
