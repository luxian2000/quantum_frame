#!/usr/bin/env python3
"""真机 NPU 噪声路径全面探针：找 complex64 kernel 缺口。

覆盖 hotpath 未触及的噪声表面：
  - 全部单/双比特 Kraus 信道类型（各自 matmul(matmul(k,ρ),k†) + backend.add）
  - 密度态 State 方法：purity（matmul+trace）、von_neumann_entropy（eigvalsh）、
    partial_trace、expectation_dm、is_pure
  - evolve_density_gatewise / noise_sensitivity（逐门密度演化 + 保真度）
  - 噪声 + observables（走 expectation_dm）
  - 噪声 + 末端 return_state=True（密度矩阵聚合 + Result.reduce 偏迹）

每个 case 独立 try/except，逐条打印 PASS/FAIL + 首行错误，末尾汇总。
用法（NPU 机器，仓库根目录）:
    python scripts/npu/noise_probe.py                      # 严格 NPU
    python scripts/npu/noise_probe.py --allow-cpu-fallback # 仅本地开发
    python scripts/npu/noise_probe.py --n-qubits 6
"""
from __future__ import annotations

import argparse
import traceback

import numpy as np

from aicir import Circuit, Measure, NPUBackend, cx, hadamard, rx, ry, measure
from aicir.backends.npu_backend import is_npu_available
from aicir.noise import (
    AmplitudeDampingChannel,
    BitFlipChannel,
    CorrelatedTwoQubitPauliChannel,
    DepolarizingChannel,
    GeneralizedAmplitudeDampingChannel,
    NoiseModel,
    PauliChannel,
    PhaseDampingChannel,
    PhaseFlipChannel,
    TwoQubitDepolarizingChannel,
    noise_sensitivity,
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


def _base_circuit(n: int, backend: NPUBackend) -> Circuit:
    gates = [hadamard(0)] + [cx(q, [q - 1]) for q in range(1, n)] + [ry(0.3, 0), rx(0.2, n - 1)]
    return Circuit(*gates, n_qubits=n, backend=backend)


def _run_noisy(backend, n, channel, *, shots=8, return_state=False, observables=None):
    circ = _base_circuit(n, backend)
    circ.noise_model = NoiseModel().add_channel(channel)
    return Measure(backend).run(circ, shots=shots, seed=7, return_state=return_state,
                                observables=observables, return_probabilities=True)


# ---------- 逐信道 Kraus 类型 ----------
def _channel_cases(n):
    """返回 (名称, channel) 列表，覆盖全部 Kraus 信道类型。"""
    q = min(1, n - 1)
    q2 = min(2, n - 1)
    return [
        ("depolarizing", DepolarizingChannel(target_qubit=0, p=0.05)),
        ("bit_flip", BitFlipChannel(target_qubit=0, p=0.1)),
        ("phase_flip", PhaseFlipChannel(target_qubit=q, p=0.1)),
        ("amplitude_damping", AmplitudeDampingChannel(target_qubit=0, gamma=0.15)),
        ("phase_damping", PhaseDampingChannel(target_qubit=q, gamma=0.15)),
        ("generalized_amp_damping",
         GeneralizedAmplitudeDampingChannel(target_qubit=0, gamma=0.15, p_excited=0.3)),
        ("pauli", PauliChannel(target_qubit=q, px=0.02, py=0.03, pz=0.04)),
        ("two_qubit_depolarizing",
         TwoQubitDepolarizingChannel(qubit_1=0, qubit_2=q2, p=0.05)),
        ("correlated_two_qubit_pauli",
         CorrelatedTwoQubitPauliChannel(qubit_1=0, qubit_2=q2,
                                        probabilities={"xx": 0.02, "zz": 0.03})),
    ]


def case_channel(backend, n, name, channel):
    result = _run_noisy(backend, n, channel)
    total = float(np.sum(np.asarray(result.probabilities, dtype=float)))
    if not np.isclose(total, 1.0, atol=1e-4):
        raise AssertionError(f"{name}: probabilities 之和 {total} != 1")
    counts = result.counts(-1)
    if sum(counts.values()) != 8:
        raise AssertionError(f"{name}: 计数总和 {sum(counts.values())} != 8")


# ---------- 密度态 State 方法 ----------
def case_density_state_methods(backend, n):
    circ = _base_circuit(n, backend)
    circ.noise_model = NoiseModel().add_channel(AmplitudeDampingChannel(target_qubit=0, gamma=0.2))
    result = Measure(backend).run(circ, shots=4, seed=1, return_state=True)
    rho = result.final_state
    if not rho.is_density:
        raise AssertionError("噪声末态应为密度矩阵")
    purity = rho.purity()                 # matmul(rho,rho) + trace（complex64）
    if not (0.0 < purity <= 1.0 + 1e-5):
        raise AssertionError(f"purity 越界: {purity}")
    entropy = rho.von_neumann_entropy()   # eigvalsh（to_numpy 后主机）
    if entropy < -1e-6:
        raise AssertionError(f"熵为负: {entropy}")
    reduced = rho.partial_trace([0])      # backend.partial_trace（complex64）
    if reduced.n_qubits != 1:
        raise AssertionError("偏迹结果比特数应为 1")
    tr = float(np.real(np.trace(np.asarray(reduced.to_numpy()))))
    if not np.isclose(tr, 1.0, atol=1e-4):
        raise AssertionError(f"约化密度矩阵迹 {tr} != 1")


def case_density_observables(backend, n):
    zz = np.zeros((1 << n, 1 << n), dtype=complex)
    diag = np.array([1.0 if bin(i).count("1") % 2 == 0 else -1.0 for i in range(1 << n)])
    np.fill_diagonal(zz, diag)
    result = _run_noisy(backend, n, DepolarizingChannel(target_qubit=0, p=0.1),
                        return_state=True, observables={"parity": zz})
    val = result.expectation_values["parity"]   # expectation_dm: matmul+trace
    if not (-1.0 - 1e-5 <= val <= 1.0 + 1e-5):
        raise AssertionError(f"期望值越界: {val}")


def case_noise_sensitivity(backend, n):
    circ = _base_circuit(n, backend)
    model = NoiseModel().add_channel(AmplitudeDampingChannel(target_qubit=0, gamma=0.1))
    res = noise_sensitivity(circ, backend=backend, noise_model=model)
    if not (0.0 <= res.avg_fidelity_loss <= 1.0 + 1e-5):
        raise AssertionError(f"fidelity loss 越界: {res.avg_fidelity_loss}")


def case_noise_with_incircuit_measure(backend, n):
    # 噪声 + in-circuit measure：密度路径下的联合 Pauli 投影 + 噪声共存
    gates = [hadamard(q) for q in range(n)] + [measure([0, 1], basis="X")]
    circ = Circuit(*gates, n_qubits=n, backend=backend)
    circ.noise_model = NoiseModel().add_channel(PhaseFlipChannel(target_qubit=0, p=0.05))
    result = Measure(backend).run(circ, shots=6, seed=3, return_state=False)
    if sum(result.counts(-1).values()) != 6:
        raise AssertionError("噪声 + in-circuit measure 计数总和不符")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--n-qubits", type=int, default=5)
    args = parser.parse_args()

    backend = _backend(args.allow_cpu_fallback)
    n = args.n_qubits

    cases = [(f"channel:{name}", (lambda nm=name, ch=ch: case_channel(backend, n, nm, ch)))
             for name, ch in _channel_cases(n)]
    cases += [
        ("density_state_methods", lambda: case_density_state_methods(backend, n)),
        ("density_observables", lambda: case_density_observables(backend, n)),
        ("noise_sensitivity", lambda: case_noise_sensitivity(backend, n)),
        ("noise_with_incircuit_measure", lambda: case_noise_with_incircuit_measure(backend, n)),
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

    print(f"noise_probe: {passed}/{len(cases)} cases passed")
    if failures:
        print("\n===== 失败详情 =====")
        for name, tb in failures:
            print(f"\n--- {name} ---\n{tb}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
