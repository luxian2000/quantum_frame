#!/usr/bin/env python3
"""真机 NPU 流水线探针：transpile / io / universal / optimization。

验证电路经变换/序列化后仍在设备上产生正确末态（概率与原电路一致）：
  - transpile.optimize_circuit：优化后设备演化概率与原电路一致
  - QASM 2.0 round-trip：circuit_to_qasm → circuit_from_qasm → 设备演化一致
  - JSON round-trip：circuit_to_json → circuit_from_json → 设备演化一致
  - universal.qft_circuit：QFT 设备演化范数=1
  - optimization.qubo QAOA：QuboBuilder → BasicQAOA 设备能量有限

每 case 独立兜底。用法：
    python scripts/npu/pipeline_probe.py [--allow-cpu-fallback] [--n-qubits N]
"""
from __future__ import annotations

import argparse
import traceback

import numpy as np

from aicir import (
    Circuit,
    NPUBackend,
    circuit_from_json,
    circuit_from_qasm,
    circuit_to_json,
    circuit_to_qasm,
    cx,
    hadamard,
    rx,
    ry,
    rzz,
)
from aicir.backends.npu_backend import is_npu_available
from aicir.core.state import State
from aicir.transpile import optimize_circuit
from aicir.universal.qft import qft_circuit
from aicir.optimization.qubo.modeling.builder import QuboBuilder
from aicir.optimization.qubo.qaoa import builder_to_basic_qaoa


def _backend(allow_cpu_fallback: bool) -> NPUBackend:
    if not allow_cpu_fallback and not is_npu_available():
        raise SystemExit("NPU is unavailable. Use --allow-cpu-fallback only for local script validation.")
    backend = NPUBackend.from_distributed_env(fallback_to_cpu=allow_cpu_fallback)
    if not allow_cpu_fallback and getattr(backend._device, "type", None) != "npu":
        raise AssertionError(f"strict NPU probe resolved device={backend._device!r}, expected npu")
    print(f"[backend] {backend.name}")
    print(f"[runtime] {backend.runtime_context}")
    return backend


def _probs(circ, backend):
    c = Circuit(*circ.gates, n_qubits=circ.n_qubits, backend=backend)
    state = State.zero_state(c.n_qubits, backend).evolve(c.unitary(backend=backend))
    return np.abs(np.asarray(state.to_numpy()).reshape(-1)) ** 2


def _bench_circuit(n):
    gates = [hadamard(0)]
    for q in range(1, n):
        gates.append(cx(q, [q - 1]))
    gates += [ry(0.4, 0), rx(0.3, n - 1), rzz(0.2, 0, n - 1)]
    return Circuit(*gates, n_qubits=n)


def case_transpile_optimize(backend, n):
    circ = _bench_circuit(n)
    ref = _probs(circ, backend)
    opt = optimize_circuit(circ)
    got = _probs(opt, backend)
    if not np.allclose(ref, got, atol=1e-4):
        raise AssertionError("优化后设备概率与原电路不符")


def case_qasm_roundtrip(backend, n):
    circ = _bench_circuit(n)
    ref = _probs(circ, backend)
    restored = circuit_from_qasm(circuit_to_qasm(circ, version="2.0"))
    got = _probs(restored, backend)
    if not np.allclose(ref, got, atol=1e-4):
        raise AssertionError("QASM round-trip 后设备概率不符")


def case_json_roundtrip(backend, n):
    circ = _bench_circuit(n)
    ref = _probs(circ, backend)
    restored = circuit_from_json(circuit_to_json(circ))
    got = _probs(restored, backend)
    if not np.allclose(ref, got, atol=1e-4):
        raise AssertionError("JSON round-trip 后设备概率不符")


def case_qft(backend, n):
    circ = qft_circuit(n)
    state = State.zero_state(n, backend).evolve(
        Circuit(*circ.gates, n_qubits=n, backend=backend).unitary(backend=backend))
    norm = float(np.linalg.norm(np.asarray(state.to_numpy()).reshape(-1)))
    if not np.isclose(norm, 1.0, atol=1e-3):
        raise AssertionError(f"QFT 末态范数 {norm} 偏离 1")


def case_qubo_qaoa(backend, n):
    # 简单 max-cut 型 QUBO：QuboBuilder → BasicQAOA 在设备上求能量
    builder = QuboBuilder()
    ids = [builder.registry.get_or_create(f"x{q}") for q in range(n)]
    for q in range(n - 1):
        builder.add_quadratic(ids[q], ids[q + 1], 1.0)
        builder.add_linear(ids[q], -0.5)
    solver = builder_to_basic_qaoa(builder, p=1, seed=0)
    energy = solver.energy(np.array([0.3, 0.2]), backend=backend)
    if not np.isfinite(energy):
        raise AssertionError(f"QUBO QAOA 能量非有限: {energy}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--n-qubits", type=int, default=4)
    args = parser.parse_args()
    backend = _backend(args.allow_cpu_fallback)
    n = args.n_qubits

    cases = [
        ("transpile_optimize", lambda: case_transpile_optimize(backend, n)),
        ("qasm_roundtrip", lambda: case_qasm_roundtrip(backend, n)),
        ("json_roundtrip", lambda: case_json_roundtrip(backend, n)),
        ("qft", lambda: case_qft(backend, n)),
        ("qubo_qaoa", lambda: case_qubo_qaoa(backend, n)),
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
    print(f"pipeline_probe: {passed}/{len(cases)} cases passed")
    if failures:
        print("\n===== 失败详情 =====")
        for name, tb in failures:
            print(f"\n--- {name} ---\n{tb}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
