#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable

import numpy as np

from aicir import (
    Circuit,
    CircuitIR,
    Hamiltonian,
    Measure,
    Measurement,
    NPUBackend,
    Observable,
    Operation,
    Parameter,
    circuit_from_json,
    circuit_to_json,
    circuit_to_qasm,
    crx,
    cx,
    hadamard,
    rx,
    ry,
    rz,
    rxx,
    rzz,
)
from aicir.backends.npu_backend import is_npu_available
from aicir.core.gates import apply_gate_to_state, gate_to_matrix
from aicir.metrics._utils import count_two_qubit_gates, depth_proxy
from aicir.primitives import StatevectorEstimator
from aicir.qml import ad, auto, fd, psr
from aicir.transpile import PassManager, optimize_circuit


Case = tuple[str, Callable[[NPUBackend], None]]


def _assert_close(actual, expected, *, atol: float, label: str) -> None:
    if not np.allclose(np.asarray(actual), np.asarray(expected), atol=atol):
        raise AssertionError(f"{label}: actual={actual!r} expected={expected!r} atol={atol}")


def _assert_normalized(probs, *, label: str) -> None:
    total = float(np.sum(np.asarray(probs, dtype=float)))
    if not np.isclose(total, 1.0, atol=1e-5):
        raise AssertionError(f"{label}: probabilities sum to {total}")


def _scalar(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    array = np.asarray(value)
    if np.iscomplexobj(array):
        array = array.real
    return float(array.reshape(()))


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


def case_typed_circuit_surface(backend: NPUBackend) -> None:
    theta = Parameter("theta")
    circuit = Circuit(
        ry(theta, 0),
        crx(0.21, 1, [0]),
        rzz(0.33, 0, 1),
        n_qubits=2,
        backend=backend,
    )

    if not all(isinstance(gate, Operation) for gate in circuit.gates):
        raise AssertionError("Circuit.gates must expose typed Operation objects")
    if circuit.legacy_gates[0]["type"] != "ry":
        raise AssertionError("legacy_gates must preserve dict interop")
    if circuit.gates[0]["type"] != "ry":
        raise AssertionError("Operation must preserve legacy key access")

    bound = circuit.bind_parameters([0.4])
    if bound.parameters:
        raise AssertionError("bound circuit should not keep symbolic parameters")
    if not isinstance(bound.gates[0], Operation):
        raise AssertionError("bind_parameters must keep typed operations")

    ir = circuit.ir
    rebuilt = ir.to_circuit(backend=backend)
    if not isinstance(ir, CircuitIR) or not isinstance(rebuilt.gates[0], Operation):
        raise AssertionError("Circuit.ir/to_circuit must keep typed IR")


def case_typed_ir_interop_metrics_transpile(backend: NPUBackend) -> None:
    ir = CircuitIR(
        (
            Operation("hadamard", qubits=(0,)),
            Operation("cx", qubits=(1,), controls=(0,)),
            Operation("rz", qubits=(1,), params=(0.25,)),
        ),
        n_qubits=2,
    )

    qasm = circuit_to_qasm(ir)
    if "h q[0];" not in qasm or "cx q[0],q[1];" not in qasm:
        raise AssertionError(f"QASM output missing typed IR gates:\n{qasm}")

    rebuilt = circuit_from_json(circuit_to_json(ir))
    if rebuilt.n_qubits != ir.n_qubits or rebuilt.legacy_gates != ir.to_gate_dicts():
        raise AssertionError("JSON round-trip changed typed IR operations")

    if count_two_qubit_gates(ir) != 1 or depth_proxy(ir) != 3.0:
        raise AssertionError("typed IR metrics changed")

    optimized = PassManager(["cancel_inverse", "merge_rotations"]).run(
        CircuitIR(
            (
                Operation("hadamard", qubits=(0,)),
                Operation("hadamard", qubits=(0,)),
                Operation("rx", qubits=(0,), params=(0.1,)),
                Operation("rx", qubits=(0,), params=(0.2,)),
            ),
            n_qubits=1,
        )
    )
    optimized_via_legacy = optimize_circuit(optimized)
    if optimized.legacy_gates != optimized_via_legacy.legacy_gates:
        raise AssertionError("transpile/optimizer typed IR round-trip mismatch")

    matrix = gate_to_matrix(Operation("rzz", qubits=(0, 1), params=(math.pi / 2,)), cir_qubits=2, backend=backend)
    if tuple(matrix.shape) != (4, 4):
        raise AssertionError("gate_to_matrix must accept Operation directly on NPU backend")


def case_typed_ir_measurement_execution(backend: NPUBackend) -> None:
    bell = CircuitIR(
        (
            Operation("hadamard", qubits=(0,)),
            Operation("cx", qubits=(1,), controls=(0,)),
            Measurement((0, 1)),
        ),
        n_qubits=2,
    )
    result = Measure(backend).run(bell, shots=None, return_state=False)
    _assert_close(result.probabilities, [0.5, 0.0, 0.0, 0.5], atol=1e-5, label="bell typed IR probs")

    ansatz = CircuitIR(
        (
            Operation("ry", qubits=(0,), params=(0.37,)),
            Operation("rx", qubits=(1,), params=(-0.22,)),
            Operation("cx", qubits=(1,), controls=(0,)),
            Operation("rxx", qubits=(0, 1), params=(0.41,)),
        ),
        n_qubits=2,
    )
    exact = Measure(backend).run(ansatz, shots=None, return_state=True)
    _assert_normalized(exact.probabilities, label="ansatz typed IR probs")

    density = np.zeros((4, 4), dtype=np.complex64)
    density[0, 0] = 1.0
    dm_result = Measure(backend).run(ansatz, shots=None, initial_density_matrix=density, return_state=False)
    _assert_normalized(dm_result.probabilities, label="typed IR density matrix probs")


def case_typed_observable_on_npu(backend: NPUBackend) -> None:
    circuit = CircuitIR((Operation("ry", qubits=(0,), params=(0.4,)),), n_qubits=1)
    observable = Observable.hamiltonian(Hamiltonian([("Z", 1.0)]))

    grad, value = ad(circuit, observable, backend=backend, return_value=True)

    _assert_close(value, math.cos(0.4), atol=1e-5, label="Observable.hamiltonian value")
    _assert_close(grad, [-math.sin(0.4)], atol=1e-5, label="Observable.hamiltonian ad grad")


def _npu_energy(backend: NPUBackend, theta):
    hamiltonian = Hamiltonian([("ZI", 0.7), ("IZ", -0.2), ("XX", 0.35)])
    operator = hamiltonian.to_matrix(backend)
    state = backend.zeros_state(2)
    gates = (
        Operation("ry", qubits=(0,), params=(theta[0],)),
        Operation("rz", qubits=(1,), params=(theta[1],)),
        Operation("rxx", qubits=(0, 1), params=(theta[2],)),
        Operation("cx", qubits=(1,), controls=(0,)),
    )
    for gate in gates:
        state = apply_gate_to_state(gate, state, 2, backend)
    return backend.expectation_sv(state, operator)


def case_deriv_auto_matches_psr_on_npu(backend: NPUBackend) -> None:
    theta = np.array([0.31, -0.27, 0.43], dtype=float)

    grad_auto = auto(lambda values: _npu_energy(backend, values), theta, backend=backend)
    grad_psr = psr(lambda values: _scalar(_npu_energy(backend, values)), theta)

    _assert_close(grad_auto, grad_psr, atol=2e-3, label="auto vs psr on typed Operation energy")


def case_deriv_ad_matches_analytic_on_typed_ir(backend: NPUBackend) -> None:
    circuit = CircuitIR((Operation("ry", qubits=(0,), params=(0.4,)),), n_qubits=1)
    h_z = Hamiltonian([("Z", 1.0)])

    grad, value = ad(circuit, h_z, backend=backend, return_value=True)

    _assert_close(value, math.cos(0.4), atol=1e-5, label="ad value")
    _assert_close(grad, [-math.sin(0.4)], atol=1e-5, label="ad gradient")


def case_deriv_psr_fd_estimator_binding(backend: NPUBackend) -> None:
    template = Circuit(ry(Parameter("theta"), 0), n_qubits=1, backend=backend)
    h_z = Hamiltonian([("Z", 1.0)])
    estimator = StatevectorEstimator(backend)

    psr_result = estimator.gradient(template, h_z, parameter_values=[0.3], method="psr")
    fd_result = estimator.gradient(template, h_z, parameter_values=[0.3], method="fd")

    expected = [-math.sin(0.3)]
    _assert_close(psr_result.gradient, expected, atol=1e-5, label="StatevectorEstimator psr gradient")
    _assert_close(fd_result.gradient, expected, atol=1e-3, label="StatevectorEstimator fd gradient")


TYPED_IR_CASES: tuple[Case, ...] = (
    ("typed_circuit_surface", case_typed_circuit_surface),
    ("typed_ir_interop_metrics_transpile", case_typed_ir_interop_metrics_transpile),
    ("typed_ir_measurement_execution", case_typed_ir_measurement_execution),
    ("typed_observable_on_npu", case_typed_observable_on_npu),
)

DERIV_CASES: tuple[Case, ...] = (
    ("deriv_auto_matches_psr_on_npu", case_deriv_auto_matches_psr_on_npu),
    ("deriv_ad_matches_analytic_on_typed_ir", case_deriv_ad_matches_analytic_on_typed_ir),
    ("deriv_psr_fd_estimator_binding", case_deriv_psr_fd_estimator_binding),
)


def _selected_cases(section: str) -> tuple[Case, ...]:
    if section == "typed-ir":
        return TYPED_IR_CASES
    if section == "deriv":
        return DERIV_CASES
    return TYPED_IR_CASES + DERIV_CASES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Strict NPU probe for typed IR and derivative paths."
    )
    parser.add_argument(
        "--section",
        choices=("all", "typed-ir", "deriv"),
        default="all",
        help="Subset to run.",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Allow CPU fallback for local validation. Omit this on real NPU runs.",
    )
    args = parser.parse_args(argv)

    backend = _backend(args.allow_cpu_fallback)
    failed: list[tuple[str, str]] = []
    for name, fn in _selected_cases(args.section):
        try:
            fn(backend)
        except Exception as exc:  # noqa: BLE001
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"[PASS] {name}")

    if failed:
        print("\nSummary: FAILED")
        for name, message in failed:
            print(f"- {name}: {message}")
        return 1

    print("\nSummary: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
