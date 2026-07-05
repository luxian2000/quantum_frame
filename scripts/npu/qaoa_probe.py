#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections.abc import Callable

import numpy as np

from aicir import Hamiltonian, NPUBackend, NumpyBackend
from aicir.backends.npu_backend import is_npu_available
from aicir.ir import instruction_name
from aicir.vqc import BasicQAOA


Case = tuple[str, Callable[[NPUBackend, int], None]]


def _backend(allow_cpu_fallback: bool) -> NPUBackend:
    if not allow_cpu_fallback and not is_npu_available():
        raise SystemExit("NPU is unavailable. Use --allow-cpu-fallback only for local script validation.")
    backend = NPUBackend.from_distributed_env(fallback_to_cpu=allow_cpu_fallback)
    device_type = getattr(backend._device, "type", None)
    if not allow_cpu_fallback and device_type != "npu":
        raise AssertionError(f"strict QAOA NPU probe resolved device={backend._device!r}, expected npu")
    print(f"[backend] {backend.name}")
    print(f"[runtime] {backend.runtime_context}")
    return backend


def _assert_close(actual, expected, *, atol: float, label: str) -> None:
    actual_arr = np.asarray(actual)
    expected_arr = np.asarray(expected)
    if not np.allclose(actual_arr, expected_arr, atol=atol):
        raise AssertionError(f"{label}: actual={actual_arr!r} expected={expected_arr!r} atol={atol}")


def _assert_finite(value: float, *, label: str) -> None:
    if not math.isfinite(float(value)):
        raise AssertionError(f"{label}: expected finite value, got {value!r}")


def _numpy_energy(qaoa: BasicQAOA, params: np.ndarray) -> float:
    return qaoa.energy(params, backend=NumpyBackend())


def case_diagonal_gate_level_energy_and_sampling(backend: NPUBackend, shots: int) -> None:
    hamiltonian = Hamiltonian(
        n_qubits=3,
        terms=[
            ("Z", [0], 0.4),
            ("ZZ", [0, 2], -0.75),
            ("III", 0.2),
        ],
    )
    params = np.array([0.31, -0.17], dtype=float)
    qaoa = BasicQAOA(problem_hamiltonian=hamiltonian, p=1)

    circuit = qaoa.build_circuit(params, backend=backend)
    names = [instruction_name(gate) for gate in circuit.operations]
    for required in ("hadamard", "rz", "rzz", "rx"):
        if required not in names:
            raise AssertionError(f"diagonal QAOA circuit missing {required!r}: {names!r}")

    actual = qaoa.energy(params, backend=backend)
    expected = _numpy_energy(qaoa, params)
    _assert_close(actual, expected, atol=2e-5, label="diagonal QAOA exact energy")

    counts = qaoa.sample(params, shots=shots, backend=backend, seed=7)
    if sum(counts.values()) != shots:
        raise AssertionError(f"sample counts do not sum to shots: counts={counts!r}, shots={shots}")
    sampled_energy = qaoa.energy(params, shots=shots, backend=backend, seed=7)
    _assert_finite(sampled_energy, label="diagonal QAOA sampled energy")


def case_non_diagonal_trotter_order_1_exact_energy(backend: NPUBackend, shots: int) -> None:
    del shots
    hamiltonian = Hamiltonian(
        n_qubits=2,
        terms=[
            ("X", [0], 0.35),
            ("YZ", [0, 1], -0.2),
            ("ZZ", [0, 1], 0.5),
        ],
    )
    params = np.array([0.23, -0.19], dtype=float)
    qaoa = BasicQAOA(problem_hamiltonian=hamiltonian, p=1, trotter_steps=2, trotter_order=1)

    actual = qaoa.energy(params, backend=backend)
    expected = _numpy_energy(qaoa, params)
    _assert_close(actual, expected, atol=3e-5, label="non-diagonal first-order QAOA exact energy")

    try:
        qaoa.energy(params, shots=16, backend=backend, seed=11)
    except ValueError as exc:
        if "non-diagonal" not in str(exc):
            raise
    else:
        raise AssertionError("non-diagonal shots-based QAOA energy must require a Pauli estimator")


def case_non_diagonal_trotter_order_2_exact_energy(backend: NPUBackend, shots: int) -> None:
    del shots
    hamiltonian = Hamiltonian(
        n_qubits=2,
        terms=[
            ("X", [0], 0.35),
            ("Z", [1], -0.6),
            ("XY", [0, 1], 0.25),
        ],
    )
    params = np.array([0.37, 0.21], dtype=float)
    qaoa = BasicQAOA(problem_hamiltonian=hamiltonian, p=1, trotter_steps=2, trotter_order=2)

    circuit = qaoa.build_circuit(params, backend=backend)
    names = [instruction_name(gate) for gate in circuit.operations]
    if names.count("rx") < 3 or "rz" not in names:
        raise AssertionError(f"second-order Trotter circuit missing expected rotations: {names!r}")

    actual = qaoa.energy(params, backend=backend)
    expected = _numpy_energy(qaoa, params)
    _assert_close(actual, expected, atol=3e-5, label="non-diagonal second-order QAOA exact energy")


def case_short_qaoa_run_on_npu(backend: NPUBackend, shots: int) -> None:
    del shots
    hamiltonian = Hamiltonian(n_qubits=1, terms=[("Z", [0], 1.0)])
    qaoa = BasicQAOA(problem_hamiltonian=hamiltonian, p=1, seed=3)
    result = qaoa.run(
        max_iters=2,
        lr=0.05,
        init_params=np.array([0.12, -0.08], dtype=float),
        backend=backend,
    )
    _assert_finite(result.energy, label="short QAOA run energy")
    if result.parameters is None or result.parameters.shape != (2,):
        raise AssertionError(f"QAOA run returned invalid parameters: {result.parameters!r}")
    if len(result.energy_history) != 2:
        raise AssertionError(f"QAOA run expected 2 history entries, got {result.energy_history!r}")


CASES: tuple[Case, ...] = (
    ("diagonal_gate_level_energy_and_sampling", case_diagonal_gate_level_energy_and_sampling),
    ("non_diagonal_trotter_order_1_exact_energy", case_non_diagonal_trotter_order_1_exact_energy),
    ("non_diagonal_trotter_order_2_exact_energy", case_non_diagonal_trotter_order_2_exact_energy),
    ("short_qaoa_run_on_npu", case_short_qaoa_run_on_npu),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strict NPU probe for gate-level BasicQAOA.")
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Allow CPU fallback for local script validation. Do not use for real NPU validation.",
    )
    parser.add_argument("--shots", type=int, default=64, help="Shots for diagonal QAOA sampling checks.")
    args = parser.parse_args(argv)

    if args.shots <= 0:
        raise SystemExit("--shots must be positive")

    backend = _backend(args.allow_cpu_fallback)
    for name, case in CASES:
        print(f"[case] {name}")
        case(backend, int(args.shots))
    print("[ok] QAOA NPU probe completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
