#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Suite:
    name: str
    description: str
    targets: tuple[str, ...]
    scripts: tuple[tuple[str, ...], ...] = ()


SUITES: dict[str, Suite] = {
    "smoke": Suite(
        name="smoke",
        description="Fast NPU backend sanity checks and the backend-bound circuit smoke path.",
        targets=(
            "tests/backends/test_npu_backend.py::TestNPUBackend::test_fallback_or_npu_device_selection",
            "tests/backends/test_npu_backend.py::TestNPUBackend::test_statevector_pipeline",
            "tests/backends/test_npu_backend.py::TestNPUBackend::test_parameterized_gate_matrices_have_no_complex_backward",
            "tests/smoke_npu_new_path.py",
        ),
    ),
    "backend": Suite(
        name="backend",
        description="Complete NPUBackend regression suite, including distributed helpers.",
        targets=(
            "tests/backends/test_npu_backend.py",
            "tests/backends/test_take_add.py",
            "tests/backends/test_contract_primitives.py",
        ),
    ),
    "ops": Suite(
        name="ops",
        description="NPU-safe complex operator decompositions and backend gradient regressions.",
        targets=(
            "tests/backends/test_npu_hamiltonian_grad.py",
            "tests/backends/test_npu_backend.py::TestNPUBackend::test_npu_complex_matmul_function_matches_native_gradient",
            "tests/backends/test_npu_backend.py::TestNPUBackend::test_npu_real_complex_matmul_function_gradient",
            "tests/backends/test_npu_backend.py::TestNPUBackend::test_npu_expectation_function_matches_native_gradient",
            "tests/backends/test_npu_backend.py::TestNPUBackend::test_npu_circuit_backward_runs_without_complex_ops",
            "tests/gates/test_matrix_autograd.py",
        ),
    ),
    "capacity": Suite(
        name="capacity",
        description="NPU capability probe, capacity guards, and sharding/distributed batch helpers.",
        targets=(
            "tests/backends/test_npu_probe.py",
            "tests/backends/test_npu_backend_caps.py",
            "tests/primitives/test_sharding.py",
            "tests/test_supernet_sharding.py",
            "tests/test_supernet_sharding_dist.py",
        ),
    ),
    "typed_ir": Suite(
        name="typed_ir",
        description="Typed CircuitIR/Operation/Measurement surface, interop, metrics, transpile, and NPU execution.",
        targets=(
            "tests/circuit/test_operation_ir.py",
            "tests/circuit/test_typed_factories.py",
            "tests/circuit/test_circuit_typed_gates_api.py",
            "tests/circuit/test_typed_internal_gate_access.py",
            "tests/circuit/test_typed_ir_internal_migration.py",
            "tests/circuit/io/test_json_qasm_io.py",
            "tests/circuit/io/test_wuyue_interop.py",
            "tests/gates/test_matrix_dispatch_consistency.py",
        ),
        scripts=(("scripts/npu/typed_ir_deriv_probe.py", "--section", "typed-ir"),),
    ),
    "circuit": Suite(
        name="circuit",
        description="Circuit execution, measurement, typed-gate API, and JSON/QASM interop around NPU paths.",
        targets=(
            "tests/circuit/test_state_unified.py",
            "tests/circuit/test_circuit_backend_unitary.py",
            "tests/circuit/test_typed_ir_internal_migration.py",
            "tests/circuit/test_circuit_typed_gates_api.py",
            "tests/measure/test_measure.py",
            "tests/measure/test_unified_run.py",
            "tests/measure/test_estimator.py",
            "tests/circuit/io/test_json_qasm_io.py",
        ),
    ),
    "deriv": Suite(
        name="deriv",
        description="Typed IR derivative paths on NPU: ad, auto, psr/fd, backend autograd, and estimator gradients.",
        targets=(
            "tests/qml",
            "tests/primitives/test_estimator_gradient.py",
            "tests/gates/test_matrix_autograd.py",
            "tests/circuit/test_typed_ir_internal_migration.py::test_circuit_ir_is_accepted_by_measure_and_adjoint_gradient",
            "tests/backends/test_npu_backend.py::TestNPUBackend::test_npu_complex_matmul_function_matches_native_gradient",
            "tests/backends/test_npu_backend.py::TestNPUBackend::test_npu_real_complex_matmul_function_gradient",
            "tests/backends/test_npu_backend.py::TestNPUBackend::test_npu_expectation_function_matches_native_gradient",
            "tests/backends/test_npu_backend.py::TestNPUBackend::test_npu_circuit_energy_gradient_matches_gpu",
            "tests/backends/test_npu_backend.py::TestNPUBackend::test_npu_circuit_backward_runs_without_complex_ops",
            "tests/backends/test_npu_hamiltonian_grad.py",
        ),
        scripts=(("scripts/npu/typed_ir_deriv_probe.py", "--section", "deriv"),),
    ),
    "qml": Suite(
        name="qml",
        description="QML gradients, parameter-shift, qlayer integration, and optimizer parameter registry.",
        targets=(
            "tests/qml",
            "tests/qfun/test_qfun.py",
            "tests/vqc/test_parameter_shift_uses_qml.py",
            "tests/primitives/test_estimator_gradient.py",
            "tests/optimizer/test_params.py",
            "tests/optimizer/test_params_diff_registry.py",
        ),
    ),
    "qaoa": Suite(
        name="qaoa",
        description="Gate-level QAOA on NPU: Hamiltonian circuits, Trotter order 1/2, exact energy, and sampling.",
        targets=(
            "tests/vqc/test_qaoa_canonical.py",
            "tests/vqc/test_qaoa_qfun.py",
            "tests/optimization/qubo/test_qaoa_helpers.py",
        ),
        scripts=(("scripts/npu/qaoa_probe.py",),),
    ),
    "tensor": Suite(
        name="tensor",
        description="Tensor-network simulator, cotengra API, contraction primitives, and NPU demo importability.",
        targets=(
            "tests/simulator/test_gate_tensors.py",
            "tests/simulator/test_contract.py",
            "tests/simulator/test_contract_cotengra.py",
            "tests/simulator/test_api_cotengra.py",
            "tests/simulator/test_tn_autograd.py",
            "tests/simulator/test_npu_demo_importable.py",
        ),
    ),
    "qas": Suite(
        name="qas",
        description="QAS/VQE modules that exercise NPU-friendly batched statevector and gradient workloads.",
        targets=(
            "tests/test_vqa_qas.py",
            "tests/test_qas_runner.py",
            "tests/algorithms/test_vqe_qas_modules.py",
            "tests/algorithms/test_architecture_evaluation.py",
            "tests/algorithms/test_dqas.py",
            "tests/qas/test_strategy_registry.py",
            "tests/vqc/test_vqe_orchestration.py",
        ),
    ),
    "demos": Suite(
        name="demos",
        description="Demo and molecule smoke tests that are reasonable to run before long NPU jobs.",
        targets=(
            "tests/test_reset_demo.py",
            "tests/test_lih_demo.py",
            "tests/test_h2o_demo.py",
            "tests/test_maxcut_demo.py",
            "tests/test_h2_excitation_vqe.py",
            "tests/test_chemistry_demo_ansatz_pools.py",
            "tests/simulator/test_npu_demo_importable.py",
        ),
    ),
}

DEFAULT_SUITES = tuple(SUITES)


def quote_cmd(parts: Iterable[str | os.PathLike[str]]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def npu_check_cmd(python: str) -> list[str]:
    check = (
        "from aicir.backends.npu_backend import is_npu_available; "
        "raise SystemExit(0 if is_npu_available() else 1)"
    )
    return [python, "-c", check]


def pytest_cmd(
    python: str,
    suite: Suite,
    *,
    fail_fast: bool,
    pytest_args: tuple[str, ...],
) -> list[str]:
    cmd = [python, "-m", "pytest"]
    if fail_fast:
        cmd.append("-x")
    cmd.extend(suite.targets)
    cmd.extend(pytest_args)
    return cmd


def script_cmd(python: str, script: tuple[str, ...], *, strict_npu: bool) -> list[str]:
    cmd = [python, *script]
    if not strict_npu:
        cmd.append("--allow-cpu-fallback")
    return cmd


def selected_suites(names: list[str] | None) -> list[Suite]:
    if not names:
        names = list(DEFAULT_SUITES)
    unknown = [name for name in names if name not in SUITES]
    if unknown:
        valid = ", ".join(sorted(SUITES))
        raise SystemExit(f"unknown suite(s): {', '.join(unknown)}; valid suites: {valid}")
    return [SUITES[name] for name in names]


def print_suite_list() -> None:
    for name in DEFAULT_SUITES:
        suite = SUITES[name]
        print(f"{suite.name}: {suite.description}")
        for script in suite.scripts:
            print(f"  - python {quote_cmd(script)}")
        for target in suite.targets:
            print(f"  - {target}")


def normalize_pytest_arg_tokens(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--pytest-arg" and i + 1 < len(argv):
            normalized.append(f"--pytest-arg={argv[i + 1]}")
            i += 2
            continue
        normalized.append(token)
        i += 1
    return normalized


def run_command(cmd: list[str], *, env: dict[str, str]) -> int:
    print(f"+ {quote_cmd(cmd)}", flush=True)
    completed = subprocess.run(cmd, cwd=ROOT, env=env)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run layered AICIR NPU test suites from the repository root."
    )
    parser.add_argument("--list", action="store_true", help="List available suites and pytest targets.")
    parser.add_argument(
        "--suite",
        action="append",
        choices=tuple(DEFAULT_SUITES),
        help="Suite to run. Repeat to run several. Defaults to all suites.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--strict-npu",
        action="store_true",
        help="Fail before pytest unless a real NPU runtime is available.",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Pass -x to pytest.")
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to pytest. Repeat for multiple args.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for checks and pytest commands.",
    )
    raw_argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(normalize_pytest_arg_tokens(list(raw_argv)))

    if args.list:
        print_suite_list()
        return 0

    suites = selected_suites(args.suite)
    env = os.environ.copy()
    if args.strict_npu:
        env["AICIR_REQUIRE_REAL_NPU"] = "1"

    commands: list[list[str]] = []
    if args.strict_npu:
        commands.append(npu_check_cmd(args.python))
    for suite in suites:
        commands.extend(script_cmd(args.python, script, strict_npu=args.strict_npu) for script in suite.scripts)
        commands.append(
            pytest_cmd(
                args.python,
                suite,
                fail_fast=args.fail_fast,
                pytest_args=tuple(args.pytest_arg),
            )
        )

    if args.dry_run:
        print(f"cd {shlex.quote(str(ROOT))}")
        if args.strict_npu:
            print("AICIR_REQUIRE_REAL_NPU=1")
        for cmd in commands:
            print(quote_cmd(cmd))
        return 0

    for cmd in commands:
        rc = run_command(cmd, env=env)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
