"""可选 ``estimator=`` 注入点（Phase 3b §3b.3）：``crlqas._energy_of_gates`` 与
``vqe_loop.fair_vqe._evaluate_pauli_state_energy``/``evaluate_vqe_energy``。

两处都遵循同一契约：``estimator=None``（默认）时数值路径与注入前逐字节一致；
传入实现了 ``aicir.primitives`` ``BaseEstimator.run(circuit, observable) -> EstimateResult``
契约的对象时，能量改走 ``estimator.run(...).value``。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class _RecordingEstimator:
    """记录调用、返回固定值的桩 estimator（不做真实物理计算）。"""

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[tuple] = []

    def run(self, circuit, observable, **kwargs):
        self.calls.append((circuit, observable))
        return SimpleNamespace(value=self.value)


# ──────────────────────────────────────────────────────────────────────────────
# crlqas._energy_of_gates
# ──────────────────────────────────────────────────────────────────────────────


def test_energy_of_gates_default_path_unchanged_when_estimator_none():
    from aicir.backends.numpy_backend import NumpyBackend
    from aicir.qas.algorithms.crlqas import _energy_of_gates

    backend = NumpyBackend()
    gates = [{"type": "pauli_x", "target_qubit": 0}]
    hamiltonian_matrix = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex64)

    energy = _energy_of_gates(gates, hamiltonian_matrix, n_qubits=1, backend=backend)

    # |1> 态下 Z 期望为 -1（不传 estimator 时走原有态向量路径，数值与改造前一致）。
    assert energy == pytest.approx(-1.0, abs=1e-6)


def test_energy_of_gates_estimator_injection_changes_energy_source():
    from aicir.backends.numpy_backend import NumpyBackend
    from aicir.qas.algorithms.crlqas import _energy_of_gates

    backend = NumpyBackend()
    gates = [{"type": "pauli_x", "target_qubit": 0}]
    hamiltonian_matrix = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex64)

    stub = _RecordingEstimator(value=42.0)
    energy = _energy_of_gates(gates, hamiltonian_matrix, n_qubits=1, backend=backend, estimator=stub)

    assert energy == 42.0
    assert len(stub.calls) == 1


def test_energy_of_gates_real_estimator_matches_default_numeric_path():
    from aicir.backends.numpy_backend import NumpyBackend
    from aicir.primitives.estimator import StatevectorEstimator
    from aicir.qas.algorithms.crlqas import _energy_of_gates

    backend = NumpyBackend()
    gates = [{"type": "pauli_x", "target_qubit": 0}]
    hamiltonian_matrix = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex64)

    default_energy = _energy_of_gates(gates, hamiltonian_matrix, n_qubits=1, backend=backend)
    estimator_energy = _energy_of_gates(
        gates, hamiltonian_matrix, n_qubits=1, backend=backend, estimator=StatevectorEstimator(backend)
    )

    assert estimator_energy == pytest.approx(default_energy, abs=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# vqe_loop.fair_vqe.evaluate_vqe_energy / _evaluate_pauli_state_energy
# ──────────────────────────────────────────────────────────────────────────────


def _tiny_architecture_and_problem():
    from aicir.qas.core._types import ArchitectureSpec
    from aicir.qas.problems.hamiltonians import VQEProblem

    architecture = ArchitectureSpec.from_gates("empty_1q", [], n_qubits=1)
    problem = VQEProblem(name="z0_1q", n_qubits=1, hamiltonian=((1.0, "Z"),), reference_energy=-1.0)
    return architecture, problem


def test_evaluate_vqe_energy_default_path_unchanged_when_estimator_none():
    from aicir.qas.vqe_loop.fair_vqe import evaluate_vqe_energy

    architecture, problem = _tiny_architecture_and_problem()

    energy = evaluate_vqe_energy(architecture, problem)

    # 空线路（|0>）下 <Z> = 1（不传 estimator 时数值路径与改造前一致）。
    assert energy == pytest.approx(1.0, abs=1e-6)


def test_evaluate_vqe_energy_estimator_injection_changes_energy_source():
    from aicir.qas.vqe_loop.fair_vqe import evaluate_vqe_energy

    architecture, problem = _tiny_architecture_and_problem()
    stub = _RecordingEstimator(value=-7.0)

    energy = evaluate_vqe_energy(architecture, problem, estimator=stub)

    assert energy == -7.0
    assert len(stub.calls) == 1


def test_evaluate_vqe_energy_real_estimator_matches_default_numeric_path():
    from aicir.backends.numpy_backend import NumpyBackend
    from aicir.primitives.estimator import StatevectorEstimator
    from aicir.qas.vqe_loop.fair_vqe import evaluate_vqe_energy

    architecture, problem = _tiny_architecture_and_problem()
    backend = NumpyBackend()

    default_energy = evaluate_vqe_energy(architecture, problem, backend=backend)
    estimator_energy = evaluate_vqe_energy(
        architecture, problem, backend=backend, estimator=StatevectorEstimator(backend)
    )

    assert estimator_energy == pytest.approx(default_energy, abs=1e-6)
