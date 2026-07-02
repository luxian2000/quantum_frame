"""BasicVQE 接入 Target：按设备能力自动选择 Estimator 执行路径（NEXT.md §3 / §4）。

``BasicVQE(..., target=Target(...))`` 在未显式注入 ``energy_estimator`` 时，经
``estimator_for_target`` 构造并注入对应 Estimator，使 VQE 的能量求值走 primitives
（phase-1 item 4：BasicVQE 优先调用 Estimator）。
"""

import numpy as np

from aicir import Circuit, Hamiltonian, NumpyBackend, Parameter, ry
from aicir.devices import Target
from aicir.primitives import ShotEstimator, StatevectorEstimator
from aicir.vqc import BasicVQE


def _z_hamiltonian():
    return Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])


def _single_ry_template():
    return Circuit(ry(Parameter("theta"), 0), n_qubits=1)


def test_statevector_target_injects_statevector_estimator():
    vqe = BasicVQE(
        _z_hamiltonian(),
        ansatz=_single_ry_template(),
        backend=NumpyBackend(),
        target=Target(n_qubits=1, supports_statevector=True),
    )
    assert isinstance(vqe.energy_estimator, StatevectorEstimator)


def test_target_path_energy_matches_exact():
    vqe = BasicVQE(
        _z_hamiltonian(),
        ansatz=_single_ry_template(),
        backend=NumpyBackend(),
        target=Target(n_qubits=1, supports_statevector=True),
    )
    # <Z> = cos(theta) — Estimator 路径应与精确解析一致。
    assert np.isclose(vqe.energy(np.array([0.0])), 1.0)
    assert np.isclose(vqe.energy(np.array([np.pi / 3])), np.cos(np.pi / 3), atol=1e-6)


def test_shots_target_injects_shot_estimator():
    vqe = BasicVQE(
        _z_hamiltonian(),
        ansatz=_single_ry_template(),
        backend=NumpyBackend(),
        shots=2048,
        target=Target(n_qubits=1, supports_statevector=False, supports_shots=True),
    )
    assert isinstance(vqe.energy_estimator, ShotEstimator)


def test_explicit_estimator_takes_precedence_over_target():
    explicit = ShotEstimator(NumpyBackend(), shots=128)
    vqe = BasicVQE(
        _z_hamiltonian(),
        ansatz=_single_ry_template(),
        backend=NumpyBackend(),
        energy_estimator=explicit,
        target=Target(n_qubits=1, supports_statevector=True),
    )
    assert vqe.energy_estimator is explicit


def test_no_target_keeps_exact_default():
    vqe = BasicVQE(_z_hamiltonian(), ansatz=_single_ry_template(), backend=NumpyBackend())
    assert vqe.energy_estimator == "exact"
