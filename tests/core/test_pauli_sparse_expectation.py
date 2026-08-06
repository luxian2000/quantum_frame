"""Pauli 哈密顿量的稀疏期望值路径。

历史问题：`Hamiltonian.expectation` 与 `StatevectorEstimator` 都经
`Hamiltonian.to_matrix()` 构造稠密 `2^n × 2^n` 矩阵。实测 `n=12`、23 项的 TFIM
单次能量求值 3.77 s，其中 3.59 s 花在建矩阵，真正的期望只要 0.017 s；且稠密矩阵
在 `n=14` 就要 4.3 GB、`n=16` 要 68 GB，使**变分栈卡在 n≈13**，与模拟器无关
（态矢量演化在 n=20 毫无压力）。

正确做法：Pauli 串对基态只做"比特翻转 + 相位"，
``P|b⟩ = i^{n_Y} · (-1)^{popcount(b & z_mask)} · |b ⊕ x_mask⟩``，
故 ``⟨ψ|P|ψ⟩`` 每项 O(2^n)、无需任何矩阵。

本文件以**稠密路径为正确性 oracle**，并钉住稠密路径不可能达到的规模。
"""

import time

import numpy as np
import pytest

from aicir import Hamiltonian, NumpyBackend, PauliString
from aicir.core.state import State


def _random_state(n_qubits, seed=0):
    rng = np.random.default_rng(seed)
    dim = 1 << n_qubits
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    vec = vec / np.linalg.norm(vec)
    return State.from_array(vec.astype(np.complex128), backend=NumpyBackend(dtype=np.complex128))


def _dense_expectation(hamiltonian, state, backend):
    """oracle：显式走稠密矩阵。"""
    return float(np.real(state.expectation(hamiltonian.to_matrix(backend))))


class TestMatchesDenseOracle:
    """稀疏路径必须与稠密路径逐值一致。"""

    @pytest.mark.parametrize("pauli", ["I", "X", "Y", "Z"])
    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 5])
    def test_single_pauli_on_every_qubit(self, pauli, n_qubits):
        backend = NumpyBackend(dtype=np.complex128)
        state = _random_state(n_qubits, seed=hash((pauli, n_qubits)) % 1000)
        for qubit in range(n_qubits):
            ham = Hamiltonian(n_qubits=n_qubits, terms=[(pauli, [qubit], 1.0)])
            assert ham.expectation(state, backend) == pytest.approx(
                _dense_expectation(ham, state, backend), abs=1e-10
            ), f"{pauli} on qubit {qubit} of {n_qubits}"

    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 6])
    def test_multi_qubit_strings_including_y(self, n_qubits):
        """Y 是最容易写错的一项（Y = iXZ，相位用的是**翻转前**的比特）。"""

        backend = NumpyBackend(dtype=np.complex128)
        state = _random_state(n_qubits, seed=n_qubits + 40)
        rng = np.random.default_rng(n_qubits)
        for _ in range(8):
            labels = "".join(rng.choice(list("IXYZ"), size=n_qubits))
            ham = Hamiltonian([(labels, 1.0)])
            assert ham.expectation(state, backend) == pytest.approx(
                _dense_expectation(ham, state, backend), abs=1e-10
            ), f"labels={labels}"

    def test_multi_term_hamiltonian_with_real_coefficients(self):
        backend = NumpyBackend(dtype=np.complex128)
        n_qubits = 5
        state = _random_state(n_qubits, seed=99)
        ham = Hamiltonian(
            n_qubits=n_qubits,
            terms=[("ZZ", [q, q + 1], 1.0) for q in range(n_qubits - 1)]
            + [("X", [q], 0.5) for q in range(n_qubits)]
            + [("Y", [0], -0.25)],
        )
        assert ham.expectation(state, backend) == pytest.approx(
            _dense_expectation(ham, state, backend), abs=1e-10
        )

    def test_identity_only_term_equals_coefficient(self):
        backend = NumpyBackend(dtype=np.complex128)
        state = _random_state(3, seed=5)
        ham = Hamiltonian([("III", 2.5)])
        assert ham.expectation(state, backend) == pytest.approx(2.5, abs=1e-10)

    def test_matches_dense_on_computational_basis_state(self):
        backend = NumpyBackend(dtype=np.complex128)
        n_qubits = 4
        vec = np.zeros(1 << n_qubits, dtype=np.complex128)
        vec[0b0110] = 1.0
        state = State.from_array(vec, backend=backend)
        ham = Hamiltonian(n_qubits=n_qubits, terms=[("ZZ", [1, 2], 1.0), ("Z", [0], 1.0)])
        # |0110>: q0=0,q1=1,q2=1,q3=0 → Z1Z2 = (-1)(-1) = +1；Z0 = +1
        assert ham.expectation(state, backend) == pytest.approx(2.0, abs=1e-12)
        assert ham.expectation(state, backend) == pytest.approx(
            _dense_expectation(ham, state, backend), abs=1e-12
        )


class TestDensityMatrix:
    """密度矩阵形态 Tr(ρH) 同样要正确。"""

    def test_matches_dense_for_mixed_state(self):
        backend = NumpyBackend(dtype=np.complex128)
        n_qubits = 3
        pure = _random_state(n_qubits, seed=11)
        rho = pure.to_density_matrix() if hasattr(pure, "to_density_matrix") else None
        if rho is None:
            pytest.skip("State 未提供 to_density_matrix")
        ham = Hamiltonian(n_qubits=n_qubits, terms=[("XY", [0, 2], 0.75), ("Z", [1], -1.0)])
        assert ham.expectation(rho, backend) == pytest.approx(
            _dense_expectation(ham, rho, backend), abs=1e-10
        )


class TestScaleBeyondDenseLimit:
    """稠密路径根本到不了的规模。"""

    def test_n14_completes_quickly(self):
        """n=14 的稠密 H 需 4.3 GB；稀疏路径必须在一秒内完成。"""

        backend = NumpyBackend(dtype=np.complex128)
        n_qubits = 14
        state = _random_state(n_qubits, seed=14)
        ham = Hamiltonian(
            n_qubits=n_qubits,
            terms=[("ZZ", [q, q + 1], 1.0) for q in range(n_qubits - 1)]
            + [("X", [q], 0.5) for q in range(n_qubits)],
        )
        start = time.perf_counter()
        value = ham.expectation(state, backend)
        elapsed = time.perf_counter() - start
        assert np.isfinite(value)
        assert elapsed < 1.0, f"n=14 期望值耗时 {elapsed:.2f}s，稀疏路径应远快于此"

    def test_n16_does_not_allocate_dense_matrix(self):
        """n=16 的稠密 H 需 68 GB——能跑完本身就证明没有走稠密路径。"""

        backend = NumpyBackend(dtype=np.complex128)
        n_qubits = 16
        state = _random_state(n_qubits, seed=16)
        ham = Hamiltonian(n_qubits=n_qubits, terms=[("ZZ", [0, 15], 1.0), ("X", [7], 0.5)])
        value = ham.expectation(state, backend)
        assert np.isfinite(value)


class TestEstimatorUsesSparsePath:
    """`StatevectorEstimator` 是 VQE 的默认能量路径，必须同样受益。"""

    def test_estimator_matches_dense_and_scales(self):
        import aicir as A
        from aicir.primitives import StatevectorEstimator

        n_qubits = 14
        gates = [A.hadamard(q) for q in range(n_qubits)]
        circuit = A.Circuit(*gates, n_qubits=n_qubits)
        ham = Hamiltonian(n_qubits=n_qubits, terms=[("X", [q], 1.0) for q in range(n_qubits)])

        start = time.perf_counter()
        value = StatevectorEstimator().run(circuit, ham).value
        elapsed = time.perf_counter() - start

        # |+>^n 是每个 X 的 +1 本征态 → 期望 = 比特数
        assert value == pytest.approx(float(n_qubits), abs=1e-8)
        assert elapsed < 2.0, f"n=14 estimator 耗时 {elapsed:.2f}s"
