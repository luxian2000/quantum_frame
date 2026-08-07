"""因子化（乘积态）引擎：容器、稠密化、门应用与合并。

本文件以**稠密路径为唯一 oracle**：因子化只是表示方式不同，物理态必须逐值一致。
"""

import numpy as np
import pytest

import aicir as A
from aicir import Circuit, Measure, NumpyBackend
from aicir.simulator.factored import FactoredState


class TestZeroState:
    def test_zero_state_is_fully_factored(self):
        st = FactoredState.zero_state(5, NumpyBackend())
        assert st.n_qubits == 5
        assert st.n_factors == 5           # |0>^5 完全可分
        assert st.max_factor_width == 1

    def test_zero_state_factors_cover_every_qubit_exactly_once(self):
        st = FactoredState.zero_state(4, NumpyBackend())
        seen = [q for qubits, _ in st.factors for q in qubits]
        assert sorted(seen) == [0, 1, 2, 3]

    def test_factor_index_of_locates_qubit(self):
        st = FactoredState.zero_state(3, NumpyBackend())
        idx = st.factor_index_of(1)
        assert 1 in st.factors[idx][0]

    def test_factor_qubits_are_sorted(self):
        st = FactoredState.zero_state(3, NumpyBackend())
        for qubits, _ in st.factors:
            assert list(qubits) == sorted(qubits)

    def test_amplitudes_follow_backend_dtype(self):
        st = FactoredState.zero_state(2, NumpyBackend(dtype=np.complex64))
        assert np.asarray(st.factors[0][1]).dtype == np.complex64

    def test_rejects_non_positive_qubit_count(self):
        with pytest.raises(ValueError):
            FactoredState.zero_state(0, NumpyBackend())


class TestMaterialisation:
    """kron 得到的比特序是各因子 qubits 的**拼接**顺序，不是 0..n-1，必须再置换。"""

    def test_zero_state_materialises_to_computational_zero(self):
        st = FactoredState.zero_state(3, NumpyBackend())
        vec = np.asarray(st.to_statevector().to_numpy()).reshape(-1)
        expected = np.zeros(8, dtype=np.complex128)
        expected[0] = 1.0
        np.testing.assert_allclose(vec, expected, atol=1e-12)

    def test_single_qubit_factors_kron_in_canonical_order(self):
        backend = NumpyBackend()
        # qubit0=|1>, qubit1=|0>, qubit2=|1>  ->  |101> = index 5
        one = np.array([0.0, 1.0], dtype=np.complex128)
        zero = np.array([1.0, 0.0], dtype=np.complex128)
        st = FactoredState([((0,), one), ((1,), zero), ((2,), one)], 3, backend)
        vec = np.asarray(st.to_statevector().to_numpy()).reshape(-1)
        assert np.argmax(np.abs(vec)) == 0b101

    def test_out_of_order_factors_still_materialise_correctly(self):
        """因子列表顺序不应影响结果——排列由 qubits 决定，不由列表位置决定。"""
        backend = NumpyBackend()
        one = np.array([0.0, 1.0], dtype=np.complex128)
        zero = np.array([1.0, 0.0], dtype=np.complex128)
        st = FactoredState([((2,), one), ((0,), one), ((1,), zero)], 3, backend)
        vec = np.asarray(st.to_statevector().to_numpy()).reshape(-1)
        assert np.argmax(np.abs(vec)) == 0b101

    def test_multi_qubit_factor_materialises_correctly(self):
        """跨 qubit 0 与 2 的双比特因子，中间夹着独立的 qubit 1——交错才暴露置换错误。"""
        backend = NumpyBackend()
        pair = np.zeros(4, dtype=np.complex128)
        pair[3] = 1.0                      # 因子 (0,2) 处于 |11>
        zero = np.array([1.0, 0.0], dtype=np.complex128)
        st = FactoredState([((0, 2), pair), ((1,), zero)], 3, backend)
        vec = np.asarray(st.to_statevector().to_numpy()).reshape(-1)
        assert np.argmax(np.abs(vec)) == 0b101

    def test_materialisation_preserves_norm(self):
        backend = NumpyBackend()
        rng = np.random.default_rng(0)
        a = rng.normal(size=2) + 1j * rng.normal(size=2)
        a /= np.linalg.norm(a)
        b = rng.normal(size=4) + 1j * rng.normal(size=4)
        b /= np.linalg.norm(b)
        st = FactoredState([((1,), a), ((0, 2), b)], 3, backend)
        vec = np.asarray(st.to_statevector().to_numpy()).reshape(-1)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-12
