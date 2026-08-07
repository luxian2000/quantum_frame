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
