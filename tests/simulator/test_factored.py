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


_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128
)


class TestIntraFactorGate:
    def test_single_qubit_gate_stays_within_its_factor(self):
        backend = NumpyBackend()
        st = FactoredState.zero_state(3, backend)
        out = st.apply_local(_H, (1,))
        assert out.n_factors == 3          # 未发生合并
        assert out.max_factor_width == 1
        vec = np.asarray(out.to_statevector().to_numpy()).reshape(-1)
        nonzero = np.flatnonzero(np.abs(vec) > 1e-9)
        assert sorted(nonzero.tolist()) == [0b000, 0b010]

    def test_two_qubit_gate_inside_one_factor_does_not_merge(self):
        backend = NumpyBackend()
        pair = np.zeros(4, dtype=np.complex128)
        pair[0] = 1.0
        zero = np.array([1.0, 0.0], dtype=np.complex128)
        st = FactoredState([((0, 1), pair), ((2,), zero)], 3, backend)
        assert st.apply_local(_CNOT, (0, 1)).n_factors == 2

    def test_apply_local_does_not_mutate_input(self):
        backend = NumpyBackend()
        st = FactoredState.zero_state(2, backend)
        before = np.asarray(st.to_statevector().to_numpy()).copy()
        st.apply_local(_X, (0,))
        after = np.asarray(st.to_statevector().to_numpy())
        np.testing.assert_array_equal(before, after)

    def test_apply_local_rejects_qubits_spanning_factors(self):
        backend = NumpyBackend()
        st = FactoredState.zero_state(3, backend)
        with pytest.raises(ValueError, match="跨因子"):
            st.apply_local(_CNOT, (0, 1))


class TestJoining:
    def test_join_merges_two_single_qubit_factors(self):
        backend = NumpyBackend()
        joined = FactoredState.zero_state(3, backend).join_for((0, 2))
        assert joined.n_factors == 2
        assert sorted(len(qs) for qs, _ in joined.factors) == [1, 2]

    def test_joined_factor_qubits_are_sorted(self):
        backend = NumpyBackend()
        joined = FactoredState.zero_state(3, backend).join_for((2, 0))
        merged = [qs for qs, _ in joined.factors if len(qs) == 2][0]
        assert merged == (0, 2)

    def test_join_preserves_the_physical_state(self):
        """合并只是换表示，物理态必须逐值不变。"""
        backend = NumpyBackend()
        rng = np.random.default_rng(3)
        amps = []
        for _ in range(3):
            a = rng.normal(size=2) + 1j * rng.normal(size=2)
            amps.append(a / np.linalg.norm(a))
        st = FactoredState([((0,), amps[0]), ((1,), amps[1]), ((2,), amps[2])], 3, backend)
        before = np.asarray(st.to_statevector().to_numpy()).reshape(-1)
        after = np.asarray(st.join_for((0, 2)).to_statevector().to_numpy()).reshape(-1)
        np.testing.assert_allclose(before, after, atol=1e-12)

    def test_join_is_idempotent_when_already_together(self):
        backend = NumpyBackend()
        pair = np.zeros(4, dtype=np.complex128)
        pair[0] = 1.0
        st = FactoredState([((0, 1), pair)], 2, backend)
        assert st.join_for((0, 1)).n_factors == 1


def _dense(circuit, backend):
    return np.asarray(
        Measure(backend=backend).run(circuit, shots=None).final_state
    ).reshape(-1)


class TestDriverMatchesDense:
    """稠密路径是唯一 oracle：因子化只是表示方式不同，物理态必须一致。"""

    @pytest.mark.parametrize("n", [2, 3, 5, 7])
    def test_ghz_matches_dense(self, n):
        from aicir.simulator.factored import factored_statevector

        backend = NumpyBackend()
        gates = [A.hadamard(0)] + [A.cnot(q + 1, [q]) for q in range(n - 1)]
        circuit = Circuit(*gates, n_qubits=n)
        got = np.asarray(
            factored_statevector(circuit, backend).to_statevector().to_numpy()
        ).reshape(-1)
        np.testing.assert_allclose(got, _dense(circuit, backend), atol=1e-10)

    def test_product_circuit_never_joins(self):
        """全单比特门的线路应保持完全因子化——这就是本引擎的收益来源。"""
        from aicir.simulator.factored import factored_statevector

        backend = NumpyBackend()
        gates = [A.rx(0.3, q) for q in range(6)] + [A.ry(0.7, q) for q in range(6)]
        st = factored_statevector(Circuit(*gates, n_qubits=6), backend)
        assert st.n_factors == 6
        assert st.max_factor_width == 1

    def test_disjoint_entanglers_produce_pairwise_factors(self):
        from aicir.simulator.factored import factored_statevector

        backend = NumpyBackend()
        gates = [A.hadamard(0), A.cnot(1, [0]), A.hadamard(2), A.cnot(3, [2])]
        st = factored_statevector(Circuit(*gates, n_qubits=4), backend)
        assert st.n_factors == 2
        assert st.max_factor_width == 2

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_random_circuit_matches_dense(self, seed):
        from aicir.simulator.factored import factored_statevector

        backend = NumpyBackend()
        rng = np.random.default_rng(seed)
        n = 5
        gates = []
        for _ in range(12):
            if rng.random() < 0.5:
                gates.append(A.ry(float(rng.uniform(0, 6.28)), int(rng.integers(n))))
            else:
                a = int(rng.integers(n - 1))
                gates.append(A.cnot(a + 1, [a]))
        circuit = Circuit(*gates, n_qubits=n)
        got = np.asarray(
            factored_statevector(circuit, backend).to_statevector().to_numpy()
        ).reshape(-1)
        np.testing.assert_allclose(got, _dense(circuit, backend), atol=1e-10)

    def test_non_adjacent_entangler_matches_dense(self):
        """非相邻 CNOT 会合并出交错的因子——比特序最容易错的地方。"""
        from aicir.simulator.factored import factored_statevector

        backend = NumpyBackend()
        gates = [A.hadamard(0), A.ry(0.4, 1), A.cnot(3, [0]), A.ry(0.2, 2)]
        circuit = Circuit(*gates, n_qubits=4)
        got = np.asarray(
            factored_statevector(circuit, backend).to_statevector().to_numpy()
        ).reshape(-1)
        np.testing.assert_allclose(got, _dense(circuit, backend), atol=1e-10)

    def test_rejects_control_flow(self):
        from aicir.core.circuit import if_
        from aicir.core.classical import ClassicalRegister
        from aicir.simulator.factored import factored_statevector

        backend = NumpyBackend()
        creg = ClassicalRegister(1, "c")
        body = Circuit(A.pauli_x(0), n_qubits=1)
        circuit = Circuit(if_(creg[0] == 1, body), n_qubits=1)
        with pytest.raises(ValueError, match="控制流"):
            factored_statevector(circuit, backend)
