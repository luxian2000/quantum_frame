"""``unitary`` 自定义门必须作用在它声明的 ``qubits`` 上。

历史 bug：`apply_gate_to_state` 与 `gate_to_matrix` 的 unitary 分支都无视指令自带的
``qubits``——前者传 ``list(range(gate_qubits))``，后者用 kron 右填充单位阵，两者都把
门钉死在比特 ``0..k-1``。作用在从 0 开始的相邻比特时结果恰好正确（README §4.5 的
示例正是 ``qubits=(0, 1)``），因此长期未被发现，非相邻比特则**静默算错**。

注意 ``gate_tensors``（张量网络路径）一直是对的（走 ``_local_target_qubits``），
所以修复前三条路径互相矛盾。本文件同时钉住三者一致。
"""

import numpy as np
import pytest

from aicir import Circuit, Measure, NumpyBackend
from aicir.core.gates import apply_gate_to_state, gate_to_matrix

THETA = 0.7
PHASE = np.exp(1j * THETA)

#: 受控相位：对称矩阵，用于验证"作用在哪两个比特"。
CP = np.diag([1.0, 1.0, 1.0, PHASE]).astype(np.complex128)

#: CNOT（局部高位为控制）：**非对称**，用于验证 qubits 的顺序也被尊重。
CNOT_LOCAL = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128
)


def _unitary_gate(matrix, qubits):
    return {"type": "unitary", "qubits": list(qubits), "parameter": matrix}


def _basis_state(bits):
    n = len(bits)
    index = int("".join(str(b) for b in bits), 2)
    state = np.zeros((1 << n, 1), dtype=np.complex128)
    state[index, 0] = 1.0
    return state, index


def _run_statevector(gate, n_qubits, initial):
    circuit = Circuit(gate, n_qubits=n_qubits)
    backend = NumpyBackend(dtype=np.complex128)
    result = Measure(backend=backend).run(circuit, shots=None, initial_state=initial)
    return np.asarray(result.final_state).reshape(-1)


class TestControlledPhaseTargeting:
    """CP 只在两个目标比特同时为 1 时加相位。"""

    # |011>: qubit0=0, qubit1=1, qubit2=1 —— 能区分 (0,1) 与 (1,2)
    BITS = (0, 1, 1)

    @pytest.mark.parametrize(
        "qubits,expected_phase",
        [
            ((0, 1), 1.0),      # q0=0 → 不加相位
            ((1, 2), PHASE),    # q1=q2=1 → 加相位
            ((0, 2), 1.0),      # q0=0 → 不加相位
        ],
    )
    def test_statevector_path_honours_qubits(self, qubits, expected_phase):
        initial, index = _basis_state(self.BITS)
        out = _run_statevector(_unitary_gate(CP, qubits), 3, initial)
        assert out[index] == pytest.approx(expected_phase, abs=1e-12)

    @pytest.mark.parametrize(
        "qubits,expected_phase",
        [((0, 1), 1.0), ((1, 2), PHASE), ((0, 2), 1.0)],
    )
    def test_full_matrix_path_honours_qubits(self, qubits, expected_phase):
        _, index = _basis_state(self.BITS)
        full = np.asarray(gate_to_matrix(_unitary_gate(CP, qubits), 3))
        assert full[index, index] == pytest.approx(expected_phase, abs=1e-12)


class TestQubitOrderIsRespected:
    """非对称矩阵：``qubits`` 的**顺序**决定谁是局部高位。"""

    def test_reversed_qubits_give_different_results(self):
        # |010>: q0=0, q1=1, q2=0
        initial, _ = _basis_state((0, 1, 0))
        forward = _run_statevector(_unitary_gate(CNOT_LOCAL, (1, 2)), 3, initial)
        reversed_ = _run_statevector(_unitary_gate(CNOT_LOCAL, (2, 1)), 3, initial)
        # (1,2): control=q1=1 → 翻转 q2 → |011>
        assert forward[int("011", 2)] == pytest.approx(1.0, abs=1e-12)
        # (2,1): control=q2=0 → 不动 → |010>
        assert reversed_[int("010", 2)] == pytest.approx(1.0, abs=1e-12)


class TestAllThreePathsAgree:
    """态矢量 / 全矩阵 / 张量网络三条路径必须一致。"""

    @pytest.mark.parametrize("qubits", [(0, 1), (1, 2), (0, 2), (2, 1), (2, 0)])
    def test_paths_agree_on_random_state(self, qubits):
        rng = np.random.default_rng(int("".join(map(str, qubits))))
        vec = rng.normal(size=8) + 1j * rng.normal(size=8)
        vec = (vec / np.linalg.norm(vec)).astype(np.complex128).reshape(8, 1)

        gate = _unitary_gate(CNOT_LOCAL, qubits)
        via_state = _run_statevector(gate, 3, vec)
        via_matrix = np.asarray(gate_to_matrix(gate, 3)) @ vec.reshape(-1)
        np.testing.assert_allclose(via_state, via_matrix, atol=1e-12)

    @pytest.mark.parametrize("qubits", [(1, 2), (0, 2)])
    def test_tensor_network_path_agrees(self, qubits):
        from aicir.simulator import tn_statevector

        # TN 引擎从 |0…0⟩ 起演化，故先用 H 造出叠加再作用自定义门。
        import aicir as A

        gates = [A.hadamard(q) for q in range(3)] + [_unitary_gate(CP, qubits)]
        circuit = Circuit(*gates, n_qubits=3)

        tn = np.asarray(tn_statevector(circuit)).reshape(-1)
        sv = np.asarray(
            Measure(backend=NumpyBackend(dtype=np.complex128)).run(circuit, shots=None).final_state
        ).reshape(-1)
        np.testing.assert_allclose(tn, sv, atol=1e-10)


class TestSingleQubitUnitary:
    def test_single_qubit_unitary_targets_declared_qubit(self):
        flip = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        initial, _ = _basis_state((0, 0, 0))
        out = _run_statevector(_unitary_gate(flip, (2,)), 3, initial)
        assert out[int("001", 2)] == pytest.approx(1.0, abs=1e-12)


class TestBackwardCompatibility:
    """从 0 开始的相邻比特是历史上唯一正确的情形，必须保持不变。"""

    def test_leading_qubits_behaviour_unchanged(self):
        initial, _ = _basis_state((1, 1, 0))
        out = _run_statevector(_unitary_gate(CP, (0, 1)), 3, initial)
        assert out[int("110", 2)] == pytest.approx(PHASE, abs=1e-12)

    def test_matrix_dimension_mismatch_still_raises(self):
        with pytest.raises(ValueError):
            gate_to_matrix(_unitary_gate(CP, (0,)), 3)
