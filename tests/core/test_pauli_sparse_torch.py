"""torch/NPU 上的稀疏 Pauli 期望值（real/imag 分解）。

稀疏公式 ``P|b⟩ = i^{n_Y}·(-1)^{popcount(b & z_mask)}·|b ⊕ x_mask⟩`` 在 torch 上
不能照搬 numpy 写法：昇腾缺三类内核，而稀疏路径恰好会命中全部三类——

- ``aclnnIndex``  (DT_COMPLEX64)：复数张量的高级索引/`index_select`；
- ``aclnnAdd`` / ``aclnnMul`` (DT_COMPLEX64)：复数加/乘；
- 复数归约：对复数张量求 sum/dot。

拆成实部/虚部后三类全部消失：索引变成两次实数 `index_select`，
``conj(ψ)·φ = (ac+bd) + i(ad-bc)`` 变成四次实数点积。与
``aicir/simulator/mps.py:_permute_basis`` 同法。

本文件在 CPU 上用 ``TorchDispatchMode`` **复现昇腾的内核缺口**，从而无需真机即可
证明该路径 NPU-safe：任何复数 add/mul/sub/index/sum/matmul 都会直接抛错。
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from aicir import GPUBackend, Hamiltonian, NumpyBackend  # noqa: E402
from aicir.core.state import State  # noqa: E402


class _BanUnsupportedComplexOps(TorchDispatchMode):
    """在 CPU 上复现昇腾缺失的 complex64 内核。

    比 ``tests/backends/test_batched_statevector.py`` 的版本更严：除加/乘/减外，
    还拦截复数的**索引**与**归约**——稀疏路径正是靠这两类操作，不拦就证明不了
    NPU-safe。``real``/``imag``/``view_as_real`` 是取实数视图的官方手段
    （见 CLAUDE.md），不在拦截之列。
    """

    BANNED = ("add", "mul", "sub", "index", "sum", "dot", "matmul", "mm", "gather")
    ALLOWED = ("view_as_real", "real", "imag", "conj_physical", "_conj", "resolve_conj")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        if not any(tag in name for tag in self.ALLOWED):
            if any(tag in name for tag in self.BANNED):
                for value in list(args) + list(kwargs.values()):
                    if isinstance(value, torch.Tensor) and torch.is_complex(value):
                        raise RuntimeError(f"NPU 不支持的复数算子: {name}")
        return func(*args, **kwargs)


def _numpy_reference(ham, vec, n_qubits):
    """numpy 稀疏路径作为数值 oracle（已由 test_pauli_sparse_expectation 验证）。"""

    backend = NumpyBackend(dtype=np.complex128)
    state = State.from_array(vec.astype(np.complex128), backend=backend)
    return ham.expectation(state, backend)


def _torch_state(vec, backend):
    return State.from_array(vec, backend=backend)


def _random_vec(n_qubits, seed=0):
    rng = np.random.default_rng(seed)
    dim = 1 << n_qubits
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return vec / np.linalg.norm(vec)


class TestMatchesNumpyReference:
    """torch 路径必须与已验证的 numpy 稀疏路径一致。"""

    @pytest.mark.parametrize("pauli", ["I", "X", "Y", "Z"])
    @pytest.mark.parametrize("n_qubits", [1, 3, 5])
    def test_single_pauli_on_every_qubit(self, pauli, n_qubits):
        backend = GPUBackend(device="cpu", dtype=torch.complex128)
        vec = _random_vec(n_qubits, seed=n_qubits)
        for qubit in range(n_qubits):
            ham = Hamiltonian(n_qubits=n_qubits, terms=[(pauli, [qubit], 1.0)])
            got = ham.expectation(_torch_state(vec, backend), backend)
            want = _numpy_reference(ham, vec, n_qubits)
            assert got == pytest.approx(want, abs=1e-9), f"{pauli}@{qubit}/{n_qubits}"

    @pytest.mark.parametrize("n_qubits", [2, 4, 6])
    def test_random_strings_including_y(self, n_qubits):
        backend = GPUBackend(device="cpu", dtype=torch.complex128)
        vec = _random_vec(n_qubits, seed=n_qubits + 7)
        rng = np.random.default_rng(n_qubits)
        for _ in range(6):
            labels = "".join(rng.choice(list("IXYZ"), size=n_qubits))
            ham = Hamiltonian([(labels, 1.0)])
            got = ham.expectation(_torch_state(vec, backend), backend)
            want = _numpy_reference(ham, vec, n_qubits)
            assert got == pytest.approx(want, abs=1e-9), f"labels={labels}"

    def test_multi_term_hamiltonian(self):
        backend = GPUBackend(device="cpu", dtype=torch.complex128)
        n_qubits = 5
        vec = _random_vec(n_qubits, seed=123)
        ham = Hamiltonian(
            n_qubits=n_qubits,
            terms=[("ZZ", [q, q + 1], 1.0) for q in range(n_qubits - 1)]
            + [("X", [q], 0.5) for q in range(n_qubits)]
            + [("Y", [2], -0.75)],
        )
        got = ham.expectation(_torch_state(vec, backend), backend)
        assert got == pytest.approx(_numpy_reference(ham, vec, n_qubits), abs=1e-9)


class TestNPUSafety:
    """核心断言：整条路径不触碰任何昇腾缺失的复数内核。"""

    def test_expectation_uses_no_unsupported_complex_op(self):
        backend = GPUBackend(device="cpu", dtype=torch.complex64)
        n_qubits = 6
        vec = _random_vec(n_qubits, seed=3)
        state = _torch_state(vec, backend)
        ham = Hamiltonian(
            n_qubits=n_qubits,
            terms=[("ZZ", [0, 1], 1.0), ("XY", [2, 3], 0.5), ("Y", [5], -0.25)],
        )
        with _BanUnsupportedComplexOps():
            value = ham.expectation(state, backend)
        assert np.isfinite(value)

    def test_ban_mode_actually_catches_the_dense_path(self):
        """反证：稠密路径在同一拦截下必须失败，否则这道测试形同虚设。"""

        backend = GPUBackend(device="cpu", dtype=torch.complex64)
        n_qubits = 4
        vec = _random_vec(n_qubits, seed=4)
        state = _torch_state(vec, backend)
        ham = Hamiltonian(n_qubits=n_qubits, terms=[("ZZ", [0, 1], 1.0)])
        with pytest.raises(RuntimeError, match="NPU 不支持的复数算子"):
            with _BanUnsupportedComplexOps():
                state.expectation(ham.to_matrix(backend))


class TestScale:
    def test_n14_needs_no_dense_matrix(self):
        """n=14 的 complex64 稠密 H 要 2.1 GB；跑得完即证明没走稠密。"""

        backend = GPUBackend(device="cpu", dtype=torch.complex64)
        n_qubits = 14
        vec = _random_vec(n_qubits, seed=14)
        ham = Hamiltonian(
            n_qubits=n_qubits,
            terms=[("ZZ", [q, q + 1], 1.0) for q in range(n_qubits - 1)],
        )
        value = ham.expectation(_torch_state(vec, backend), backend)
        assert np.isfinite(value)
