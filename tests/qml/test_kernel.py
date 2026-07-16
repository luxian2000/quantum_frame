"""量子核（qml 成熟化 #5）：BatchSV 整批演化 + 实/虚 gram。

K[i,j] = |<Φ(xᵢ)|Φ(zⱼ)>|²。批量演化 N 个特征态一次（O(N)），再实/虚分离矩阵
乘算 gram（NPU 安全），替代逐对重演化（O(N²)）。
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aicir import Circuit, Parameter, cx, rx, ry
from aicir.backends.gpu_backend import GPUBackend
from aicir.core.state import State
from aicir.qml import QuantumKernel, angle_feature_map


def _feature_map(n):
    xs = [Parameter(f"x{i}") for i in range(n)]
    gates = [rx(xs[i], i) for i in range(n)] + [cx(1, [0])] + [ry(xs[i], i) for i in range(n)]
    return Circuit(*gates, n_qubits=n)


def _naive_kernel(fmap, X, Z, backend):
    """逐对单态路径参考：|<ψᵢ|ζⱼ>|²。"""
    def states(A):
        out = []
        for row in A:
            bound = fmap.bind_parameters(np.asarray(row, dtype=float))
            psi = State.zero_state(fmap.n_qubits, backend).evolve(bound.unitary(backend=backend))
            out.append(np.asarray(psi.to_numpy()).reshape(-1))
        return out
    sx, sz = states(X), states(Z)
    K = np.zeros((len(sx), len(sz)))
    for i, a in enumerate(sx):
        for j, b in enumerate(sz):
            K[i, j] = abs(np.vdot(a, b)) ** 2
    return K


def test_kernel_matrix_matches_naive():
    n = 2
    fmap = _feature_map(n)
    backend = GPUBackend()
    kernel = QuantumKernel(fmap, backend=backend)
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(5, n))
    K = np.asarray(kernel.matrix(X))
    ref = _naive_kernel(fmap, X, X, backend)
    np.testing.assert_allclose(K, ref, atol=1e-4)


def test_kernel_symmetric_and_unit_diagonal():
    fmap = _feature_map(2)
    kernel = QuantumKernel(fmap, backend=GPUBackend())
    X = np.random.default_rng(1).uniform(-1, 1, size=(6, 2))
    K = np.asarray(kernel.matrix(X))
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-5)   # 归一态自内积 =1
    np.testing.assert_allclose(K, K.T, atol=1e-5)


def test_kernel_cross_matrix_shape():
    fmap = _feature_map(2)
    kernel = QuantumKernel(fmap, backend=GPUBackend())
    X = np.random.default_rng(2).uniform(-1, 1, size=(4, 2))
    Z = np.random.default_rng(3).uniform(-1, 1, size=(7, 2))
    K = np.asarray(kernel.matrix(X, Z))
    assert K.shape == (4, 7)


def test_kernel_evolves_once_per_dataset(monkeypatch):
    # O(N) 演化：matrix(X) 只实例化 1 个 BatchSV（对称），matrix(X,Z) 实例化 2 个，
    # 而非逐对 N² 次。
    import aicir.qml.kernel as kmod

    count = {"n": 0}
    orig = kmod.BatchSV

    class Counting(orig):
        def __init__(self, *a, **k):
            count["n"] += 1
            super().__init__(*a, **k)

    monkeypatch.setattr(kmod, "BatchSV", Counting)
    fmap = _feature_map(2)
    kernel = QuantumKernel(fmap, backend=GPUBackend())
    X = np.random.default_rng(4).uniform(-1, 1, size=(8, 2))

    count["n"] = 0
    kernel.matrix(X)
    assert count["n"] == 1        # 8 样本，1 次批量演化（非 64 次逐对）

    count["n"] = 0
    kernel.matrix(X, X[:3])
    assert count["n"] == 2


def test_kernel_single_pair_call_matches_matrix():
    fmap = _feature_map(2)
    kernel = QuantumKernel(fmap, backend=GPUBackend())
    x = np.array([0.3, -0.5]); z = np.array([0.1, 0.7])
    k = float(kernel(x, z))
    K = np.asarray(kernel.matrix(x.reshape(1, -1), z.reshape(1, -1)))
    assert abs(k - K[0, 0]) < 1e-6


def test_angle_feature_map_usable():
    fmap = angle_feature_map(n_qubits=3)
    assert len(fmap.parameters) == 3
    kernel = QuantumKernel(fmap, backend=GPUBackend())
    K = np.asarray(kernel.matrix(np.random.default_rng(5).uniform(-1, 1, size=(4, 3))))
    assert K.shape == (4, 4)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-5)
