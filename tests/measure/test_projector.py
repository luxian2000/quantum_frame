import numpy as np
from aicir.core.state import State
from aicir.backends.numpy_backend import NumpyBackend
from aicir.measure import projector


def _sv(vec):
    b = NumpyBackend()
    n = int(np.log2(len(vec)))
    return State(b.cast(np.asarray(vec, dtype=complex).reshape(-1, 1)), n, b)


def test_basis_change_x_maps_plus_to_zero():
    plus = _sv([1, 1])
    plus = State(plus.backend.cast((plus.to_numpy() / np.sqrt(2))), 1, plus.backend)
    out = projector.pauli_basis_change(plus, [0], "X", inverse=False)
    v = out.to_numpy().reshape(-1)
    assert np.allclose(v, [1, 0], atol=1e-6)


def test_basis_change_round_trip_identity():
    psi = _sv([0.5, 0.5, 0.5, 0.5])
    fwd = projector.pauli_basis_change(psi, [0, 1], "Y", inverse=False)
    back = projector.pauli_basis_change(fwd, [0, 1], "Y", inverse=True)
    assert np.allclose(back.to_numpy().reshape(-1), psi.to_numpy().reshape(-1), atol=1e-6)


def test_basis_change_z_is_noop():
    psi = _sv([0.6, 0.8])
    out = projector.pauli_basis_change(psi, [0], "Z", inverse=False)
    assert np.allclose(out.to_numpy().reshape(-1), [0.6, 0.8], atol=1e-6)


def _bell():
    b = NumpyBackend()
    v = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    return State(b.cast(v.reshape(-1, 1)), 2, b)


def test_joint_zz_on_bell_is_deterministic_plus_and_keeps_entanglement():
    psi = _bell()
    rng = np.random.default_rng(0)
    out, lam = projector.measure_joint_pauli(psi, [0, 1], "Z", rng)
    assert lam == 1  # Bell 态 Z⊗Z 恒 +1
    v = out.to_numpy().reshape(-1)
    assert np.allclose(np.abs(v), [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], atol=1e-6)


def test_joint_measure_single_qubit_not_in_pa_eigenstate():
    b = NumpyBackend()
    v = np.array([1, 1, 1, -1], dtype=complex) / 2
    psi = State(b.cast(v.reshape(-1, 1)), 2, b)
    rng = np.random.default_rng(1)
    out, lam = projector.measure_joint_pauli(psi, [0, 1], "X", rng)
    assert lam in (1, -1)
    red = out.partial_trace([0]).to_numpy().reshape(2, 2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    exp_x = np.real(np.trace(red @ X))
    assert abs(exp_x) < 1 - 1e-6


def test_joint_probs_match_born():
    b = NumpyBackend()
    v = np.array([np.sqrt(0.3), 0, 0, np.sqrt(0.7)], dtype=complex)
    psi = State(b.cast(v.reshape(-1, 1)), 2, b)
    p_plus, p_minus = projector.joint_parity_probs(psi, [0, 1], "Z")
    assert np.isclose(p_plus, 1.0, atol=1e-6)
    assert np.isclose(p_minus, 0.0, atol=1e-6)


def test_reset_product_qubit_stays_pure():
    b = NumpyBackend()
    v = np.kron([0, 1], np.array([1, 1]) / np.sqrt(2)).astype(complex)
    psi = State(b.cast(v.reshape(-1, 1)), 2, b)
    out = projector.reset_channel(psi, [0])
    assert not out.is_density
    expect = np.kron([1, 0], np.array([1, 1]) / np.sqrt(2))
    assert np.allclose(out.to_numpy().reshape(-1), expect, atol=1e-6)


def test_reset_entangled_qubit_promotes_to_density_matrix():
    b = NumpyBackend()
    v = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    psi = State(b.cast(v.reshape(-1, 1)), 2, b)
    out = projector.reset_channel(psi, [0])
    assert out.is_density
    rho = out.to_numpy().reshape(4, 4)
    expect = np.zeros((4, 4), dtype=complex)
    expect[0, 0] = 0.5
    expect[1, 1] = 0.5
    assert np.allclose(rho, expect, atol=1e-6)


def test_reset_on_density_matrix_input():
    b = NumpyBackend()
    rho = np.zeros((2, 2), dtype=complex)
    rho[1, 1] = 1.0
    st = State.from_matrix(rho, 1, b)
    out = projector.reset_channel(st, [0])
    assert np.allclose(out.to_numpy().reshape(2, 2), np.array([[1, 0], [0, 0]]), atol=1e-6)


def test_reset_phased_product_qubit_stays_pure():
    # 带相对相位的乘积态：(|0> + i|1>)/√2 ⊗ |+>，reset(0) 后仍应是纯态 |0> ⊗ |+>
    b = NumpyBackend()
    qubit = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    rest = np.array([1, 1], dtype=complex) / np.sqrt(2)
    psi = np.kron(qubit, rest)
    st = State(b.cast(psi.reshape(-1, 1)), 2, b)
    out = projector.reset_channel(st, [0])
    assert not out.is_density  # 可分离目标必须保持纯态向量
    expect = np.kron([1, 0], rest)
    assert np.allclose(out.to_numpy().reshape(-1), expect, atol=1e-6)


def test_terminal_z_full_register_collapses_to_basis():
    psi = _bell()
    rng = np.random.default_rng(3)
    out, eig = projector.terminal_z_measure(psi, [0, 1], rng)
    v = out.to_numpy().reshape(-1)
    nz = np.flatnonzero(np.abs(v) > 1e-9)
    assert len(nz) == 1
    assert eig in ([1, 1], [-1, -1])


def test_terminal_z_subset_keeps_other_qubit():
    b = NumpyBackend()
    v = np.kron([1, 0], np.array([1, 1]) / np.sqrt(2)).astype(complex)
    psi = State(b.cast(v.reshape(-1, 1)), 2, b)
    rng = np.random.default_rng(4)
    out, eig = projector.terminal_z_measure(psi, [0], rng)
    assert eig == [1]
    red = out.partial_trace([1]).to_numpy().reshape(2, 2)
    plus = np.array([1, 1]) / np.sqrt(2)
    assert np.allclose(red, np.outer(plus, plus.conj()), atol=1e-6)


def test_terminal_order_preserved():
    b = NumpyBackend()
    v = np.kron([1, 0], [0, 1]).astype(complex)  # |0>_0 ⊗ |1>_1
    psi = State(b.cast(v.reshape(-1, 1)), 2, b)
    rng = np.random.default_rng(5)
    _, eig = projector.terminal_z_measure(psi, [1, 0], rng)
    assert eig == [-1, 1]


def test_measure_joint_pauli_rotates_once_each_way(monkeypatch):
    # 修复重复基变换：measure_joint_pauli 只应做一次正向 + 一次逆向旋转
    # （此前 joint_parity_probs 内部对同一态重复做正向旋转，共 3 次）
    calls = []
    original = projector.pauli_basis_change

    def counting(state, qubits, basis, inverse):
        calls.append(bool(inverse))
        return original(state, qubits, basis, inverse)

    monkeypatch.setattr(projector, "pauli_basis_change", counting)

    plus = _sv([1 / np.sqrt(2), 1 / np.sqrt(2)])
    rng = np.random.default_rng(0)
    out, lam = projector.measure_joint_pauli(plus, [0], "X", rng)

    assert calls == [False, True]  # 一次正向、一次逆向
    assert lam == 1  # |+> 的 X 测量恒为 +1
    assert np.allclose(out.to_numpy().reshape(-1), plus.to_numpy().reshape(-1), atol=1e-6)


def test_born_probs_density_no_full_rho_transfer(monkeypatch):
    # 密度态 _born_probs 只应经 State.probabilities 下传 2^n 对角线，
    # 不应 to_numpy 整个 (2^n,2^n) 密度矩阵
    b = NumpyBackend()
    rho = np.diag([0.25, 0.25, 0.25, 0.25]).astype(complex)
    state = State(b.cast(rho), 2, b)

    sizes = []
    original = b.to_numpy

    def spying(tensor):
        arr = original(tensor)
        sizes.append(arr.size)
        return arr

    monkeypatch.setattr(b, "to_numpy", spying)

    probs = projector._born_probs(state)
    np.testing.assert_allclose(probs, [0.25] * 4, atol=1e-6)
    assert max(sizes) <= 4  # 只传 2^n 向量，未传 4^n 矩阵


def test_density_projection_avoids_outer_mask(monkeypatch):
    # 密度分支投影不应物化 (2^n,2^n) 布尔掩码（np.outer），改行列置零；
    # 数值与手写期望一致
    def _boom(*_a, **_k):
        raise AssertionError("密度投影不应构造 outer 布尔掩码")

    monkeypatch.setattr(projector.np, "outer", _boom)

    # Bell 密度矩阵投影 q0=0 → |00><00|
    bell = np.zeros((4, 1), dtype=complex)
    bell[0, 0] = bell[3, 0] = 1 / np.sqrt(2)
    rho = bell @ bell.conj().T
    b = NumpyBackend()
    state = State(b.cast(rho), 2, b)

    out = projector._project_subset_outcome(state, [0], [0])
    expected = np.zeros((4, 4), dtype=complex)
    expected[0, 0] = 1.0
    np.testing.assert_allclose(out.to_numpy(), expected, atol=1e-6)

    # 宇称投影同样不应用 outer：|Φ+> 密度上投影 X⊗X=+1 保持原态
    rotated = projector.pauli_basis_change(state, [0, 1], "X", inverse=False)
    proj = projector._project_parity_rotated(rotated, [0, 1], 1)
    back = projector.pauli_basis_change(proj, [0, 1], "X", inverse=True)
    np.testing.assert_allclose(back.to_numpy(), rho, atol=1e-6)


def test_reset_dm_matches_kraus_definition():
    # _reset_dm 与信道定义 K0 ρ K0† + K1 ρ K1† 逐元素一致（随机密度矩阵）
    rng = np.random.default_rng(3)
    n, q = 3, 1
    dim = 1 << n
    m = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = m @ m.conj().T
    rho = rho / np.trace(rho)

    k0_1q = np.array([[1, 0], [0, 0]], dtype=complex)
    k1_1q = np.array([[0, 1], [0, 0]], dtype=complex)

    def embed(op1q):
        full = np.array([[1.0]], dtype=complex)
        for qi in range(n):
            full = np.kron(full, op1q if qi == q else np.eye(2, dtype=complex))
        return full

    k0, k1 = embed(k0_1q), embed(k1_1q)
    expected = k0 @ rho @ k0.conj().T + k1 @ rho @ k1.conj().T
    np.testing.assert_allclose(projector._reset_dm(rho, n, q), expected, atol=1e-12)


def test_reset_dm_vectorized_scales():
    # 向量化后 n=11（dim=2048）应远快于旧 O(4^n) Python 双循环（旧实现数秒）
    import time
    n = 13
    dim = 1 << n
    rho = np.eye(dim, dtype=complex) / dim
    t0 = time.perf_counter()
    out = projector._reset_dm(rho, n, 5)
    elapsed = time.perf_counter() - t0
    assert np.isclose(np.trace(out).real, 1.0, atol=1e-9)
    assert elapsed < 1.0  # 旧 O(4^n) Python 双循环在本机约 3s
