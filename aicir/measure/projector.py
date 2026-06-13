"""测量投影 / 重置 / 末端读出的后端无关纯函数（numpy 主机计算）。

约定：bit_order="msb"，qubit q 对应 flat index 第 (n-1-q) 位、reshape [2]*n 后第 q 轴。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..core.state import State

_H = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
_S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
_SDG = np.array([[1, 0], [0, -1j]], dtype=np.complex128)


def _apply_1q_sv(psi_col: np.ndarray, n: int, q: int, U: np.ndarray) -> np.ndarray:
    """对纯态向量施加单比特门 U，作用在第 q 个量子比特上。"""
    t = psi_col.reshape([2] * n)
    t = np.tensordot(U, t, axes=([1], [q]))
    t = np.moveaxis(t, 0, q)
    return t.reshape(-1, 1)


def _apply_1q_dm(rho: np.ndarray, n: int, q: int, U: np.ndarray) -> np.ndarray:
    """对密度矩阵施加单比特门 U，作用在第 q 个量子比特上：U ρ U†。"""
    dim = 1 << n
    t = rho.reshape([2] * (2 * n))
    t = np.tensordot(U, t, axes=([1], [q]))
    t = np.moveaxis(t, 0, q)
    t = np.tensordot(U.conj(), t, axes=([1], [n + q]))
    t = np.moveaxis(t, 0, n + q)
    return t.reshape(dim, dim)


def _basis_change_seq(basis: str, inverse: bool) -> List[np.ndarray]:
    """返回把单比特从 basis 旋到 Z 所需、按施加顺序排列的单比特门。

    X: 前向/逆向均为 H（自逆）。
    Y: 前向 Sdg 然后 H；逆向 H 然后 S。
    Z: 无操作（计算基）。
    """
    basis = basis.upper()
    if basis == "Z":
        return []
    if basis == "X":
        return [_H]
    if basis == "Y":
        return [_SDG, _H] if not inverse else [_H, _S]
    raise ValueError(f"未知 basis {basis!r}")


def pauli_basis_change(state: State, qubits: Sequence[int], basis: str, inverse: bool) -> State:
    """对指定量子比特施加 Pauli 基变换，将计算基旋转到目标基（或逆操作）。

    参数:
        state:   输入量子态（纯态或密度矩阵）
        qubits:  需要进行基变换的量子比特列表
        basis:   目标 Pauli 基，"X" / "Y" / "Z"
        inverse: False 表示旋到目标基；True 表示逆变换（旋回计算基）

    返回:
        变换后的新 State 对象
    """
    backend = state.backend
    n = state.n_qubits
    seq = _basis_change_seq(basis, inverse)
    if not seq:
        return state
    if state.is_density:
        rho = backend.to_numpy(state.data).reshape(1 << n, 1 << n).astype(np.complex128)
        for q in qubits:
            for U in seq:
                rho = _apply_1q_dm(rho, n, int(q), U)
        return State(backend.cast(rho), n, backend)
    psi = backend.to_numpy(state.data).reshape(-1, 1).astype(np.complex128)
    for q in qubits:
        for U in seq:
            psi = _apply_1q_sv(psi, n, int(q), U)
    return State(backend.cast(psi), n, backend, bit_order=state.bit_order)


def _parity_mask(n: int, qubits: Sequence[int]) -> int:
    """构造比特掩码：选中 qubits 列表对应的 flat-index 比特位（msb 约定）。"""
    m = 0
    for q in qubits:
        m |= 1 << (n - 1 - int(q))
    return m


def _parities(dim: int, mask: int) -> np.ndarray:
    """返回每个 flat index 在 mask 选中比特上的宇称（0=偶,1=奇）。"""
    idx = np.arange(dim, dtype=np.int64) & mask
    p = idx.copy()
    shift = 32
    while shift:
        p ^= p >> shift
        shift >>= 1
    return (p & 1).astype(np.int64)


def joint_parity_probs(state: State, qubits: Sequence[int], basis: str) -> Tuple[float, float]:
    """计算联合 Pauli 串 P 取本征值 +1 / -1 的概率（Born 规则）。

    做法：把指定比特旋到 Z 基后，按选中比特的宇称把概率分桶——
    偶宇称对应 P=+1，奇宇称对应 P=-1。
    返回 (p_plus, p_minus)。
    """
    n = state.n_qubits
    rotated = pauli_basis_change(state, qubits, basis, inverse=False)
    backend = state.backend
    par = _parities(1 << n, _parity_mask(n, qubits))
    if rotated.is_density:
        rho = backend.to_numpy(rotated.data).reshape(1 << n, 1 << n)
        diag = np.real(np.diag(rho))
        p_plus = float(diag[par == 0].sum())
    else:
        psi = backend.to_numpy(rotated.data).reshape(-1)
        probs = np.abs(psi) ** 2
        p_plus = float(probs[par == 0].sum())
    p_plus = min(max(p_plus, 0.0), 1.0)
    return p_plus, 1.0 - p_plus


def _project_parity_rotated(rotated: State, qubits: Sequence[int], lam: int) -> State:
    """在已旋到 Z 的态上，投影到联合宇称 lam(±1) 子空间并归一化（保持子空间内相干）。"""
    backend = rotated.backend
    n = rotated.n_qubits
    par = _parities(1 << n, _parity_mask(n, qubits))
    keep = (par == (0 if lam == 1 else 1))
    if rotated.is_density:
        rho = backend.to_numpy(rotated.data).reshape(1 << n, 1 << n).copy()
        mask2d = np.outer(keep, keep)
        rho = np.where(mask2d, rho, 0.0)
        tr = np.real(np.trace(rho))
        if tr > 0:
            rho = rho / tr
        return State(backend.cast(rho), n, backend)
    psi = backend.to_numpy(rotated.data).reshape(-1, 1).copy()
    psi[~keep, 0] = 0.0
    norm = np.linalg.norm(psi)
    if norm > 0:
        psi = psi / norm
    return State(backend.cast(psi), n, backend, bit_order=rotated.bit_order)


def measure_joint_pauli(state: State, qubits: Sequence[int], basis: str, rng) -> Tuple[State, int]:
    """非破坏性联合 Pauli 投影测量：返回 (坍缩后完整态, 本征值 lam∈{+1,-1})。

    实现的是真正的联合本征空间投影 Π_λ=(I+λP)/2，而非逐比特测量的乘积：
    单次两结果投影只坍缩 ±1 宇称子空间，保持子空间内部相干。
    """
    rotated = pauli_basis_change(state, qubits, basis, inverse=False)
    p_plus, _ = joint_parity_probs(state, qubits, basis)
    lam = 1 if rng.random() < p_plus else -1
    projected = _project_parity_rotated(rotated, qubits, lam)
    restored = pauli_basis_change(projected, qubits, basis, inverse=True)
    return restored, lam


def _replace_bit(index: int, n: int, q: int, bit: int) -> int:
    """将 flat index 在第 q 个量子比特（msb 约定）上的比特置为 bit，返回新 index。"""
    mask = 1 << (n - 1 - q)
    return (index | mask) if bit else (index & ~mask)


def _reset_dm(rho: np.ndarray, n: int, q: int) -> np.ndarray:
    """对密度矩阵施加重置信道 R_q(ρ)=K0 ρ K0† + K1 ρ K1†（K0=|0><0|, K1=|0><1|）。

    结果中目标比特恒处 |0>：仅在目标比特行/列均为 0 的子块上累加
    out[r,c] = rho[r,c] + rho[r1,c1]（r1/c1 为把 r/c 的目标比特置 1 后的 index）。
    """
    dim = 1 << n
    out = np.zeros_like(rho)
    mask = 1 << (n - 1 - q)
    rows = [r for r in range(dim) if not (r & mask)]
    for r in rows:
        r1 = _replace_bit(r, n, q, 1)
        for c in rows:
            c1 = _replace_bit(c, n, q, 1)
            out[r, c] = rho[r, c] + rho[r1, c1]
    return out


def _reset_one(state: State, q: int) -> State:
    """对单个量子比特施加重置信道：

    - 密度矩阵输入：直接对 ρ 施加信道；
    - 纯态且目标比特与其余比特可分离（product）：结果仍为纯态 |0>_q ⊗ (rest)；
    - 纯态且目标比特纠缠：结果为混合态，升级为密度矩阵。
    """
    backend = state.backend
    n = state.n_qubits
    if state.is_density:
        rho = backend.to_numpy(state.data).reshape(1 << n, 1 << n).astype(np.complex128)
        return State(backend.cast(_reset_dm(rho, n, int(q))), n, backend)

    psi = backend.to_numpy(state.data).reshape([2] * n).astype(np.complex128)
    sl0 = [slice(None)] * n; sl0[q] = 0
    sl1 = [slice(None)] * n; sl1[q] = 1
    a = psi[tuple(sl0)].reshape(-1)
    b = psi[tuple(sl1)].reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)

    # 判断目标比特是否与其余比特可分离：q=0 切片 a 与 q=1 切片 b 是否平行
    parallel = False
    if na < 1e-12 or nb < 1e-12:
        parallel = True
    else:
        # 相对阈值：残差是 b 相对 a 的正交分量，可分离时仅为后端精度噪声
        # （complex64 约 1e-7·‖b‖），纠缠时为 O(‖b‖)，故按 ‖b‖ 归一化判定
        c = np.vdot(a, b) / (na * na)
        parallel = np.linalg.norm(b - c * a) < 1e-6 * nb

    if parallel:
        # 可分离：其余比特的纯态即非零切片归一化，目标比特重置为 |0>
        v = a if na >= nb else b
        v = v / np.linalg.norm(v)
        out = np.zeros([2] * n, dtype=np.complex128)
        out[tuple(sl0)] = v.reshape(out[tuple(sl0)].shape)
        return State(backend.cast(out.reshape(-1, 1)), n, backend, bit_order=state.bit_order)

    # 纠缠：构造 ρ=|ψ><ψ| 后施加信道，升级为密度矩阵
    flat = psi.reshape(-1, 1)
    rho = (flat @ flat.conj().T)
    rho = _reset_dm(rho, n, int(q))
    return State(backend.cast(rho), n, backend)


def reset_channel(state: State, qubits: Sequence[int]) -> State:
    """对一组量子比特依次施加重置信道 R_S(ρ)=|0><0|_S ⊗ Tr_S(ρ)。

    无需事先测量；逐比特处理，纠缠目标会把纯态升级为密度矩阵。
    """
    for q in qubits:
        state = _reset_one(state, int(q))
    return state
