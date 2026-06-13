"""测量投影 / 重置 / 末端读出的后端无关纯函数（numpy 主机计算）。

约定：bit_order="msb"，qubit q 对应 flat index 第 (n-1-q) 位、reshape [2]*n 后第 q 轴。
"""

from __future__ import annotations

from typing import List, Sequence

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
