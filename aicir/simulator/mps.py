"""MPS（矩阵乘积态）近似模拟引擎：bond 截断的纯态演化（Spec 2）。

每比特一个 rank-3 张量 (Dl, 2, Dr)，维护正交中心做 TEBD 式演化。
数值走 Backend 张量原语（tensordot/transpose/reshape/conj/svd），
NumPy/GPU 后端通用；GPU 上对参数门可微。仅纯态、无噪声、1/2 比特门。
"""

from __future__ import annotations

import numpy as np

from ..core.state import State


class MPSState:
    """bond 截断的矩阵乘积态。"""

    def __init__(self, tensors, n_qubits, backend, *, max_bond_dim=None, cutoff=1e-10):
        self.tensors = list(tensors)
        self.n_qubits = int(n_qubits)
        self.backend = backend
        self.max_bond_dim = None if max_bond_dim is None else int(max_bond_dim)
        self.cutoff = float(cutoff)
        self.truncation_error = 0.0
        self.oc = 0
        self.logical_at = list(range(self.n_qubits))
        self.site_of = list(range(self.n_qubits))

    @classmethod
    def zero_state(cls, n_qubits, backend, *, max_bond_dim=None, cutoff=1e-10):
        n = int(n_qubits)
        if n <= 0:
            raise ValueError("n_qubits 必须为正整数")
        tensors = []
        for _ in range(n):
            arr = np.zeros((1, 2, 1), dtype=np.complex64)
            arr[0, 0, 0] = 1.0
            tensors.append(backend.reshape(backend.cast(arr), (1, 2, 1)))
        return cls(tensors, n, backend, max_bond_dim=max_bond_dim, cutoff=cutoff)

    def to_statevector(self):
        """收缩为稠密态矢量并按逻辑比特序还原，返回 State（仅供小 n 验证）。"""
        bk = self.backend
        cur = bk.reshape(self.tensors[0], (2, self.tensors[0].shape[2]))
        for s in range(1, self.n_qubits):
            t = self.tensors[s]
            dl = t.shape[0]
            cur = bk.tensordot(cur, t, ([cur.ndim - 1], [0]))  # (..., 2, Dr)
            new_rows = 1
            shape = np.asarray(bk.to_numpy(cur)).shape
            for d in shape[:-1]:
                new_rows *= int(d)
            cur = bk.reshape(cur, (new_rows, shape[-1]))
        phys = bk.reshape(cur, (2,) * self.n_qubits)  # 物理 site 序
        perm = [self.site_of[q] for q in range(self.n_qubits)]  # 逻辑序
        logical = bk.transpose(phys, perm)
        data = bk.reshape(logical, (1 << self.n_qubits, 1))
        return State(data, self.n_qubits, bk)
