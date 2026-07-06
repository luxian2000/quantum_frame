"""MPS（矩阵乘积态）近似模拟引擎：bond 截断的纯态演化（Spec 2）。

每比特一个 rank-3 张量 (Dl, 2, Dr)，维护正交中心做 TEBD 式演化。
数值走 Backend 张量原语（tensordot/transpose/reshape/conj/svd），
NumPy/GPU 后端通用；GPU 上对参数门可微。仅纯态、无噪声、1/2 比特门。
"""

from __future__ import annotations

import numpy as np

from ..core.state import State


def _keep_count(s_np, max_bond_dim, cutoff):
    """按 cutoff（相对最大奇异值）+ max_bond_dim 决定保留的奇异值个数（≥1）。"""
    if s_np.size == 0:
        return 0
    smax = float(s_np[0]) if s_np[0] > 0 else 1.0
    k = int((s_np > cutoff * smax).sum())
    k = max(k, 1)
    if max_bond_dim is not None:
        k = min(k, int(max_bond_dim))
    return k


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

    def _apply_one_site(self, m2, site):
        """单比特门就地作用于物理 site 的物理指标（不改正交性、不截断）。"""
        bk = self.backend
        t = self.tensors[site]  # (Dl, 2, Dr)
        # tensordot(m2 (o,i), t (Dl,i,Dr)) over i -> (o, Dl, Dr) -> (Dl, o, Dr)
        out = bk.tensordot(m2, t, ([1], [1]))
        self.tensors[site] = bk.transpose(out, [1, 0, 2])

    def _move_center_right(self, i):
        """把正交中心从 i 移到 i+1（SVD，不截断）。"""
        bk = self.backend
        t = self.tensors[i]  # (Dl, 2, Dr)
        dl, dr = t.shape[0], t.shape[2]
        mat = bk.reshape(t, (dl * 2, dr))
        u, s, vh = bk.svd(mat)  # u:(dl*2,K) s:(K,) vh:(K,dr)
        k = int(np.asarray(bk.to_numpy(s)).shape[0])
        self.tensors[i] = bk.reshape(u, (dl, 2, k))
        s_c = bk.cast(s)
        carry = vh * bk.reshape(s_c, (k, 1))  # (K, dr)
        nxt = self.tensors[i + 1]  # (dr, 2, Dr2)
        self.tensors[i + 1] = bk.tensordot(carry, nxt, ([1], [0]))  # (K, 2, Dr2)
        self.oc = i + 1

    def _move_center_left(self, i):
        """把正交中心从 i 移到 i-1（SVD，不截断）。"""
        bk = self.backend
        t = self.tensors[i]  # (Dl, 2, Dr)
        dl, dr = t.shape[0], t.shape[2]
        mat = bk.reshape(t, (dl, 2 * dr))
        u, s, vh = bk.svd(mat)  # u:(dl,K) s:(K,) vh:(K,2*dr)
        k = int(np.asarray(bk.to_numpy(s)).shape[0])
        self.tensors[i] = bk.reshape(vh, (k, 2, dr))
        s_c = bk.cast(s)
        carry = u * bk.reshape(s_c, (1, k))  # (dl, K)
        prev = self.tensors[i - 1]  # (Dm, 2, dl)
        self.tensors[i - 1] = bk.tensordot(prev, carry, ([2], [0]))  # (Dm, 2, K)
        self.oc = i - 1

    def _ensure_center(self, p):
        while self.oc < p:
            self._move_center_right(self.oc)
        while self.oc > p:
            self._move_center_left(self.oc)

    def _apply_two_site(self, op4, s, *, truncate):
        """作用双比特算子 op4=(out_s,out_{s+1},in_s,in_{s+1}) 于物理 site (s, s+1)。"""
        bk = self.backend
        self._ensure_center(s)  # 中心在 s -> 右侧 s+1 为右规范
        a, b = self.tensors[s], self.tensors[s + 1]
        dl, dr = a.shape[0], b.shape[2]
        theta = bk.tensordot(a, b, ([2], [0]))  # (Dl, 2, 2, Dr)
        # 作用 op4 于两个物理指标 (axis 1,2)
        applied = bk.tensordot(op4, theta, ([2, 3], [1, 2]))  # (out_s, out_{s+1}, Dl, Dr)
        applied = bk.transpose(applied, [2, 0, 1, 3])  # (Dl, out_s, out_{s+1}, Dr)
        mat = bk.reshape(applied, (dl * 2, 2 * dr))
        u, sv, vh = bk.svd(mat)
        s_np = np.asarray(bk.to_numpy(sv)).real
        if truncate:
            k = _keep_count(s_np, self.max_bond_dim, self.cutoff)
        else:
            k = int(s_np.shape[0])
        total = float((s_np ** 2).sum()) or 1.0
        discarded = float((s_np[k:] ** 2).sum()) / total
        self.truncation_error += discarded
        u_k = u[:, :k]
        vh_k = vh[:k, :]
        s_k = bk.cast(sv[:k])
        self.tensors[s] = bk.reshape(u_k, (dl, 2, k))
        vh_scaled = vh_k * bk.reshape(s_k, (k, 1))  # 把奇异值吸收进右张量
        self.tensors[s + 1] = bk.reshape(vh_scaled, (k, 2, dr))
        self.oc = s + 1
