"""IQP (Instantaneous Quantum Polynomial) feature-map encoder.

实现 Havlicek et al., "Supervised learning with quantum-enhanced feature
spaces" (Nature 567, 209 (2019), arXiv:1804.11326) 提出的 IQP 特征映射：

    |Phi(x)> = U_Phi(x) H^n U_Phi(x) H^n |0>^n
    U_Phi(x) = exp(i * sum_S phi_S(x) * prod_{i in S} Z_i),  |S| <= 2

默认系数取论文选择：phi_{i}(x) = x_i，phi_{i,j}(x) = (pi - x_i)(pi - x_j)。
对角相位门用 rz / rzz 精确实现（exp(i*phi*Z) == rz(-2*phi)，无全局相位差）。
"""

from __future__ import annotations

import numpy as np

from .abstract import BaseEncoder
from .angle import _default_backend, _emit_circuit
from ..core.circuit import Circuit, hadamard, rz, rzz
from ..core.state import State


def _paper_map(values):
    """论文默认数据映射：单比特取 x_i，双比特取 (pi - x_i)(pi - x_j)。"""
    if len(values) == 1:
        return float(values[0])
    return float(np.prod([np.pi - v for v in values]))


class IQPEncoder(BaseEncoder):
    """IQP encoding: x -> (U_Phi(x) H^n)^reps |0>^n with diagonal U_Phi(x)."""

    def __init__(self, n_qubits=None, reps=2, entanglement="full", data_map=None):
        self.n_qubits = n_qubits
        self.reps = int(reps)
        if self.reps < 1:
            raise ValueError("reps must be >= 1")
        self.entanglement = entanglement
        self.data_map = _paper_map if data_map is None else data_map

    def _pairs(self, n):
        if isinstance(self.entanglement, str):
            if self.entanglement == "full":
                return [(i, j) for i in range(n) for j in range(i + 1, n)]
            if self.entanglement == "linear":
                return [(i, i + 1) for i in range(n - 1)]
            raise ValueError("entanglement must be 'full', 'linear' or pair list")
        pairs = [(int(i), int(j)) for i, j in self.entanglement]
        for i, j in pairs:
            if i == j or not (0 <= i < n and 0 <= j < n):
                raise ValueError(f"invalid entanglement pair ({i}, {j}) for {n} qubits")
        return pairs

    def circuit(self, data):
        """构建特征映射线路（不做态演化），返回 Circuit。"""
        arr = np.asarray(data, dtype=np.float64).ravel()
        data_len = len(arr)

        if self.n_qubits is not None:
            n = self.n_qubits
            if data_len > n:
                raise ValueError(f"data length {data_len} > n_qubits={n}")
            if data_len < n:
                arr = np.concatenate([arr, np.zeros(n - data_len)])
        else:
            n = data_len
        if n < 1:
            raise ValueError("data must contain at least one value")

        pairs = self._pairs(n)
        gates = []
        for _ in range(self.reps):
            gates.extend(hadamard(q) for q in range(n))
            # exp(i*phi*Z) == rz(-2*phi)；exp(i*phi*ZZ) == rzz(-2*phi)
            gates.extend(rz(-2.0 * self.data_map([arr[q]]), q) for q in range(n))
            gates.extend(
                rzz(-2.0 * self.data_map([arr[i], arr[j]]), i, j) for i, j in pairs
            )
        return Circuit(*gates, n_qubits=n)

    def encode(self, data, *, cir="dict", backend=None):
        bk = _default_backend(backend)
        circuit = self.circuit(data)

        zero = State.zero_state(circuit.n_qubits, bk)
        state = zero.evolve(circuit.unitary(backend=bk))

        return _emit_circuit(circuit, cir), state

    def decode(self, quantum_state):
        """IQP 特征映射不可逆，无法从量子态恢复经典数据。"""
        raise NotImplementedError("IQP feature map is not invertible; decode unsupported")

    def kernel(self, x, z, *, backend=None):
        """核函数 K(x, z) = |<Phi(x)|Phi(z)>|^2。"""
        bk = _default_backend(backend)
        _, sx = self.encode(x, backend=bk)
        _, sz = self.encode(z, backend=bk)
        overlap = np.vdot(sx.to_numpy().ravel(), sz.to_numpy().ravel())
        return float(np.abs(overlap) ** 2)

    def kernel_matrix(self, xs, zs=None, *, backend=None):
        """批量核矩阵：K[i, j] = kernel(xs[i], zs[j])；zs 缺省时为对称 Gram 矩阵。"""
        bk = _default_backend(backend)
        xs = [np.asarray(x, dtype=np.float64).ravel() for x in xs]
        sx = [self.encode(x, backend=bk)[1].to_numpy().ravel() for x in xs]
        if zs is None:
            sz = sx
        else:
            zs = [np.asarray(z, dtype=np.float64).ravel() for z in zs]
            sz = [self.encode(z, backend=bk)[1].to_numpy().ravel() for z in zs]
        mat = np.empty((len(sx), len(sz)), dtype=np.float64)
        for i, a in enumerate(sx):
            for j, b in enumerate(sz):
                mat[i, j] = np.abs(np.vdot(a, b)) ** 2
        return mat
