"""因子化（乘积态）模拟引擎：把纯态保存为若干互不纠缠的因子。

与稠密态矢量互补，**不替代**它：门只作用在自己所属的因子上，只有当一个门跨越
两个因子时才把它们合并（kron）。低纠缠线路因此只在小张量上演算。

v1 **只合并不拆分**——不做可分性检测。拆分需要 Schmidt/SVD，而昇腾没有复数 SVD
内核（real-embedding 变通在秩亏时对阈值极敏感），把它排除在 v1 之外可使本引擎
仅依赖 ``backend.apply_statevector_local`` 与 ``backend.kron`` 两个已 NPU-safe 的原语。
"""

from __future__ import annotations

from ..dtypes import resolve_dtype

__all__ = ["FactoredState"]


class FactoredState:
    """乘积态：``factors`` 为 ``[(sorted_qubits, amplitude_tensor), ...]``。

    不变量（每次修改后都应成立）：
      - 所有因子的 qubits 并集恰为 ``range(n_qubits)``，两两不交；
      - 每个因子的 qubits 升序排列；
      - 第 i 个因子的张量长度为 ``2 ** len(qubits_i)``，比特序为大端
        （该因子 qubits 列表中的第一个是最高位）。
    """

    def __init__(self, factors, n_qubits: int, backend):
        self._factors = [(tuple(int(q) for q in qs), amp) for qs, amp in factors]
        self._n_qubits = int(n_qubits)
        self._backend = backend

    @classmethod
    def zero_state(cls, n_qubits: int, backend) -> "FactoredState":
        """|0…0⟩：完全可分，每比特一个因子。"""
        n_qubits = int(n_qubits)
        if n_qubits <= 0:
            raise ValueError(f"n_qubits 必须为正整数，收到 {n_qubits}")
        dtype = resolve_dtype(backend)
        factors = []
        for qubit in range(n_qubits):
            amp = backend.zeros((2,), dtype=dtype)
            amp = backend.cast(_with_first_one(amp, backend))
            factors.append(((qubit,), amp))
        return cls(factors, n_qubits, backend)

    @property
    def factors(self):
        return list(self._factors)

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def backend(self):
        return self._backend

    @property
    def n_factors(self) -> int:
        return len(self._factors)

    @property
    def max_factor_width(self) -> int:
        return max((len(qs) for qs, _ in self._factors), default=0)

    def factor_index_of(self, qubit: int) -> int:
        """返回 ``qubit`` 所在因子的下标。"""
        for index, (qubits, _) in enumerate(self._factors):
            if qubit in qubits:
                return index
        raise ValueError(f"qubit {qubit} 不在任何因子中")


def _with_first_one(amp, backend):
    """把长度 2 的零张量的第 0 个分量置 1（避免依赖具体张量类型的原地赋值语义）。"""
    import numpy as np

    arr = np.asarray(backend.to_numpy(amp)).reshape(-1).copy()
    arr[0] = 1.0
    return arr
