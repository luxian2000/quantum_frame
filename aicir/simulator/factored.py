"""因子化（乘积态）模拟引擎：把纯态保存为若干互不纠缠的因子。

与稠密态矢量互补，**不替代**它：门只作用在自己所属的因子上，只有当一个门跨越
两个因子时才把它们合并（kron）。低纠缠线路因此只在小张量上演算。

v1 **只合并不拆分**——不做可分性检测。拆分需要 Schmidt/SVD，而昇腾没有复数 SVD
内核（real-embedding 变通在秩亏时对阈值极敏感），把它排除在 v1 之外可使本引擎
仅依赖 ``backend.apply_statevector_local`` 与 ``backend.kron`` 两个已 NPU-safe 的原语。
"""

from __future__ import annotations

import numpy as np

from ..core.state import State
from ..dtypes import resolve_dtype
from .mps import _permute_basis

__all__ = ["FactoredState"]


def _kron_all(tensors, backend):
    """按给定顺序做 Kronecker 积。

    走 ``backend.kron`` 而非 ``*``：NPU 上该方法已是 real/imag 分解（4 次实数
    kron），直接用复数乘会命中昇腾缺失的 ``aclnnMul``。
    """

    result = tensors[0]
    for tensor in tensors[1:]:
        result = backend.kron(result, tensor)
    return result


def _canonical_permutation(qubit_order, n_qubits):
    """构造把 ``qubit_order`` 位序还原成 0..n-1 的基态置换。

    ``kron`` 后的第 k 位对应 ``qubit_order[k]``（大端）。返回 ``src``，
    使 ``out[i] = flat[src[i]]``。

    索引运算全部用 Python int / numpy 完成：昇腾没有 ``bitwise_right_shift``
    内核，用 torch 位运算会静默回落 CPU（见 CLAUDE.md 的缺口清单）。
    """

    dim = 1 << n_qubits
    position = {q: k for k, q in enumerate(qubit_order)}
    src = np.zeros(dim, dtype=np.int64)
    for i in range(dim):
        source = 0
        for j in range(n_qubits):
            if (i >> (n_qubits - 1 - j)) & 1:      # 目标态第 j 位（即 qubit j）
                source |= 1 << (n_qubits - 1 - position[j])
        src[i] = source
    return src


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

    def to_statevector(self) -> State:
        """合并所有因子并还原成规范比特序的稠密 ``State``。

        ``kron`` 得到的比特序是各因子 qubits 的**拼接**顺序，不是 0..n-1，
        故必须再做一次基态置换。置换走 ``mps._permute_basis``——torch 分支用
        实/虚部 ``index_select``，因为昇腾不支持复数张量的高级索引。
        """

        ordered = sorted(self._factors, key=lambda item: item[0][0])
        qubit_order = [q for qubits, _ in ordered for q in qubits]
        amplitudes = _kron_all(
            [self._backend.cast(amp).reshape(-1) for _, amp in ordered], self._backend
        )
        flat = self._backend.cast(amplitudes).reshape(-1)

        src = _canonical_permutation(qubit_order, self._n_qubits)
        if not np.array_equal(src, np.arange(1 << self._n_qubits)):
            flat = _permute_basis(flat, src, self._backend)

        return State(
            self._backend.cast(flat).reshape(1 << self._n_qubits, 1),
            self._n_qubits,
            self._backend,
        )

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
