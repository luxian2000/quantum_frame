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


def _canonical_permutation_local(local_order, width):
    """与 ``_canonical_permutation`` 同构，但作用在因子局部的 ``width`` 个比特上。

    ``local_order[k]`` 表示 kron 结果的第 k 位在排序后应处的位置。
    """

    dim = 1 << width
    src = np.zeros(dim, dtype=np.int64)
    for i in range(dim):
        source = 0
        for k, position in enumerate(local_order):
            if (i >> (width - 1 - position)) & 1:
                source |= 1 << (width - 1 - k)
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

    def apply_local(self, matrix, qubits) -> "FactoredState":
        """把 ``matrix`` 作用在**同一个因子内**的 ``qubits`` 上，返回新状态。

        作用轴是各 qubit 在该因子内的局部下标，因子宽度就是局部的 n_qubits——
        这正是 ``backend.apply_statevector_local`` 的契约，故不需要任何新原语，
        NPU 上也自动复用其已验证过的实现。
        """

        qubits = tuple(int(q) for q in qubits)
        indices = {self.factor_index_of(q) for q in qubits}
        if len(indices) != 1:
            raise ValueError(f"qubits {qubits} 跨因子，请先调用 join_for(qubits)")

        index = indices.pop()
        factor_qubits, amplitudes = self._factors[index]
        axes = [factor_qubits.index(q) for q in qubits]
        width = len(factor_qubits)

        updated = self._backend.apply_statevector_local(
            self._backend.cast(amplitudes).reshape(1 << width, 1),
            self._backend.cast(matrix),
            axes,
            width,
        )
        if updated is None:
            raise ValueError(f"后端不支持 {len(qubits)} 比特局部门")

        factors = list(self._factors)
        factors[index] = (factor_qubits, self._backend.cast(updated).reshape(-1))
        return FactoredState(factors, self._n_qubits, self._backend)

    def join_for(self, qubits) -> "FactoredState":
        """把包含 ``qubits`` 中任一比特的所有因子合并成一个因子。

        合并即 Kronecker 积；合并后 qubit 列表按升序排列，故还需按新比特序重排
        幅度——与 ``to_statevector`` 是同一个置换问题，只是范围限于被合并的比特。
        """

        qubits = tuple(int(q) for q in qubits)
        target = sorted({self.factor_index_of(q) for q in qubits})
        if len(target) == 1:
            return FactoredState(self._factors, self._n_qubits, self._backend)

        merged_qubits = []
        tensors = []
        for index in target:
            factor_qubits, amplitudes = self._factors[index]
            merged_qubits.extend(factor_qubits)
            tensors.append(self._backend.cast(amplitudes).reshape(-1))

        combined = _kron_all(tensors, self._backend)
        width = len(merged_qubits)
        sorted_qubits = tuple(sorted(merged_qubits))
        local_order = [sorted_qubits.index(q) for q in merged_qubits]
        src = _canonical_permutation_local(local_order, width)
        if not np.array_equal(src, np.arange(1 << width)):
            combined = _permute_basis(self._backend.cast(combined).reshape(-1), src, self._backend)

        chosen = set(target)
        remaining = [f for i, f in enumerate(self._factors) if i not in chosen]
        remaining.append((sorted_qubits, self._backend.cast(combined).reshape(-1)))
        remaining.sort(key=lambda item: item[0][0])
        return FactoredState(remaining, self._n_qubits, self._backend)

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
