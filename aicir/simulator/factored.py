"""因子化（乘积态）模拟引擎：把纯态保存为若干互不纠缠的因子。

与稠密态矢量互补，**不替代**它：门只作用在自己所属的因子上，只有当一个门跨越
两个因子时才把它们合并（kron）。低纠缠线路因此只在小张量上演算。

v1 **只合并不拆分**——不做可分性检测。拆分需要 Schmidt/SVD，而昇腾没有复数 SVD
内核（real-embedding 变通在秩亏时对阈值极敏感），把它排除在 v1 之外可使本引擎
只依赖两样东西：``_apply_local_matrix_to_state``（稠密路径的同一个门应用入口）
与 ``backend.kron``（NPU 上已是 real/imag 分解）。

跨后端可移植性由此**构造性**成立，无需任何后端专用分支。注意不要改调
``backend.apply_statevector_local``——那是可选优化，基类默认返回 ``None``，
只有 ``NumpyBackend`` 实现，直接依赖它会让本引擎在 GPU/NPU 上失效。
"""

from __future__ import annotations

import numpy as np

from ..core.gates import _apply_local_matrix_to_state, gate_tensors
from ..core.state import State
from ..dtypes import resolve_dtype
from ..ir import ControlFlow
from .mps import _permute_basis

__all__ = ["FactoredState", "factored_statevector", "factored_expectation"]


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
        # 先用 numpy 造 |0>，再交给 backend.cast 转成后端张量并取后端 dtype。
        # 不要写成 backend.zeros((2,), dtype=resolve_dtype(backend))——那会把
        # **numpy** dtype 传给 torch 后端，GPUBackend 直接 TypeError。
        amp = np.zeros(2, dtype=resolve_dtype(backend))
        amp[0] = 1.0
        factors = [((qubit,), backend.cast(amp.copy())) for qubit in range(n_qubits)]
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

        作用轴是各 qubit 在该因子内的局部下标，因子宽度就是局部的 n_qubits。

        走 ``_apply_local_matrix_to_state``（稠密路径的同一个入口）而**不是**
        ``backend.apply_statevector_local``：后者是可选优化，基类默认返回 ``None``
        表示"请用通用回退"，只有 ``NumpyBackend`` 实现了它。直接调用会让本引擎
        在 GPU/NPU 上直接失效。通用入口内部会自行选择快路径或回退，因此跨后端
        可移植性来自复用它，而非来自某个特定后端方法。
        """

        qubits = tuple(int(q) for q in qubits)
        indices = {self.factor_index_of(q) for q in qubits}
        if len(indices) != 1:
            raise ValueError(f"qubits {qubits} 跨因子，请先调用 join_for(qubits)")

        index = indices.pop()
        factor_qubits, amplitudes = self._factors[index]
        axes = [factor_qubits.index(q) for q in qubits]
        width = len(factor_qubits)

        updated = _apply_local_matrix_to_state(
            self._backend.cast(amplitudes).reshape(1 << width, 1),
            self._backend.cast(matrix),
            axes,
            width,
            self._backend,
        )

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


def factored_statevector(circuit, backend=None) -> FactoredState:
    """按因子化表示演化 ``circuit``，返回 ``FactoredState``。

    门经 ``gate_tensors`` 降解为 ``[(matrix, axes)]``——与稠密路径、张量网络引擎
    同一个来源，故门语义不会分叉。跨因子的门先 ``join_for`` 再 ``apply_local``。
    """

    from ..backends.numpy_backend import NumpyBackend

    backend = backend if backend is not None else (circuit._backend or NumpyBackend())

    for gate in circuit.gates:
        if isinstance(gate, ControlFlow):
            raise ValueError("控制流指令无法用因子化引擎执行；请用 Measure.run 的轨迹路径")

    state = FactoredState.zero_state(circuit.n_qubits, backend)
    for gate in circuit.gates:
        for matrix, axes in gate_tensors(gate, backend):
            qubits = tuple(int(a) for a in axes)
            if len({state.factor_index_of(q) for q in qubits}) > 1:
                state = state.join_for(qubits)
            state = state.apply_local(matrix, qubits)
    return state


def _restrict_labels(labels, factor_qubits):
    """把全局 Pauli 标签限制到某因子的比特上，返回该因子的局部标签串。"""

    return "".join(labels[q] for q in factor_qubits)


def factored_expectation(state: FactoredState, observable) -> float:
    """在因子化表示上求 Pauli 可观测量期望，**不构造完整态矢量**。

    Pauli 串在互不纠缠的因子间是乘性的：``⟨P⟩ = ∏ᵢ ⟨Pᵢ⟩``，其中 ``Pᵢ`` 是 ``P``
    限制到因子 i 的部分。逐因子求值再连乘，代价只与**最宽的因子**有关而非 ``2^n``
    ——这是本引擎内存优势真正兑现的地方。

    各因子上的局部期望仍复用 ``Hamiltonian.expectation`` 的稀疏路径，故 Y 的符号
    约定等易错细节只有一份实现。
    """

    from ..core.operators import Hamiltonian

    terms = observable.terms if isinstance(observable, Hamiltonian) else [observable]

    total = 0.0
    for term in terms:
        labels = term.qubit_labels
        product = 1.0
        for factor_qubits, amplitudes in state.factors:
            local_labels = _restrict_labels(labels, factor_qubits)
            if set(local_labels) == {"I"}:
                continue                      # 恒等因子贡献 1
            width = len(factor_qubits)
            local_state = State(
                state.backend.cast(amplitudes).reshape(1 << width, 1),
                width,
                state.backend,
            )
            product *= Hamiltonian(
                n_qubits=width, terms=[(local_labels, 1.0)]
            ).expectation(local_state, state.backend)
        total += float(np.real(term.coefficient)) * product
    return float(total)
