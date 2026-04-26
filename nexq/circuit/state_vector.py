"""
nexq/circuit/state_vector.py

纯量子态 |ψ⟩ 的面向对象封装。

设计原则：
- 内部数据始终保持列向量形式，shape (2^n, 1)
- 所有数值运算委托给注入的 Backend 实例
- 返回新对象而非原地修改（不可变风格），便于函数式组合
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    from ..channel.backends.base import Backend
    from .density_matrix import DensityMatrix


def _normalize_bit_order(bit_order: Optional[str], default: str = "lsb") -> str:
    order = default if bit_order is None else bit_order.lower()
    if order not in {"lsb", "msb"}:
        raise ValueError("bit_order 只能是 'lsb' 或 'msb'")
    return order


def _reverse_bits(index: int, n_qubits: int) -> int:
    bits = f"{index:0{n_qubits}b}"
    return int(bits[::-1], 2)


def _basis_label(index: int, n_qubits: int, bit_order: str) -> str:
    bits = f"{index:0{n_qubits}b}"
    return bits[::-1] if bit_order == "lsb" else bits


def _format_real_amplitude(value: float, tol: float) -> str:
    if abs(value - 1.0) < tol:
        return "1"
    if abs(value + 1.0) < tol:
        return "-1"

    magnitude = abs(value)
    for denom in range(2, 33):
        target = 1.0 / np.sqrt(denom)
        if abs(magnitude - target) < tol:
            sign = "-" if value < 0 else ""
            return f"{sign}1/\\sqrt{{{denom}}}"

    return f"{value:.6g}"


def _format_amplitude(value: complex, tol: float) -> str:
    real = float(np.real(value))
    imag = float(np.imag(value))

    if abs(imag) < tol:
        return _format_real_amplitude(real, tol)
    if abs(real) < tol:
        if abs(imag - 1.0) < tol:
            return "1j"
        if abs(imag + 1.0) < tol:
            return "-1j"
        return f"{imag:.6g}j"

    sign = "+" if imag >= 0 else "-"
    return f"({real:.6g}{sign}{abs(imag):.6g}j)"


class State:
    """
    纯量子态 |ψ⟩ 的面向对象封装。

    形状约定：内部数据 shape (2^n, 1)，复数类型。

    示例::

        from nexq.channel.backends import TorchBackend
        from nexq.circuit import State

        bk = TorchBackend()
        sv = State.zero_state(2, bk)                # |00⟩
        U  = ...                                    # 某个 4×4 酉矩阵
        sv2 = sv.evolve(U)                          # |ψ'⟩ = U|ψ⟩
        print(sv2.probabilities())
        print(sv2.measure(shots=1024))
    """

    def __init__(self, data, n_qubits: int, backend: "Backend", bit_order: str = "lsb"):
        """
        参数:
            data:     后端张量，shape (2^n, 1) 或 (2^n,)
            n_qubits: 量子比特数
            backend:  计算后端实例
        """
        self._backend = backend
        self._n_qubits = n_qubits
        self._bit_order = _normalize_bit_order(bit_order)

        np_data = backend.to_numpy(data)
        if np_data.ndim == 1:
            np_data = np_data.reshape(-1, 1)
        self._data = backend.cast(np_data)

        expected = 1 << n_qubits
        if self._data.shape[0] != expected:
            raise ValueError(
                f"数据长度 {self._data.shape[0]} 与 n_qubits={n_qubits} 不符（期望 {expected}）"
            )

    @classmethod
    def zero_state(
        cls,
        n_qubits: int,
        backend: "Backend",
        bit_order: str = "lsb",
    ) -> "State":
        """创建 |0⊗n⟩ 计算基基态。"""
        data = backend.zeros_state(n_qubits)
        return cls(data, n_qubits, backend, bit_order=bit_order)

    @classmethod
    def from_array(
        cls,
        array,
        n_qubits: int,
        backend: "Backend",
        bit_order: str = "lsb",
    ) -> "State":
        """
        从 numpy array / list 构造态向量。

        参数:
            array:    长度为 2^n 的一维或 (2^n,1) 的复数序列
            n_qubits: 量子比特数
            backend:  计算后端
        """
        data = backend.cast(np.asarray(array, dtype=np.complex64))
        return cls(data, n_qubits, backend, bit_order=bit_order)

    @property
    def data(self):
        """后端原生张量，shape (2^n, 1)。"""
        return self._data

    @property
    def n_qubits(self) -> int:
        """量子比特数。"""
        return self._n_qubits

    @property
    def dim(self) -> int:
        """希尔伯特空间维数 2^n。"""
        return 1 << self._n_qubits

    @property
    def backend(self) -> "Backend":
        """当前使用的计算后端。"""
        return self._backend

    @property
    def bit_order(self) -> str:
        """当前状态采用的基态标签端序。"""
        return self._bit_order

    def evolve(self, unitary) -> "State":
        """
        将酉矩阵作用于当前态，返回新的 State（不修改原对象）。

        参数:
            unitary: (2^n, 2^n) 酉矩阵（后端原生张量）
        返回:
            State — 演化后的新态
        """
        new_data = self._backend.apply_unitary(self._data, unitary)
        return State(new_data, self._n_qubits, self._backend, bit_order=self._bit_order)

    def probabilities(self):
        """
        计算计算基上的测量概率分布。

        返回:
            shape (2^n,) 实数概率向量（后端原生张量），总和为 1
        """
        return self._backend.measure_probs(self._data)

    def measure(self, shots: int = 1024, bit_order: Optional[str] = None) -> Dict[str, int]:
        """
        模拟 shots 次投影测量，返回各基态出现次数。

        参数:
            shots: 测量次数（正整数）
        返回:
            {"|00⟩": count, "|11⟩": count, ...} 字典（仅含非零项）
        """
        order = _normalize_bit_order(bit_order, default=self._bit_order)
        probs = self.probabilities()
        counts_arr = self._backend.sample(probs, shots)
        counts_np = self._backend.to_numpy(counts_arr).astype(int).reshape(-1)
        return {
            f"|{_basis_label(idx, self._n_qubits, order)}>": int(c)
            for idx, c in enumerate(counts_np)
            if c > 0
        }

    def expectation(self, operator) -> float:
        """
        计算期望值 ⟨ψ|O|ψ⟩，返回实数。

        参数:
            operator: (2^n, 2^n) Hermitian 算符（后端原生张量）
        """
        return self._backend.expectation_sv(self._data, operator)

    def inner_product(self, other: "State"):
        """
        计算内积 ⟨self|other⟩。

        返回:
            复数标量张量
        """
        if self._n_qubits != other._n_qubits:
            raise ValueError("内积要求两态具有相同 n_qubits")
        return self._backend.inner_product(self._data, other._data)

    def format(self, bit_order: Optional[str] = None, atol: float = 1e-6) -> str:
        """
        以 ket 叠加形式格式化量子态，例如：1/\\sqrt{2}|00>+1/\\sqrt{2}|11>。
        """
        order = _normalize_bit_order(bit_order, default=self._bit_order)
        amplitudes = self.to_numpy()
        terms = []
        for idx, amplitude in enumerate(amplitudes):
            if abs(amplitude) < atol:
                continue
            coeff = _format_amplitude(amplitude, atol)
            label = _basis_label(idx, self._n_qubits, order)
            terms.append(f"{coeff}|{label}>")

        if not terms:
            return "0"

        expression = terms[0]
        for term in terms[1:]:
            expression += term if term.startswith("-") else f"+{term}"
        return expression

    def reorder_endianness(self, bit_order: str) -> "State":
        """
        在 LSB / MSB 约定之间转换底层基态顺序，并返回新状态。
        """
        target_order = _normalize_bit_order(bit_order)
        if target_order == self._bit_order or self._n_qubits <= 1:
            return State(self._data, self._n_qubits, self._backend, bit_order=target_order)

        amplitudes = self.to_numpy()
        reordered = np.zeros_like(amplitudes)
        for idx, amplitude in enumerate(amplitudes):
            reordered[_reverse_bits(idx, self._n_qubits)] = amplitude

        return State.from_array(
            reordered,
            n_qubits=self._n_qubits,
            backend=self._backend,
            bit_order=target_order,
        )

    def to_big_endian(self) -> "State":
        """转换为大端序（MSB）表示。"""
        return self.reorder_endianness("msb")

    def to_little_endian(self) -> "State":
        """转换为小端序（LSB）表示。"""
        return self.reorder_endianness("lsb")

    def norm(self) -> float:
        """
        计算态向量的范数（归一化时应约等于 1.0）。
        """
        probs_np = self._backend.to_numpy(self.probabilities()).real
        return float(probs_np.sum()) ** 0.5

    def to_density_matrix(self) -> "DensityMatrix":
        """
        将纯态转换为密度矩阵 ρ = |ψ⟩⟨ψ|。
        """
        from .density_matrix import DensityMatrix

        bk = self._backend
        rho = bk.matmul(self._data, bk.dagger(self._data))
        return DensityMatrix(rho, self._n_qubits, bk)

    def to_numpy(self) -> np.ndarray:
        """
        导出为 numpy 一维复数数组，shape (2^n,)。
        """
        return self._backend.to_numpy(self._data).reshape(-1)

    def __len__(self) -> int:
        return self.dim

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        return (
            f"State(n_qubits={self._n_qubits}, "
            f"backend={self._backend.name}, bit_order={self._bit_order})"
        )


StateVector = State