"""
aicir/core/state.py

量子态的统一面向对象封装：纯态（向量形态 (2^n,1)）与混合态（密度矩阵形态 (2^n,2^n)）。

设计原则：
- 向量形态内部数据 shape (2^n, 1)，密度矩阵形态 shape (2^n, 2^n)，均为复数类型
- 所有数值运算委托给注入的 Backend 实例
- 返回新对象而非原地修改（不可变风格），便于函数式组合
- `.array` / `.matrix` / `.ket` 提供统一的用户访问接口
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    from ..channel.backends.base import Backend
    from .density import DensityMatrix


def _normalize_bit_order(bit_order: Optional[str], default: str = "msb") -> str:
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


def _format_ket(amplitudes, n_qubits: int, bit_order: str, atol: float = 1e-6) -> str:
    """Σ aᵢ|i> 形式。"""
    terms = []
    for idx, amplitude in enumerate(amplitudes):
        if abs(amplitude) < atol:
            continue
        coeff = _format_amplitude(amplitude, atol)
        label = _basis_label(idx, n_qubits, bit_order)
        terms.append(f"{coeff}|{label}>")
    if not terms:
        return "0"
    expr = terms[0]
    for term in terms[1:]:
        expr += term if term.startswith("-") else f"+{term}"
    return expr


def _format_density_amplitude(value: complex, tol: float) -> str:
    """密度矩阵元素的格式化：实数用十进制，不使用 sqrt 启发式。"""
    real = float(np.real(value))
    imag = float(np.imag(value))
    if abs(imag) < tol:
        if abs(real - 1.0) < tol:
            return "1"
        if abs(real + 1.0) < tol:
            return "-1"
        return f"{real:.6g}"
    if abs(real) < tol:
        if abs(imag - 1.0) < tol:
            return "1j"
        if abs(imag + 1.0) < tol:
            return "-1j"
        return f"{imag:.6g}j"
    sign = "+" if imag >= 0 else "-"
    return f"({real:.6g}{sign}{abs(imag):.6g}j)"


def _format_density_ket(matrix, n_qubits: int, bit_order: str, atol: float = 1e-6) -> str:
    """Σ ρ_ij|i><j| 形式（遍历所有非零矩阵元）。"""
    terms = []
    dim = matrix.shape[0]
    for i in range(dim):
        for j in range(dim):
            val = matrix[i, j]
            if abs(val) < atol:
                continue
            coeff = _format_density_amplitude(val, atol)
            li = _basis_label(i, n_qubits, bit_order)
            lj = _basis_label(j, n_qubits, bit_order)
            terms.append(f"{coeff}|{li}><{lj}|")
    if not terms:
        return "0"
    expr = terms[0]
    for term in terms[1:]:
        expr += term if term.startswith("-") else f"+{term}"
    return expr


def _infer_n_qubits(dim: int) -> int:
    if dim <= 0:
        raise ValueError(f"维数 {dim} 必须为正整数")
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"维数 {dim} 不是 2 的幂")
    return n


def _default_backend():
    from ..channel.backends import NumpyBackend
    return NumpyBackend()


class State:
    """
    量子态的面向对象封装，同时支持纯态（向量形态）与混合态（密度矩阵形态）。

    形状约定：
    - 向量形态：内部数据 shape (2^n, 1)，复数类型
    - 密度矩阵形态：内部数据 shape (2^n, 2^n)，复数类型

    用户接口属性：`.array`（纯态振幅向量）、`.matrix`（密度矩阵）、`.ket`（Dirac 记号字符串）。

    示例::

        from aicir.channel.backends import GPUBackend
        from aicir.core import State

        bk = GPUBackend()
        sv = State.zero_state(2, bk)                # |00⟩（向量形态）
        U  = ...                                    # 某个 4×4 酉矩阵
        sv2 = sv.evolve(U)                          # |ψ'⟩ = U|ψ⟩
        print(sv2.ket)                              # Dirac 记号
        print(sv2.probabilities())
        print(sv2.measure(shots=1024))
    """

    def __init__(self, data, n_qubits: int, backend: "Backend" = None, bit_order: str = "msb"):
        """
        参数:
            data:     后端张量。1D/(2^n,1) 视为纯态向量；(2^n,2^n) 视为密度矩阵
            n_qubits: 量子比特数
            backend:  计算后端实例；None 时使用默认 NumpyBackend
        """
        self._backend = backend if backend is not None else _default_backend()
        self._n_qubits = n_qubits
        self._bit_order = _normalize_bit_order(bit_order)
        self._array_cache = None
        self._matrix_cache = None

        casted = self._backend.cast(data)
        if casted is None:
            raise TypeError("backend.cast 返回了 None，无法构造 State")
        shape = tuple(int(axis) for axis in casted.shape)
        dim = 1 << n_qubits

        if len(shape) == 2 and shape[0] == dim and shape[1] == dim and dim > 1:
            self._kind = "matrix"
            self._data = casted
            return

        if len(shape) == 1:
            casted = casted.reshape(-1, 1)
            shape = tuple(int(axis) for axis in casted.shape)
        elif len(shape) == 2 and shape[1] != 1:
            raise ValueError(f"无法识别的数据形状 {shape}（n_qubits={n_qubits}）")
        self._kind = "vector"
        if shape[0] != dim:
            raise ValueError(
                f"数据长度 {shape[0]} 与 n_qubits={n_qubits} 不符（期望 {dim}）"
            )
        self._data = casted

    @classmethod
    def zero_state(cls, n_qubits: int, backend: "Backend" = None, bit_order: str = "msb") -> "State":
        """创建 |0⊗n⟩ 计算基基态（向量形态）。"""
        backend = backend if backend is not None else _default_backend()
        data = backend.zeros_state(n_qubits)
        return cls(data, n_qubits, backend, bit_order=bit_order)

    @classmethod
    def from_array(cls, array, n_qubits: int = None, backend: "Backend" = None, bit_order: str = "msb") -> "State":
        """从 numpy array / list 构造态向量（自动归一化）。n_qubits 省略时由长度推断。"""
        backend = backend if backend is not None else _default_backend()
        np_array = np.asarray(array, dtype=np.complex64).reshape(-1)
        if n_qubits is None:
            n_qubits = _infer_n_qubits(np_array.shape[0])
        norm = float(np.linalg.norm(np_array))
        if norm <= 0:
            raise ValueError("输入数组范数必须大于 0")
        data = backend.cast(np_array / norm)
        return cls(data, n_qubits, backend, bit_order=bit_order)

    @classmethod
    def from_matrix(cls, matrix, n_qubits: int = None, backend: "Backend" = None) -> "State":
        """从密度矩阵 (2^n,2^n) 构造混合/纯态（matrix 形态）。n_qubits 省略时由形状推断。"""
        backend = backend if backend is not None else _default_backend()
        np_m = np.asarray(matrix, dtype=np.complex64)
        if np_m.ndim != 2 or np_m.shape[0] != np_m.shape[1]:
            raise ValueError("from_matrix 需要方阵 (2^n, 2^n)")
        if n_qubits is None:
            n_qubits = _infer_n_qubits(np_m.shape[0])
        return cls(backend.cast(np_m), n_qubits, backend)

    @property
    def is_density(self) -> bool:
        """当前是否以密度矩阵形态存储。"""
        return self._kind == "matrix"

    @property
    def data(self):
        """后端原生张量。向量形态 shape (2^n, 1)；密度矩阵形态 shape (2^n, 2^n)。"""
        return self._data

    @property
    def array(self):
        """纯态返回 numpy (2^n,) 振幅向量；混合态返回 None。"""
        if self._array_cache is not None:
            return self._array_cache
        if self._kind == "vector":
            self._array_cache = self._backend.to_numpy(self._data).reshape(-1)
            return self._array_cache
        rho = self.matrix
        if not self.is_pure():
            return None
        evals, evecs = np.linalg.eigh(rho)
        idx = int(np.argmax(evals.real))
        vec = evecs[:, idx]
        nz = int(np.argmax(np.abs(vec) > 1e-9))
        phase = np.exp(-1j * np.angle(vec[nz])) if abs(vec[nz]) > 0 else 1.0
        self._array_cache = (vec * phase).astype(np.complex64)
        return self._array_cache

    @property
    def matrix(self) -> np.ndarray:
        """恒返回 numpy (2^n, 2^n) 密度矩阵。"""
        if self._matrix_cache is None:
            self._matrix_cache = self._backend.to_numpy(self._matrix_data())
        return self._matrix_cache

    @property
    def ket(self) -> str:
        """可打印 Dirac 记号：纯态 Σaᵢ|i>；混合态 Σρ_ij|i><j|。"""
        return self.format()

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

    def _matrix_data(self):
        """返回密度矩阵形态的后端原生张量（向量形态时计算 |ψ><ψ|）。"""
        if self._kind == "matrix":
            return self._data
        bk = self._backend
        return bk.matmul(self._data, bk.dagger(self._data))

    def evolve(self, unitary) -> "State":
        """酉演化：向量形态 U|ψ⟩；矩阵形态 UρU†。返回新 State。"""
        bk = self._backend
        if self._kind == "vector":
            new_data = bk.apply_unitary(self._data, unitary)
        else:
            new_data = bk.matmul(bk.matmul(unitary, self._data), bk.dagger(unitary))
        return State(new_data, self._n_qubits, bk, bit_order=self._bit_order)

    def probabilities(self):
        """计算基测量概率。向量形态返回后端张量；矩阵形态返回 numpy。"""
        if self._kind == "vector":
            return self._backend.measure_probs(self._data)
        diag = self._backend.to_numpy(self._backend.real(self._data.diagonal()))
        diag = np.clip(np.asarray(diag), 0, None)
        total = diag.sum()
        return diag / total if total > 0 else diag

    def measure(self, shots: int = 1024, bit_order: Optional[str] = None) -> Dict[str, int]:
        """模拟 shots 次测量，返回各基态计数（仅非零项）。"""
        order = _normalize_bit_order(bit_order, default=self._bit_order)
        if self._kind == "vector":
            counts_arr = self._backend.sample(self.probabilities(), shots)
            counts_np = self._backend.to_numpy(counts_arr).astype(int).reshape(-1)
        else:
            probs = self.probabilities()
            indices = np.random.choice(len(probs), size=shots, p=probs)
            counts_np = np.bincount(indices, minlength=len(probs))
        return {
            f"|{_basis_label(idx, self._n_qubits, order)}>": int(c)
            for idx, c in enumerate(counts_np)
            if c > 0
        }

    def expectation(self, operator) -> float:
        """期望值：向量形态 ⟨ψ|O|ψ⟩；矩阵形态 Tr(ρO)。"""
        if self._kind == "vector":
            value = self._backend.expectation_sv(self._data, operator)
        else:
            value = self._backend.expectation_dm(self._data, operator)
        if value is None:
            raise TypeError("backend expectation 返回了 None")
        return float(value)

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
        """格式化为 Dirac 记号。纯态用 ket 叠加；混合态用 ρ_ij|i><j| 展开。"""
        order = _normalize_bit_order(bit_order, default=self._bit_order)
        arr = self.array
        if arr is None:
            return _format_density_ket(self.matrix, self._n_qubits, order, atol)
        return _format_ket(arr, self._n_qubits, order, atol)

    def reorder_endianness(self, bit_order: str) -> "State":
        """
        在 LSB / MSB 约定之间转换底层基态顺序，并返回新状态。
        仅向量形态支持；矩阵形态会抛出 TypeError。
        """
        if self._kind == "matrix":
            raise TypeError("矩阵形态 State 不支持端序重排；请先取纯态向量")
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

    def msb(self) -> "State":
        """转换为大端序（MSB）表示。"""
        return self.reorder_endianness("msb")

    def lsb(self) -> "State":
        """转换为小端序（LSB）表示。"""
        return self.reorder_endianness("lsb")

    def norm(self) -> float:
        """
        计算态向量的范数（归一化时应约等于 1.0）。
        """
        probs_np = self._backend.to_numpy(self.probabilities()).real
        return float(probs_np.sum()) ** 0.5

    def partial_trace(self, keep) -> "State":
        """对子系统求偏迹，返回 matrix 形态 State（形状 2^k×2^k，k=len(keep)）。"""
        bk = self._backend
        rho_red = bk.partial_trace(self._matrix_data(), keep, self._n_qubits)
        return State(rho_red, len(keep), bk)

    def purity(self) -> float:
        """纯度 Tr(ρ²)。向量形态恒为 1.0。"""
        if self._kind == "vector":
            return 1.0
        bk = self._backend
        val = bk.trace(bk.matmul(self._data, self._data))
        return float(np.real(bk.to_numpy(val)))

    def eigenvalues(self) -> np.ndarray:
        """密度矩阵特征值（升序），numpy array。"""
        return np.linalg.eigvalsh(self._backend.to_numpy(self._matrix_data()))

    def von_neumann_entropy(self) -> float:
        """冯·诺依曼熵 S(ρ) = -Tr(ρ ln ρ)。"""
        eigs = self.eigenvalues()
        eigs = eigs[eigs > 1e-15]
        return float(-np.sum(eigs * np.log(eigs)))

    def is_pure(self, tol: float = 1e-5) -> bool:
        """是否纯态（purity ≈ 1）。"""
        return abs(self.purity() - 1.0) < tol

    @classmethod
    def maximally_mixed(cls, n_qubits: int, backend: "Backend" = None) -> "State":
        """最大混合态 ρ = I / 2^n（matrix 形态）。"""
        backend = backend if backend is not None else _default_backend()
        dim = 1 << n_qubits
        rho_np = np.eye(dim, dtype=np.complex64) / dim
        return cls(backend.cast(rho_np), n_qubits, backend)

    def to_density_matrix(self) -> "State":
        """转为密度矩阵形态 State：ρ = |ψ⟩⟨ψ|（向量形态）或原样（矩阵形态）。"""
        return State(self._matrix_data(), self._n_qubits, self._backend)

    def to_numpy(self) -> np.ndarray:
        """向量形态导出 (2^n,)；矩阵形态导出 (2^n, 2^n)。"""
        arr = self._backend.to_numpy(self._data)
        return arr.reshape(-1) if self._kind == "vector" else arr

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
