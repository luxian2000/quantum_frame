"""
aicir/operators.py

量子算符层：PauliOp、PauliString、Hamiltonian。

提供从物理描述到矩阵表示的转换，与具体后端解耦。

示例::

    from aicir.backends import GPUBackend
    from aicir.core.operators import Hamiltonian

    bk = GPUBackend()

    # H = -0.5 * Z₀Z₁  +  0.3 * X₀X₁
    H = Hamiltonian([
        ("ZZ", -0.5),
        ("XX", 0.3),
    ])
    mat = H.to_matrix(bk)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Dict, List, Union

import numpy as np

if TYPE_CHECKING:
    from ..backends.base import Backend
    from .state import State

# ── 单比特泡利矩阵（NumPy 常量，用于构造复杂算符）──────────────────────
_I = np.array([[1, 0], [0, 1]], dtype=np.complex64)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)

PAULI_MAP: Dict[str, np.ndarray] = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


# ──────────────────────────────────────────────────────────────────────────────
# PauliOp
# ──────────────────────────────────────────────────────────────────────────────

class PauliOp:
    """
    作用在单个量子比特上的泡利算符。

    示例::

        Z0 = PauliOp('Z', qubit=0)
        mat = Z0.to_matrix(n_qubits=2, backend=bk)  # 4×4 矩阵
    """

    def __init__(self, pauli: str, qubit: int = 0):
        """
        参数:
            pauli: 泡利标签 'I'/'X'/'Y'/'Z'（不区分大小写）
            qubit: 作用的量子比特索引（从 0 开始）
        """
        label = pauli.upper()
        if label not in PAULI_MAP:
            raise ValueError(f"未知泡利算符 '{pauli}'，只支持 I/X/Y/Z")
        self.pauli = label
        self.qubit = qubit

    def to_matrix(self, n_qubits: int, backend: "Backend"):
        """
        在 n_qubits 个量子比特的希尔伯特空间中构造完整矩阵。

        其他比特位填充单位矩阵 I。

        参数:
            n_qubits: 总量子比特数
            backend:  计算后端
        返回:
            shape (2^n, 2^n) 后端原生张量
        """
        if self.qubit >= n_qubits:
            raise IndexError(
                f"qubit={self.qubit} 超出 n_qubits={n_qubits} 的范围"
            )
        matrices = [
            backend.cast(PAULI_MAP[self.pauli if i == self.qubit else "I"])
            for i in range(n_qubits)
        ]
        return backend.tensor_product(*matrices)

    def __repr__(self) -> str:
        return f"PauliOp({self.pauli}, qubit={self.qubit})"


# ──────────────────────────────────────────────────────────────────────────────
# PauliString
# ──────────────────────────────────────────────────────────────────────────────

class PauliString:
    """
    多比特泡利串：coefficient × ∏_i σ_i。

    推荐用字符串描述完整或局部泡利串；也支持用 ``qubits``
    将局部泡利串放到指定比特上；还兼容
    {pauli_label: [qubit_indices]} 字典描述非单位项，其余默认为 I。

    示例::

        ps = PauliString("ZX", coefficient=0.5)
        ps_sparse = PauliString("ZZ", coefficient=-1.0, n_qubits=4, qubits=[0, 3])
    """

    def __init__(
        self,
        paulistring: str | Mapping[str, Sequence[int]] | None = None,
        coefficient: complex = 1.0,
        n_qubits: int | None = None,
        *,
        terms: str | Mapping[str, Sequence[int]] | None = None,
        qubits: Sequence[int] | None = None,
    ):
        """
        参数:
            paulistring: 完整或局部泡利串，例如 "ZIX"；或 {pauli_label: [qubit_indices]} 字典
            coefficient: 系数（复数，默认 1.0）
            n_qubits:    总量子比特数。若为 None，则从 paulistring 中自动推导
            qubits:       可选；指定字符串中每个泡利算符作用的比特下标
        """
        if paulistring is None:
            paulistring = terms
        elif terms is not None:
            raise ValueError("请只传入 paulistring 或 terms 之一")

        if paulistring is None:
            raise ValueError("paulistring 不能为空")

        if isinstance(paulistring, str):
            labels = [label.upper() for label in paulistring.strip()]
            if not labels:
                raise ValueError("Pauli 字符串不能为空")
            invalid = sorted({label for label in labels if label not in PAULI_MAP})
            if invalid:
                raise ValueError(f"未知泡利算符 '{invalid[0]}'，只支持 I/X/Y/Z")
            qubit_indices = (
                [int(qubit) for qubit in qubits]
                if qubits is not None
                else list(range(len(labels)))
            )
            if len(labels) != len(qubit_indices):
                raise ValueError("Pauli 字符串长度必须与 qubits 长度一致")
            if len(set(qubit_indices)) != len(qubit_indices):
                raise ValueError("qubits 不能包含重复下标")
            for qubit in qubit_indices:
                if qubit < 0:
                    raise IndexError(f"量子比特索引 {qubit} 不能为负数")
            if n_qubits is None:
                n_qubits = max(qubit_indices) + 1 if qubit_indices else 0
            self.n_qubits = int(n_qubits)
            if self.n_qubits <= 0:
                raise ValueError("n_qubits 必须为正整数")
            self.coefficient = complex(coefficient)
            self._qubit_labels = ["I"] * self.n_qubits
            for label, qubit in zip(labels, qubit_indices):
                if qubit >= self.n_qubits:
                    raise IndexError(
                        f"量子比特索引 {qubit} 超出范围 [0, {self.n_qubits})"
                    )
                self._qubit_labels[qubit] = label
            return

        if qubits is not None:
            raise ValueError("qubits 只能与字符串形式的 Pauli 串一起使用")

        if n_qubits is None:
            max_qubit = -1
            for qubits in paulistring.values():
                for q in qubits:
                    if q < 0:
                        raise IndexError(f"量子比特索引 {q} 不能为负数")
                    if q > max_qubit:
                        max_qubit = q
            n_qubits = max_qubit + 1 if max_qubit >= 0 else 0

        self.n_qubits = n_qubits
        self.coefficient = complex(coefficient)

        # 每个比特位的泡利标签，默认 'I'
        self._qubit_labels: List[str] = ["I"] * n_qubits
        for label, qubits in paulistring.items():
            label = label.upper()
            if label not in PAULI_MAP:
                raise ValueError(f"未知泡利算符 '{label}'，只支持 I/X/Y/Z")
            for q in qubits:
                if q < 0 or q >= n_qubits:
                    raise IndexError(
                        f"量子比特索引 {q} 超出范围 [0, {n_qubits})"
                    )
                self._qubit_labels[q] = label

    def to_matrix(self, backend: "Backend"):
        """
        构造完整的 2^n × 2^n 矩阵（系数已乘入）。

        参数:
            backend: 计算后端
        返回:
            shape (2^n, 2^n) 后端原生张量

        注意：求期望值**不要**走这里。稠密矩阵在 n=14 就要 4.3 GB、n=16 要 68 GB，
        而 ``masks()`` + 稀疏路径每项只要 O(2^n)。本方法保留给确实需要显式矩阵的
        场合（对拍、小规模调试、密度矩阵构造）。
        """
        matrices = [backend.cast(PAULI_MAP[lbl]) for lbl in self._qubit_labels]
        mat_np = backend.to_numpy(backend.tensor_product(*matrices))
        return backend.cast(self.coefficient * mat_np)

    def masks(self) -> tuple[int, int, int]:
        """返回 ``(x_mask, z_mask, y_count)``，用于免矩阵的期望值计算。

        Pauli 串对计算基只做"比特翻转 + 相位"：

            P|b⟩ = i^{n_Y} · (-1)^{popcount(b & z_mask)} · |b ⊕ x_mask⟩

        其中 ``x_mask`` 标记 X/Y 作用的比特（发生翻转），``z_mask`` 标记 Z/Y 作用的
        比特（贡献符号）。Y = iXZ 中的 Z 作用在**翻转前**的比特上，故 Y 同时进入
        两个掩码——这是最容易写错的一处。

        比特序与本仓库一致（大端）：qubit ``q`` 位于第 ``n_qubits-1-q`` 位。
        """

        x_mask = 0
        z_mask = 0
        y_count = 0
        for qubit, label in enumerate(self._qubit_labels):
            bit = 1 << (self.n_qubits - 1 - qubit)
            if label == "X":
                x_mask |= bit
            elif label == "Z":
                z_mask |= bit
            elif label == "Y":
                x_mask |= bit
                z_mask |= bit
                y_count += 1
        return x_mask, z_mask, y_count

    @property
    def qubit_labels(self) -> List[str]:
        """每个比特位对应的泡利标签列表。"""
        return list(self._qubit_labels)

    def __repr__(self) -> str:
        s = "⊗".join(self._qubit_labels)
        return f"PauliString({self.coefficient:.3g} × {s})"


def _is_qubit_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


_HamiltonianTerm = (
    PauliString
    | str
    | tuple[str | Mapping[str, Sequence[int]]]
    | tuple[str | Mapping[str, Sequence[int]], complex | Sequence[int]]
    | tuple[str, complex, Sequence[int]]
    | tuple[str, Sequence[int], complex]
)


# ──────────────────────────────────────────────────────────────────────────────
# Hamiltonian
# ──────────────────────────────────────────────────────────────────────────────

def _popcount_parity(indices: np.ndarray, mask: int) -> np.ndarray:
    """返回 ``(-1)^{popcount(indices & mask)}``，实数 ±1 数组。"""

    if mask == 0:
        return np.ones(indices.shape, dtype=np.float64)
    masked = indices & np.int64(mask)
    # 折半异或求奇偶：比逐位循环快，且与比特数无关地只做 log2 步。
    parity = masked
    shift = 32
    while shift:
        parity = parity ^ (parity >> shift)
        shift >>= 1
    return 1.0 - 2.0 * (parity & 1).astype(np.float64)


def _masked_qubits(mask: int, n_qubits: int):
    """掩码里置位的比特对应的 qubit 下标（大端：qubit q ↔ 第 n-1-q 位）。"""

    return [q for q in range(n_qubits) if mask & (1 << (n_qubits - 1 - q))]


def _apply_z_signs_(flat, z_mask: int, n_qubits: int):
    """就地施加 ``(-1)^{popcount(b & z_mask)}``：沿每个 Z 轴把 "1" 半边取负。

    **不使用任何位运算。** 昇腾没有 ``aten::bitwise_right_shift`` 内核，用移位求
    奇偶会静默回落 CPU（``npu_cpu_fallback``），每个 Pauli 项换来一次
    device→host→device 往返——实测使 n=14 的期望值从 3 ms 劣化到 50 ms。
    改用与 ``_apply_local_strided`` 相同的跨步视图：秩恒为 3，与比特数无关。
    """

    for qubit in _masked_qubits(z_mask, n_qubits):
        left = 1 << qubit
        right = 1 << (n_qubits - 1 - qubit)
        view = flat.reshape(left, 2, right)
        view[:, 1, :].neg_()
    return flat


def _apply_x_flips(flat, x_mask: int, n_qubits: int):
    """返回 ``ψ[b ⊕ x_mask]``：沿每个 X 轴交换两个半边。

    同样不使用位运算，也不物化 ``2^n`` 的 int64 索引数组（后者在 n=20 时是 8 MB
    的额外 HBM 占用，且 gather 是随机访问）。
    """

    import torch

    out = flat
    for qubit in _masked_qubits(x_mask, n_qubits):
        left = 1 << qubit
        right = 1 << (n_qubits - 1 - qubit)
        out = torch.flip(out.reshape(left, 2, right), dims=[1]).reshape(-1)
    return out


def _sparse_expectation_torch(pauli, data, is_density, x_mask, z_mask, phase, dim):
    """torch/NPU 的 real/imag 稀疏路径。

    昇腾缺三类内核，而稀疏公式恰好会命中全部三类：``aclnnIndex``(complex64)
    高级索引、``aclnnAdd``/``aclnnMul``(complex64)、复数归约。拆成实部/虚部后
    三类全部消失。

    **同样不能用位运算求奇偶。** 昇腾没有 ``aten::bitwise_right_shift`` 内核，
    移位版 popcount 会静默回落 CPU（``npu_cpu_fallback``），每个 Pauli 项换来一次
    device→host→device 往返——真机实测使 n=14 从 3 ms 劣化到 50 ms，且 n=16 的
    run-to-run 波动达 4×。因此符号与置换全部改用**实数跨步视图**（秩恒为 3，
    与比特数无关），与 ``_apply_local_strided`` 同一套写法。

    数学上：先 ``χ[c] = sign(c)·ψ[c]``（沿 Z 轴把 "1" 半边取负），
    再 ``φ[b] = χ[b⊕x]``（沿 X 轴交换两个半边），即得所需的
    ``sign(b⊕x)·ψ[b⊕x]``。
    """

    import torch

    n_qubits = pauli.n_qubits
    re = torch.real(data)
    im = torch.imag(data)

    if is_density:
        if tuple(re.shape) != (dim, dim):
            return None
        # Tr(ρP) = Σ_b ρ[b, b⊕x] · phase · sign(b)
        # 沿**列轴**做 XOR 置换后取对角，即得 ρ[b, b⊕x]；符号取在 b 上。
        perm_re, perm_im = re, im
        for qubit in _masked_qubits(x_mask, n_qubits):
            left = 1 << qubit
            right = 1 << (n_qubits - 1 - qubit)
            perm_re = torch.flip(perm_re.reshape(dim, left, 2, right), dims=[2]).reshape(dim, dim)
            perm_im = torch.flip(perm_im.reshape(dim, left, 2, right), dims=[2]).reshape(dim, dim)
        # diagonal 返回非连续视图，clone 后才能 reshape 成跨步视图并就地取负。
        diag_re = torch.diagonal(perm_re).clone()
        diag_im = torch.diagonal(perm_im).clone()
        _apply_z_signs_(diag_re, z_mask, n_qubits)
        _apply_z_signs_(diag_im, z_mask, n_qubits)
        total = complex(float(torch.sum(diag_re)), float(torch.sum(diag_im)))
        return float((pauli.coefficient * phase * total).real)

    re = re.reshape(-1)
    im = im.reshape(-1)
    if re.shape[0] != dim:
        return None

    work_re = re.clone()
    work_im = im.clone()
    _apply_z_signs_(work_re, z_mask, n_qubits)
    _apply_z_signs_(work_im, z_mask, n_qubits)
    perm_re = _apply_x_flips(work_re, x_mask, n_qubits)
    perm_im = _apply_x_flips(work_im, x_mask, n_qubits)

    # ⟨ψ|φ⟩ = Σ conj(ψ)·φ = Σ(re·pr + im·pi) + i·Σ(re·pi − im·pr)
    # 用 sum(a*b) 而非 torch.dot：CLAUDE.md 明确不把 torch.dot 当作 NPU 规避手段。
    total = complex(
        float(torch.sum(re * perm_re + im * perm_im)),
        float(torch.sum(re * perm_im - im * perm_re)),
    )
    return float((pauli.coefficient * phase * total).real)


def _sparse_expectation(pauli, state, backend):
    """免矩阵的 ``⟨ψ|P|ψ⟩`` / ``Tr(ρP)``；无法处理时返回 ``None``。

    利用 ``P|b⟩ = i^{n_Y}·(-1)^{popcount(b & z_mask)}·|b ⊕ x_mask⟩``，每项只要
    O(2^n)，不构造任何 ``2^n × 2^n`` 矩阵——后者在 n=14 就要 4.3 GB，是变分栈
    卡在 n≈13 的**唯一**原因（态矢量演化本身在 n=20 毫无压力）。

    numpy 与 torch/NPU 各有实现：numpy 直接用复数运算；torch 走
    ``_sparse_expectation_torch`` 的 real/imag 分解，以避开昇腾缺失的
    complex64 索引/加乘/归约内核。其余类型返回 ``None`` 回退稠密路径。
    """

    from .state import State

    if isinstance(state, State):
        data = state.data
        is_density = state.is_density
    else:
        data = state
        is_density = getattr(data, "ndim", 1) == 2 and data.shape[0] == data.shape[1] and data.shape[1] > 1

    x_mask, z_mask, y_count = pauli.masks()
    dim = 1 << pauli.n_qubits
    # i^{n_Y}：只有四种取值，避免引入复数幂运算。
    phase = (1.0, 1j, -1.0, -1j)[y_count % 4]

    if not isinstance(data, np.ndarray):
        try:
            import torch
        except ImportError:
            return None
        if not isinstance(data, torch.Tensor):
            return None
        return _sparse_expectation_torch(pauli, data, is_density, x_mask, z_mask, phase, dim)

    indices = np.arange(dim, dtype=np.int64)
    signs = _popcount_parity(indices, z_mask)

    if is_density:
        rho = np.asarray(data)
        if rho.shape != (dim, dim):
            return None
        # Tr(ρP) = Σ_b ρ[b, b⊕x] · P[b⊕x, b] = Σ_b ρ[b, b⊕x] · phase · sign(b)
        # 注意符号取在 **b** 上，与态矢量分支取在 b⊕x 上不同：那里算符作用在 ket，
        # 这里配对的是 P 的 (b⊕x, b) 元素。
        cols = indices ^ np.int64(x_mask)
        diag = rho[indices, cols]
        total = complex(np.sum(diag * signs))
        return float(np.real(pauli.coefficient * phase * total))

    psi = np.asarray(data).reshape(-1)
    if psi.shape[0] != dim:
        return None
    # (Pψ)[b] = phase · sign(b⊕x) · ψ[b⊕x]：符号取在**被作用**的下标 b⊕x 上，
    # 不是 b 上。二者仅在 x_mask=0（纯 Z）或 z_mask=0（纯 X）时重合，所以写错
    # 时只有含 Y 的串会露馅。
    cols = indices if x_mask == 0 else (indices ^ np.int64(x_mask))
    permuted = psi if x_mask == 0 else psi[cols]
    col_signs = signs if x_mask == 0 else _popcount_parity(cols, z_mask)
    total = complex(np.vdot(psi, permuted * col_signs))
    return float(np.real(pauli.coefficient * phase * total))


class Hamiltonian:
    """
    哈密顿量：加权 PauliString 的线性组合  H = Σ_i cᵢ Pᵢ。

    推荐使用 Pauli 在前的构造方式：
    ``Hamiltonian([("ZI", 0.3), ("XX", 0.5)])``。若只想写局部 Pauli
    串，可以额外传入比特下标：``Hamiltonian(n_qubits=4, terms=[("ZZ", [0, 3], -1.0)])``。

    示例::

        bk = GPUBackend()

        # H = -Z₀Z₁  +  0.5 X₀X₁  +  0.3 Z₀
        H = Hamiltonian([
            ("ZZ", -1.0),
            ("XX", 0.5),
            ("ZI", 0.3),
        ])
        H03 = Hamiltonian(n_qubits=4, terms=[("ZZ", [0, 3], -1.0)])

        mat = H.to_matrix(bk)
        print(H.expectation(sv, bk))   # sv 是 State
    """

    def __init__(
        self,
        n_qubits: int | Iterable[_HamiltonianTerm] | None = None,
        terms: Iterable[_HamiltonianTerm] | None = None,
    ):
        if terms is None and n_qubits is not None and not isinstance(n_qubits, (int, np.integer)):
            terms = n_qubits
            n_qubits = None
        if n_qubits is not None and not isinstance(n_qubits, (int, np.integer)):
            raise TypeError("n_qubits 必须是整数；若要传入 Pauli 项，请使用 terms= 或省略 n_qubits")
        width = int(n_qubits) if n_qubits is not None else None

        raw_terms = list(terms or ())
        if width is None and raw_terms:
            width = max(self._infer_term_width(term) for term in raw_terms)

        parsed_terms = [self._parse_term(term, width) for term in raw_terms]
        if width is None:
            if not parsed_terms:
                raise ValueError("n_qubits 不能为 None，除非 terms 中至少包含一个 Pauli 字符串")
            width = parsed_terms[0].n_qubits
        if width <= 0:
            raise ValueError("n_qubits 必须为正整数")
        for term in parsed_terms:
            if term.n_qubits != width:
                raise ValueError(
                    f"Pauli 项宽度 {term.n_qubits} 与 n_qubits={width} 不一致"
                )
        self.n_qubits = width
        self._terms: List[PauliString] = parsed_terms

    @staticmethod
    def _infer_term_width(term: _HamiltonianTerm) -> int:
        if isinstance(term, PauliString):
            return term.n_qubits
        if isinstance(term, str):
            return len(term.strip())
        if not isinstance(term, tuple):
            raise TypeError("Hamiltonian terms 必须是 PauliString 或 Pauli 项元组")
        if len(term) not in {1, 2, 3}:
            raise TypeError("Hamiltonian terms 必须是 (pauli,), (pauli, coefficient), (pauli, qubits) 或 (pauli, coefficient, qubits)")

        pauli = term[0]
        if len(term) == 2 and isinstance(pauli, str) and _is_qubit_sequence(term[1]):
            qubit_indices = [int(qubit) for qubit in term[1]]
            return max(qubit_indices) + 1 if qubit_indices else 0
        elif len(term) == 3:
            qubits = term[1] if _is_qubit_sequence(term[1]) else term[2]
            qubit_indices = [int(qubit) for qubit in qubits]
            return max(qubit_indices) + 1 if qubit_indices else 0
        if isinstance(pauli, str):
            return len(pauli.strip())
        if isinstance(pauli, Mapping):
            max_qubit = -1
            for qubits in pauli.values():
                for qubit in qubits:
                    max_qubit = max(max_qubit, int(qubit))
            return max_qubit + 1 if max_qubit >= 0 else 0
        raise TypeError("Hamiltonian 项的第一个元素必须是 Pauli 字符串或 Pauli 字典")

    @staticmethod
    def _parse_term(
        term: _HamiltonianTerm,
        n_qubits: int | None,
    ) -> PauliString:
        if isinstance(term, PauliString):
            return term
        if isinstance(term, str):
            return PauliString(term, coefficient=1.0, n_qubits=n_qubits)
        if not isinstance(term, tuple):
            raise TypeError("Hamiltonian terms 必须是 PauliString 或 Pauli 项元组")
        if len(term) == 1:
            pauli = term[0]
            return PauliString(pauli, coefficient=1.0, n_qubits=n_qubits)
        if len(term) == 2:
            pauli, value = term
            if isinstance(pauli, str) and _is_qubit_sequence(value):
                return PauliString(pauli, coefficient=1.0, n_qubits=n_qubits, qubits=value)
            coefficient = value
            return PauliString(pauli, coefficient=coefficient, n_qubits=n_qubits)
        if len(term) == 3:
            pauli = term[0]
            if not isinstance(pauli, str):
                raise ValueError("带 qubits 的 Hamiltonian 项必须使用字符串形式的 Pauli 串")
            if _is_qubit_sequence(term[1]):
                qubits = term[1]
                coefficient = term[2]
            else:
                coefficient = term[1]
                qubits = term[2]
            return PauliString(pauli, coefficient=coefficient, n_qubits=n_qubits, qubits=qubits)
        raise TypeError("Hamiltonian terms 必须是 (pauli,), (pauli, coefficient), (pauli, qubits) 或 (pauli, coefficient, qubits)")

    @classmethod
    def from_list(
        cls,
        terms: Iterable[
            str
            | tuple[str | Mapping[str, Sequence[int]]]
            | tuple[str | Mapping[str, Sequence[int]], complex | Sequence[int]]
            | tuple[str, complex, Sequence[int]]
            | tuple[str, Sequence[int], complex]
        ],
        n_qubits: int | None = None,
    ) -> "Hamiltonian":
        """用 ``[(pauli, coefficient), ...]`` 或带 qubits 的三元组构造哈密顿量。"""

        return cls(n_qubits=n_qubits, terms=terms)

    def to_matrix(self, backend: "Backend"):
        """
        构造完整哈密顿量矩阵，shape (2^n, 2^n)。

        参数:
            backend: 计算后端
        返回:
            后端原生张量（complex）
        """
        dim = 1 << self.n_qubits
        result_np = np.zeros((dim, dim), dtype=np.complex64)
        for term in self._terms:
            mat_np = backend.to_numpy(term.to_matrix(backend))
            result_np = result_np + mat_np
        return backend.cast(result_np)

    def expectation(
        self,
        state: "State",
        backend: "Backend",
    ) -> float:
        """
        计算量子态对哈密顿量的期望值。

        参数:
            state:   State 实例（向量或密度矩阵形态）
            backend: 计算后端
        返回:
            实数期望值
        """
        from .state import State

        # 稀疏路径：逐项累加 ⟨ψ|Pᵢ|ψ⟩，全程不构造 2^n × 2^n 矩阵。
        # 任一项无法稀疏处理（如 torch/NPU 张量）就整体回退，避免两条路径混用。
        total = 0.0
        for term in self._terms:
            value = _sparse_expectation(term, state, backend)
            if value is None:
                total = None
                break
            total += value
        if total is not None:
            return float(total)

        # 回退：稠密矩阵（见 _sparse_expectation 的说明）。
        H_mat = self.to_matrix(backend)
        if isinstance(state, State):
            return state.expectation(H_mat)
        # 兼容原始后端张量（态向量）
        return backend.expectation_sv(state, H_mat)

    @property
    def terms(self) -> List[PauliString]:
        """返回所有 PauliString 项（只读副本）。"""
        return list(self._terms)

    def __len__(self) -> int:
        return len(self._terms)

    def __repr__(self) -> str:
        return f"Hamiltonian(n_qubits={self.n_qubits}, terms={len(self._terms)})"
