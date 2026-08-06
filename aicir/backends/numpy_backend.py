"""
aicir/backends/numpy_backend.py

基于 NumPy 的 CPU 计算后端（参考实现）。

特点：
- 无外部深度学习依赖，环境要求最低
- 不支持自动微分（参数优化需用 GPUBackend）
- 适合小规模验证和算法原型
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from ..dtypes import get_default_dtype, to_numpy_complex_dtype
from .base import Backend


def _blas_safe_matmul(a, b):
    """复数 matmul，屏蔽 BLAS 内核的伪警告。

    某些 BLAS 实现（如 Apple Accelerate）会在复数 matmul 内核内部由中间量触发
    divide/overflow/invalid 的 RuntimeWarning，而输出本身有限且正确。这里只对
    该次调用抑制这三个标志，不改动 numpy 全局错误状态。

    历史上此处还会把两个操作数无条件提升到 complex128 再转回——那是同一问题的
    第一版尝试，errstate 才是真正的修复。提升会让 complex64 路径在每个门上白付
    一次转换，因此已移除。
    """

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return np.asarray(a) @ np.asarray(b)


def _local_offsets(axes, n_qubits: int):
    offsets = []
    for local_index in range(1 << len(axes)):
        offset = 0
        for local_pos, axis in enumerate(axes):
            bit = (local_index >> (len(axes) - 1 - local_pos)) & 1
            offset |= bit << (int(n_qubits) - 1 - int(axis))
        offsets.append(np.int64(offset))
    return offsets


def _zero_target_bases(start: int, stop: int, axes, n_qubits: int) -> np.ndarray:
    basis = np.arange(start, stop, dtype=np.int64)
    mask = np.ones(basis.shape, dtype=bool)
    for axis in axes:
        shift = int(n_qubits) - 1 - int(axis)
        mask &= ((basis >> shift) & 1) == 0
    return basis[mask]


def _apply_local_gather(state, local_matrix, axes, n_qubits: int, dtype, chunk_size: int = 1 << 20):
    """gather/scatter 版局部门应用（**昇腾策略的 CPU 参照实现**）。

    物化 int64 索引数组、fancy indexing 取值、matmul、再散回。这是 NPU 上的必需
    做法——`aclnnComplex` 限 8 维，复数态无法 reshape 成 `(2,)*n`，只能走扁平位
    置换。NumPy 没有该限制，因此 CPU 热路径改用 `_apply_local_strided`，这里保留
    两个用途：跨路径正确性 oracle（见 `tests/backends/test_numpy_strided_apply.py`），
    以及 NPU 策略的可读参照。
    """

    axes = tuple(int(axis) for axis in axes)
    dim_local = 1 << len(axes)
    local = np.asarray(local_matrix, dtype=dtype)
    flat = np.asarray(state).reshape(-1)
    dim = 1 << int(n_qubits)

    out = np.empty_like(flat)
    chunk_size = max(int(chunk_size), 1)
    offsets = _local_offsets(axes, n_qubits)

    for start in range(0, dim, chunk_size):
        stop = min(start + chunk_size, dim)
        bases = _zero_target_bases(start, stop, axes, n_qubits)
        if bases.size == 0:
            continue
        gathered_indices = [bases | offset for offset in offsets]
        gathered = np.stack([flat[index] for index in gathered_indices], axis=0)
        updated = _blas_safe_matmul(local, gathered)
        for index, values in zip(gathered_indices, updated):
            out[index] = values

    return out.reshape(dim, 1)


def _apply_local_strided(state, local_matrix, axes, n_qubits: int, dtype):
    """跨步视图版局部门应用（CPU 快路径）。

    把扁平态 reshape 成**秩恒定**的分块视图，直接按块读写：

    - 单比特门（作用在 axis ``k``）→ ``(2^k, 2, 2^(n-1-k))``，秩 3；
    - 双比特门（作用在 ``lo < hi``）→ ``(2^lo, 2, 2^(hi-lo-1), 2, 2^(n-1-hi))``，秩 5。

    秩为 ``2*len(axes)+1``，**与比特数无关**，因此既不触碰 numpy 的维度上限，也
    不需要 `(2,)*n` 那种随 n 增长的 reshape。相比 gather 版省掉了：整份 int64
    索引数组的构造、随机访问的 gather/scatter，以及 `np.stack` 的中间拷贝。

    注意局部矩阵的比特序跟随 **给定的 axes 顺序**（``axes[0]`` 是局部高位），
    而不是排序后的物理顺序——反序时必须换算，否则结果安静地错。
    """

    axes = tuple(int(axis) for axis in axes)
    local = np.asarray(local_matrix, dtype=dtype)
    flat = np.asarray(state, dtype=dtype).reshape(-1)
    dim = 1 << int(n_qubits)
    out = np.empty_like(flat)

    if len(axes) == 1:
        axis = axes[0]
        left = 1 << axis
        right = 1 << (n_qubits - 1 - axis)
        src = flat.reshape(left, 2, right)
        dst = out.reshape(left, 2, right)
        s0 = src[:, 0, :]
        s1 = src[:, 1, :]
        # 两个目标块都在写入前算完，故 src 是不是 out 的别名都不影响正确性。
        dst[:, 0, :] = local[0, 0] * s0 + local[0, 1] * s1
        dst[:, 1, :] = local[1, 0] * s0 + local[1, 1] * s1
        return out.reshape(dim, 1)

    first, second = axes
    lo, hi = (first, second) if first < second else (second, first)
    left = 1 << lo
    middle = 1 << (hi - lo - 1)
    right = 1 << (n_qubits - 1 - hi)
    shape = (left, 2, middle, 2, right)
    src = flat.reshape(shape)
    dst = out.reshape(shape)

    def _local_index(bit_lo: int, bit_hi: int) -> int:
        """把 (低位轴比特, 高位轴比特) 换算成局部矩阵的行/列号。"""

        bit_first = bit_lo if first == lo else bit_hi
        bit_second = bit_hi if second == hi else bit_lo
        return (bit_first << 1) | bit_second

    blocks = {(i, j): src[:, i, :, j, :] for i in (0, 1) for j in (0, 1)}
    for i in (0, 1):
        for j in (0, 1):
            row = _local_index(i, j)
            acc = None
            for p in (0, 1):
                for q in (0, 1):
                    coeff = local[row, _local_index(p, q)]
                    if coeff == 0:
                        continue
                    term = coeff * blocks[(p, q)]
                    acc = term if acc is None else acc + term
            dst[:, i, :, j, :] = acc if acc is not None else 0
    return out.reshape(dim, 1)


class NumpyBackend(Backend):
    """基于 NumPy 的 CPU 计算后端（无自动微分支持）。"""

    def __init__(self, dtype=None):
        """
        参数:
            dtype: 复数数据类型，默认取 ``aicir.get_default_dtype()``（complex128）
        """
        self._dtype = to_numpy_complex_dtype(dtype) if dtype is not None else get_default_dtype()

    # ──────────────────────── 元信息 ────────────────────────────

    @property
    def name(self) -> str:
        return f"NumpyBackend(dtype={self._dtype.__name__ if hasattr(self._dtype, '__name__') else self._dtype})"

    # ──────────────────────── 张量工厂 ──────────────────────────

    def zeros(self, shape: tuple, dtype=None):
        return np.zeros(shape, dtype=dtype or self._dtype)

    def eye(self, dim: int):
        return np.eye(dim, dtype=self._dtype)

    def cast(self, array, dtype=None):
        if isinstance(array, np.ndarray):
            return array.astype(dtype or self._dtype)
        return np.array(array, dtype=dtype or self._dtype)

    def to_numpy(self, tensor) -> np.ndarray:
        return np.asarray(tensor)

    # ──────────────────────── 量子态初始化 ──────────────────────

    def zeros_state(self, n_qubits: int):
        dim = 1 << n_qubits
        state = np.zeros((dim, 1), dtype=self._dtype)
        state[0, 0] = 1.0 + 0j
        return state

    # ──────────────────────── 线性代数 ──────────────────────────

    def matmul(self, a, b):
        return _blas_safe_matmul(a, b).astype(self._dtype, copy=False)

    def kron(self, a, b):
        return np.kron(a, b)

    def dagger(self, matrix):
        return np.conj(matrix).T

    def trace(self, matrix):
        return np.trace(matrix)

    def real(self, tensor):
        return np.real(tensor)

    def tensordot(self, a, b, axes):
        out = np.tensordot(np.asarray(a), np.asarray(b), axes=axes)
        return out.astype(self._dtype)

    def transpose(self, a, axes):
        return np.transpose(np.asarray(a), axes)

    def reshape(self, a, shape):
        return np.reshape(np.asarray(a), tuple(shape))

    def conj(self, a):
        return np.conj(np.asarray(a))

    def svd(self, matrix):
        m = np.asarray(matrix, dtype=np.complex128)
        u, s, vh = np.linalg.svd(m, full_matrices=False)
        return u.astype(self._dtype), s.astype(np.float64), vh.astype(self._dtype)

    def take(self, a, axis, index):
        return np.take(np.asarray(a), int(index), axis=int(axis))

    def add(self, a, b):
        return np.asarray(a) + np.asarray(b)

    def mul(self, a, b):
        return np.asarray(a) * np.asarray(b)

    def div(self, a, b):
        return np.asarray(a) / np.asarray(b)

    def abs_sq(self, tensor):
        return np.abs(tensor) ** 2

    # ──────────────────────── 量子操作 ──────────────────────────

    def apply_unitary(self, state, unitary):
        return unitary @ state

    def apply_statevector_local(self, state, local_matrix, axes, n_qubits: int):
        axes = tuple(int(axis) for axis in axes)
        n_qubits = int(n_qubits)
        if len(axes) == 0:
            return state
        if len(axes) > 2:
            return None
        if len(set(axes)) != len(axes):
            raise ValueError("局部门作用的量子比特不能重复")
        if any(axis < 0 or axis >= n_qubits for axis in axes):
            raise ValueError("局部门作用的量子比特索引超出范围")

        dim_local = 1 << len(axes)
        local = np.asarray(local_matrix, dtype=self._dtype)
        if local.shape != (dim_local, dim_local):
            raise ValueError(
                f"local gate matrix shape {local.shape} does not match {len(axes)} target qubit(s)"
            )

        flat = np.asarray(state).reshape(-1)
        dim = 1 << n_qubits
        if flat.size != dim:
            raise ValueError(f"state length {flat.size} does not match n_qubits={n_qubits}")

        # CPU 走跨步视图快路径。gather 版是**昇腾**的必需策略（`aclnnComplex` 限
        # 8 维），NumPy 没有该限制，不该替 NPU 交这笔税：实测跨步版在态仍驻留
        # cache 的区间快 1.2–1.9×，n=20 时快约 7×。
        return _apply_local_strided(state, local, axes, n_qubits, self._dtype)

    def inner_product(self, bra, ket):
        b = np.asarray(bra).reshape(-1)
        k = np.asarray(ket).reshape(-1)
        return np.conj(b) @ k

    def measure_probs(self, state):
        probs = (np.abs(np.asarray(state).reshape(-1)) ** 2).real
        total = probs.sum()
        if total > 0:
            probs = probs / total
        return probs

    def partial_trace(self, rho, keep: List[int], n_qubits: int):
        rho = np.asarray(rho)
        if n_qubits is None:
            n_qubits = int(math.log2(rho.shape[0]))

        keep = sorted(set(int(k) for k in keep))
        trace_out = [i for i in range(n_qubits) if i not in keep]
        if not trace_out:
            return rho.copy()

        reshaped = rho.reshape([2] * n_qubits + [2] * n_qubits)
        perm = (keep + trace_out
                + [k + n_qubits for k in keep]
                + [t + n_qubits for t in trace_out])
        permuted = reshaped.transpose(perm)

        d_keep = 1 << len(keep)
        d_trace = 1 << len(trace_out)
        permuted = permuted.reshape(d_keep, d_trace, d_keep, d_trace)
        return np.einsum("abcb->ac", permuted)

    def sample(self, probs, shots: int):
        probs_real = np.real(np.asarray(probs)).astype(np.float64)
        probs_real = np.clip(probs_real, 0, None)
        total = probs_real.sum()
        if total == 0:
            raise ValueError("概率全为零，无法采样")
        probs_real = probs_real / total
        indices = np.random.choice(len(probs_real), size=shots, p=probs_real)
        return np.bincount(indices, minlength=len(probs_real))

    def expectation_sv(self, state, operator):
        # 见 matmul() 注释：部分 BLAS 构建在此类复数矩阵乘法上会给出无害的
        # divide/overflow/invalid RuntimeWarning，errstate 仅抑制这几类标志。
        s = np.asarray(state).reshape(-1, 1)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            val = (np.conj(s).T @ np.asarray(operator) @ s)[0, 0]
        return float(np.real(val))

    def expectation_dm(self, rho, operator):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            val = np.trace(np.asarray(rho) @ np.asarray(operator))
        return float(np.real(val))
