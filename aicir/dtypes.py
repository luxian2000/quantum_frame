"""复数精度策略：后端是 dtype 的单一真源。

设计要点
--------
精度在 aicir 中是**后端能力**，不是全局常量：

- ``NumpyBackend`` 默认 ``complex128`` —— 研究型 CPU 路径以正确性优先，
  也与 Qiskit-Aer / Qulacs 的双精度默认对齐，便于横向对比；
- ``GPUBackend`` 默认 ``complex64`` —— PyTorch 生态是 float32-native（量子层读出
  float64 会直接让 ``nn.Linear`` 报 dtype 不匹配），且消费级 GPU 的 fp64 吞吐
  仅为 fp32 的 1/32；
- ``NPUBackend`` 硬件锁定 ``complex64`` —— 昇腾没有 complex128 内核，
  ``aicir.distributed`` 的成对实数（paired-real）通道同样按 float32 设计；
- 门矩阵一律在**最宽精度**下构造，再由后端在边界处窄化（见 ``aicir/core/gates.py``）。

``set_default_dtype`` 提供进程级默认覆盖（对标 TensorCircuit 的 ``tc.set_dtype``），
只影响**此后新建**的后端，不会改写已存在的后端实例。一旦显式调用，torch 后端也会
服从该全局值（而非自身的 complex64 默认）；``reset_default_dtype`` 可还原初始语义。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "get_default_dtype",
    "set_default_dtype",
    "default_dtype",
    "default_dtype_was_set_explicitly",
    "reset_default_dtype",
    "resolve_dtype",
    "is_complex_dtype",
    "to_numpy_complex_dtype",
]

#: 进程级默认复数精度。研究型框架以正确性优先，故默认双精度。
_DEFAULT_DTYPE = np.complex128

#: 用户是否显式调用过 ``set_default_dtype``。
#:
#: 用于区分“没人动过默认值”与“用户明确要求全局精度”：前者让各后端按自身生态
#: 惯例取默认（torch 后端走 complex64），后者则一律服从用户。
_DEFAULT_EXPLICITLY_SET = False


def default_dtype_was_set_explicitly() -> bool:
    """返回用户是否显式设置过全局默认精度。"""

    return _DEFAULT_EXPLICITLY_SET

_NUMPY_COMPLEX_DTYPES = (np.complex64, np.complex128)


def is_complex_dtype(dtype) -> bool:
    """判断 dtype 是否为受支持的复数类型（numpy 或 torch 皆可）。"""

    if dtype is None:
        return False
    try:
        return bool(np.issubdtype(np.dtype(dtype), np.complexfloating))
    except TypeError:
        # torch dtype 走这里：torch.complex64 无法被 np.dtype 解析。
        return bool(getattr(dtype, "is_complex", False))


def get_default_dtype():
    """返回当前进程级默认复数精度。"""

    return _DEFAULT_DTYPE


def set_default_dtype(dtype):
    """设置进程级默认复数精度，返回设置后的值。

    参数:
        dtype: ``np.complex64`` 或 ``np.complex128``。

    仅影响此后新建的后端；已有后端实例保持自身 dtype 不变。
    """

    global _DEFAULT_DTYPE, _DEFAULT_EXPLICITLY_SET
    if not is_complex_dtype(dtype):
        raise ValueError(f"默认 dtype 必须是复数类型，收到 {dtype!r}；可选 complex64 / complex128")
    resolved = to_numpy_complex_dtype(dtype)
    if resolved not in _NUMPY_COMPLEX_DTYPES:
        raise ValueError(f"不支持的复数精度 {dtype!r}；可选 complex64 / complex128")
    _DEFAULT_DTYPE = resolved
    _DEFAULT_EXPLICITLY_SET = True
    return _DEFAULT_DTYPE


def reset_default_dtype():
    """恢复出厂默认精度，并清除“用户显式设置过”的标记。

    清除标记是必要的：``set_default_dtype`` 一旦调用就会让 torch 后端服从全局值，
    仅把数值改回去并不能还原“各后端按自身生态取默认”的初始语义。
    """

    global _DEFAULT_DTYPE, _DEFAULT_EXPLICITLY_SET
    _DEFAULT_DTYPE = np.complex128
    _DEFAULT_EXPLICITLY_SET = False
    return _DEFAULT_DTYPE


def default_dtype():
    """``get_default_dtype`` 的别名，便于在表达式中书写。"""

    return _DEFAULT_DTYPE


def to_numpy_complex_dtype(dtype):
    """把任意后端 dtype（含 torch dtype）映射到对应的 numpy 复数 dtype。"""

    if dtype is None:
        return _DEFAULT_DTYPE
    try:
        return np.dtype(dtype).type
    except TypeError:
        pass
    name = str(getattr(dtype, "name", dtype))
    if "128" in name:
        return np.complex128
    if "64" in name:
        return np.complex64
    return _DEFAULT_DTYPE


def resolve_dtype(backend=None, dtype=None):
    """解析应当使用的 numpy 复数 dtype。

    优先级：显式 ``dtype`` > 后端自身 dtype > 进程级默认。
    """

    if dtype is not None:
        return to_numpy_complex_dtype(dtype)
    if backend is not None:
        backend_dtype = getattr(backend, "dtype", None)
        if backend_dtype is None:
            backend_dtype = getattr(backend, "_dtype", None)
        if backend_dtype is not None:
            return to_numpy_complex_dtype(backend_dtype)
    return _DEFAULT_DTYPE
