"""适配器协议。

每个框架实现一个适配器，把框架无关的 ``CircuitSpec`` 翻译成自家线路并执行。

**比特序约定**：本仓库统一用 **大端**（qubit 0 是最高位，与 ``aicir`` 一致）。
Qiskit 的态矢量是小端，必须在适配器内翻转后再返回——这正是 ``test_bench_harness``
里跨适配器 parity 测试要钉死的东西：比特序错了，计时表比的就是两条不同线路。
"""

from __future__ import annotations

import abc

import numpy as np

from ..core.spec import CircuitSpec

__all__ = ["Adapter", "reverse_qubit_order"]


def reverse_qubit_order(state: np.ndarray, n_qubits: int) -> np.ndarray:
    """在大端与小端之间转换态矢量。

    做法是把 ``2^n`` 维向量视作 ``(2,)*n`` 张量、反转轴序后再展平，
    等价于把每个基态的比特串前后颠倒。
    """

    if n_qubits == 0:
        return state
    return np.transpose(np.asarray(state).reshape((2,) * n_qubits), tuple(reversed(range(n_qubits)))).reshape(-1)


class Adapter(abc.ABC):
    """一个被基准的框架。"""

    #: 适配器名（清单里的键）
    name: str = ""

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        """框架是否已安装。缺失时基准跳过该行并在清单中记录。"""

    @classmethod
    def version(cls) -> str:
        """框架版本，写入清单以保证数字可追溯。"""

        return "unknown"

    @abc.abstractmethod
    def build(self, spec: CircuitSpec):
        """把规格翻译成框架自己的线路对象（**构建**阶段，单独计时）。"""

    @abc.abstractmethod
    def run(self, circuit, spec: CircuitSpec):
        """执行线路并返回末态（**执行**阶段，单独计时）。"""

    def statevector(self, spec: CircuitSpec) -> np.ndarray:
        """构建 + 执行，返回**大端**归一化的一维态矢量。parity 测试用。"""

        state = np.asarray(self.run(self.build(spec), spec)).reshape(-1)
        return state.astype(np.complex128, copy=False)

    def precision(self) -> str:
        """本次运行的复数精度，用于保证跨框架比较是同精度的。"""

        return "complex128"
