"""aicir 适配器（基准的被测对象与 parity 参考）。"""

from __future__ import annotations

import numpy as np

from ..core.spec import CircuitSpec
from .base import Adapter

__all__ = ["AicirAdapter"]

_GATE_BUILDERS = {}


def _builders():
    """惰性构造门工厂映射，避免导入期就绑定 aicir。

    每个 builder 返回**门列表**，因为部分规格门在 aicir 里没有一一对应的原生门，
    需要就地降级展开。
    """

    global _GATE_BUILDERS
    if _GATE_BUILDERS:
        return _GATE_BUILDERS

    import aicir as A

    def _cp(q, p):
        """受控相位 CP(θ)=diag(1,1,1,e^{iθ})，作为**单条**双比特 unitary 指令。

        aicir 没有原生 phase/cp 门。可以用恒等式 ``CP(θ) ≡ CRZ(θ)·RZ_control(θ/2)``
        （差一个全局相位）拆成两个门，但那样 aicir 在 QFT 上要跑 1.9 倍于
        Qiskit/Cirq 的门数——比较的就不再是同等工作量，基准会把适配器的选择
        误记成引擎的差距。这里直接给出 4×4 矩阵，三个框架门数一致。

        注意 **CRZ 不等于 CP**（两者在 |10⟩ 上差一个相位），不能拿 CRZ 冒充。
        """

        matrix = np.diag([1.0, 1.0, 1.0, np.exp(1j * p[0])]).astype(np.complex128)
        return [{"type": "unitary", "qubits": [q[0], q[1]], "parameter": matrix}]

    _GATE_BUILDERS = {
        "h": lambda q, p: [A.hadamard(q[0])],
        "x": lambda q, p: [A.pauli_x(q[0])],
        "y": lambda q, p: [A.pauli_y(q[0])],
        "z": lambda q, p: [A.pauli_z(q[0])],
        "s": lambda q, p: [A.s_gate(q[0])],
        "t": lambda q, p: [A.t_gate(q[0])],
        "rx": lambda q, p: [A.rx(p[0], q[0])],
        "ry": lambda q, p: [A.ry(p[0], q[0])],
        "rz": lambda q, p: [A.rz(p[0], q[0])],
        "cx": lambda q, p: [A.cnot(q[1], [q[0]])],
        "cz": lambda q, p: [A.cz(q[1], [q[0]])],
        "cp": _cp,
        "swap": lambda q, p: [A.swap(q[0], q[1])],
    }
    return _GATE_BUILDERS


class AicirAdapter(Adapter):
    name = "aicir"

    def __init__(self, dtype=None, backend=None):
        self._dtype = dtype if dtype is not None else np.complex128
        self._backend = backend

    @classmethod
    def is_available(cls) -> bool:
        try:
            import aicir  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def version(cls) -> str:
        import aicir

        return getattr(aicir, "__version__", "unknown")

    def _resolve_backend(self):
        if self._backend is not None:
            return self._backend
        from aicir import NumpyBackend

        return NumpyBackend(dtype=self._dtype)

    def build(self, spec: CircuitSpec):
        from aicir import Circuit

        builders = _builders()
        gates = []
        for op in spec.operations:
            gates.extend(builders[op.gate](op.qubits, op.params))
        return Circuit(*gates, n_qubits=spec.n_qubits)

    def run(self, circuit, spec: CircuitSpec):
        from aicir import Measure

        result = Measure(backend=self._resolve_backend()).run(circuit, shots=None)
        return np.asarray(result.final_state).reshape(-1)

    def precision(self) -> str:
        return np.dtype(self._dtype).name
