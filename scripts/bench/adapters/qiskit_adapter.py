"""Qiskit 适配器。

两点必须小心：

1. **比特序**：Qiskit 的 ``Statevector`` 是小端（qubit 0 是最低位），本仓库统一
   大端，故返回前必须翻转，否则 parity 测试会红——这正是它存在的意义。
2. **构建 vs 执行**：``QuantumCircuit`` 的组装归入 build 阶段，态矢量演化归入 run，
   两者分开计时，避免把编译开销算作模拟速度。
"""

from __future__ import annotations

import numpy as np

from ..core.spec import CircuitSpec
from .base import Adapter, reverse_qubit_order

__all__ = ["QiskitAdapter", "QiskitAerAdapter"]


class QiskitAdapter(Adapter):
    """Qiskit 参考态矢量路径（``quantum_info.Statevector``，纯 Python/NumPy）。"""

    name = "qiskit"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import qiskit  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def version(cls) -> str:
        import qiskit

        return qiskit.__version__

    def build(self, spec: CircuitSpec):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(spec.n_qubits)
        for op in spec.operations:
            q = op.qubits
            p = op.params
            if op.gate == "h":
                qc.h(q[0])
            elif op.gate == "x":
                qc.x(q[0])
            elif op.gate == "y":
                qc.y(q[0])
            elif op.gate == "z":
                qc.z(q[0])
            elif op.gate == "s":
                qc.s(q[0])
            elif op.gate == "t":
                qc.t(q[0])
            elif op.gate == "rx":
                qc.rx(p[0], q[0])
            elif op.gate == "ry":
                qc.ry(p[0], q[0])
            elif op.gate == "rz":
                qc.rz(p[0], q[0])
            elif op.gate == "cx":
                qc.cx(q[0], q[1])
            elif op.gate == "cz":
                qc.cz(q[0], q[1])
            elif op.gate == "cp":
                qc.cp(p[0], q[0], q[1])
            elif op.gate == "swap":
                qc.swap(q[0], q[1])
            else:  # pragma: no cover - SUPPORTED_GATES 已挡住
                raise ValueError(f"Qiskit 适配器未覆盖门 {op.gate!r}")
        return qc

    def run(self, circuit, spec: CircuitSpec):
        from qiskit.quantum_info import Statevector

        state = np.asarray(Statevector(circuit).data)
        # Qiskit 小端 → 本仓库大端
        return reverse_qubit_order(state, spec.n_qubits)


class QiskitAerAdapter(QiskitAdapter):
    """Qiskit Aer 高性能模拟器路径（C++ 后端）。"""

    name = "qiskit_aer"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import qiskit_aer  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def version(cls) -> str:
        import qiskit_aer

        return qiskit_aer.__version__

    def __init__(self, precision: str = "double"):
        #: Aer 支持 "single"/"double"，用于与 aicir 的 complex64/complex128 对齐。
        self._precision = precision

    def build(self, spec: CircuitSpec):
        circuit = super().build(spec)
        circuit.save_statevector()
        return circuit

    def run(self, circuit, spec: CircuitSpec):
        from qiskit_aer import AerSimulator

        sim = AerSimulator(method="statevector", precision=self._precision)
        result = sim.run(circuit).result()
        state = np.asarray(result.get_statevector(circuit))
        return reverse_qubit_order(state, spec.n_qubits)

    def precision(self) -> str:
        return "complex128" if self._precision == "double" else "complex64"
