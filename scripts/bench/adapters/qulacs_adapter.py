"""Qulacs 适配器。

Qulacs 只有 **double** 精度（无单精度构建），因此它只会出现在 complex128 那张表里，
这一点必须在论文表格中写明，否则读者会以为它缺席是因为跑不动。

比特序：Qulacs 的 ``QuantumState.get_vector()`` 是小端，需翻转成本仓库的大端。
"""

from __future__ import annotations

import numpy as np

from ..core.spec import CircuitSpec
from .base import Adapter, reverse_qubit_order

__all__ = ["QulacsAdapter"]


class QulacsAdapter(Adapter):
    name = "qulacs"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import qulacs  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def version(cls) -> str:
        import qulacs

        return getattr(qulacs, "__version__", "unknown")

    def build(self, spec: CircuitSpec):
        from qulacs import QuantumCircuit
        from qulacs.gate import CNOT, CZ, SWAP, H, S, T, X, Y, Z, RX, RY, RZ, DenseMatrix

        circuit = QuantumCircuit(spec.n_qubits)
        for op in spec.operations:
            q = op.qubits
            p = op.params
            if op.gate == "h":
                circuit.add_gate(H(q[0]))
            elif op.gate == "x":
                circuit.add_gate(X(q[0]))
            elif op.gate == "y":
                circuit.add_gate(Y(q[0]))
            elif op.gate == "z":
                circuit.add_gate(Z(q[0]))
            elif op.gate == "s":
                circuit.add_gate(S(q[0]))
            elif op.gate == "t":
                circuit.add_gate(T(q[0]))
            elif op.gate in ("rx", "ry", "rz"):
                # Qulacs 的 R* 用 exp(+i θ/2 P) 约定，与 Qiskit/aicir 的
                # exp(-i θ/2 P) 差一个符号——不取负号会安静地算出共轭结果。
                factory = {"rx": RX, "ry": RY, "rz": RZ}[op.gate]
                circuit.add_gate(factory(q[0], -p[0]))
            elif op.gate == "cx":
                circuit.add_gate(CNOT(q[0], q[1]))
            elif op.gate == "cz":
                circuit.add_gate(CZ(q[0], q[1]))
            elif op.gate == "swap":
                circuit.add_gate(SWAP(q[0], q[1]))
            elif op.gate == "cp":
                matrix = np.diag([1.0, 1.0, 1.0, np.exp(1j * p[0])]).astype(np.complex128)
                circuit.add_gate(DenseMatrix([q[0], q[1]], matrix))
            else:  # pragma: no cover
                raise ValueError(f"Qulacs 适配器未覆盖门 {op.gate!r}")
        return circuit

    def run(self, circuit, spec: CircuitSpec):
        from qulacs import QuantumState

        state = QuantumState(spec.n_qubits)
        state.set_zero_state()
        circuit.update_quantum_state(state)
        return reverse_qubit_order(np.asarray(state.get_vector()), spec.n_qubits)

    def precision(self) -> str:
        # Qulacs 无单精度构建。
        return "complex128"
