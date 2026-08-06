"""Cirq 适配器。

Cirq 的 ``final_state_vector`` 已是大端（``LineQubit(0)`` 为最高位），与本仓库
约定一致，因此**不**做比特序翻转——这与 Qiskit 适配器正好相反，也是 parity
测试值得存在的又一个理由。
"""

from __future__ import annotations

import numpy as np

from ..core.spec import CircuitSpec
from .base import Adapter

__all__ = ["CirqAdapter"]


class CirqAdapter(Adapter):
    name = "cirq"

    def __init__(self, dtype=np.complex128):
        self._dtype = dtype

    @classmethod
    def is_available(cls) -> bool:
        try:
            import cirq  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def version(cls) -> str:
        import cirq

        return cirq.__version__

    def build(self, spec: CircuitSpec):
        import cirq

        qubits = cirq.LineQubit.range(spec.n_qubits)
        moments = []
        for op in spec.operations:
            q = [qubits[i] for i in op.qubits]
            p = op.params
            if op.gate == "h":
                moments.append(cirq.H(q[0]))
            elif op.gate == "x":
                moments.append(cirq.X(q[0]))
            elif op.gate == "y":
                moments.append(cirq.Y(q[0]))
            elif op.gate == "z":
                moments.append(cirq.Z(q[0]))
            elif op.gate == "s":
                moments.append(cirq.S(q[0]))
            elif op.gate == "t":
                moments.append(cirq.T(q[0]))
            elif op.gate == "rx":
                moments.append(cirq.rx(p[0])(q[0]))
            elif op.gate == "ry":
                moments.append(cirq.ry(p[0])(q[0]))
            elif op.gate == "rz":
                moments.append(cirq.rz(p[0])(q[0]))
            elif op.gate == "cx":
                moments.append(cirq.CNOT(q[0], q[1]))
            elif op.gate == "cz":
                moments.append(cirq.CZ(q[0], q[1]))
            elif op.gate == "cp":
                # cirq 的 CZPowGate 以 π 为单位：exponent = θ/π
                moments.append(cirq.CZPowGate(exponent=p[0] / np.pi)(q[0], q[1]))
            elif op.gate == "swap":
                moments.append(cirq.SWAP(q[0], q[1]))
            else:  # pragma: no cover
                raise ValueError(f"Cirq 适配器未覆盖门 {op.gate!r}")
        return cirq.Circuit(moments)

    def run(self, circuit, spec: CircuitSpec):
        import cirq

        simulator = cirq.Simulator(dtype=self._dtype)
        result = simulator.simulate(circuit, qubit_order=cirq.LineQubit.range(spec.n_qubits))
        # Cirq 已是大端，无需翻转。
        return np.asarray(result.final_state_vector)

    def precision(self) -> str:
        return np.dtype(self._dtype).name
