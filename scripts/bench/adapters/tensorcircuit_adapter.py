"""TensorCircuit 适配器。

TensorCircuit 走 ML 后端（NumPy/JAX/TensorFlow/PyTorch）。**JIT 预热必须排除**：
首次调用包含编译时间，计进去测的就不是执行速度。预热由 ``timing.time_callable``
的 ``warmup`` 统一处理，这里只需保证同一个可调用对象被重复调用。

比特序：TensorCircuit 的 ``state()`` 是大端，与本仓库一致，不翻转。
"""

from __future__ import annotations

import numpy as np

from ..core.spec import CircuitSpec
from .base import Adapter

__all__ = ["TensorCircuitAdapter"]


class TensorCircuitAdapter(Adapter):
    name = "tensorcircuit"

    def __init__(self, backend: str = "numpy", dtype: str = "complex128"):
        self._backend = backend
        self._dtype = dtype

    @classmethod
    def is_available(cls) -> bool:
        try:
            import tensorcircuit  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def version(cls) -> str:
        import tensorcircuit

        return getattr(tensorcircuit, "__version__", "unknown")

    def build(self, spec: CircuitSpec):
        import tensorcircuit as tc

        tc.set_backend(self._backend)
        tc.set_dtype(self._dtype)

        circuit = tc.Circuit(spec.n_qubits)
        for op in spec.operations:
            q = op.qubits
            p = op.params
            if op.gate == "h":
                circuit.h(q[0])
            elif op.gate == "x":
                circuit.x(q[0])
            elif op.gate == "y":
                circuit.y(q[0])
            elif op.gate == "z":
                circuit.z(q[0])
            elif op.gate == "s":
                circuit.s(q[0])
            elif op.gate == "t":
                circuit.t(q[0])
            elif op.gate == "rx":
                circuit.rx(q[0], theta=p[0])
            elif op.gate == "ry":
                circuit.ry(q[0], theta=p[0])
            elif op.gate == "rz":
                circuit.rz(q[0], theta=p[0])
            elif op.gate == "cx":
                circuit.cnot(q[0], q[1])
            elif op.gate == "cz":
                circuit.cz(q[0], q[1])
            elif op.gate == "cp":
                circuit.cphase(q[0], q[1], phi=p[0])
            elif op.gate == "swap":
                circuit.swap(q[0], q[1])
            else:  # pragma: no cover
                raise ValueError(f"TensorCircuit 适配器未覆盖门 {op.gate!r}")
        return circuit

    def run(self, circuit, spec: CircuitSpec):
        # TensorCircuit 已是大端，无需翻转。
        return np.asarray(circuit.state()).reshape(-1)

    def precision(self) -> str:
        return self._dtype
