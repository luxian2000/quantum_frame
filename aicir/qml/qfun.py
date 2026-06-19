"""PennyLane 风格轻量量子函数 ``qfun``（NEXT.md §5）。

把"量子函数 + 设备 + 测量 + 梯度"统一为一个可调用对象：用户函数构造并
返回一个 :class:`~aicir.core.circuit.Circuit`，观测量在装饰器上声明；调用得
期望值，``.grad`` 得梯度（梯度方法经 §6 的 ``aicir.qml.diff`` 注册表分发）。

```python
from aicir import Circuit, Hamiltonian, ry
from aicir.qml import qfun

H = Hamiltonian([("Z", 1.0)])

@qfun(device="numpy", differential="psr", observable=H)
def cost(theta):
    c = Circuit(n_qubits=1)
    c.append(ry(theta, 0))
    return c

value = cost(0.1)        # 期望值 <H>
grad = cost.grad(0.1)    # 梯度
```

约定：函数体**返回 Circuit**（不依赖全局 tape），观测量由 ``observable=``
声明；单个可训练位置参数（标量或一维数组）。
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import numpy as np

from ..core.circuit import Circuit
from ..ir import circuit_gate_dicts
from ..measure import Measure
from .diff import resolve_diff, select_diff


def _make_backend(device: Any):
    dev = str(device).lower()
    if dev in {"numpy", "cpu"}:
        from ..backends.numpy_backend import NumpyBackend

        return NumpyBackend()
    if dev in {"gpu", "torch"}:
        from ..backends.gpu_backend import GPUBackend

        return GPUBackend()
    if dev == "npu":
        from ..backends.npu_backend import NPUBackend

        return NPUBackend()
    raise ValueError(f"unknown device {device!r}; choose from numpy/gpu/npu")


class QFun:
    """绑定了设备、观测量与梯度方法的量子函数。由 :func:`qfun` 构造。"""

    def __init__(
        self,
        fn: Callable[[Any], Circuit],
        *,
        device: Any = "numpy",
        differential: str = "psr",
        observable: Any = None,
        shots: Any = None,
    ) -> None:
        if observable is None:
            raise ValueError("qfun 需要 observable=（如 Hamiltonian）")
        self._fn = fn
        self.device = device
        self.differential = differential
        self.observable = observable
        self.shots = shots
        self._backend = _make_backend(device)
        functools.update_wrapper(self, fn)

    def _circuit(self, param: Any) -> Circuit:
        circuit = self._fn(param)
        if not isinstance(circuit, Circuit):
            raise TypeError("qfun 函数体必须返回 Circuit")
        if circuit.parameters:
            names = ", ".join(p.name for p in circuit.parameters)
            raise ValueError(f"返回线路含未绑定参数: {names}")
        if circuit.backend is None:
            circuit = Circuit(
                *circuit_gate_dicts(circuit), n_qubits=circuit.n_qubits, backend=self._backend
            )
        return circuit

    def _energy(self, param: Any) -> float:
        circuit = self._circuit(param)
        backend = circuit.backend
        observable = self.observable.to_matrix(backend)
        measurement = Measure(backend).run(
            circuit, shots=self.shots, observables={"H": observable}, return_state=False
        )
        return float(measurement.expectation_values["H"])

    def __call__(self, param: Any) -> float:
        return self._energy(param)

    def _gradient_fn(self) -> Callable[..., Any]:
        name = str(self.differential).lower()
        if name == "auto":
            name = select_diff(backend=self._backend, shots=self.shots, noisy=False)
        return resolve_diff(name)

    def grad(self, param: Any) -> Any:
        x = np.asarray(param, dtype=float)
        scalar = x.ndim == 0
        grad_fn = self._gradient_fn()

        def energy(p: np.ndarray) -> float:
            arg = float(p) if (p.ndim == 0 or p.size == 1) else p
            return self._energy(arg)

        g = np.asarray(grad_fn(energy, x), dtype=float)
        return float(g) if scalar else g


def qfun(
    *,
    device: Any = "numpy",
    differential: str = "psr",
    observable: Any = None,
    shots: Any = None,
) -> Callable[[Callable[[Any], Circuit]], QFun]:
    """装饰器：把"返回 Circuit 的量子函数"包成带 ``.grad`` 的 :class:`QFun`。"""

    def deco(fn: Callable[[Any], Circuit]) -> QFun:
        return QFun(fn, device=device, differential=differential, observable=observable, shots=shots)

    return deco
