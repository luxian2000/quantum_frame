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
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..core.circuit import Circuit
from ..ir import circuit_gate_dicts
from ..measure import Measure
from .diff import resolve_diff, select_diff


# ── 测量返回构造器（§5）：函数体可返回这些对象表达测量意图，替代/补充装饰器
# observable=。不依赖全局 tape，故显式携带 circuit。 ────────────────────────


@dataclass(frozen=True)
class Expval:
    """期望值测量：``<observable>``（可微）。"""

    circuit: Circuit
    observable: Any


@dataclass(frozen=True)
class Probs:
    """概率向量测量；``wires=None`` 取整寄存器。"""

    circuit: Circuit
    wires: Any = None


@dataclass(frozen=True)
class Sample:
    """采样 counts 测量；shots 取自装饰器。``wires=None`` 取整寄存器。"""

    circuit: Circuit
    wires: Any = None


def expval(circuit: Circuit, observable: Any) -> Expval:
    """函数体返回构造器：对 ``circuit`` 测 ``observable`` 期望值。"""
    return Expval(circuit, observable)


def probs(circuit: Circuit, wires: Any = None) -> Probs:
    """函数体返回构造器：``circuit`` 末态概率向量（可限制到 ``wires``）。"""
    return Probs(circuit, wires)


def sample(circuit: Circuit, wires: Any = None) -> Sample:
    """函数体返回构造器：``circuit`` 采样 counts（需装饰器 ``shots=``）。"""
    return Sample(circuit, wires)


_Measurement = (Expval, Probs, Sample)


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
        noise_model: Any = None,
    ) -> None:
        # observable 可选：函数体可经 expval(...) 自带观测量。仅当函数体返回裸
        # Circuit 且无装饰器 observable 时，于调用期报错（见 _resolve）。
        if observable is None:
            self._multi = False
            self._observables = None
        else:
            self._multi = isinstance(observable, (list, tuple))
            observables = list(observable) if self._multi else [observable]
            if not observables or any(o is None for o in observables):
                raise ValueError("observable= 不能含 None")
            self._observables = observables
        self._fn = fn
        self.device = device
        self.differential = differential
        self.observable = observable
        self.shots = shots
        self.noise_model = noise_model
        self._backend = _make_backend(device)
        functools.update_wrapper(self, fn)

    def _prepare(self, circuit: Circuit) -> Circuit:
        if not isinstance(circuit, Circuit):
            raise TypeError("qfun 函数体必须返回 Circuit 或 expval/probs/sample 测量对象")
        if circuit.parameters:
            names = ", ".join(p.name for p in circuit.parameters)
            raise ValueError(f"返回线路含未绑定参数: {names}")
        if circuit.backend is None:
            circuit = Circuit(
                *circuit_gate_dicts(circuit), n_qubits=circuit.n_qubits, backend=self._backend
            )
        if self.noise_model is not None:
            # 噪声路径：附加到线路，由 Measure.run 走密度矩阵模拟读取
            circuit.noise_model = self.noise_model
        return circuit

    def _resolve(self, param: Any) -> tuple[str, Circuit, list | None, Any, bool]:
        """解析函数体返回值为 ``(kind, circuit, observables, wires, multi)``。"""
        out = self._fn(param)
        if isinstance(out, Expval):
            return "expval", self._prepare(out.circuit), [out.observable], None, False
        if isinstance(out, Probs):
            return "probs", self._prepare(out.circuit), None, out.wires, False
        if isinstance(out, Sample):
            return "sample", self._prepare(out.circuit), None, out.wires, False
        if isinstance(out, Circuit):
            if self._observables is None:
                raise ValueError("返回裸 Circuit 时需装饰器 observable=（或函数体返回 expval(...)）")
            return "expval", self._prepare(out), self._observables, None, self._multi
        raise TypeError("qfun 函数体必须返回 Circuit 或 expval/probs/sample 测量对象")

    def _expectations(self, circuit: Circuit, observables: list) -> np.ndarray:
        """对 ``observables`` 求期望值，返回形如 ``(n_obs,)`` 的数组。"""
        backend = circuit.backend
        obs = {f"H{i}": o.to_matrix(backend) for i, o in enumerate(observables)}
        measurement = Measure(backend).run(
            circuit, shots=self.shots, observables=obs, return_state=False
        )
        return np.array(
            [float(measurement.expectation_values[f"H{i}"]) for i in range(len(observables))],
            dtype=float,
        )

    def _probs(self, circuit: Circuit, wires: Any) -> np.ndarray:
        result = Measure(circuit.backend).run(circuit, shots=None, return_state=False)
        full = np.asarray(result.probabilities, dtype=float).reshape(-1)
        if wires is None:
            return full
        return _marginalize(full, list(wires), int(circuit.n_qubits))

    def _sample(self, circuit: Circuit, wires: Any) -> dict:
        if self.shots is None:
            raise ValueError("sample(...) 需装饰器 shots=")
        qubits = list(wires) if wires is not None else list(range(int(circuit.n_qubits)))
        result = Measure(circuit.backend).run(
            circuit, shots=self.shots, measure_qubits=qubits, return_state=False
        )
        return result.counts(-1)

    def __call__(self, param: Any) -> Any:
        kind, circuit, observables, wires, multi = self._resolve(param)
        if kind == "probs":
            return self._probs(circuit, wires)
        if kind == "sample":
            return self._sample(circuit, wires)
        values = self._expectations(circuit, observables)
        return values if multi else float(values[0])

    def _gradient_fn(self) -> Callable[..., Any]:
        name = str(self.differential).lower()
        if name == "auto":
            noisy = self.noise_model is not None
            name = select_diff(backend=self._backend, shots=self.shots, noisy=noisy)
        return resolve_diff(name)

    def _probs_jacobian(self, x: np.ndarray, scalar: bool) -> np.ndarray:
        """probs 返回的参数移位 Jacobian。

        每个基态概率 p_i = <ψ|i><i|ψ> 是投影算符期望，故对整条概率向量施同一
        参数移位规则即得 (D, P) Jacobian（标量参为 (D,)）。逐输出分量复用注册表
        梯度方法（``self.differential``，单一事实来源），并用按参数元组缓存的
        ``_probs`` 使 D 个分量共享同一批底层线路求值（psr 下共 2P 次）。
        """
        grad_fn = self._gradient_fn()
        cache: dict[tuple, np.ndarray] = {}

        def probs_at(p: np.ndarray) -> np.ndarray:
            arg = float(p) if np.ndim(p) == 0 else np.asarray(p, dtype=float)
            key = tuple(np.round(np.atleast_1d(np.asarray(arg, dtype=float)).reshape(-1), 12))
            cached = cache.get(key)
            if cached is None:
                _, circuit, _, wires, _ = self._resolve(arg)
                cached = self._probs(circuit, wires)
                cache[key] = cached
            return cached

        dim = int(probs_at(x).shape[0])
        rows = [
            np.asarray(grad_fn(lambda p, i=i: float(probs_at(p)[i]), x), dtype=float)
            for i in range(dim)
        ]
        # 标量参：每行为标量 → (D,)；向量参：每行 (P,) → (D, P)。
        return np.stack(rows, axis=0)

    def grad(self, param: Any) -> Any:
        x = np.asarray(param, dtype=float)
        scalar = x.ndim == 0
        kind, _, observables, _, multi = self._resolve(float(x) if scalar else x)
        if kind == "sample":
            raise ValueError(".grad 不支持 sample 返回（离散采样无梯度）")
        if kind == "probs":
            return self._probs_jacobian(x, scalar)
        n_obs = len(observables)
        grad_fn = self._gradient_fn()

        def energy_of(index: int) -> Callable[[np.ndarray], float]:
            def energy(p: np.ndarray) -> float:
                # 0 维（标量参）喂标量给用户函数；1 维数组按原样传，保留下标语义。
                arg = float(p) if p.ndim == 0 else p
                _, circuit, obs, _, _ = self._resolve(arg)
                return float(self._expectations(circuit, obs)[index])

            return energy

        if not multi:
            g = np.asarray(grad_fn(energy_of(0), x), dtype=float)
            return float(g) if scalar else g

        # 多观测量：逐观测量求梯度，按 n_obs 堆叠成 Jacobian。
        # 标量参 → (n_obs,)；向量参 → (n_obs, n_param)。
        rows = [np.asarray(grad_fn(energy_of(i), x), dtype=float) for i in range(n_obs)]
        return np.stack(rows, axis=0)


def _marginalize(probs: np.ndarray, wires: list[int], n_qubits: int) -> np.ndarray:
    """把整寄存器概率向量边缘化到 ``wires`` 子集（按 wires 顺序定位元）。"""
    out = np.zeros(1 << len(wires), dtype=float)
    for index, p in enumerate(probs):
        key = 0
        for wire in wires:
            bit = (index >> (n_qubits - 1 - int(wire))) & 1
            key = (key << 1) | bit
        out[key] += float(p)
    return out


def qfun(
    *,
    device: Any = "numpy",
    differential: str = "psr",
    observable: Any = None,
    shots: Any = None,
    noise_model: Any = None,
) -> Callable[[Callable[[Any], Circuit]], QFun]:
    """装饰器：把"返回 Circuit 的量子函数"包成带 ``.grad`` 的 :class:`QFun`。"""

    def deco(fn: Callable[[Any], Circuit]) -> QFun:
        return QFun(
            fn,
            device=device,
            differential=differential,
            observable=observable,
            shots=shots,
            noise_model=noise_model,
        )

    return deco
