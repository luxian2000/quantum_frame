"""面向真实硬件/远端服务的扩展点 primitives（NEXT.md 第 4 节）。

本仓不内置任何 QPU 或远端后端，因此这里提供的是**注入式扩展点**：把用户
传入的 ``runner`` 可调用对象包装成统一的 Sampler/Estimator 接口，由 runner
负责真正的执行（向硬件/远端排队、轮询、读取结果）。这样上层算法只依赖
primitives，而把"在哪执行"留给注入的 runner。

runner 契约：

- 采样：``runner(circuit, *, shots) -> Mapping[str, int]``（counts），或直接返回
  现成的 :class:`SampleResult`。
- 估计：``runner(circuit, observable, *, shots) -> float``（期望值），或直接返回
  现成的 :class:`EstimateResult`。
"""

from __future__ import annotations

from typing import Any, Callable

from .base import BaseEstimator, BaseSampler, normalize_run_inputs, pair_observables
from .results import EstimateResult, SampleResult


class BackendSampler(BaseSampler):
    """把采样 ``runner`` 包装成统一 Sampler。"""

    def __init__(self, runner: Callable[..., Any], *, shots: int = 1024) -> None:
        if not callable(runner):
            raise TypeError("runner 必须可调用：runner(circuit, *, shots)")
        self.runner = runner
        self.shots = int(shots)

    def _wrap(self, raw: Any, shots: int) -> SampleResult:
        if isinstance(raw, SampleResult):
            return raw
        counts = {str(k): int(v) for k, v in dict(raw).items()}
        return SampleResult(
            counts=counts, probs={}, shots=shots, measured_qubits=(), metadata={"method": "backend"}
        )

    def run(self, circuits, *, shots: int | None = None, parameter_values=None):
        items, single = normalize_run_inputs(circuits, parameter_values)
        use_shots = self.shots if shots is None else int(shots)
        results = [self._wrap(self.runner(circuit, shots=use_shots), use_shots) for circuit in items]
        return results[0] if single else results


class BackendEstimator(BaseEstimator):
    """把期望值 ``runner`` 包装成统一 Estimator。"""

    def __init__(self, runner: Callable[..., Any], *, shots: int | None = None) -> None:
        if not callable(runner):
            raise TypeError("runner 必须可调用：runner(circuit, observable, *, shots)")
        self.runner = runner
        self.shots = shots

    def _wrap(self, raw: Any, shots: int | None) -> EstimateResult:
        if isinstance(raw, EstimateResult):
            return raw
        return EstimateResult(value=float(raw), shots=shots, metadata={"method": "backend"})

    def run(self, circuits, observables, *, shots: int | None = None, parameter_values=None):
        use_shots = self.shots if shots is None else shots
        items, single = normalize_run_inputs(circuits, parameter_values)
        paired = pair_observables(items, observables)
        results = [
            self._wrap(self.runner(circuit, observable, shots=use_shots), use_shots)
            for circuit, observable in zip(items, paired)
        ]
        return results[0] if single else results
