"""期望值估计 primitives：精确态向量路径与有限 shots 路径。"""

from __future__ import annotations

import numpy as np

from ..backends import NumpyBackend
from ..measure import Measure
from ..measure.estimator import PauliEstimator
from .base import BaseEstimator, normalize_run_inputs, pair_observables
from .results import EstimateResult


class StatevectorEstimator(BaseEstimator):
    """精确期望：演化态向量后计算 ``<psi|H|psi>``，无采样噪声。

    可观测量接受 :class:`~aicir.operators.Hamiltonian`（经
    ``to_matrix``）或现成的稠密矩阵。
    """

    def __init__(self, backend=None) -> None:
        self.backend = backend if backend is not None else NumpyBackend()

    def _expectation(self, circuit, observable) -> float:
        result = Measure(self.backend).run(circuit, shots=None, return_state=True)
        state = self.backend.cast(result.state.to_numpy())
        if hasattr(observable, "to_matrix"):
            matrix = observable.to_matrix(self.backend)
        else:
            matrix = self.backend.cast(np.asarray(observable))
        raw = self.backend.expectation_sv(state, matrix)
        return float(np.real(complex(raw)))

    def run(self, circuits, observables, *, shots: int | None = None, parameter_values=None):
        if shots is not None:
            raise ValueError("StatevectorEstimator 为精确路径，不接受 shots；请用 ShotEstimator")
        items, single = normalize_run_inputs(circuits, parameter_values)
        paired = pair_observables(items, observables)
        results = [
            EstimateResult(value=self._expectation(circuit, observable), metadata={"method": "statevector"})
            for circuit, observable in zip(items, paired)
        ]
        return results[0] if single else results


class _EnergyResult:
    """``BasicVQE(energy_estimator=...)`` 契约的最小载体：仅暴露 ``energy``。"""

    __slots__ = ("energy",)

    def __init__(self, energy: float) -> None:
        self.energy = float(energy)


class NoisyEstimator(BaseEstimator):
    """带噪声期望：把 ``noise_model`` 附加到线路，经密度矩阵模拟读取期望值。

    ``shots=None`` 给出确定性密度矩阵期望（仅退相干、无采样噪声）；``shots>=1``
    则叠加散粒统计。同样暴露 :meth:`estimate`，可作 ``BasicVQE`` 的
    ``energy_estimator`` 注入。
    """

    def __init__(self, noise_model, backend=None, *, shots: int | None = None) -> None:
        if noise_model is None:
            raise ValueError("NoisyEstimator 需要 noise_model=")
        self.noise_model = noise_model
        self.backend = backend if backend is not None else NumpyBackend()
        self.shots = shots

    def _expectation(self, circuit, observable, shots) -> float:
        circuit.noise_model = self.noise_model
        if hasattr(observable, "to_matrix"):
            matrix = observable.to_matrix(self.backend)
        else:
            matrix = self.backend.cast(np.asarray(observable))
        result = Measure(self.backend).run(
            circuit, shots=shots, observables={"H": matrix}, return_state=False
        )
        return float(np.real(complex(result.expectation_values["H"])))

    def estimate(self, circuit, hamiltonian, *, shots: int | None = None, **_ignored):
        """直通期望（BasicVQE energy_estimator 契约）。"""

        use_shots = self.shots if shots is None else shots
        return _EnergyResult(self._expectation(circuit, hamiltonian, use_shots))

    def run(self, circuits, observables, *, shots: int | None = None, parameter_values=None):
        use_shots = self.shots if shots is None else shots
        items, single = normalize_run_inputs(circuits, parameter_values)
        paired = pair_observables(items, observables)
        results = [
            EstimateResult(
                value=self._expectation(circuit, observable, use_shots),
                shots=use_shots,
                metadata={"method": "noisy_dm"},
            )
            for circuit, observable in zip(items, paired)
        ]
        return results[0] if single else results


class ShotEstimator(BaseEstimator):
    """有限 shots 能量估计：包装 :class:`~aicir.measure.estimator.PauliEstimator`
    （qubit-wise commuting 分组、基变换测量、shot 分配）。

    同时暴露 :meth:`estimate`（返回原始 ``PauliEstimateResult``），可直接
    作为 ``BasicVQE(energy_estimator=...)`` 注入。
    """

    def __init__(self, backend=None, *, shots: int = 1024, **pauli_estimator_kwargs) -> None:
        self._inner = PauliEstimator(backend, shots=shots, **pauli_estimator_kwargs)

    @property
    def backend(self):
        return self._inner.backend

    @property
    def shots(self) -> int:
        return self._inner.shots

    def estimate(self, circuit, hamiltonian, **kwargs):
        """直通底层 PauliEstimator（BasicVQE energy_estimator 契约）。"""

        return self._inner.estimate(circuit, hamiltonian, **kwargs)

    def run(self, circuits, observables, *, shots: int | None = None, parameter_values=None):
        items, single = normalize_run_inputs(circuits, parameter_values)
        paired = pair_observables(items, observables)
        results = []
        for circuit, observable in zip(items, paired):
            raw = self._inner.estimate(circuit, observable, shots=shots)
            results.append(
                EstimateResult(
                    value=float(raw.energy),
                    variance=float(raw.variance),
                    shots=int(raw.shots),
                    term_results=tuple(raw.term_results),
                    metadata={"method": "pauli_shots", "groups": raw.groups, **raw.metadata},
                )
            )
        return results[0] if single else results
