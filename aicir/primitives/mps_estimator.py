"""MPS 近似期望估计 primitive：经 bond 截断的 MPS 求 <psi|H|psi>。"""

from __future__ import annotations

import numpy as np

from ..backends import NumpyBackend
from .base import BaseEstimator, normalize_run_inputs, pair_observables
from .estimator import _EnergyResult
from .results import EstimateResult


class MPSEstimator(BaseEstimator):
    """MPS 近似期望：bond 截断的纯态演化后求期望，无采样噪声。

    ``max_bond_dim`` 越大越接近精确（None 表示无硬上限）；``cutoff`` 为相对
    奇异值截断阈值。暴露 :meth:`estimate` 供 ``BasicVQE(energy_estimator=...)`` 注入。
    """

    def __init__(self, *, max_bond_dim=None, cutoff=1e-10, backend=None) -> None:
        self.backend = backend if backend is not None else NumpyBackend()
        self.max_bond_dim = max_bond_dim
        self.cutoff = float(cutoff)

    def _expectation(self, circuit, observable):
        from ..simulator import mps_statevector
        from ..simulator.mps import _expectation_from_mps

        # 只构建一次 MPS，复用同一份张量求期望，避免重复演化电路
        mps = mps_statevector(circuit, max_bond_dim=self.max_bond_dim, cutoff=self.cutoff, backend=self.backend)
        raw = _expectation_from_mps(mps, observable, self.backend)
        value = float(np.real(complex(self.backend.to_numpy(raw) if hasattr(raw, "shape") else raw)))
        return value, float(mps.truncation_error)

    def estimate(self, circuit, hamiltonian, **_ignored):
        value, _err = self._expectation(circuit, hamiltonian)
        return _EnergyResult(value)

    def run(self, circuits, observables, *, shots=None, parameter_values=None):
        if shots is not None:
            raise ValueError("MPSEstimator 为（近似）精确路径，不接受 shots")
        items, single = normalize_run_inputs(circuits, parameter_values)
        paired = pair_observables(items, observables)
        results = []
        for circuit, observable in zip(items, paired):
            value, err = self._expectation(circuit, observable)
            results.append(
                EstimateResult(
                    value=value,
                    metadata={"method": "mps", "max_bond_dim": self.max_bond_dim, "truncation_error": err},
                )
            )
        return results[0] if single else results
