"""
quantum_sim/measure/result.py

统一测量结果对象：承载概率分布、采样计数、期望值以及末态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class Result:
    """一次电路测量的统一结果容器。"""

    n_qubits: int
    backend_name: str
    probabilities: np.ndarray
    counts: Optional[Dict[str, int]] = None
    shots: Optional[int] = None
    expectation_values: Dict[str, float] = field(default_factory=dict)
    expectation_variances: Dict[str, float] = field(default_factory=dict)
    final_state: Optional[np.ndarray] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def most_probable(self):
        idx = int(np.argmax(self.probabilities))
        bitstr = f"|{idx:0{self.n_qubits}b}>"
        return bitstr, float(self.probabilities[idx])

    def variance(self, observable_name: str) -> Optional[float]:
        if observable_name not in self.expectation_variances:
            return None
        return float(self.expectation_variances[observable_name])

    def stddev(self, observable_name: str) -> Optional[float]:
        var = self.variance(observable_name)
        if var is None:
            return None
        return float(np.sqrt(max(var, 0.0)))

    def summary(self) -> str:
        peak_state, peak_prob = self.most_probable()
        lines = [
            f"Result(n_qubits={self.n_qubits}, backend={self.backend_name})",
            f"peak={peak_state}, prob={peak_prob:.6f}",
        ]
        if self.shots is not None:
            lines.append(f"shots={self.shots}")
        if self.expectation_values:
            lines.append(f"expectations={self.expectation_values}")
        if self.expectation_variances:
            lines.append(f"variances={self.expectation_variances}")
        return " | ".join(lines)

    def __repr__(self) -> str:
        return self.summary()