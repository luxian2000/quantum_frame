"""统一测量结果对象（统一测量模型，见 README §4 与设计文档）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from ..core.state import State


@dataclass
class MeasureSpec:
    """一个线路内 measure 操作的登记项。"""
    op_index: int
    id: Optional[str]
    qubits: List[int]
    basis: str


def _reduced_density(rho_flat: np.ndarray, n: int, keep: Sequence[int]) -> np.ndarray:
    keep = list(keep)
    rho = np.asarray(rho_flat).reshape(1 << n, 1 << n).reshape([2] * (2 * n))
    traced = [q for q in range(n) if q not in set(keep)]
    perm = keep + traced + [n + q for q in keep] + [n + q for q in traced]
    m, k = len(keep), len(traced)
    t = np.transpose(rho, perm).reshape(1 << m, 1 << k, 1 << m, 1 << k)
    return np.einsum("akbk->ab", t)


@dataclass
class Result:
    n_qubits: int
    backend_name: str
    probabilities: np.ndarray
    shots: Optional[int] = None
    measurement_specs: List[MeasureSpec] = field(default_factory=list)
    incircuit_outputs: Dict[int, object] = field(default_factory=dict)
    incircuit_counts: Dict[int, Dict[int, int]] = field(default_factory=dict)
    terminal_output: Optional[np.ndarray] = None
    terminal_counts: Optional[Dict[str, int]] = None
    terminal_qubits: Optional[List[int]] = None
    state: Optional[State] = None
    final_state: Optional[State] = None
    final_state_kind: Optional[str] = None
    expectation_values: Dict[str, float] = field(default_factory=dict)
    expectation_variances: Dict[str, float] = field(default_factory=dict)
    snapshot_states: Dict[int, State] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def _resolve(self, target: Union[int, str]) -> int:
        if isinstance(target, str):
            for spec in self.measurement_specs:
                if spec.id == target:
                    return spec.op_index
            raise ValueError(f"未找到 id={target!r} 的 measure 操作")
        return int(target)

    def output(self, target: Union[int, str]):
        if target == -1:
            if self.terminal_output is None:
                raise ValueError("未执行末端测量：output(-1) 不可用（tm=False / measure_qubits=[] / shots∈{None,0}）")
            return self.terminal_output
        idx = self._resolve(target)
        if idx not in self.incircuit_outputs:
            raise ValueError(f"操作下标 {idx} 不是线路内 measure 操作")
        return self.incircuit_outputs[idx]

    def counts(self, target: Union[int, str]):
        if self.shots is None:
            raise RuntimeError("单轨迹模式（shots=None/0）不支持统计结果")
        if target == -1:
            if self.terminal_counts is None:
                raise ValueError("未执行末端测量：counts(-1) 不可用")
            return dict(self.terminal_counts)
        idx = self._resolve(target)
        if idx not in self.incircuit_counts:
            raise ValueError(f"操作下标 {idx} 不是线路内 measure 操作")
        return dict(self.incircuit_counts[idx])

    def prob(self, target: Union[int, str]):
        counts = self.counts(target)
        total = sum(counts.values()) or 1
        return {k: v / total for k, v in counts.items()}

    def snap(self, op_index: int) -> Optional[State]:
        return self.snapshot_states.get(int(op_index))

    def reduce(self, R: Sequence[int], pos: str = "final") -> np.ndarray:
        src = self.final_state if pos == "final" else self.state
        if src is None:
            raise ValueError(f"{pos} 态不可用，无法 reduce")
        arr = np.asarray(src)
        if arr.ndim == 1 or arr.shape[0] != arr.shape[1]:
            vec = arr.reshape(-1, 1)
            arr = vec @ vec.conj().T
        return _reduced_density(arr, self.n_qubits, list(R))

    def most_probable(self):
        idx = int(np.argmax(self.probabilities))
        return f"|{idx:0{self.n_qubits}b}>", float(self.probabilities[idx])

    def variance(self, name: str) -> Optional[float]:
        return None if name not in self.expectation_variances else float(self.expectation_variances[name])

    def stddev(self, name: str) -> Optional[float]:
        var = self.variance(name)
        return None if var is None else float(np.sqrt(max(var, 0.0)))

    def summary(self) -> str:
        peak, p = self.most_probable()
        lines = [f"Result(n_qubits={self.n_qubits}, backend={self.backend_name})", f"peak={peak}, prob={p:.6f}"]
        if self.shots is not None:
            lines.append(f"shots={self.shots}")
        if self.expectation_values:
            lines.append(f"expectations={self.expectation_values}")
        if self.expectation_variances:
            lines.append(f"variances={self.expectation_variances}")
        return " | ".join(lines)

    def __repr__(self) -> str:
        return self.summary()
