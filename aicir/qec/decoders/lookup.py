"""查表解码器：枚举权重 ≤ t 的错误，建 syndrome → 最小权修正表。

**与「承重约束」的边界**：约束的是**运行器交给解码器什么**——运行器只经
reset(layout) + update(round, events) 传递信息。解码器作者可以在**构造时**注入
额外知识；LookupDecoder 就是构造时吃进 code 来建表的。代价是它只能用于构造时
那个码，而只依赖 layout 的解码器可以跑任何码、包括 M2 里 Stim 采样的事件流。

M2 的 DEM 正是用来消除这个特例的。M1 的构造时注入是**已知的临时做法**，
不应被后续实现当作范例推广。
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np

from ..code import pauli_to_gf2
from . import DecodeStep, register_decoder


class LookupDecoder:
    """逐轮即时提交的查表解码器。"""

    name = "lookup"
    window = 1
    commit_lag = 0

    def __init__(self, code, t: int | None = None, error_basis: str = "XYZ"):
        self._code = code
        self._basis = "".join(sorted(set(str(error_basis).upper())))
        if not self._basis or any(ch not in "XYZ" for ch in self._basis):
            raise ValueError(f"error_basis 只能由 X/Y/Z 组成，收到 {error_basis!r}")
        if t is None:
            t = (code.distance() - 1) // 2
            if t < 1:
                raise ValueError(
                    f"[{code.name}] 由 distance()={code.distance()} 推出的 t={t} 无意义"
                    f"（重复码的 distance() 是 1，它对未受保护的基毫无防护）。"
                    f"请显式传 t，例如 LookupDecoder(code, t=1, error_basis='X')"
                )
        self._t = int(t)
        self._table: dict[bytes, np.ndarray] = {}
        self._layout = None
        self._committed = -1

    def _build_table(self) -> None:
        """按权重升序枚举，先到先得 → 表中即为最小权修正。"""
        code = self._code
        self._table = {np.zeros(code.m, dtype=np.uint8).tobytes():
                       np.zeros(2 * code.n, dtype=np.uint8)}
        for weight in range(1, self._t + 1):
            for support in combinations(range(code.n), weight):
                for labels in product(self._basis, repeat=weight):
                    chars = ["I"] * code.n
                    for q, ch in zip(support, labels):
                        chars[q] = ch
                    err = pauli_to_gf2("".join(chars), code.n)
                    key = code.syndrome(err).tobytes()
                    if key not in self._table:
                        self._table[key] = err

    def reset(self, layout) -> None:
        self._layout = layout
        self._committed = -1
        if not self._table:
            self._build_table()

    def correction_for_syndrome(self, syndrome) -> np.ndarray:
        """综合征 → 修正 Pauli 的 (2n,) 向量；查不到时返回恒等（放弃修正）。"""
        key = np.asarray(syndrome, dtype=np.uint8).tobytes()
        found = self._table.get(key)
        return found.copy() if found is not None else np.zeros(2 * self._code.n, dtype=np.uint8)

    def cost_of(self, round_index: int, events) -> float:
        """声明代价：一次哈希查表，记为 1 个单位。"""
        return 1.0

    def update(self, round_index: int, events) -> DecodeStep:
        correction = self.correction_for_syndrome(events)
        self._committed = int(round_index)
        return DecodeStep(
            frame_flips=self._code.logical_class(correction).ravel().copy(),
            corrections=_to_gate_list(correction, self._code.n),
            committed_through=self._committed,
            cost=self.cost_of(round_index, events),
        )

    def flush(self) -> DecodeStep:
        """逐轮即时提交 → flush 无未决可提交。"""
        return DecodeStep(frame_flips=None, corrections=None,
                          committed_through=self._committed, cost=0.0)


def _to_gate_list(correction: np.ndarray, n: int) -> list:
    """(2n,) 修正向量 → [(qubit, pauli), ...] 门列表。"""
    out = []
    for q in range(n):
        x, z = int(correction[q]), int(correction[n + q])
        if x and z:
            out.append((q, "Y"))
        elif x:
            out.append((q, "X"))
        elif z:
            out.append((q, "Z"))
    return out


register_decoder("lookup", LookupDecoder)
