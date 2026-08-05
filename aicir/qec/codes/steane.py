"""Steane [[7,1,3]] CSS 码，基于 [7,4,3] Hamming 码的校验矩阵。

校验行 h1={3,4,5,6}、h2={1,2,5,6}、h3={0,2,4,6}；任意两行交集均为偶数，
故 X 型与 Z 型生成元逐对对易（CSS 条件）。
"""

from __future__ import annotations

from ..code import StabilizerCode
from . import register_code

X_GENERATORS = ["IIIXXXX", "IXXIIXX", "XIXIXIX"]
Z_GENERATORS = ["IIIZZZZ", "IZZIIZZ", "ZIZIZIZ"]


def build() -> StabilizerCode:
    return StabilizerCode.from_paulis(
        X_GENERATORS + Z_GENERATORS,
        logical_x=["XXXXXXX"], logical_z=["ZZZZZZZ"], name="steane",
        coords={q: (0, q) for q in range(7)},
    )


register_code("steane", build)
