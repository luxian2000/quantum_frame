"""Shor [[9,1,3]] 级联码：三组三比特相位块，块内 Z 型、块间 X 型。"""

from __future__ import annotations

from ..code import StabilizerCode
from . import register_code

Z_GENERATORS = [
    "ZZIIIIIII", "IZZIIIIII",   # 块 0
    "IIIZZIIII", "IIIIZZIII",   # 块 1
    "IIIIIIZZI", "IIIIIIIZZ",   # 块 2
]
X_GENERATORS = ["XXXXXXIII", "IIIXXXXXX"]


def build() -> StabilizerCode:
    return StabilizerCode.from_paulis(
        Z_GENERATORS + X_GENERATORS,
        logical_x=["XXXXXXXXX"], logical_z=["ZZZZZZZZZ"], name="shor",
        coords={q: (q // 3, q % 3) for q in range(9)},
    )


register_code("shor", build)
