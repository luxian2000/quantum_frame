"""[[5,1,3]] 完美码。生成元为 XZZXI 的四个循环移位。"""

from __future__ import annotations

from ..code import StabilizerCode
from . import register_code

GENERATORS = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]


def build() -> StabilizerCode:
    return StabilizerCode.from_paulis(
        GENERATORS, logical_x=["XXXXX"], logical_z=["ZZZZZ"], name="five_qubit",
        coords={q: (0, q) for q in range(5)},
    )


register_code("five_qubit", build)
