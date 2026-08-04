"""旋转表面码 d=3，9 个 data 比特按行主序排在 3x3 网格上：

    q0 q1 q2
    q3 q4 q5
    q6 q7 q8

X 型稳定子：X(0,1,3,4)、X(4,5,7,8)、X(2,5)、X(3,6)
Z 型稳定子：Z(1,2,4,5)、Z(3,4,6,7)、Z(0,1)、Z(7,8)
逻辑 X = X(0,1,2)（顶行），逻辑 Z = Z(0,3,6)（左列）。
所有 X/Z 生成元对的交集大小均为偶数，逐对已验证对易。
"""

from __future__ import annotations

from ..code import StabilizerCode
from . import register_code

X_SUPPORTS = [(0, 1, 3, 4), (4, 5, 7, 8), (2, 5), (3, 6)]
Z_SUPPORTS = [(1, 2, 4, 5), (3, 4, 6, 7), (0, 1), (7, 8)]


def _label(support, pauli: str, n: int = 9) -> str:
    chars = ["I"] * n
    for q in support:
        chars[q] = pauli
    return "".join(chars)


def build(d: int = 3) -> StabilizerCode:
    if int(d) != 3:
        raise ValueError(f"M1 只内置 d=3 的旋转表面码，收到 d={d}")
    gens = [_label(s, "X") for s in X_SUPPORTS] + [_label(s, "Z") for s in Z_SUPPORTS]
    return StabilizerCode.from_paulis(
        gens,
        logical_x=[_label((0, 1, 2), "X")],
        logical_z=[_label((0, 3, 6), "Z")],
        name="surface_d3",
        coords={q: (q // 3, q % 3) for q in range(9)},
    )


register_code("surface", build)
