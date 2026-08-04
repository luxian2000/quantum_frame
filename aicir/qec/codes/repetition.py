"""重复码：n=d，d−1 个两体生成元。

basis="Z"：稳定子为 Z_iZ_{i+1}，保护 X（bit-flip）错误。
basis="X"：稳定子为 X_iX_{i+1}，保护 Z（phase-flip）错误。

注意：重复码的**真实码距是 1**——basis="Z" 时 logical_z=Z₀ 权重为 1，
该码对 Z 错误无任何保护。参数 d 是它针对受保护基的有效距离。
"""

from __future__ import annotations

from ..code import StabilizerCode
from . import register_code


def build(d: int = 3, basis: str = "Z") -> StabilizerCode:
    d = int(d)
    if d < 3 or d % 2 == 0:
        raise ValueError(f"重复码的 d 必须是 ≥3 的奇数，收到 {d}")
    basis = str(basis).strip().upper()
    if basis not in ("Z", "X"):
        raise ValueError(f"basis 只支持 'Z' 或 'X'，收到 {basis!r}")

    gens = []
    for i in range(d - 1):
        chars = ["I"] * d
        chars[i] = chars[i + 1] = basis
        gens.append("".join(chars))

    if basis == "Z":
        logical_x, logical_z = [ "X" * d ], [ "Z" + "I" * (d - 1) ]
    else:
        logical_x, logical_z = [ "X" + "I" * (d - 1) ], [ "Z" * d ]

    return StabilizerCode.from_paulis(
        gens, logical_x=logical_x, logical_z=logical_z,
        name=f"repetition_d{d}_{basis}",
        coords={q: (0, q) for q in range(d)},
    )


register_code("repetition", build)
