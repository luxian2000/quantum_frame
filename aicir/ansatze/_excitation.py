"""UCCSD 激发的精确电路 builder（fermionic-SWAP 网络）。

非相邻 orbital 间的费米子激发需处理 JW Z-string。用 fSWAP = CZ·SWAP（JW 一致的
费米子交换，在 |11⟩ 上带 −1）把远端 orbital 逐格搬到与近端相邻，施加既有相邻激发
gate，再逐格搬回。参数原样流入既有 gate，不做任何算术；只用既有 gate，不引入新类型。
"""

from __future__ import annotations

from ..core.circuit import cz, double_excitation, single_excitation, swap


def fswap_ops(i: int, j: int) -> list:
    """fSWAP(i, j) = CZ·SWAP：JW 一致的费米子交换。"""

    return [cz(j, [i]), swap(i, j)]


def _bring_adjacent(op_list: list, high: int, target: int) -> int:
    """把 qubit ``high`` 逐格 fswap 到 ``target``（target < high），返回它现在的位置。"""

    pos = high
    while pos > target:
        op_list.extend(fswap_ops(pos - 1, pos))
        pos -= 1
    return pos


def _undo_adjacent(op_list: list, low: int, high: int) -> None:
    """还原 ``_bring_adjacent(op_list, high, low)`` 的搬运。"""

    for pos in range(low, high):
        op_list.extend(fswap_ops(pos, pos + 1))


def single_excitation_ops(param, p: int, q: int) -> list:
    """orbital p<q 间单激发的 op 列表。"""

    if p >= q:
        raise ValueError(f"single excitation 需 p<q，得到 p={p}, q={q}")
    if q == p + 1:
        return [single_excitation(param, p, q)]
    ops: list = []
    _bring_adjacent(ops, q, p + 1)          # q 搬到 p+1
    ops.append(single_excitation(param, p, p + 1))
    _undo_adjacent(ops, p + 1, q)
    return ops


def _local_permute_ops(base: int, current: list, target: list) -> list:
    """在物理位置 [base, base+len-1] 上，用相邻 fSWAP（selection sort）把 current 顺序重排为 target 顺序。

    fSWAP 是自身的逆，撤销时把返回的 op 列表整体倒序重放即可（见调用方）。
    """

    current = list(current)
    ops: list = []
    for i, want in enumerate(target):
        j = current.index(want, i)
        while j > i:
            ops.extend(fswap_ops(base + j - 1, base + j))
            current[j - 1], current[j] = current[j], current[j - 1]
            j -= 1
    return ops


def double_excitation_ops(param, p: int, q: int, r: int, s: int) -> list:
    """四 orbital 间双激发的 op 列表：创生对 (p,q)、湮灭对 (r,s)（``a_p^dag a_q^dag a_r a_s - h.c.``）。

    p、q、r、s 只需互不相同，无需数值有序——角色（创生对 vs 湮灭对）由参数
    *位置*（前两个 vs 后两个）决定，与四者数值大小无关。这对 HF 占据/未占据
    在比特序上交错分布的分子（如占据 {1,3}、未占据 {0,2}）是必需的：若简单按
    数值排序会把交错的占据/未占据对错误拆成"数值上更小的两个"与"更大的两个"，
    与目标 fermionic 生成元的创生/湮灭配对不符，导致该双激发在 HF 态上完全不
    起作用（矩阵只连接 |两个更小 orbital 占据⟩ 与 |两个更大 orbital 占据⟩ 两个基
    态，HF 态两者都不是）。

    实现：先用 fSWAP 网络把四个 orbital 聚到相邻块（按数值升序聚拢，复用既有
    聚拢/撤销原语），再在块内用相邻 fSWAP 把顺序从"数值升序"局部重排为
    ``(p, q, r, s)`` 的角色顺序，施加既有 ``double_excitation`` gate，最后按相反顺序
    撤销局部重排与聚拢。当 (p,q,r,s) 本身已数值升序时，局部重排为空操作，退化为
    原算法（``tests/vqc/test_excitation_circuits.py`` 的既有 oracle 用例覆盖）。
    """

    modes = (p, q, r, s)
    if len(set(modes)) != 4:
        raise ValueError(f"double excitation 需 4 个互不相同的 qubit，得到 {modes!r}")
    a, b, c, d = sorted(modes)
    base = a

    ops: list = []
    # 依次把 b、c、d 聚到 base+1, base+2, base+3（数值升序聚拢，与旧算法一致）
    _bring_adjacent(ops, b, base + 1)
    _bring_adjacent(ops, c, base + 2)
    _bring_adjacent(ops, d, base + 3)

    # 块内从数值升序 (a,b,c,d) 局部重排为角色顺序 (p,q,r,s)
    permute_ops = _local_permute_ops(base, [a, b, c, d], [p, q, r, s])
    ops.extend(permute_ops)

    ops.append(double_excitation(param, base, base + 1, base + 2, base + 3))

    # 逆序撤销：局部重排 -> 聚拢
    ops.extend(reversed(permute_ops))
    _undo_adjacent(ops, base + 3, d)
    _undo_adjacent(ops, base + 2, c)
    _undo_adjacent(ops, base + 1, b)
    return ops
