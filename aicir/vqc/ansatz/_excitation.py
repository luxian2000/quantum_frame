"""UCCSD 激发的精确电路 builder（fermionic-SWAP 网络）。

非相邻 orbital 间的费米子激发需处理 JW Z-string。用 fSWAP = CZ·SWAP（JW 一致的
费米子交换，在 |11⟩ 上带 −1）把远端 orbital 逐格搬到与近端相邻，施加既有相邻激发
gate，再逐格搬回。参数原样流入既有 gate，不做任何算术；只用既有 gate，不引入新类型。
"""

from __future__ import annotations

from ...core.circuit import cz, double_excitation, single_excitation, swap


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


def double_excitation_ops(param, p: int, q: int, r: int, s: int) -> list:
    """orbital p<q<r<s 间双激发的 op 列表。

    把四个 orbital 用 fSWAP 网络聚到相邻位 (p, p+1, p+2, p+3)，施加既有
    ``double_excitation`` gate，再还原。

    喂给 gate 的 qubit 顺序：``double_excitation(param, p, p+2, p+1, p+3)`` ——
    即 (p, r, q, s) 而非字面上的 (p, q, r, s)。已用酉矩阵等价测试守住（对照
    Qiskit Nature ``UCC`` 的 double 激发算符，全局相位内相等）：既有
    ``double_excitation`` gate 的内部 4-qubit 基约定是把前两个参数位、后两个
    参数位分别看作一对占据轨道（即在 (a,b,c,d) 参数序下混合 |a=b=1,c=d=0> 与
    |a=b=0,c=d=1>），因此要把物理上配对的 (p,r) 与 (q,s) —— 对应 JW 编号下
    两个电子分别从 p→r、q→s 激发时保持占据数配对的轨道组合 —— 分别放在
    gate 参数的前两位、后两位。
    """

    if not (p < q < r < s):
        raise ValueError(f"double excitation 需 p<q<r<s，得到 {(p, q, r, s)}")
    ops: list = []
    # 依次把 q、r、s 聚到 p+1, p+2, p+3（从最靠近的开始，避免互相错位）
    _bring_adjacent(ops, q, p + 1)
    _bring_adjacent(ops, r, p + 2)
    _bring_adjacent(ops, s, p + 3)
    ops.append(double_excitation(param, p, p + 2, p + 1, p + 3))
    _undo_adjacent(ops, p + 3, s)
    _undo_adjacent(ops, p + 2, r)
    _undo_adjacent(ops, p + 1, q)
    return ops
