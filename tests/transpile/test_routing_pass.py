"""RoutingPass 测试：置换跟踪路由——插入 SWAP 满足耦合拓扑，
并把比特置换向前携带（不复位），整条线路与原线路等价**至最终置换**。"""

import numpy as np
import pytest

from aicir.core.circuit import Circuit, cx, cz, hadamard, rz, swap, toffoli
from aicir.devices import Target
from aicir.ir import circuit_gate_dicts, instruction_controls, instruction_qubits
from aicir.transpile import RoutingPass


def _equiv(a: Circuit, b: Circuit) -> bool:
    return np.allclose(np.asarray(a.unitary()), np.asarray(b.unitary()), atol=1e-5)


def _all_two_qubit_coupled(circuit: Circuit, target: Target) -> bool:
    for gate in circuit_gate_dicts(circuit):
        qubits = tuple(dict.fromkeys((*instruction_qubits(gate), *instruction_controls(gate))))
        if len(qubits) == 2 and not target.coupled(qubits[0], qubits[1]):
            return False
    return True


def _restore_swaps(final_layout: dict[int, int], n: int) -> list[dict]:
    """生成把每个逻辑比特从 ``final_layout`` 末位移回 home 线的 SWAP 序列。"""
    pos = dict(final_layout)
    cur_inv = {w: l for l, w in pos.items()}
    out: list[dict] = []
    for home in range(n):
        w = pos[home]
        if w == home:
            continue
        out.append(swap(home, w))
        l_at_home = cur_inv[home]
        pos[l_at_home] = w
        pos[home] = home
        cur_inv[w] = l_at_home
        cur_inv[home] = home
    return out


def _equiv_up_to_permutation(original: Circuit, routed: Circuit, final_layout: dict[int, int]) -> bool:
    """路由后线路追加复位 SWAP 后应与原线路完全幺正等价。"""
    n = int(routed.n_qubits)
    restored = Circuit(
        *circuit_gate_dicts(routed),
        *[g for g in _restore_swaps(final_layout, n)],
        n_qubits=n,
    )
    return _equiv(original, restored)


def test_distant_cx_is_routed_and_equiv_up_to_permutation():
    cir = Circuit(hadamard(0), cx(3, [0]), n_qubits=4)
    target = Target(n_qubits=4, coupling_map=[(0, 1), (1, 2), (2, 3)])
    rp = RoutingPass(target=target)
    out = rp.run(cir)
    assert "swap" in {g["type"] for g in circuit_gate_dicts(out)}
    assert _all_two_qubit_coupled(out, target)
    assert _equiv_up_to_permutation(cir, out, rp.final_layout)


def test_permutation_is_carried_forward_not_restored():
    # 单个距离-3 的 cx：置换跟踪只插 2 个 SWAP（不复位），而非旧的 4 个。
    cir = Circuit(cx(3, [0]), n_qubits=4)
    target = Target(n_qubits=4, coupling_map=[(0, 1), (1, 2), (2, 3)])
    rp = RoutingPass(target=target)
    out = rp.run(cir)
    n_swaps = sum(1 for g in circuit_gate_dicts(out) if g["type"] == "swap")
    assert n_swaps == 2
    # 置换被携带：final_layout 非恒等
    assert rp.final_layout != {q: q for q in range(4)}


def test_second_gate_reuses_carried_permutation():
    # 两个相同的距离 cx：第一个把 0、3 移到相邻；第二个无需再插 SWAP。
    cir = Circuit(cx(3, [0]), cx(3, [0]), n_qubits=4)
    target = Target(n_qubits=4, coupling_map=[(0, 1), (1, 2), (2, 3)])
    rp = RoutingPass(target=target)
    out = rp.run(cir)
    n_swaps = sum(1 for g in circuit_gate_dicts(out) if g["type"] == "swap")
    assert n_swaps == 2  # 仅第一个 cx 需要路由
    assert _all_two_qubit_coupled(out, target)
    assert _equiv_up_to_permutation(cir, out, rp.final_layout)


def test_single_qubit_gate_follows_permutation():
    # cx(3,[0]) 后 0 被搬动；后续作用在逻辑 3 上的门必须跟随到其新物理线。
    cir = Circuit(cx(3, [0]), hadamard(3), rz(0.4, 0), n_qubits=4)
    target = Target(n_qubits=4, coupling_map=[(0, 1), (1, 2), (2, 3)])
    rp = RoutingPass(target=target)
    out = rp.run(cir)
    assert _all_two_qubit_coupled(out, target)
    assert _equiv_up_to_permutation(cir, out, rp.final_layout)


def test_adjacent_gate_is_not_modified():
    cir = Circuit(cx(2, [1]), cz(1, [0]), n_qubits=3)
    target = Target(n_qubits=3, coupling_map=[(0, 1), (1, 2)])
    rp = RoutingPass(target=target)
    out = rp.run(cir)
    assert "swap" not in {g["type"] for g in circuit_gate_dicts(out)}
    assert circuit_gate_dicts(out) == circuit_gate_dicts(cir)
    assert rp.final_layout == {0: 0, 1: 1, 2: 2}
    assert rp.last_layout is None


def test_routing_a_swap_gate():
    cir = Circuit(swap(0, 3), n_qubits=4)
    target = Target(n_qubits=4, coupling_map=[(0, 1), (1, 2), (2, 3)])
    rp = RoutingPass(target=target)
    out = rp.run(cir)
    assert _all_two_qubit_coupled(out, target)
    assert _equiv_up_to_permutation(cir, out, rp.final_layout)


def test_fully_connected_target_is_noop():
    cir = Circuit(cx(3, [0]), n_qubits=4)
    target = Target(n_qubits=4)
    rp = RoutingPass(target=target)
    out = rp.run(cir)
    assert circuit_gate_dicts(out) == circuit_gate_dicts(cir)
    assert rp.final_layout == {q: q for q in range(4)}
    assert rp.last_layout is None


def test_multi_qubit_gate_raises():
    cir = Circuit(toffoli(2, [0, 1]), n_qubits=3)
    target = Target(n_qubits=3, coupling_map=[(0, 1), (1, 2)])
    with pytest.raises(NotImplementedError):
        RoutingPass(target=target).run(cir)


def test_disconnected_coupling_raises():
    cir = Circuit(cx(2, [0]), n_qubits=3)
    target = Target(n_qubits=3, coupling_map=[(0, 1)])  # qubit 2 isolated
    with pytest.raises(ValueError):
        RoutingPass(target=target).run(cir)
