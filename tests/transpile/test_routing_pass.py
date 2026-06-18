"""RoutingPass 测试：插入 SWAP 满足耦合拓扑，且保持幺正等价。"""

import numpy as np
import pytest

from aicir.core.circuit import Circuit, cx, cz, hadamard, swap, toffoli
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


def test_distant_cx_is_routed_and_stays_equivalent():
    cir = Circuit(hadamard(0), cx(3, [0]), n_qubits=4)
    target = Target(n_qubits=4, coupling_map=[(0, 1), (1, 2), (2, 3)])
    out = RoutingPass(target=target).run(cir)
    assert "swap" in {g["type"] for g in circuit_gate_dicts(out)}
    assert _all_two_qubit_coupled(out, target)
    assert _equiv(cir, out)


def test_adjacent_gate_is_not_modified():
    cir = Circuit(cx(2, [1]), cz(1, [0]), n_qubits=3)
    target = Target(n_qubits=3, coupling_map=[(0, 1), (1, 2)])
    out = RoutingPass(target=target).run(cir)
    assert "swap" not in {g["type"] for g in circuit_gate_dicts(out)}
    assert circuit_gate_dicts(out) == circuit_gate_dicts(cir)


def test_routing_a_swap_gate():
    cir = Circuit(swap(0, 3), n_qubits=4)
    target = Target(n_qubits=4, coupling_map=[(0, 1), (1, 2), (2, 3)])
    out = RoutingPass(target=target).run(cir)
    assert _all_two_qubit_coupled(out, target)
    assert _equiv(cir, out)


def test_fully_connected_target_is_noop():
    cir = Circuit(cx(3, [0]), n_qubits=4)
    target = Target(n_qubits=4)
    out = RoutingPass(target=target).run(cir)
    assert circuit_gate_dicts(out) == circuit_gate_dicts(cir)


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


def test_scratch_qubit_is_restored_for_later_gate():
    # cx(3,[0]) 在 0-1-2-3 链上路由时，比特 1、2 被用作 SWAP 暂存；
    # 之后作用在比特 1 上的门必须看到已复位的状态——靠整体幺正等价锁定该不变量。
    cir = Circuit(hadamard(1), cx(3, [0]), hadamard(1), cx(1, [2]), n_qubits=4)
    target = Target(n_qubits=4, coupling_map=[(0, 1), (1, 2), (2, 3)])
    out = RoutingPass(target=target).run(cir)
    assert _all_two_qubit_coupled(out, target)
    assert _equiv(cir, out)


def test_swap_insertion_is_symmetric():
    # swap-and-restore：插入的 SWAP 必须前后成对（数量为偶数），保证置换被复位。
    cir = Circuit(cx(3, [0]), n_qubits=4)
    target = Target(n_qubits=4, coupling_map=[(0, 1), (1, 2), (2, 3)])
    out = RoutingPass(target=target).run(cir)
    n_swaps = sum(1 for g in circuit_gate_dicts(out) if g["type"] == "swap")
    assert n_swaps > 0 and n_swaps % 2 == 0
