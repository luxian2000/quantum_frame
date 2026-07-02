"""LayoutPass 测试：逻辑->物理比特重标号。"""

import numpy as np
import pytest

from aicir.core.circuit import Circuit, cx, hadamard, rz
from aicir.devices import Target
from aicir.ir import circuit_gate_dicts
from aicir.transpile import LayoutPass, PassManager, RoutingPass


def test_trivial_layout_is_identity():
    cir = Circuit(hadamard(0), cx(1, [0]), rz(0.3, 1), n_qubits=2)
    out = LayoutPass().run(cir)
    assert np.allclose(np.asarray(out.unitary()), np.asarray(cir.unitary()), atol=1e-6)
    assert circuit_gate_dicts(out) == circuit_gate_dicts(cir)


def test_explicit_layout_relabels_qubits_and_uses_target_width():
    cir = Circuit(hadamard(0), cx(1, [0]), rz(0.3, 1), n_qubits=2)
    target = Target(n_qubits=4, coupling_map=[(0, 1), (1, 2), (2, 3)])
    out = LayoutPass(initial_layout={0: 2, 1: 3}, target=target).run(cir)
    assert out.n_qubits == 4
    gates = circuit_gate_dicts(out)
    assert gates[0]["target_qubit"] == 2
    assert gates[1]["target_qubit"] == 3 and gates[1]["control_qubits"] == [2]
    assert gates[2]["target_qubit"] == 3


def test_sequence_layout_form():
    cir = Circuit(cx(1, [0]), n_qubits=2)
    out = LayoutPass([3, 1]).run(cir)  # logical 0->3, logical 1->1
    gate = circuit_gate_dicts(out)[0]
    assert gate["target_qubit"] == 1 and gate["control_qubits"] == [3]
    assert out.n_qubits == 4


def test_non_injective_layout_rejected():
    cir = Circuit(cx(1, [0]), n_qubits=2)
    with pytest.raises(ValueError):
        LayoutPass({0: 1, 1: 1}).run(cir)


def test_physical_out_of_target_range_rejected():
    cir = Circuit(cx(1, [0]), n_qubits=2)
    target = Target(n_qubits=2)
    with pytest.raises(ValueError):
        LayoutPass({0: 0, 1: 5}, target=target).run(cir)


def _swaps(circuit: Circuit) -> int:
    return sum(1 for g in circuit_gate_dicts(circuit) if g["type"] == "swap")


def test_auto_layout_places_interacting_qubits_adjacent():
    # 逻辑 0、2 频繁交互；line 拓扑下平凡布局需路由，auto 布局应让其相邻 -> 0 SWAP。
    cir = Circuit(cx(2, [0]), cx(2, [0]), hadamard(1), n_qubits=3)
    target = Target(n_qubits=3, coupling_map=[(0, 1), (1, 2)])

    trivial = RoutingPass(target=target).run(LayoutPass(target=target).run(cir))
    auto = RoutingPass(target=target).run(LayoutPass("auto", target=target).run(cir))

    assert _swaps(trivial) > 0
    assert _swaps(auto) == 0
    # 交互对映射到相邻物理比特
    m = LayoutPass("auto", target=target)
    m.run(cir)
    assert target.coupled(m.last_layout[0], m.last_layout[2])


def test_auto_layout_is_injective_and_in_range():
    cir = Circuit(cx(1, [0]), cx(3, [2]), n_qubits=4)
    target = Target(n_qubits=4, coupling_map=[(0, 1), (1, 2), (2, 3)])
    lp = LayoutPass("auto", target=target)
    lp.run(cir)
    physical = list(lp.last_layout.values())
    assert len(set(physical)) == len(physical)
    assert all(0 <= p < 4 for p in physical)


def test_auto_layout_requires_target():
    cir = Circuit(cx(1, [0]), n_qubits=2)
    with pytest.raises(ValueError):
        LayoutPass("auto").run(cir)


def test_auto_layout_fully_connected_is_identity():
    cir = Circuit(cx(2, [0]), n_qubits=3)
    target = Target(n_qubits=3)  # fully connected
    out = LayoutPass("auto", target=target).run(cir)
    assert circuit_gate_dicts(out) == circuit_gate_dicts(cir)


def test_runs_inside_passmanager_by_name():
    cir = Circuit(hadamard(0), n_qubits=1)
    out = PassManager(["layout"]).run(cir)
    assert circuit_gate_dicts(out) == circuit_gate_dicts(cir)
