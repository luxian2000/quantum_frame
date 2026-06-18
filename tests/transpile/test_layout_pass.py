"""LayoutPass 测试：逻辑->物理比特重标号。"""

import numpy as np
import pytest

from aicir.core.circuit import Circuit, cx, hadamard, rz
from aicir.devices import Target
from aicir.ir import circuit_gate_dicts
from aicir.transpile import LayoutPass, PassManager


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


def test_runs_inside_passmanager_by_name():
    cir = Circuit(hadamard(0), n_qubits=1)
    out = PassManager(["layout"]).run(cir)
    assert circuit_gate_dicts(out) == circuit_gate_dicts(cir)
