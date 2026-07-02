"""transpile 优化：相邻同操作数 excitation 门按角度相加合并（G(θ1)·G(θ2)=G(θ1+θ2)）。"""

import numpy as np

from aicir.core.circuit import Circuit, double_excitation, hadamard, single_excitation
from aicir.ir import circuit_gate_dicts
from aicir.transpile import MergeRotationsPass, optimize


def _equiv(a: Circuit, b: Circuit) -> bool:
    return np.allclose(np.asarray(a.unitary()), np.asarray(b.unitary()), atol=1e-6)


def test_adjacent_single_excitations_merge():
    cir = Circuit(single_excitation(0.3, 0, 1), single_excitation(0.5, 0, 1), n_qubits=2)
    out = optimize(cir)
    gates = circuit_gate_dicts(out)
    assert len(gates) == 1
    assert gates[0]["type"] == "single_excitation"
    assert gates[0]["parameter"] == 0.8
    assert _equiv(cir, out)


def test_inverse_single_excitations_cancel():
    cir = Circuit(single_excitation(0.7, 0, 1), single_excitation(-0.7, 0, 1), n_qubits=2)
    out = optimize(cir)
    assert circuit_gate_dicts(out) == []
    assert _equiv(cir, out)


def test_adjacent_double_excitations_merge():
    cir = Circuit(
        double_excitation(0.2, 0, 1, 2, 3),
        double_excitation(0.5, 0, 1, 2, 3),
        n_qubits=4,
    )
    out = MergeRotationsPass().run(cir)
    gates = circuit_gate_dicts(out)
    assert len(gates) == 1
    assert gates[0]["type"] == "double_excitation"
    assert gates[0]["parameter"] == 0.7
    assert _equiv(cir, out)


def test_different_operands_do_not_merge():
    cir = Circuit(single_excitation(0.3, 0, 1), single_excitation(0.5, 1, 2), n_qubits=3)
    out = optimize(cir)
    assert len(circuit_gate_dicts(out)) == 2


def test_non_adjacent_excitations_not_merged_across_blocker():
    # 中间隔了作用在重叠比特的门，不应合并。
    cir = Circuit(
        single_excitation(0.3, 0, 1),
        hadamard(1),
        single_excitation(0.5, 0, 1),
        n_qubits=2,
    )
    out = optimize(cir)
    types = [g["type"] for g in circuit_gate_dicts(out)]
    assert types.count("single_excitation") == 2
