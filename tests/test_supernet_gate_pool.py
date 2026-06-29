"""Characterization: supernet build_circuit emits aicir-canonical gate dicts.

Locks the gate-dict output so the "use aicir gates" refactor stays
byte-identical (types, qubit fields, control_states, parameter placement).
"""

import pytest

pytest.importorskip("torch")

from aicir.gates import canonical_gate_name, registered_gate_names
from aicir.qas.algorithms.supernet import (
    Architecture,
    LayerArchitecture,
    Supernet,
    SupernetConfig,
)


def _qas():
    return Supernet(
        SupernetConfig(
            n_qubits=3,
            layers=1,
            single_qubit_gates=("i", "h", "rx", "ry", "rz"),
            two_qubit_gates=("cx", "rzz"),
            two_qubit_pairs=((0, 1), (1, 2)),
            seed=0,
        )
    )


def test_mixed_architecture_emits_expected_gate_dicts():
    qas = _qas()
    arch = Architecture((LayerArchitecture(("h", "rx", "rz"), ("cx", "rzz")),))
    circuit, _keys, _tensors = qas.build_circuit(arch, supernet_id=0)
    gates = circuit.gates

    assert [g["type"] for g in gates] == ["hadamard", "rx", "rz", "cx", "rzz"]

    assert gates[0] == {"type": "hadamard", "target_qubit": 0}
    assert gates[1]["type"] == "rx" and gates[1]["target_qubit"] == 1 and "parameter" in gates[1]
    assert gates[2]["type"] == "rz" and gates[2]["target_qubit"] == 2 and "parameter" in gates[2]
    # cx on pair (0, 1): target = pair[1], control = pair[0]
    assert gates[3] == {
        "type": "cx",
        "target_qubit": 1,
        "control_qubits": [0],
        "control_states": [1],
    }
    # rzz on pair (1, 2): qubit_1 = pair[0], qubit_2 = pair[1]
    assert gates[4]["type"] == "rzz" and gates[4]["qubit_1"] == 1 and gates[4]["qubit_2"] == 2
    assert "parameter" in gates[4]


def test_every_emitted_type_is_a_registered_aicir_gate():
    qas = _qas()
    arch = Architecture((LayerArchitecture(("h", "rx", "ry"), ("cx", "rzz")),))
    circuit, _keys, _tensors = qas.build_circuit(arch, supernet_id=0)
    valid = set(registered_gate_names())
    for gate in circuit.gates:
        assert canonical_gate_name(gate["type"]) in valid


def test_identity_token_emits_no_gate():
    qas = _qas()
    arch = Architecture((LayerArchitecture(("i", "i", "i"), ("none", "none")),))
    circuit, _keys, _tensors = qas.build_circuit(arch, supernet_id=0)
    assert circuit.gates == []


def test_hf_reference_and_excitation_pool_emit_chemistry_gates():
    qas = Supernet(
        SupernetConfig(
            n_qubits=4,
            layers=1,
            single_qubit_gates=("i",),
            two_qubit_gates=("single_excitation",),
            two_qubit_pairs=((0, 2),),
            four_qubit_gates=("double_excitation",),
            four_qubit_groups=((0, 2, 1, 3),),
            hf_occupied_qubits=(1, 3),
            seed=0,
        )
    )
    arch = Architecture(
        (
            LayerArchitecture(
                ("i", "i", "i", "i"),
                ("single_excitation",),
                ("double_excitation",),
            ),
        )
    )
    circuit, keys, _tensors = qas.build_circuit(arch, supernet_id=0)
    gates = circuit.gates

    assert [g["type"] for g in gates] == [
        "pauli_x",
        "pauli_x",
        "single_excitation",
        "double_excitation",
    ]
    assert [g["target_qubit"] for g in gates[:2]] == [1, 3]
    assert gates[2]["qubit_1"] == 0 and gates[2]["qubit_2"] == 2
    assert tuple(gates[3]["qubits"]) == (0, 2, 1, 3)
    assert [key[0] for key in keys] == ["tq", "fq"]
