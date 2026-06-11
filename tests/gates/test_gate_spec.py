"""GateSpec 注册表测试：门元信息单点注册，别名解析，未知门保持宽松。"""

import pytest

from aicir.gates import (
    GateSpec,
    canonical_gate_name,
    get_gate_spec,
    register_gate,
    registered_gate_names,
)


def test_standard_single_qubit_gates_are_registered():
    spec = get_gate_spec("rz")
    assert spec is not None
    assert spec.name == "rz"
    assert spec.num_qubits == 1
    assert spec.num_params == 1

    assert get_gate_spec("hadamard").num_params == 0
    assert get_gate_spec("u3").num_params == 3
    assert get_gate_spec("u2").num_params == 2


def test_aliases_resolve_to_same_spec():
    assert get_gate_spec("X") is get_gate_spec("pauli_x")
    assert get_gate_spec("H") is get_gate_spec("hadamard")
    assert get_gate_spec("cnot") is get_gate_spec("cx")
    assert get_gate_spec("ccnot") is get_gate_spec("toffoli")


def test_controlled_and_pair_gate_specs():
    cx = get_gate_spec("cx")
    assert cx.num_qubits == 1
    assert cx.num_params == 0
    assert cx.controlled is True

    cry = get_gate_spec("cry")
    assert cry.num_params == 1
    assert cry.controlled is True

    swap = get_gate_spec("swap")
    assert swap.num_qubits == 2
    assert swap.num_params == 0
    assert swap.controlled is False

    rzz = get_gate_spec("rzz")
    assert rzz.num_qubits == 2
    assert rzz.num_params == 1


def test_variable_qubit_gates_use_none():
    unitary = get_gate_spec("unitary")
    assert unitary.num_qubits is None
    # unitary 矩阵在绘图占位场景可缺省，参数个数为可变（None）
    assert unitary.num_params is None

    measure = get_gate_spec("measure")
    assert measure.num_qubits is None
    assert measure.num_params == 0

    # identity 允许整寄存器形式（无 target_qubit），目标比特数可变
    assert get_gate_spec("identity").num_qubits is None


def test_unknown_gate_returns_none():
    assert get_gate_spec("definitely_not_a_gate") is None


def test_canonical_gate_name_resolves_aliases():
    assert canonical_gate_name("X") == "pauli_x"
    assert canonical_gate_name("H") == "hadamard"
    assert canonical_gate_name("cnot") == "cx"
    assert canonical_gate_name("ccnot") == "toffoli"
    assert canonical_gate_name("measurement") == "measure"
    # 规范名与未注册名原样返回
    assert canonical_gate_name("rz") == "rz"
    assert canonical_gate_name("my_custom_block") == "my_custom_block"


def test_register_custom_gate_and_duplicate_protection():
    spec = GateSpec(name="my_iswap", num_qubits=2, num_params=0, aliases=("MYISWAP",))
    register_gate(spec)
    try:
        assert get_gate_spec("my_iswap") is spec
        assert get_gate_spec("MYISWAP") is spec
        assert "my_iswap" in registered_gate_names()
        with pytest.raises(ValueError):
            register_gate(GateSpec(name="my_iswap", num_qubits=2, num_params=0))
        # overwrite=True 显式允许覆盖
        replacement = GateSpec(name="my_iswap", num_qubits=2, num_params=1)
        register_gate(replacement, overwrite=True)
        assert get_gate_spec("my_iswap") is replacement
    finally:
        from aicir.gates.registry import unregister_gate

        unregister_gate("my_iswap")
    assert get_gate_spec("my_iswap") is None


def test_alias_collision_is_rejected():
    with pytest.raises(ValueError):
        register_gate(GateSpec(name="brand_new", num_qubits=1, num_params=0, aliases=("X",)))
