"""GateSpec.num_controls：受控门的控制位数量作为注册表数据。"""

from aicir.gates import get_gate_spec, registered_gate_names


def test_num_controls_defaults_to_zero_for_plain_gates():
    assert get_gate_spec("rx").num_controls == 0
    assert get_gate_spec("ry").num_controls == 0
    assert get_gate_spec("rz").num_controls == 0
    assert get_gate_spec("hadamard").num_controls == 0
    assert get_gate_spec("rzz").num_controls == 0
    assert get_gate_spec("swap").num_controls == 0


def test_num_controls_set_for_controlled_gates():
    assert get_gate_spec("cx").num_controls == 1
    assert get_gate_spec("cy").num_controls == 1
    assert get_gate_spec("cz").num_controls == 1
    assert get_gate_spec("crx").num_controls == 1
    assert get_gate_spec("cry").num_controls == 1
    assert get_gate_spec("crz").num_controls == 1
    assert get_gate_spec("toffoli").num_controls == 2


def test_num_controls_consistent_with_controlled_flag():
    for name in registered_gate_names():
        spec = get_gate_spec(name)
        assert spec.controlled == (spec.num_controls > 0), name
