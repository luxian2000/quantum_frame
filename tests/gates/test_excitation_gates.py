"""粒子数守恒激发门 + shift_rule 元数据测试。"""

from aicir.gates import get_gate_spec, gate_shift_rule


def test_shift_rule_defaults_to_none():
    assert get_gate_spec("rx").shift_rule is None
    assert get_gate_spec("rzz").shift_rule is None


def test_gate_shift_rule_helper_reads_registry():
    # 未注册门返回 None；标准门 None
    assert gate_shift_rule("rx") is None
    assert gate_shift_rule("not_a_gate") is None
