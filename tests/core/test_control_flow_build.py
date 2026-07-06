import pytest
from aicir.core.circuit import Circuit, if_, pauli_x, while_
from aicir.core.classical import ClassicalRegister
from aicir.ir import ControlFlow
from aicir.ir.accessors import as_instruction, instruction_name


def _body():
    return Circuit(pauli_x(1), n_qubits=2)


def test_if_build_and_fields():
    reg = ClassicalRegister(1, "c")
    cf = if_(reg[0] == 1, _body())
    assert isinstance(cf, ControlFlow)
    assert cf.name == "if" and cf.n_qubits == 2
    assert cf.condition.op == "==" and cf.else_gates is None
    assert cf.body.n_qubits == 2 and len(cf.body.gates) == 1


def test_if_else():
    reg = ClassicalRegister(1, "c")
    cf = if_(reg[0] == 1, _body(), else_body=Circuit(pauli_x(0), n_qubits=2))
    assert cf.else_body.gates[0]["target_qubit"] == 0


def test_while_requires_max_iterations():
    reg = ClassicalRegister(1, "c")
    with pytest.raises(TypeError):
        while_(reg[0] == 1, _body())  # max_iterations 必填
    cf = while_(reg[0] == 1, _body(), max_iterations=10)
    assert cf.name == "while" and cf.max_iterations == 10


def test_body_nqubits_must_match():
    reg = ClassicalRegister(1, "c")
    with pytest.raises(ValueError, match="n_qubits"):
        Circuit(if_(reg[0] == 1, Circuit(pauli_x(1), n_qubits=3)), n_qubits=2)
    # 直接构造时以 body 自身 n_qubits 为准，父线路校验在 append 时


def test_roundtrip_dict():
    reg = ClassicalRegister(1, "c")
    cf = if_(reg[0] == 1, _body(), else_body=Circuit(pauli_x(0), n_qubits=2))
    d = cf.to_dict()
    assert d["type"] == "if"
    back = ControlFlow.from_dict(d)
    assert back.to_dict() == d
    assert back.condition.register_name == "c"


def test_as_instruction_routes_control_flow():
    reg = ClassicalRegister(1, "c")
    d = if_(reg[0] == 1, _body()).to_dict()
    inst = as_instruction(d)
    assert isinstance(inst, ControlFlow)
    assert instruction_name(inst) == "if"


def test_circuit_stores_control_flow():
    reg = ClassicalRegister(1, "c")
    c = Circuit(pauli_x(0), if_(reg[0] == 1, _body()), n_qubits=2)
    assert c.n_qubits == 2
    names = [as_instruction(g).name if not hasattr(g, "name") else g.name
             for g in c.gates]
    # 末条应是 if 指令 dict
    assert c.gates[-1]["type"] == "if"
