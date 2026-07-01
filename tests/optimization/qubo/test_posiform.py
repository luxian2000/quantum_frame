from aicir.optimization.qubo import Binary
from aicir.optimization.qubo.modeling import VariableRegistry
from aicir.optimization.qubo.modeling.posiform import to_posiform


def test_positive_quadratic_passes_through():
    reg = VariableRegistry()
    a = Binary("a", registry=reg)  # id 0
    b = Binary("b", registry=reg)  # id 1
    posi = to_posiform(a * b)      # terms {(0, 1): 1}
    assert posi.terms == {(0, 2): 1.0}   # literal nodes 2*0=0, 2*1=2
    assert posi.constant == 0.0


def test_negative_linear_complements():
    reg = VariableRegistry()
    a = Binary("a", registry=reg)  # id 0
    posi = to_posiform(-1 * a)     # -x0 = -1 + x̄0
    assert posi.terms == {(1,): 1.0}     # x̄0 is node 1
    assert posi.constant == -1.0


def test_negative_quadratic_complements():
    reg = VariableRegistry()
    a = Binary("a", registry=reg)  # id 0
    b = Binary("b", registry=reg)  # id 1
    posi = to_posiform(-1 * a * b) # -x0 x1 = -1 + x̄0 + x0 x̄1
    assert posi.terms == {(1,): 1.0, (0, 3): 1.0}   # x̄0 node 1; x0 x̄1 nodes (0, 3)
    assert posi.constant == -1.0


def test_rejects_cubic():
    reg = VariableRegistry()
    a = Binary("a", registry=reg)
    b = Binary("b", registry=reg)
    c = Binary("c", registry=reg)
    try:
        to_posiform(a * b * c)
    except ValueError as exc:
        assert "degree <= 2" in str(exc)
    else:
        raise AssertionError("Expected cubic input to be rejected.")
