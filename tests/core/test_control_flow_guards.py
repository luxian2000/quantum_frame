import pytest
from aicir import Circuit, NumpyBackend, if_, pauli_x
from aicir.core.circuit import measure
from aicir.core.classical import ClassicalRegister
from aicir.simulator import tn_statevector


def _cf_circuit():
    reg = ClassicalRegister(1, "c")
    return Circuit(pauli_x(0), measure(0, creg=reg),
                   if_(reg[0] == 1, Circuit(pauli_x(0), n_qubits=1)), n_qubits=1)


def test_unitary_rejects_control_flow():
    with pytest.raises(ValueError, match="控制流"):
        _cf_circuit().unitary(backend=NumpyBackend())


def test_unitary_rejects_even_with_ignore_nonunitary():
    with pytest.raises(ValueError, match="控制流"):
        _cf_circuit().unitary(backend=NumpyBackend(), ignore_nonunitary=True)


def test_tn_rejects_control_flow():
    with pytest.raises(ValueError, match="控制流"):
        tn_statevector(_cf_circuit())
