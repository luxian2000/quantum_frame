from aicir import Circuit, Measure, NumpyBackend, hadamard, pauli_x
from aicir.core.circuit import measure
from aicir.core.classical import ClassicalRegister


def test_classical_counts_by_name_and_object():
    reg = ClassicalRegister(2, "c")
    c = Circuit(pauli_x(0), measure([0, 1], creg=reg), n_qubits=2)
    res = Measure(NumpyBackend()).run(c, shots=8, seed=1)
    # q0=1,q1=0 -> bits [1,0] -> int 1
    assert res.classical_counts(reg) == {1: 8}
    assert res.classical_counts("c") == {1: 8}


def test_classical_counts_distribution():
    reg = ClassicalRegister(1, "c")
    c = Circuit(hadamard(0), measure(0, creg=reg), n_qubits=1)
    res = Measure(NumpyBackend()).run(c, shots=400, seed=3)
    cc = res.classical_counts(reg)
    assert set(cc) == {0, 1} and sum(cc.values()) == 400


def test_classical_counts_unknown_register_empty():
    reg = ClassicalRegister(1, "c")
    c = Circuit(pauli_x(0), n_qubits=1)  # 无 measure→creg
    res = Measure(NumpyBackend()).run(c, shots=5, seed=1)
    assert res.classical_counts(reg) == {}
