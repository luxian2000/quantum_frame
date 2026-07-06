import numpy as np
import pytest

from aicir.backends import NumpyBackend
from aicir.core import Circuit
from aicir import hadamard, cnot, rx, rz, toffoli
from aicir.core.circuit import if_, measure
from aicir.core.classical import ClassicalRegister
from aicir.simulator import tn_statevector, mps_statevector


def _random_circuit(n, depth, seed):
    rng = np.random.default_rng(seed)
    c = Circuit(n_qubits=n)
    for _ in range(depth):
        for q in range(n):
            c.append(rx(float(rng.uniform(0, np.pi)), q))
            c.append(rz(float(rng.uniform(0, np.pi)), q))
        for q in range(n - 1):
            c.append(cnot(q + 1, [q]))
    return c


def test_mps_matches_exact_full_bond():
    bk = NumpyBackend()
    c = _random_circuit(5, 3, seed=1)
    mps_sv = np.asarray(mps_statevector(c, backend=bk).to_statevector().to_numpy()).reshape(-1)
    exact = np.asarray(tn_statevector(c, backend=bk).to_numpy()).reshape(-1)
    assert np.allclose(mps_sv, exact, atol=1e-5)


def test_three_qubit_gate_rejected():
    bk = NumpyBackend()
    c = Circuit(n_qubits=3)
    c.append(toffoli(2, [0, 1]))
    with pytest.raises(ValueError, match="1/2"):
        mps_statevector(c, backend=bk)


def test_control_flow_rejected():
    from aicir import pauli_x
    bk = NumpyBackend()
    reg = ClassicalRegister(1, "c")
    c = Circuit(n_qubits=1)
    c.append(measure(0, creg=reg))
    body_circuit = Circuit(n_qubits=1)
    body_circuit.append(pauli_x(0))
    c.append(if_(reg[0] == 1, body_circuit))
    with pytest.raises(ValueError):
        mps_statevector(c, backend=bk)
