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


def test_shape_reads_do_not_transfer_tensors(monkeypatch):
    # shape/长度读取不应 to_numpy 整个张量：to_statevector 与单比特门
    # 引发的中心移动全程 0 次 to_numpy（NPU 下每次都是强制 D2H + 同步）
    from aicir.backends.numpy_backend import NumpyBackend
    from aicir.core.circuit import Circuit, hadamard, ry
    from aicir.simulator import mps_statevector

    backend = NumpyBackend()
    circuit = Circuit(hadamard(0), ry(0.3, 2), hadamard(1), n_qubits=3, backend=backend)

    calls = []
    original = backend.to_numpy

    def spying(tensor):
        calls.append(True)
        return original(tensor)

    monkeypatch.setattr(backend, "to_numpy", spying)
    state = mps_statevector(circuit, backend=backend)
    psi = state.to_statevector()
    assert calls == []  # 单比特门 + to_statevector：无任何 to_numpy

    monkeypatch.undo()
    # 数值由既有 parity 测试钉住，此处只做归一化 sanity
    assert np.isclose(np.linalg.norm(psi.to_numpy()), 1.0, atol=1e-6)
