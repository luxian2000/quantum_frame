import numpy as np
import pytest

from aicir.backends import NumpyBackend
from aicir.core import Circuit
from aicir import hadamard, cnot, rx, rz
from aicir.core.operators import Hamiltonian, PauliString
from aicir.primitives import StatevectorEstimator
from aicir.simulator import mps_expectation


def _circ(seed):
    rng = np.random.default_rng(seed)
    c = Circuit(n_qubits=4)
    for q in range(4):
        c.append(rx(q, float(rng.uniform(0, np.pi))))
    for q in range(3):
        c.append(cnot(q + 1, [q]))
    for q in range(4):
        c.append(rz(q, float(rng.uniform(0, np.pi))))
    return c


def test_hamiltonian_matches_statevector_estimator():
    bk = NumpyBackend()
    c = _circ(7)
    H = Hamiltonian([("ZZII", -1.0), ("IXXI", 0.5), ("ZIIZ", 0.3)])
    got = float(np.real(complex(mps_expectation(c, H, backend=bk))))
    ref = StatevectorEstimator(bk).run(c, H).value
    assert abs(got - ref) < 1e-5


def test_paulistring_matches():
    bk = NumpyBackend()
    c = _circ(3)
    ps = PauliString("ZIZI", coefficient=0.7)
    got = float(np.real(complex(mps_expectation(c, ps, backend=bk))))
    ref = StatevectorEstimator(bk).run(c, ps).value
    assert abs(got - ref) < 1e-5


def test_dense_matrix_fallback():
    bk = NumpyBackend()
    c = _circ(5)
    op = np.diag([1.0] * 8 + [-1.0] * 8).astype(np.complex64)  # 稠密 16x16
    got = float(np.real(complex(mps_expectation(c, op, backend=bk))))
    ref = StatevectorEstimator(bk).run(c, op).value
    assert abs(got - ref) < 1e-5


def test_gpu_expectation_differentiable():
    torch = pytest.importorskip("torch")
    from aicir.backends import GPUBackend

    bk = GPUBackend(device="cpu")
    theta = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
    c = Circuit(n_qubits=2)
    c.append(rx(0, theta))
    c.append(cnot(1, [0]))
    H = Hamiltonian([("ZI", 1.0)])
    val = mps_expectation(c, H, backend=bk)
    val.backward()
    assert theta.grad is not None
