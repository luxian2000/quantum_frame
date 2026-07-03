import numpy as np
import pytest

from aicir import Circuit, Hamiltonian, NumpyBackend, cnot, hadamard, ry
from aicir.simulator import partial_amplitude, single_amplitude, tn_expectation, tn_statevector


def _circ():
    c = Circuit(n_qubits=3)
    c.append(hadamard(0))
    c.append(cnot(1, [0]))
    c.append(ry(0.3, 2))
    return c


def test_api_kwargs_accepted_and_parity():
    pytest.importorskip("cotengra")
    base = np.asarray(NumpyBackend().to_numpy(tn_statevector(_circ()).data)).reshape(-1)
    sliced = np.asarray(NumpyBackend().to_numpy(
        tn_statevector(_circ(), optimize="cotengra", memory_limit=4).data
    )).reshape(-1)
    assert np.allclose(base, sliced, atol=1e-5)

    a0 = single_amplitude(_circ(), "000")
    a1 = single_amplitude(_circ(), "000", optimize="cotengra", memory_limit=4)
    assert abs(a0 - a1) < 1e-5

    p0 = partial_amplitude(_circ(), open_qubits=[0, 1])
    p1 = partial_amplitude(_circ(), open_qubits=[0, 1], optimize="cotengra", memory_limit=4)
    assert np.allclose(p0, p1, atol=1e-5)

    h = Hamiltonian([("ZII", 1.0)])
    e0 = complex(np.asarray(NumpyBackend().to_numpy(tn_expectation(_circ(), h))).reshape(()))
    e1 = complex(np.asarray(NumpyBackend().to_numpy(
        tn_expectation(_circ(), h, optimize="cotengra", memory_limit=4)
    )).reshape(()))
    assert abs(e0 - e1) < 1e-5


def test_sliced_expectation_autograd_matches_psr():
    pytest.importorskip("cotengra")
    torch = pytest.importorskip("torch")
    from aicir.backends import GPUBackend

    bk = GPUBackend(device="cpu")
    h = Hamiltonian([("ZII", 1.0)])
    theta = torch.tensor(0.4, requires_grad=True)

    c = Circuit(n_qubits=3)
    c.append(hadamard(1))
    c.append(cnot(2, [1]))
    c.append(ry(theta, 0))

    e = tn_expectation(c, h, backend=bk, optimize="cotengra", memory_limit=4)
    torch.real(e).backward()
    got = float(theta.grad)

    # <Z0> = cos(theta)（q0 与其余比特无纠缠），解析导数 -sin(theta)
    assert abs(got - (-np.sin(0.4))) < 1e-4
