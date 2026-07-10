# tests/simulator/test_mps_npu.py
import numpy as np
import pytest

from aicir.backends.npu_backend import is_npu_available

npu_only = pytest.mark.skipif(not is_npu_available(), reason="需要真实 NPU")


def _circ():
    from aicir.core import Circuit
    from aicir import rx, rz, cnot

    rng = np.random.default_rng(0)
    c = Circuit(n_qubits=4)
    for q in range(4):
        c.append(rx(float(rng.uniform(0, np.pi)), q))
        c.append(rz(float(rng.uniform(0, np.pi)), q))
    for q in range(3):
        c.append(cnot(q + 1, [q]))
    return c


def _warm():
    torch = pytest.importorskip("torch")
    torch.ones(1).npu()


@npu_only
def test_mps_statevector_npu_matches_cpu():
    _warm()
    from aicir.backends import NumpyBackend
    from aicir.backends.npu_backend import NPUBackend
    from aicir.simulator import mps_statevector

    c = _circ()
    cpu = np.asarray(mps_statevector(c, backend=NumpyBackend()).to_statevector().to_numpy()).reshape(-1)
    npu = mps_statevector(c, backend=NPUBackend()).to_statevector().to_numpy().reshape(-1)
    assert np.allclose(npu, cpu, atol=1e-4)


@npu_only
def test_mps_expectation_npu_matches_cpu():
    _warm()
    from aicir.backends import NumpyBackend
    from aicir.backends.npu_backend import NPUBackend
    from aicir.core.operators import Hamiltonian
    from aicir.simulator import mps_expectation

    c = _circ()
    H = Hamiltonian([("ZZII", -1.0), ("IXXI", 0.5), ("IIZZ", 0.3)])
    cpu = float(np.real(complex(mps_expectation(c, H, backend=NumpyBackend()))))
    raw = mps_expectation(c, H, backend=NPUBackend())  # 后端标量（torch npu）
    npu = float(np.real(complex(NPUBackend().to_numpy(raw))))
    assert abs(npu - cpu) < 1e-3


@npu_only
def test_mps_truncated_npu_matches_cpu():
    _warm()
    from aicir.backends import NumpyBackend
    from aicir.backends.npu_backend import NPUBackend
    from aicir.simulator import mps_statevector

    c = _circ()
    cpu = np.asarray(mps_statevector(c, max_bond_dim=2, backend=NumpyBackend()).to_statevector().to_numpy()).reshape(-1)
    npu = mps_statevector(c, max_bond_dim=2, backend=NPUBackend()).to_statevector().to_numpy().reshape(-1)
    assert np.allclose(npu, cpu, atol=1e-3)


@npu_only
def test_mps_expectation_npu_psr_gradient_matches_analytic():
    _warm()
    from aicir.backends.npu_backend import NPUBackend
    from aicir.core import Circuit
    from aicir import rx, cnot
    from aicir import Parameter
    from aicir.core.operators import Hamiltonian
    from aicir.primitives import MPSEstimator

    theta = Parameter("theta")
    H = Hamiltonian([("ZI", 1.0)])
    c = Circuit(n_qubits=2)
    c.append(rx(theta, 0))
    c.append(cnot(1, [0]))

    grad = MPSEstimator(backend=NPUBackend()).gradient(
        c, H, parameter_values=[0.7], method="psr"
    )
    assert np.allclose(grad.gradient, [-np.sin(0.7)], atol=1e-3)


def test_mps_estimator_gradient_defaults_to_psr_on_torch_cpu():
    torch = pytest.importorskip("torch")
    from aicir.backends import GPUBackend
    from aicir.core import Circuit
    from aicir import rx, cnot
    from aicir import Parameter
    from aicir.core.operators import Hamiltonian
    from aicir.primitives import MPSEstimator, StatevectorEstimator

    theta = Parameter("theta")
    H = Hamiltonian([("ZI", 1.0)])
    c = Circuit(n_qubits=2)
    c.append(rx(theta, 0))
    c.append(cnot(1, [0]))
    x = np.array([0.7])

    bk = GPUBackend(device=torch.device("cpu"))
    got = MPSEstimator(backend=bk).gradient(c, H, parameter_values=x)
    ref = StatevectorEstimator(bk).gradient(c, H, parameter_values=x, method="psr")

    assert got.method == "psr"
    assert np.allclose(got.gradient, ref.gradient, atol=1e-5)
