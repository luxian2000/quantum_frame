# tests/simulator/test_mps_npu.py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aicir.backends.npu_backend import is_npu_available

pytestmark = pytest.mark.skipif(not is_npu_available(), reason="需要真实 NPU")


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
    torch.ones(1).npu()


def test_mps_statevector_npu_matches_cpu():
    _warm()
    from aicir.backends import NumpyBackend
    from aicir.backends.npu_backend import NPUBackend
    from aicir.simulator import mps_statevector

    c = _circ()
    cpu = np.asarray(mps_statevector(c, backend=NumpyBackend()).to_statevector().to_numpy()).reshape(-1)
    npu = mps_statevector(c, backend=NPUBackend()).to_statevector().to_numpy().reshape(-1)
    assert np.allclose(npu, cpu, atol=1e-4)


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


def test_mps_truncated_npu_matches_cpu():
    _warm()
    from aicir.backends import NumpyBackend
    from aicir.backends.npu_backend import NPUBackend
    from aicir.simulator import mps_statevector

    c = _circ()
    cpu = np.asarray(mps_statevector(c, max_bond_dim=2, backend=NumpyBackend()).to_statevector().to_numpy()).reshape(-1)
    npu = mps_statevector(c, max_bond_dim=2, backend=NPUBackend()).to_statevector().to_numpy().reshape(-1)
    assert np.allclose(npu, cpu, atol=1e-3)


def test_mps_expectation_npu_autograd_matches_cpu():
    _warm()
    from aicir.backends import NumpyBackend
    from aicir.backends.npu_backend import NPUBackend
    from aicir.core import Circuit
    from aicir import rx, cnot
    from aicir.core.operators import Hamiltonian
    from aicir.simulator import mps_expectation

    H = Hamiltonian([("ZI", 1.0)])

    def build(theta):
        c = Circuit(n_qubits=2)
        c.append(rx(theta, 0))
        c.append(cnot(1, [0]))
        return c

    # NPU autograd
    t_npu = torch.tensor(0.7, dtype=torch.float32, device="npu:0", requires_grad=True)
    mps_expectation(build(t_npu), H, backend=NPUBackend()).backward()
    g_npu = float(t_npu.grad.cpu())
    # CPU 参考（GPUBackend on cpu 亦可微；此处用解析 -sin）
    assert abs(g_npu - (-np.sin(0.7))) < 1e-3
