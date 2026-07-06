# tests/simulator/test_mps_truncation.py
import numpy as np

from aicir.backends import NumpyBackend
from aicir.core import Circuit
from aicir import hadamard, cnot, rx, rz
from aicir.simulator import tn_statevector, mps_statevector


def _brickwork(n, depth, seed):
    rng = np.random.default_rng(seed)
    c = Circuit(n_qubits=n)
    for d in range(depth):
        for q in range(n):
            c.append(rx(q, float(rng.uniform(0, np.pi))))
            c.append(rz(q, float(rng.uniform(0, np.pi))))
        start = d % 2
        for q in range(start, n - 1, 2):
            c.append(cnot(q + 1, [q]))
    return c


def _l2_err(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.linalg.norm(a - b))


def test_full_bond_is_exact():
    bk = NumpyBackend()
    c = _brickwork(6, 4, seed=11)
    exact = tn_statevector(c, backend=bk).to_numpy()
    full = mps_statevector(c, max_bond_dim=2 ** 3, backend=bk)
    assert _l2_err(full.to_statevector().to_numpy(), exact) < 1e-5
    assert full.truncation_error < 1e-9


def test_small_chi_is_lossy_and_monotone():
    bk = NumpyBackend()
    c = _brickwork(8, 6, seed=22)
    exact = tn_statevector(c, backend=bk).to_numpy()
    errs = []
    for chi in (1, 2, 4, 16):
        mps = mps_statevector(c, max_bond_dim=chi, backend=bk)
        errs.append(_l2_err(mps.to_statevector().to_numpy(), exact))
    # chi=1 明显有损（证明截断真实发生，而非静默精确）
    assert errs[0] > 1e-3
    # 误差随 chi 增大单调不增（允许极小数值容差）
    for i in range(len(errs) - 1):
        assert errs[i + 1] <= errs[i] + 1e-6
    # chi=1 时累计 truncation_error 非零
    mps1 = mps_statevector(c, max_bond_dim=1, backend=bk)
    assert mps1.truncation_error > 0.0
