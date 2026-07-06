import numpy as np
import pytest

from aicir import Circuit, NumpyBackend, cnot, hadamard, ry
from aicir.core.state import State
from aicir.simulator.contract import contract
from aicir.simulator.network import build_network


def _ref_state(circ):
    bk = NumpyBackend()
    return np.asarray(bk.to_numpy(
        State.zero_state(circ.n_qubits, bk).evolve(circ.unitary(backend=bk)).data
    )).reshape(-1)


def _bell3():
    c = Circuit(n_qubits=3)
    c.append(hadamard(0))
    c.append(cnot(1, [0]))
    c.append(cnot(2, [1]))
    c.append(ry(0.7, 2))
    return c


def _net(circ, bk):
    tensors, indices, open_idx = build_network(circ, bk, output_spec=[None] * circ.n_qubits)
    return tensors, [tuple(t) for t in indices], open_idx


def test_cotengra_path_parity():
    pytest.importorskip("cotengra")
    bk = NumpyBackend()
    tensors, idx, open_idx = _net(_bell3(), bk)
    got = np.asarray(bk.to_numpy(
        contract(tensors, idx, open_idx, bk, optimize="cotengra")
    )).reshape(-1)
    assert np.allclose(got, _ref_state(_bell3()), atol=1e-5)


def test_sliced_contraction_parity_and_actually_slices():
    pytest.importorskip("cotengra")
    from aicir.simulator.contract import _cotengra_path

    bk = NumpyBackend()
    tensors, idx, open_idx = _net(_bell3(), bk)
    _path, sliced = _cotengra_path(idx, open_idx, 4)
    assert len(sliced) >= 1  # 防"没切到片的假通过"

    got = np.asarray(bk.to_numpy(
        contract(tensors, idx, open_idx, bk, optimize="cotengra", memory_limit=4)
    )).reshape(-1)
    assert np.allclose(got, _ref_state(_bell3()), atol=1e-5)


def test_greedy_and_auto_parity():
    bk = NumpyBackend()
    for opt in ("greedy", "auto"):
        tensors, idx, open_idx = _net(_bell3(), bk)
        got = np.asarray(bk.to_numpy(
            contract(tensors, idx, open_idx, bk, optimize=opt)
        )).reshape(-1)
        assert np.allclose(got, _ref_state(_bell3()), atol=1e-5)


def test_bad_optimize_raises():
    bk = NumpyBackend()
    tensors, idx, open_idx = _net(_bell3(), bk)
    with pytest.raises(ValueError, match="optimize"):
        contract(tensors, idx, open_idx, bk, optimize="nope")


def test_memory_limit_with_greedy_raises():
    bk = NumpyBackend()
    tensors, idx, open_idx = _net(_bell3(), bk)
    with pytest.raises(ValueError, match="memory_limit"):
        contract(tensors, idx, open_idx, bk, optimize="greedy", memory_limit=4)
