import numpy as np
from aicir import Circuit, NumpyBackend, ry
from aicir.simulator.network import build_network
from aicir.simulator.contract import contract, _greedy_path


def test_contract_single_ry_statevector():
    bk = NumpyBackend()
    c = Circuit(ry(0.7, 0), n_qubits=1)
    tensors, indices, open_idx = build_network(c, bk, output_spec=[None])
    result = contract(tensors, indices, open_idx, bk)
    vec = bk.to_numpy(result).reshape(-1)
    assert np.allclose(vec, [np.cos(0.35), np.sin(0.35)], atol=1e-6)


def test_greedy_path_reduces_to_single_tensor():
    # 3 个张量链式共享标签 -> 2 步收缩
    indices = [(1,), (1, 2), (2,)]
    path = _greedy_path(indices)
    assert len(path) == 2


def test_contract_single_tensor_no_gates():
    # 无门电路：网络约化为单个 |0> 张量，opt_einsum 返回 (0,) 步，不应崩溃
    bk = NumpyBackend()
    c = Circuit(n_qubits=1)
    tensors, indices, open_idx = build_network(c, bk, output_spec=[None])
    result = contract(tensors, indices, open_idx, bk)
    assert np.allclose(bk.to_numpy(result).reshape(-1), [1.0, 0.0], atol=1e-6)
