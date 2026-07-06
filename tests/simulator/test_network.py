from aicir import Circuit, NumpyBackend, ry
from aicir.simulator.network import build_network


def test_build_network_statevector_shapes():
    bk = NumpyBackend()
    c = Circuit(ry(0.3, 0), n_qubits=1)
    tensors, indices, open_idx = build_network(bk_circuit := c, bk, output_spec=[None])
    # 1 个输入 |0> 向量 + 1 个门张量；输出开放
    assert len(tensors) == len(indices) == 2
    assert len(open_idx) == 1
    # 每条腿维度均为 2
    for t, ids in zip(tensors, indices):
        assert tuple(t.shape) == (2,) * len(ids)


def test_build_network_single_amplitude_no_open():
    bk = NumpyBackend()
    c = Circuit(ry(0.3, 0), n_qubits=1)
    tensors, indices, open_idx = build_network(c, bk, output_spec=[0])
    assert open_idx == ()
    # 输入向量 + 门 + 输出 bra 向量
    assert len(tensors) == 3
