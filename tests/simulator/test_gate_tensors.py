import numpy as np
from aicir import NumpyBackend, cnot, ry
from aicir.core.gates import gate_tensors


def test_single_qubit_gate_tensor():
    bk = NumpyBackend()
    (matrix, axes), = gate_tensors(ry(0.5, 0), bk)
    assert axes == [0]
    expected = np.array([[np.cos(0.25), -np.sin(0.25)], [np.sin(0.25), np.cos(0.25)]])
    assert np.allclose(bk.to_numpy(matrix), expected, atol=1e-6)


def test_cnot_gate_tensor_axes_and_shape():
    bk = NumpyBackend()
    tensors = gate_tensors(cnot(1, [0]), bk)
    assert len(tensors) == 1
    matrix, axes = tensors[0]
    # 控制在前、目标在后：axes = controls + targets
    assert axes == [0, 1]
    assert bk.to_numpy(matrix).shape == (4, 4)
