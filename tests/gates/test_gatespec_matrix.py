"""GateSpec.matrix 字段与 gate_to_matrix 自定义门回退（NEXT.md §7）。"""

import numpy as np
import pytest

import aicir.core.gates  # noqa: F401  触发标准门矩阵构造器附加
from aicir.core.gates import gate_to_matrix
from aicir.gates import GateSpec, gate_matrix, register_gate, unregister_gate
from aicir.core.circuit import Circuit
from aicir.backends.numpy_backend import NumpyBackend

# 标准门 + 代表参数（无参门 params 被忽略）。
_STANDARD_CASES = [
    ("pauli_x", (), 1),
    ("pauli_y", (), 1),
    ("pauli_z", (), 1),
    ("hadamard", (), 1),
    ("s_gate", (), 1),
    ("t_gate", (), 1),
    ("rx", 0.7, 1),
    ("ry", 1.1, 1),
    ("rz", -0.4, 1),
    ("u2", [0.3, 0.5], 1),
    ("u3", [0.3, 0.5, 0.9], 1),
    ("swap", (), 2),
    ("rzz", 0.6, 2),
    ("rxx", 0.8, 2),
    ("single_excitation", 0.5, 2),
    ("double_excitation", 0.45, 4),
]


def _gate_dict(name, params, k):
    gate = {"type": name}
    if k == 1:
        gate["target_qubit"] = 0
    elif name in ("swap", "rzz", "rxx", "single_excitation"):
        gate["qubit_1"], gate["qubit_2"] = 0, 1
    else:  # double_excitation
        gate["qubits"] = list(range(k))
    if params != ():
        gate["parameter"] = params
    return gate


@pytest.mark.parametrize("name,params,k", _STANDARD_CASES)
def test_gate_matrix_field_matches_gate_to_matrix_numpy(name, params, k):
    # 漂移护栏：嵌入后的 gate_matrix 与 gate_to_matrix 数值一致（numpy）。
    local = gate_matrix(name, params, None)
    from aicir.core.gates import _expand_local_matrix_to_full

    embedded = _expand_local_matrix_to_full(local, list(range(k)), k, backend=None)
    full = gate_to_matrix(_gate_dict(name, params, k), cir_qubits=k, backend=None)
    assert np.allclose(np.asarray(embedded), np.asarray(full), atol=1e-7)


@pytest.mark.parametrize("name,params,k", _STANDARD_CASES)
def test_gate_matrix_field_matches_gate_to_matrix_torch(name, params, k):
    torch = pytest.importorskip("torch")
    from aicir.backends.gpu_backend import GPUBackend
    from aicir.core.gates import _expand_local_matrix_to_full

    backend = GPUBackend()
    p = params
    if params != () and not isinstance(params, list):
        p = torch.tensor(float(params), dtype=torch.float32)
    elif isinstance(params, list):
        p = [torch.tensor(float(v), dtype=torch.float32) for v in params]
    local = gate_matrix(name, p, backend)
    embedded = _expand_local_matrix_to_full(local, list(range(k)), k, backend=backend)
    full = gate_to_matrix(_gate_dict(name, p, k), cir_qubits=k, backend=backend)
    assert np.allclose(backend.to_numpy(embedded), backend.to_numpy(full), atol=1e-6)


def test_controlled_and_special_gates_have_no_matrix():
    for name in ("cx", "cy", "cz", "crx", "toffoli", "measure", "reset", "unitary"):
        assert gate_matrix(name) is None


def test_custom_uncontrolled_gate_simulates_via_fallback():
    # 注册自定义不受控门，携带 matrix=，经 gate_to_matrix 回退即可模拟。
    sx = np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=complex)
    register_gate(GateSpec("sqrt_x", 1, 0, matrix=lambda params, backend: sx))
    try:
        full = gate_to_matrix({"type": "sqrt_x", "target_qubit": 0}, cir_qubits=1, backend=None)
        assert np.allclose(np.asarray(full), sx, atol=1e-7)
        # 端到端：电路模拟 sqrt_x|0> 概率 [0.5, 0.5]
        c = Circuit({"type": "sqrt_x", "target_qubit": 0}, n_qubits=1, backend=NumpyBackend())
        u = c.unitary()
        probs = np.abs(np.asarray(u)[:, 0]) ** 2
        assert np.allclose(probs, [0.5, 0.5], atol=1e-7)
    finally:
        unregister_gate("sqrt_x")


def test_unknown_gate_without_matrix_still_raises():
    register_gate(GateSpec("mystery", 1, 0))
    try:
        with pytest.raises(ValueError):
            gate_to_matrix({"type": "mystery", "target_qubit": 0}, cir_qubits=1, backend=None)
    finally:
        unregister_gate("mystery")
