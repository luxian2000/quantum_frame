"""自定义门经注册表在两条矩阵路径中一致模拟（Approach A）。

Approach A 让门局部矩阵的唯一来源是 ``GateSpec.matrix`` 注册表，两条路径
（``gate_to_matrix`` 全矩阵 / ``apply_gate_to_state`` 局部）都从中取矩阵。因此注册
一个自定义**不受控**门（携带 ``matrix=`` 局部构造器）后：

- ``gate_to_matrix`` 可构造整线路矩阵（Approach C 已支持）；
- ``apply_gate_to_state`` 也能在**快速局部路径**直接模拟它（Approach A 新增：
  旧实现对未硬编码门返回 ``None``，只能回退全矩阵路径）。

跨 numpy / torch / npu（可用才跑）比对两条路径一致。
"""

from __future__ import annotations

import numpy as np
import pytest

from aicir.backends.numpy_backend import NumpyBackend
from aicir.gates import GateSpec, register_gate, set_gate_matrix, unregister_gate

N_QUBITS = 3


def _backends():
    yield ("numpy", None, NumpyBackend())
    try:
        import torch  # noqa: F401

        from aicir.backends.gpu_backend import GPUBackend

        gpu = GPUBackend(device="cpu")
        yield ("torch", gpu, gpu)
    except Exception:
        pass
    try:
        import torch  # noqa: F401

        from aicir.backends.npu_backend import NPUBackend

        yield ("npu", NPUBackend(), NPUBackend())
    except Exception:
        pass


BACKENDS = list(_backends())
BACKEND_IDS = [b[0] for b in BACKENDS]

# 固定的自定义单比特幺正（非标准门，随机相位）。
_SQRT2 = 1.0 / np.sqrt(2.0)
_CUSTOM_1Q = np.array([[_SQRT2, 1j * _SQRT2], [1j * _SQRT2, _SQRT2]], dtype=np.complex128)


@pytest.fixture
def custom_gate():
    def build(params, backend):
        return backend.cast(_CUSTOM_1Q) if backend is not None else _CUSTOM_1Q

    register_gate(GateSpec("my_custom_1q", 1, 0, symbol="Cus"))
    set_gate_matrix("my_custom_1q", build)
    try:
        yield {"type": "my_custom_1q", "target_qubit": 1}
    finally:
        unregister_gate("my_custom_1q")


def _to_np(backend, tensor):
    return np.asarray(backend.to_numpy(tensor)).reshape(-1)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_custom_gate_runs_in_both_paths(custom_gate, backend):
    from aicir.core.gates import apply_gate_to_state, gate_to_matrix

    name, matrix_backend, apply_backend = backend
    rng = np.random.default_rng(3)
    dim = 1 << N_QUBITS
    vec = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    vec = (vec / np.linalg.norm(vec)).reshape(-1, 1)
    state = apply_backend.cast(vec)

    # 局部路径：Approach A 下自定义门不再返回 None。
    via_local = apply_gate_to_state(custom_gate, state, N_QUBITS, apply_backend)
    assert via_local is not None

    unitary = gate_to_matrix(custom_gate, N_QUBITS, backend=matrix_backend)
    if matrix_backend is None:
        unitary = apply_backend.cast(np.asarray(unitary))
    via_matrix = apply_backend.apply_unitary(state, unitary)

    np.testing.assert_allclose(
        _to_np(apply_backend, via_local), _to_np(apply_backend, via_matrix), atol=1e-6
    )


def test_custom_gate_matrix_matches_numpy_reference(custom_gate):
    """自定义门整线路矩阵应把局部 2x2 正确嵌入到 qubit 1。"""
    from aicir.core.gates import gate_to_matrix

    unitary = np.asarray(gate_to_matrix(custom_gate, N_QUBITS, backend=None))
    # 期望：I(0) ⊗ CUSTOM(1) ⊗ I(2)
    expected = np.kron(np.kron(np.eye(2), _CUSTOM_1Q), np.eye(2))
    np.testing.assert_allclose(unitary, expected, atol=1e-6)
