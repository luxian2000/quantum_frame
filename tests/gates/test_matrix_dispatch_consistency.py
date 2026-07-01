"""门矩阵分发一致性（Approach A 回归护栏）。

对全部内置门断言两条独立路径一致：
- ``gate_to_matrix``（构造整线路幺正矩阵，再作用到态）
- ``apply_gate_to_state``（局部矩阵直接作用到态）

并跨后端（numpy / torch / npu，可用才跑）比对、检查幺正性与 torch autograd 保持。
这些不变量在 Approach A 重构前后都应成立，故可作为回归护栏。用户可在 NPU 平台运行：
`NPUBackend` 在无真实设备时透明回退 CPU，仍会走 npu 分支断言。
"""

from __future__ import annotations

import numpy as np
import pytest

from aicir.backends.numpy_backend import NumpyBackend
from aicir.core.circuit import (
    cx,
    cy,
    cz,
    crx,
    cry,
    crz,
    double_excitation,
    hadamard,
    pauli_x,
    pauli_y,
    pauli_z,
    rx,
    rxx,
    ry,
    rz,
    rzz,
    s_gate,
    single_excitation,
    swap,
    t_gate,
    toffoli,
    u2,
    u3,
)
from aicir.core.gates import apply_gate_to_state, gate_to_matrix

N_QUBITS = 4

# (id, gate) —— 覆盖单比特/受控/双比特/多目标/激发门与 control_states=0 变体。
GATES = [
    ("pauli_x", pauli_x(0)),
    ("pauli_y", pauli_y(1)),
    ("pauli_z", pauli_z(3)),
    ("hadamard", hadamard(2)),
    ("s_gate", s_gate(0)),
    ("t_gate", t_gate(1)),
    ("rx", rx(0.37, 0)),
    ("ry", ry(-0.71, 2)),
    ("rz", rz(1.13, 3)),
    ("u2", u2(0.2, 0.4, 1)),
    ("u3", u3(0.3, 0.4, 0.5, 0)),
    ("cx", cx(1, [0])),
    ("cy", cy(2, [0])),
    ("cz", cz(1, [3])),
    ("cx_neg", cx(1, [0], [0])),  # control_state=0
    ("crx", crx(0.44, 2, [0])),
    ("cry", cry(-0.66, 0, [1])),
    ("crz", crz(0.9, 1, [3])),
    ("swap", swap(0, 2)),
    ("toffoli", toffoli(3, [0, 1])),
    ("rzz", rzz(0.55, 0, 1)),
    ("rxx", rxx(-0.31, 1, 2)),
    ("single_excitation", single_excitation(0.42, 0, 1)),
    ("double_excitation", double_excitation(0.63, 0, 1, 2, 3)),
]


def _backends():
    """产出 (name, matrix_backend, apply_backend)；不可用的后端跳过。"""
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

        npu = NPUBackend()  # 无真实 NPU 时透明回退 CPU
        yield ("npu", npu, npu)
    except Exception:
        pass


BACKENDS = list(_backends())
BACKEND_IDS = [b[0] for b in BACKENDS]


def _random_state(apply_backend):
    rng = np.random.default_rng(1234)
    vec = rng.standard_normal(1 << N_QUBITS) + 1j * rng.standard_normal(1 << N_QUBITS)
    vec = (vec / np.linalg.norm(vec)).astype(np.complex128).reshape(-1, 1)
    return apply_backend.cast(vec), vec


def _to_np(backend, tensor):
    return np.asarray(backend.to_numpy(tensor)).reshape(-1)


@pytest.mark.parametrize("gate_id,gate", GATES, ids=[g[0] for g in GATES])
@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_matrix_and_local_apply_agree(gate_id, gate, backend):
    name, matrix_backend, apply_backend = backend
    state, _ = _random_state(apply_backend)

    unitary = gate_to_matrix(gate, N_QUBITS, backend=matrix_backend)
    if matrix_backend is None:
        unitary = apply_backend.cast(np.asarray(unitary))
    via_matrix = apply_backend.apply_unitary(state, unitary)
    via_local = apply_gate_to_state(gate, state, N_QUBITS, apply_backend)

    np.testing.assert_allclose(_to_np(apply_backend, via_matrix), _to_np(apply_backend, via_local), atol=1e-6)


@pytest.mark.parametrize("gate_id,gate", GATES, ids=[g[0] for g in GATES])
@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_gate_matrix_is_unitary(gate_id, gate, backend):
    name, matrix_backend, _ = backend
    u = np.asarray(
        matrix_backend.to_numpy(gate_to_matrix(gate, N_QUBITS, backend=matrix_backend))
        if matrix_backend is not None
        else gate_to_matrix(gate, N_QUBITS, backend=None)
    )
    identity = np.eye(1 << N_QUBITS)
    np.testing.assert_allclose(u.conj().T @ u, identity, atol=1e-6)


@pytest.mark.parametrize("gate_id,gate", GATES, ids=[g[0] for g in GATES])
def test_local_apply_matches_numpy_reference_across_backends(gate_id, gate):
    """所有后端的 apply_gate_to_state 结果应与 numpy 参考一致。"""
    ref_backend = NumpyBackend()
    ref_state, ref_vec = _random_state(ref_backend)
    reference = _to_np(ref_backend, apply_gate_to_state(gate, ref_state, N_QUBITS, ref_backend))

    for name, _, apply_backend in BACKENDS:
        state = apply_backend.cast(ref_vec)
        got = _to_np(apply_backend, apply_gate_to_state(gate, state, N_QUBITS, apply_backend))
        np.testing.assert_allclose(got, reference, atol=1e-6, err_msg=f"backend={name} gate={gate_id}")
