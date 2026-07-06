"""aicir 精确张量网络模拟引擎：张量网络 / 单振幅 / 部分振幅。"""

from __future__ import annotations

import numpy as np

from ..core.state import State
from .contract import contract
from .network import build_network


def _resolve_backend(circuit, backend):
    if backend is not None:
        return backend
    bk = getattr(circuit, "backend", None)
    if bk is not None:
        return bk
    from ..backends import NumpyBackend
    return NumpyBackend()


def _parse_bits(bitstring, n):
    if isinstance(bitstring, str):
        bits = [int(ch) for ch in bitstring]
    else:
        bits = [int(b) for b in bitstring]
    if len(bits) != n or any(b not in (0, 1) for b in bits):
        raise ValueError(f"bitstring 必须为长度 {n} 的 0/1 串")
    return bits


def _statevector_tensor(circuit, backend, *, optimize="auto", memory_limit=None):
    n = int(circuit.n_qubits)
    tensors, indices, open_idx = build_network(circuit, backend, output_spec=[None] * n)
    result = contract(tensors, indices, open_idx, backend, optimize=optimize, memory_limit=memory_limit)
    return backend.reshape(result, (1 << n, 1))


def tn_statevector(circuit, *, backend=None, optimize="auto", memory_limit=None):
    """经张量网络收缩求整段末态，返回 State（向量形态）。支持 cotengra 切片优化。"""
    backend = _resolve_backend(circuit, backend)
    data = _statevector_tensor(circuit, backend, optimize=optimize, memory_limit=memory_limit)
    return State(data, int(circuit.n_qubits), backend)


def single_amplitude(circuit, bitstring, *, backend=None, optimize="auto", memory_limit=None):
    """求单个基态振幅 ⟨bitstring|U|0⟩（标量收缩，不构造全态矢量）。支持 cotengra 切片优化。"""
    backend = _resolve_backend(circuit, backend)
    n = int(circuit.n_qubits)
    bits = _parse_bits(bitstring, n)
    tensors, indices, open_idx = build_network(circuit, backend, output_spec=bits)
    result = contract(tensors, indices, open_idx, backend, optimize=optimize, memory_limit=memory_limit)
    return complex(np.asarray(backend.to_numpy(result)).reshape(()))


def partial_amplitude(circuit, *, open_qubits=None, bitstrings=None, backend=None, optimize="auto", memory_limit=None):
    """求部分振幅。二选一：
    - open_qubits=[...]：这些比特开放、其余输出固定 |0>，返回按开放比特升序（首个为 MSB）
      排列的 2^len 振幅向量；
    - bitstrings=[...]：枚举给定基态，返回其振幅数组。
    支持 cotengra 切片优化。
    """
    if (open_qubits is None) == (bitstrings is None):
        raise ValueError("open_qubits 与 bitstrings 必须且只能提供其一")
    backend = _resolve_backend(circuit, backend)
    n = int(circuit.n_qubits)
    if open_qubits is not None:
        openset = {int(q) for q in open_qubits}
        spec = [None if q in openset else 0 for q in range(n)]
        tensors, indices, open_idx = build_network(circuit, backend, output_spec=spec)
        result = contract(tensors, indices, open_idx, backend, optimize=optimize, memory_limit=memory_limit)
        return np.asarray(backend.to_numpy(result)).reshape(-1)
    return np.array([single_amplitude(circuit, b, backend=backend, optimize=optimize, memory_limit=memory_limit) for b in bitstrings])


def tn_expectation(circuit, observable, *, backend=None, optimize="auto", memory_limit=None):
    """经张量网络收缩求期望值 ⟨ψ|O|ψ⟩。torch/NPU 后端上对参数门可微。支持 cotengra 切片优化。"""
    backend = _resolve_backend(circuit, backend)
    psi = _statevector_tensor(circuit, backend, optimize=optimize, memory_limit=memory_limit)
    if hasattr(observable, "to_matrix"):
        operator = observable.to_matrix(backend)
    else:
        operator = backend.cast(observable)
    return backend.expectation_sv(psi, operator)


from .mps import MPSState, mps_statevector  # noqa: E402
