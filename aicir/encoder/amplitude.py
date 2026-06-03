"""Amplitude encoder implementation."""

from __future__ import annotations

import math

import numpy as np

from .abstract import BaseEncoder
from ..channel.backends.numpy_backend import NumpyBackend
from ..core.circuit import Circuit
from ..core.io.dag import circuit_to_dag
from ..core.io.qasm import circuit_to_qasm
from ..core.state import State


def _default_backend(backend):
    return NumpyBackend() if backend is None else backend


def _build_state_prep_unitary(psi):
    """Gram-Schmidt: build unitary whose first column is psi."""
    dim = len(psi)
    Q = np.zeros((dim, dim), dtype=np.complex64)
    Q[:, 0] = psi.astype(np.complex64)
    col = 1
    for i in range(dim):
        e = np.zeros(dim, dtype=np.complex64)
        e[i] = 1.0
        for k in range(col):
            e = e - np.vdot(Q[:, k], e) * Q[:, k]
        norm = np.linalg.norm(e)
        if norm > 1e-10:
            Q[:, col] = e / norm
            col += 1
            if col == dim:
                break
    return Q


def _emit_circuit(circuit, cir):
    if cir == "dict":
        return circuit
    if cir == "qasm":
        return circuit_to_qasm(circuit)
    if cir == "dag":
        gate_types = list(dict.fromkeys(g["type"] for g in circuit.gates))
        return circuit_to_dag(circuit, gate_types)
    raise ValueError(f"cir must be 'dict', 'qasm' or 'dag', got {cir!r}")


class AmplitudeEncoder(BaseEncoder):
    """Amplitude encoding by normalized state preparation."""

    def __init__(self, n_qubits=None):
        self.n_qubits = n_qubits

    def encode(self, data, *, cir="dict", backend=None):
        bk = _default_backend(backend)
        arr = np.asarray(data, dtype=np.complex64).ravel()
        data_len = len(arr)

        if self.n_qubits is not None:
            n = self.n_qubits
            expected = 1 << n
            if data_len > expected:
                raise ValueError(f"data length {data_len} > 2^n_qubits={expected}")
            if data_len < expected:
                arr = np.concatenate([arr, np.zeros(expected - data_len, dtype=arr.dtype)])
        else:
            n = max(1, math.ceil(math.log2(data_len))) if data_len > 1 else 1
            expected = 1 << n
            if data_len < expected:
                arr = np.concatenate([arr, np.zeros(expected - data_len, dtype=arr.dtype)])

        norm = float(np.linalg.norm(arr))
        if norm <= 0:
            raise ValueError("input norm is zero")
        psi = arr / norm

        U = _build_state_prep_unitary(psi)
        gate = {"type": "unitary", "parameter": U.tolist(), "n_qubits": n}
        circuit = Circuit(gate, n_qubits=n)

        state = State.from_array(psi, n_qubits=n, backend=bk)
        return _emit_circuit(circuit, cir), state

    def decode(self, quantum_state):
        return np.real(quantum_state.to_numpy().ravel())