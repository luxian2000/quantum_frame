"""Angle encoder implementation."""

from __future__ import annotations

import numpy as np

from .abstract import BaseEncoder
from ..channel.backends.numpy_backend import NumpyBackend
from ..core.circuit import Circuit
from ..core.io.dag import circuit_to_dag
from ..core.io.qasm import circuit_to_qasm
from ..core.state import State


def _default_backend(backend):
    return NumpyBackend() if backend is None else backend


def _emit_circuit(circuit, cir):
    if cir == "dict":
        return circuit
    if cir == "qasm":
        return circuit_to_qasm(circuit)
    if cir == "dag":
        gate_types = list(dict.fromkeys(g["type"] for g in circuit.gates))
        return circuit_to_dag(circuit, gate_types)
    raise ValueError(f"cir must be 'dict', 'qasm' or 'dag', got {cir!r}")


class AngleEncoder(BaseEncoder):
    """Angle encoding: x_i -> rotation(x_i) on qubit i."""

    def __init__(self, n_qubits=None, rotation="ry"):
        self.n_qubits = n_qubits
        self.rotation = rotation.lower()
        if self.rotation not in {"rx", "ry", "rz"}:
            raise ValueError("rotation must be rx/ry/rz")

    def encode(self, data, *, cir="dict", backend=None):
        bk = _default_backend(backend)
        arr = np.asarray(data, dtype=np.float64).ravel()
        data_len = len(arr)

        if self.n_qubits is not None:
            n = self.n_qubits
            if data_len > n:
                raise ValueError(f"data length {data_len} > n_qubits={n}")
            if data_len < n:
                arr = np.concatenate([arr, np.zeros(n - data_len)])
        else:
            n = data_len

        gates = [
            {"type": self.rotation, "target_qubit": i, "parameter": float(angle)}
            for i, angle in enumerate(arr)
        ]
        circuit = Circuit(*gates, n_qubits=n)

        zero = State.zero_state(n, bk)
        U = circuit.unitary(backend=bk)
        state = zero.evolve(U)

        return _emit_circuit(circuit, cir), state

    def decode(self, quantum_state):
        amplitudes = quantum_state.to_numpy()
        n = quantum_state.n_qubits
        angles = []
        for i in range(n):
            p1 = sum(
                abs(amplitudes[idx]) ** 2
                for idx in range(1 << n)
                if (idx >> (n - 1 - i)) & 1
            )
            angles.append(float(2 * np.arcsin(np.sqrt(np.clip(p1, 0.0, 1.0)))))
        return np.array(angles)