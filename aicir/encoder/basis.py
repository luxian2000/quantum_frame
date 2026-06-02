"""Basis encoder implementation."""

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


def _scaled_values(arr, n):
    x_min = float(np.min(arr))
    x_max = float(np.max(arr))
    if x_max == x_min:
        return np.zeros(arr.size, dtype=np.int64)
    return np.round((arr - x_min) / (x_max - x_min) * (2**n - 1)).astype(np.int64)


def _minimal_scale_qubits(arr):
    unique_vals = np.unique(arr)
    k = len(unique_vals)
    if k <= 1:
        return 1

    n = max(1, int(np.ceil(np.log2(k))))
    while True:
        scaled_unique = _scaled_values(unique_vals, n)
        if len(np.unique(scaled_unique)) == k:
            return n
        n += 1


class BasisEncoder(BaseEncoder):
    """Basis encoding using a divide-and-conquer decision tree."""

    def __init__(self, n_qubits=None, redundant=False, repeat=None):
        self.n_qubits = n_qubits
        if repeat is not None:
            redundant = repeat
        self.redundant = bool(redundant)

    def encode(self, data, *, cir="dict", backend=None):
        bk = _default_backend(backend)
        arr = np.asarray(data, dtype=float).ravel()
        if arr.size == 0:
            raise ValueError("data must be non-empty")

        # 先确定不会在缩放后发生碰撞的最小比特数
        min_n = _minimal_scale_qubits(arr)

        if self.n_qubits is None:
            n = min_n
        else:
            n = max(min_n, int(self.n_qubits))

        # 缩放使用 min_n，确保当 n > min_n 时高位始终为 0
        scaled = _scaled_values(arr, min_n)

        binary_strings = [format(int(v), f"0{n}b") for v in scaled]
        trie = self._build_trie(binary_strings)

        gates = []
        self._traverse_and_generate_gates(trie, "", n, gates)

        circuit = Circuit(*gates, n_qubits=n) if gates else Circuit(n_qubits=n)

        zero_state = State.zero_state(n, bk)
        U = circuit.unitary(backend=bk)
        state = zero_state.evolve(U)

        return _emit_circuit(circuit, cir), state

    def _build_trie(self, binary_strings):
        trie = {}
        frequencies = {}
        for s in binary_strings:
            if self.redundant:
                frequencies[s] = frequencies.get(s, 0) + 1
            elif s not in frequencies:
                frequencies[s] = 1

        for s, freq in frequencies.items():
            node = trie
            for bit in s:
                if bit not in node:
                    node[bit] = {}
                node = node[bit]
            node["_terminal_count"] = node.get("_terminal_count", 0) + freq

        self._compute_counts(trie)
        return trie

    def _compute_counts(self, node):
        if not isinstance(node, dict):
            return 0

        count = int(node.get("_terminal_count", 0))
        for bit in ["0", "1"]:
            if bit in node:
                count += self._compute_counts(node[bit])
        node["_count"] = count
        return count

    def _traverse_and_generate_gates(self, node, path, n, gates):
        if len(path) >= n:
            return

        j = len(path)
        c0 = node.get("0", {}).get("_count", 0)
        c1 = node.get("1", {}).get("_count", 0)
        C = c0 + c1
        if C == 0:
            return

        if c0 == 0:
            theta = np.pi
        elif c1 == 0:
            theta = 0.0
        else:
            P0 = c0 / C
            theta = 2.0 * np.arccos(np.sqrt(np.clip(P0, 0.0, 1.0)))

        if abs(theta) > 1e-10:
            control_qubits = list(range(len(path)))
            control_states = [int(bit) for bit in path]
            if len(path) == 0:
                gate = {
                    "type": "ry",
                    "target_qubit": j,
                    "parameter": float(theta),
                }
            else:
                gate = {
                    "type": "cry",
                    "target_qubit": j,
                    "control_qubits": control_qubits,
                    "control_states": control_states,
                    "parameter": float(theta),
                }
            gates.append(gate)

        if "0" in node:
            self._traverse_and_generate_gates(node["0"], path + "0", n, gates)
        if "1" in node:
            self._traverse_and_generate_gates(node["1"], path + "1", n, gates)

    def decode(self, quantum_state):
        amplitudes = quantum_state.to_numpy()
        idx = int(np.argmax(np.abs(amplitudes) ** 2))
        n = quantum_state.n_qubits
        return [(idx >> (n - 1 - i)) & 1 for i in range(n)]