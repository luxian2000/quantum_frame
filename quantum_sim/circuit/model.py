"""
quantum_sim/circuit/model.py

quantum_sim 内部 Circuit 数据结构与门字典构造器。
"""

from __future__ import annotations

import math

import numpy as np

from .gates import gate_to_matrix, identity


class Circuit:
    """量子电路类：支持门序构建、拼接和矩阵生成。"""

    def __init__(self, *gates, n_qubits):
        self.gates = list(gates)
        self.n_qubits = n_qubits

    def __add__(self, other):
        if not isinstance(other, Circuit):
            return NotImplemented
        if self.n_qubits != other.n_qubits:
            raise ValueError(
                f"Cannot compose circuits with different n_qubits: {self.n_qubits} != {other.n_qubits}"
            )
        return Circuit(*self.gates, *other.gates, n_qubits=self.n_qubits)

    def append(self, gate):
        self.gates.append(gate)
        return self

    def extend(self, *gates):
        self.gates.extend(gates)
        return self

    def unitary(self):
        if not self.gates:
            return identity(self.n_qubits)

        gate_qubits = 0
        for gate in self.gates:
            gate_type = gate["type"]
            if gate_type in [
                "pauli_x",
                "X",
                "pauli_y",
                "Y",
                "pauli_z",
                "Z",
                "hadamard",
                "H",
                "s_gate",
                "S",
                "t_gate",
                "T",
                "rx",
                "ry",
                "rz",
                "u3",
                "u2",
            ]:
                gate_qubits = max(gate_qubits, gate["target_qubit"] + 1)
            elif gate_type in ["cnot", "cx", "cz", "cy", "crx", "cry", "crz"]:
                gate_qubits = max(gate_qubits, gate["target_qubit"] + 1, max(gate["control_qubits"]) + 1)
            elif gate_type in ["toffoli", "ccnot"]:
                gate_qubits = max(gate_qubits, gate["target_qubit"] + 1, max(gate["control_qubits"]) + 1)
            elif gate_type == "swap":
                gate_qubits = max(gate_qubits, gate["qubit_1"] + 1, gate["qubit_2"] + 1)
            elif gate_type in ["identity", "I"]:
                gate_qubits = max(gate_qubits, gate["n_qubits"])
            elif gate_type == "rzz":
                gate_qubits = max(gate_qubits, gate["qubit_1"] + 1, gate["qubit_2"] + 1)
            else:
                gate_qubits = max(gate_qubits, gate["target_qubit"] + 1)

        if gate_qubits > self.n_qubits:
            raise ValueError(f"量子门的量子比特数量超出总量子比特数: {gate_qubits} > {self.n_qubits}")

        circuit_matrix = identity(self.n_qubits)
        for gate in self.gates:
            gm = gate_to_matrix(gate, self.n_qubits)
            circuit_matrix = np.matmul(gm, circuit_matrix)
        return circuit_matrix

    def matrix(self):
        return self.unitary()

    def __len__(self):
        return len(self.gates)

    def __iter__(self):
        return iter(self.gates)

    def __repr__(self):
        return f"Circuit(n_qubits={self.n_qubits}, gates={self.gates})"


def circuit(*gates, n_qubits=1):
    return Circuit(*gates, n_qubits=n_qubits).unitary()


def pauli_x(target_qubit=0):
    return {"type": "pauli_x", "target_qubit": target_qubit}


def pauli_y(target_qubit=0):
    return {"type": "pauli_y", "target_qubit": target_qubit}


def pauli_z(target_qubit=0):
    return {"type": "pauli_z", "target_qubit": target_qubit}


def hadamard(target_qubit=0):
    return {"type": "hadamard", "target_qubit": target_qubit}


def rx(theta, target_qubit=0):
    return {"type": "rx", "target_qubit": target_qubit, "parameter": theta}


def ry(theta, target_qubit=0):
    return {"type": "ry", "target_qubit": target_qubit, "parameter": theta}


def rz(theta, target_qubit=0):
    return {"type": "rz", "target_qubit": target_qubit, "parameter": theta}


def s_gate(target_qubit=0):
    return {"type": "s_gate", "target_qubit": target_qubit}


def t_gate(target_qubit=0):
    return {"type": "t_gate", "target_qubit": target_qubit}


def cx(target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    return {
        "type": "cx",
        "target_qubit": target_qubit,
        "control_qubits": control_qubits,
        "control_states": control_states,
    }


cnot = cx


def cy(target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    return {
        "type": "cy",
        "target_qubit": target_qubit,
        "control_qubits": control_qubits,
        "control_states": control_states,
    }


def cz(target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    return {
        "type": "cz",
        "target_qubit": target_qubit,
        "control_qubits": control_qubits,
        "control_states": control_states,
    }


def crx(theta, target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    return {
        "type": "crx",
        "target_qubit": target_qubit,
        "control_qubits": control_qubits,
        "parameter": theta,
        "control_states": control_states,
    }


def cry(theta, target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    return {
        "type": "cry",
        "target_qubit": target_qubit,
        "control_qubits": control_qubits,
        "parameter": theta,
        "control_states": control_states,
    }


def crz(theta, target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    return {
        "type": "crz",
        "target_qubit": target_qubit,
        "control_qubits": control_qubits,
        "parameter": theta,
        "control_states": control_states,
    }


def swap(qubit_1=0, qubit_2=1):
    return {"type": "swap", "qubit_1": qubit_1, "qubit_2": qubit_2}


def toffoli(target_qubit=2, control_qubits=(0, 1)):
    return {"type": "toffoli", "target_qubit": target_qubit, "control_qubits": list(control_qubits)}


ccnot = toffoli


def u3(theta, phi, lam, target_qubit=0):
    return {"type": "u3", "target_qubit": target_qubit, "parameter": [theta, phi, lam]}


def u2(phi, lam, target_qubit=0):
    return u3(math.pi / 2.0, phi, lam, target_qubit)


__all__ = [
    "Circuit",
    "circuit",
    "pauli_x",
    "pauli_y",
    "pauli_z",
    "hadamard",
    "rx",
    "ry",
    "rz",
    "s_gate",
    "t_gate",
    "cx",
    "cnot",
    "cy",
    "cz",
    "crx",
    "cry",
    "crz",
    "swap",
    "toffoli",
    "ccnot",
    "u3",
    "u2",
]
