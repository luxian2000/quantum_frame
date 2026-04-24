"""
nexq/circuit/gates.py

实现量子门矩阵构造逻辑。
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch

_CDTYPE = np.complex64

KET_0 = np.array([[1.0 + 0.0j], [0.0 + 0.0j]], dtype=_CDTYPE)
KET_1 = np.array([[0.0 + 0.0j], [1.0 + 0.0j]], dtype=_CDTYPE)

DENSITY_0 = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]], dtype=_CDTYPE)
DENSITY_1 = np.array([[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]], dtype=_CDTYPE)
IDENTITY_2 = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]], dtype=_CDTYPE)


def _single_qubit_from_base_backend(base_single, target_qubit: int, backend):
    matrices = []
    for i in range(target_qubit + 1):
        matrices.append(backend.cast(base_single if i == target_qubit else IDENTITY_2))
    return backend.tensor_product(*matrices)


def _controlled_from_base_backend(base_single, target_qubit: int, control_qubits: Iterable[int], control_states: Iterable[int], backend):
    control_qubits = list(control_qubits)
    control_states = list(control_states)
    if len(control_states) != len(control_qubits):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")

    all_qubits = [target_qubit] + control_qubits
    n_qubits = max(all_qubits) + 1
    delta = backend.cast(np.asarray(base_single, dtype=_CDTYPE) - IDENTITY_2)

    matrices = []
    for qubit_index in range(n_qubits):
        if qubit_index == target_qubit:
            matrices.append(delta)
        elif qubit_index in control_qubits:
            control_index = control_qubits.index(qubit_index)
            matrices.append(backend.cast(DENSITY_1 if control_states[control_index] == 1 else DENSITY_0))
        else:
            matrices.append(backend.cast(IDENTITY_2))

    result = backend.tensor_product(*matrices)
    identity_backend = backend.eye(1 << n_qubits)
    if (
        isinstance(identity_backend, torch.Tensor)
        and isinstance(result, torch.Tensor)
        and torch.is_complex(identity_backend)
        and torch.is_complex(result)
        and getattr(identity_backend.device, "type", None) == "npu"
    ):
        # NPU workaround: complex add via real/imag decomposition.
        real = torch.real(identity_backend) + torch.real(result)
        imag = torch.imag(identity_backend) + torch.imag(result)
        return torch.complex(real, imag).to(dtype=identity_backend.dtype)
    return identity_backend + result


def _swap_backend(qubit_1=0, qubit_2=1, backend=None):
    return backend.matrix_product(
        _controlled_from_base_backend(np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=_CDTYPE), qubit_1, [qubit_2], [1], backend),
        _controlled_from_base_backend(np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=_CDTYPE), qubit_2, [qubit_1], [1], backend),
        _controlled_from_base_backend(np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=_CDTYPE), qubit_1, [qubit_2], [1], backend),
    )


def _toffoli_backend(target_qubit=2, control_qubits=(0, 1), backend=None):
    return _controlled_from_base_backend(
        np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=_CDTYPE),
        target_qubit,
        control_qubits,
        [1] * len(control_qubits),
        backend,
    )


def matrix_product(*matrices):
    if len(matrices) == 0:
        raise ValueError("至少需要输入一个矩阵")
    if len(matrices) == 1:
        return np.asarray(matrices[0], dtype=_CDTYPE)
    result = np.asarray(matrices[0], dtype=_CDTYPE)
    for i in range(1, len(matrices)):
        result = np.matmul(result, np.asarray(matrices[i], dtype=_CDTYPE))
    return result


def tensor_product(*matrices):
    if len(matrices) == 0:
        raise ValueError("至少需要输入一个矩阵")
    if len(matrices) == 1:
        return np.asarray(matrices[0], dtype=_CDTYPE)
    result = np.asarray(matrices[0], dtype=_CDTYPE)
    for i in range(1, len(matrices)):
        result = np.kron(result, np.asarray(matrices[i], dtype=_CDTYPE))
    return result


def identity(n_qubits=1):
    dim = 1 << int(n_qubits)
    return np.eye(dim, dtype=_CDTYPE)


def _pauli_x(target_qubit=0):
    pauli_x = np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=_CDTYPE)
    if target_qubit > 0:
        pauli_x = np.kron(identity(target_qubit), pauli_x)
    return pauli_x


def _pauli_y(target_qubit=0):
    pauli_y = np.array([[0.0 + 0.0j, -1j], [1j, 0.0 + 0.0j]], dtype=_CDTYPE)
    if target_qubit > 0:
        pauli_y = np.kron(identity(target_qubit), pauli_y)
    return pauli_y


def _pauli_z(target_qubit=0):
    pauli_z = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]], dtype=_CDTYPE)
    if target_qubit > 0:
        pauli_z = np.kron(identity(target_qubit), pauli_z)
    return pauli_z


def _hadamard(target_qubit=0):
    sqrt2_inv = 1.0 / math.sqrt(2.0)
    hadamard = np.array(
        [[sqrt2_inv + 0.0j, sqrt2_inv + 0.0j], [sqrt2_inv + 0.0j, -sqrt2_inv + 0.0j]],
        dtype=_CDTYPE,
    )
    if target_qubit > 0:
        hadamard = np.kron(identity(target_qubit), hadamard)
    return hadamard


def _rx(theta, target_qubit=0):
    t = float(theta)
    cos = math.cos(t / 2.0)
    sin = math.sin(t / 2.0)
    neg_i_sin = -1j * sin
    rx_matrix = np.array([[cos, neg_i_sin], [neg_i_sin, cos]], dtype=_CDTYPE)
    if target_qubit > 0:
        rx_matrix = np.kron(identity(target_qubit), rx_matrix)
    return rx_matrix


def _ry(theta, target_qubit=0):
    t = float(theta)
    cos = math.cos(t / 2.0)
    sin = math.sin(t / 2.0)
    ry_matrix = np.array([[cos, -sin], [sin, cos]], dtype=_CDTYPE)
    if target_qubit > 0:
        ry_matrix = np.kron(identity(target_qubit), ry_matrix)
    return ry_matrix


def _rz(theta, target_qubit=0):
    t = float(theta)
    exp_neg = np.exp(-1j * t / 2.0)
    exp_pos = np.exp(1j * t / 2.0)
    rz_matrix = np.array([[exp_neg, 0.0 + 0.0j], [0.0 + 0.0j, exp_pos]], dtype=_CDTYPE)
    if target_qubit > 0:
        rz_matrix = np.kron(identity(target_qubit), rz_matrix)
    return rz_matrix


def _s_gate(target_qubit=0):
    s_matrix = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1j]], dtype=_CDTYPE)
    if target_qubit > 0:
        s_matrix = np.kron(identity(target_qubit), s_matrix)
    return s_matrix


def _t_gate(target_qubit=0):
    t_val = np.exp(1j * math.pi / 4.0)
    t_matrix = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, t_val]], dtype=_CDTYPE)
    if target_qubit > 0:
        t_matrix = np.kron(identity(target_qubit), t_matrix)
    return t_matrix


def _controlled_from_base(base_single, target_qubit: int, control_qubits: Iterable[int], control_states: Iterable[int]):
    control_qubits = list(control_qubits)
    control_states = list(control_states)
    if len(control_states) != len(control_qubits):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")

    all_qubits = [target_qubit] + control_qubits
    n_qubits = max(all_qubits) + 1
    delta = np.asarray(base_single, dtype=_CDTYPE) - IDENTITY_2

    matrices = []
    for qubit_index in range(n_qubits):
        if qubit_index == target_qubit:
            matrices.append(delta)
        elif qubit_index in control_qubits:
            control_index = control_qubits.index(qubit_index)
            matrices.append(DENSITY_1 if control_states[control_index] == 1 else DENSITY_0)
        else:
            matrices.append(IDENTITY_2)

    result = matrices[0]
    for i in range(1, len(matrices)):
        result = np.kron(result, matrices[i])
    return identity(n_qubits) + result


def _cx(target_qubit, control_qubits, control_states):
    return _controlled_from_base(_pauli_x(), target_qubit, control_qubits, control_states)


def _cy(target_qubit, control_qubits, control_states):
    return _controlled_from_base(_pauli_y(), target_qubit, control_qubits, control_states)


def _cz(target_qubit, control_qubits, control_states):
    return _controlled_from_base(_pauli_z(), target_qubit, control_qubits, control_states)


def _crx(theta, target_qubit, control_qubits, control_states):
    return _controlled_from_base(_rx(theta), target_qubit, control_qubits, control_states)


def _cry(theta, target_qubit, control_qubits, control_states):
    return _controlled_from_base(_ry(theta), target_qubit, control_qubits, control_states)


def _crz(theta, target_qubit, control_qubits, control_states):
    return _controlled_from_base(_rz(theta), target_qubit, control_qubits, control_states)


def _swap(qubit_1=0, qubit_2=1):
    return matrix_product(
        _cx(qubit_1, [qubit_2], [1]),
        _cx(qubit_2, [qubit_1], [1]),
        _cx(qubit_1, [qubit_2], [1]),
    )


def _toffoli(target_qubit=2, control_qubits=(0, 1)):
    c0, c1 = list(control_qubits)
    n_qubits = max(target_qubit, c0, c1) + 1

    matrices_0 = [IDENTITY_2] * n_qubits
    matrices_0[c0] = DENSITY_1
    matrices_0[c1] = DENSITY_1
    matrices_0[target_qubit] = _pauli_x()
    result_0 = matrices_0[0]
    for i in range(1, n_qubits):
        result_0 = np.kron(result_0, matrices_0[i])

    matrices_1 = [IDENTITY_2] * n_qubits
    matrices_1[c0] = DENSITY_0
    result_1 = matrices_1[0]
    for i in range(1, n_qubits):
        result_1 = np.kron(result_1, matrices_1[i])

    matrices_2 = [IDENTITY_2] * n_qubits
    matrices_2[c0] = DENSITY_1
    matrices_2[c1] = DENSITY_0
    result_2 = matrices_2[0]
    for i in range(1, n_qubits):
        result_2 = np.kron(result_2, matrices_2[i])

    return result_0 + result_1 + result_2


def _u3(theta, phi, lam, target_qubit=0):
    theta = float(theta)
    phi = float(phi)
    lam = float(lam)

    cos = math.cos(theta / 2.0)
    sin = math.sin(theta / 2.0)
    exp_iphi = np.exp(1j * phi)
    exp_ilam = np.exp(1j * lam)
    exp_iphi_lam = np.exp(1j * (phi + lam))

    u3_matrix = np.array(
        [[cos, -exp_ilam * sin], [exp_iphi * sin, exp_iphi_lam * cos]],
        dtype=_CDTYPE,
    )
    if target_qubit > 0:
        u3_matrix = np.kron(identity(target_qubit), u3_matrix)
    return u3_matrix


def _u2(phi, lam, target_qubit=0):
    return _u3(math.pi / 2.0, phi, lam, target_qubit)


def _rzz(theta, qubit_1=0, qubit_2=1):
    _ = (qubit_1, qubit_2)
    t = float(theta)
    exp_neg = np.exp(-1j * t / 2.0)
    exp_pos = np.exp(1j * t / 2.0)
    return np.array(
        [
            [exp_neg, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, exp_pos, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, exp_pos, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, exp_neg],
        ],
        dtype=_CDTYPE,
    )


def gate_to_matrix(gate, cir_qubits=1, backend=None):
    gate_type = gate["type"]
    gate_parameter = gate.get("parameter", None)

    if backend is None:
        if gate_type in ["pauli_x", "X"]:
            gate_qubits = gate["target_qubit"] + 1
            gate_matrix = _pauli_x(gate["target_qubit"])
        elif gate_type in ["pauli_y", "Y"]:
            gate_qubits = gate["target_qubit"] + 1
            gate_matrix = _pauli_y(gate["target_qubit"])
        elif gate_type in ["pauli_z", "Z"]:
            gate_qubits = gate["target_qubit"] + 1
            gate_matrix = _pauli_z(gate["target_qubit"])
        elif gate_type in ["hadamard", "H"]:
            gate_qubits = gate["target_qubit"] + 1
            gate_matrix = _hadamard(gate["target_qubit"])
        elif gate_type in ["s_gate", "S"]:
            gate_qubits = gate["target_qubit"] + 1
            gate_matrix = _s_gate(gate["target_qubit"])
        elif gate_type in ["t_gate", "T"]:
            gate_qubits = gate["target_qubit"] + 1
            gate_matrix = _t_gate(gate["target_qubit"])
        elif gate_type == "rx":
            gate_qubits = gate["target_qubit"] + 1
            gate_matrix = _rx(gate_parameter, gate["target_qubit"])
        elif gate_type == "ry":
            gate_qubits = gate["target_qubit"] + 1
            gate_matrix = _ry(gate_parameter, gate["target_qubit"])
        elif gate_type == "rz":
            gate_qubits = gate["target_qubit"] + 1
            gate_matrix = _rz(gate_parameter, gate["target_qubit"])
        elif gate_type == "u3":
            gate_qubits = gate["target_qubit"] + 1
            gate_matrix = _u3(gate_parameter[0], gate_parameter[1], gate_parameter[2], gate["target_qubit"])
        elif gate_type == "u2":
            gate_qubits = gate["target_qubit"] + 1
            gate_matrix = _u2(gate_parameter[0], gate_parameter[1], gate["target_qubit"])
        elif gate_type in ["cnot", "cx"]:
            gate_qubits = max(gate["target_qubit"], max(gate["control_qubits"])) + 1
            gate_matrix = _cx(gate["target_qubit"], gate["control_qubits"], gate["control_states"])
        elif gate_type == "cy":
            gate_qubits = max(gate["target_qubit"], max(gate["control_qubits"])) + 1
            gate_matrix = _cy(gate["target_qubit"], gate["control_qubits"], gate["control_states"])
        elif gate_type == "cz":
            gate_qubits = max(gate["target_qubit"], max(gate["control_qubits"])) + 1
            gate_matrix = _cz(gate["target_qubit"], gate["control_qubits"], gate["control_states"])
        elif gate_type == "crx":
            gate_qubits = max(gate["target_qubit"], max(gate["control_qubits"])) + 1
            gate_matrix = _crx(gate_parameter, gate["target_qubit"], gate["control_qubits"], gate["control_states"])
        elif gate_type == "cry":
            gate_qubits = max(gate["target_qubit"], max(gate["control_qubits"])) + 1
            gate_matrix = _cry(gate_parameter, gate["target_qubit"], gate["control_qubits"], gate["control_states"])
        elif gate_type == "crz":
            gate_qubits = max(gate["target_qubit"], max(gate["control_qubits"])) + 1
            gate_matrix = _crz(gate_parameter, gate["target_qubit"], gate["control_qubits"], gate["control_states"])
        elif gate_type == "swap":
            gate_qubits = max(gate["qubit_1"], gate["qubit_2"]) + 1
            gate_matrix = _swap(gate["qubit_1"], gate["qubit_2"])
        elif gate_type in ["toffoli", "ccnot"]:
            gate_qubits = max(gate["target_qubit"], gate["control_qubits"][0], gate["control_qubits"][1]) + 1
            gate_matrix = _toffoli(gate["target_qubit"], gate["control_qubits"])
        elif gate_type in ["identity", "I"]:
            return identity(gate["n_qubits"])
        elif gate_type == "rzz":
            gate_qubits = max(gate["qubit_1"], gate["qubit_2"]) + 1
            gate_matrix = _rzz(gate_parameter, gate["qubit_1"], gate["qubit_2"])
        else:
            raise ValueError(f"不支持的门类型: {gate_type}")
    else:
        if gate_type in ["pauli_x", "X"]:
            gate_qubits = gate["target_qubit"] + 1
            base = np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=_CDTYPE)
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type in ["pauli_y", "Y"]:
            gate_qubits = gate["target_qubit"] + 1
            base = np.array([[0.0 + 0.0j, -1j], [1j, 0.0 + 0.0j]], dtype=_CDTYPE)
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type in ["pauli_z", "Z"]:
            gate_qubits = gate["target_qubit"] + 1
            base = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]], dtype=_CDTYPE)
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type in ["hadamard", "H"]:
            gate_qubits = gate["target_qubit"] + 1
            sqrt2_inv = 1.0 / math.sqrt(2.0)
            base = np.array(
                [[sqrt2_inv + 0.0j, sqrt2_inv + 0.0j], [sqrt2_inv + 0.0j, -sqrt2_inv + 0.0j]],
                dtype=_CDTYPE,
            )
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type in ["s_gate", "S"]:
            gate_qubits = gate["target_qubit"] + 1
            base = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1j]], dtype=_CDTYPE)
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type in ["t_gate", "T"]:
            gate_qubits = gate["target_qubit"] + 1
            t_val = np.exp(1j * math.pi / 4.0)
            base = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, t_val]], dtype=_CDTYPE)
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type == "rx":
            gate_qubits = gate["target_qubit"] + 1
            t = float(gate_parameter)
            cos = math.cos(t / 2.0)
            sin = math.sin(t / 2.0)
            base = np.array([[cos, -1j * sin], [-1j * sin, cos]], dtype=_CDTYPE)
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type == "ry":
            gate_qubits = gate["target_qubit"] + 1
            t = float(gate_parameter)
            cos = math.cos(t / 2.0)
            sin = math.sin(t / 2.0)
            base = np.array([[cos, -sin], [sin, cos]], dtype=_CDTYPE)
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type == "rz":
            gate_qubits = gate["target_qubit"] + 1
            t = float(gate_parameter)
            base = np.array([[np.exp(-1j * t / 2.0), 0.0 + 0.0j], [0.0 + 0.0j, np.exp(1j * t / 2.0)]], dtype=_CDTYPE)
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type == "u3":
            gate_qubits = gate["target_qubit"] + 1
            theta, phi, lam = float(gate_parameter[0]), float(gate_parameter[1]), float(gate_parameter[2])
            cos = math.cos(theta / 2.0)
            sin = math.sin(theta / 2.0)
            base = np.array(
                [[cos, -np.exp(1j * lam) * sin], [np.exp(1j * phi) * sin, np.exp(1j * (phi + lam)) * cos]],
                dtype=_CDTYPE,
            )
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type == "u2":
            gate_qubits = gate["target_qubit"] + 1
            phi, lam = float(gate_parameter[0]), float(gate_parameter[1])
            cos = math.cos(math.pi / 4.0)
            sin = math.sin(math.pi / 4.0)
            base = np.array(
                [[cos, -np.exp(1j * lam) * sin], [np.exp(1j * phi) * sin, np.exp(1j * (phi + lam)) * cos]],
                dtype=_CDTYPE,
            )
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type in ["cnot", "cx"]:
            gate_qubits = max(gate["target_qubit"], max(gate["control_qubits"])) + 1
            base = np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=_CDTYPE)
            gate_matrix = _controlled_from_base_backend(base, gate["target_qubit"], gate["control_qubits"], gate["control_states"], backend)
        elif gate_type == "cy":
            gate_qubits = max(gate["target_qubit"], max(gate["control_qubits"])) + 1
            base = np.array([[0.0 + 0.0j, -1j], [1j, 0.0 + 0.0j]], dtype=_CDTYPE)
            gate_matrix = _controlled_from_base_backend(base, gate["target_qubit"], gate["control_qubits"], gate["control_states"], backend)
        elif gate_type == "cz":
            gate_qubits = max(gate["target_qubit"], max(gate["control_qubits"])) + 1
            base = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]], dtype=_CDTYPE)
            gate_matrix = _controlled_from_base_backend(base, gate["target_qubit"], gate["control_qubits"], gate["control_states"], backend)
        elif gate_type == "crx":
            gate_qubits = max(gate["target_qubit"], max(gate["control_qubits"])) + 1
            t = float(gate_parameter)
            cos = math.cos(t / 2.0)
            sin = math.sin(t / 2.0)
            base = np.array([[cos, -1j * sin], [-1j * sin, cos]], dtype=_CDTYPE)
            gate_matrix = _controlled_from_base_backend(base, gate["target_qubit"], gate["control_qubits"], gate["control_states"], backend)
        elif gate_type == "cry":
            gate_qubits = max(gate["target_qubit"], max(gate["control_qubits"])) + 1
            t = float(gate_parameter)
            cos = math.cos(t / 2.0)
            sin = math.sin(t / 2.0)
            base = np.array([[cos, -sin], [sin, cos]], dtype=_CDTYPE)
            gate_matrix = _controlled_from_base_backend(base, gate["target_qubit"], gate["control_qubits"], gate["control_states"], backend)
        elif gate_type == "crz":
            gate_qubits = max(gate["target_qubit"], max(gate["control_qubits"])) + 1
            t = float(gate_parameter)
            base = np.array([[np.exp(-1j * t / 2.0), 0.0 + 0.0j], [0.0 + 0.0j, np.exp(1j * t / 2.0)]], dtype=_CDTYPE)
            gate_matrix = _controlled_from_base_backend(base, gate["target_qubit"], gate["control_qubits"], gate["control_states"], backend)
        elif gate_type == "swap":
            gate_qubits = max(gate["qubit_1"], gate["qubit_2"]) + 1
            gate_matrix = _swap_backend(gate["qubit_1"], gate["qubit_2"], backend)
        elif gate_type in ["toffoli", "ccnot"]:
            gate_qubits = max(gate["target_qubit"], gate["control_qubits"][0], gate["control_qubits"][1]) + 1
            gate_matrix = _toffoli_backend(gate["target_qubit"], gate["control_qubits"], backend)
        elif gate_type in ["identity", "I"]:
            return backend.eye(1 << gate["n_qubits"])
        elif gate_type == "rzz":
            gate_qubits = max(gate["qubit_1"], gate["qubit_2"]) + 1
            gate_matrix = backend.cast(_rzz(gate_parameter, gate["qubit_1"], gate["qubit_2"]))
        else:
            raise ValueError(f"不支持的门类型: {gate_type}")

    if gate_qubits < cir_qubits:
        for _ in range(gate_qubits, cir_qubits):
            if backend is None:
                gate_matrix = np.kron(gate_matrix, IDENTITY_2)
            else:
                gate_matrix = backend.kron(gate_matrix, backend.cast(IDENTITY_2))
    elif gate_qubits > cir_qubits:
        raise ValueError(f"量子门的量子比特数量超出总量子比特数: {gate_qubits} > {cir_qubits}")
    return gate_matrix
