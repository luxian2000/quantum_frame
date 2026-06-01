"""
nexq/core/model.py

nexq 内部 Circuit 数据结构与门字典构造器。
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
import sys

import numpy as np

from .gates import gate_to_matrix, identity


@dataclass(frozen=True)
class Parameter:
    """Symbolic placeholder for a numeric circuit parameter."""

    name: str

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError("Parameter name must be a string")
        name = self.name.strip()
        if not name:
            raise ValueError("Parameter name cannot be empty")
        object.__setattr__(self, "name", name)

    def __str__(self):
        return self.name

    def __float__(self):
        raise TypeError(f"Parameter {self.name!r} is unbound; call Circuit.bind_parameters(...) first")


def _iter_parameters(value):
    if isinstance(value, Parameter):
        yield value
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_parameters(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_parameters(item)


def _collect_parameters(gates):
    parameters = {}
    for gate in gates:
        for parameter in _iter_parameters(gate):
            parameters.setdefault(parameter.name, parameter)
    return tuple(parameters.values())


def _parameter_key_name(key):
    if isinstance(key, Parameter):
        return key.name
    if isinstance(key, str):
        name = key.strip()
        if not name:
            raise ValueError("Parameter name cannot be empty")
        return name
    raise TypeError("Parameter bindings must use string or Parameter keys")


def _normalize_parameter_bindings(values, parameters):
    parameter_names = {parameter.name for parameter in parameters}
    if isinstance(values, Mapping):
        bindings = {_parameter_key_name(key): value for key, value in values.items()}
        unknown = sorted(set(bindings) - parameter_names)
        if unknown:
            raise ValueError(f"Unknown parameter binding(s): {', '.join(unknown)}")
        return bindings

    if isinstance(values, (str, bytes)):
        raise TypeError("Parameter bindings must be a mapping or a non-string sequence")
    try:
        sequence = list(values)
    except TypeError as exc:
        raise TypeError("Parameter bindings must be a mapping or a sequence") from exc

    if len(sequence) != len(parameters):
        raise ValueError(f"Expected {len(parameters)} parameter value(s), got {len(sequence)}")
    return {parameter.name: value for parameter, value in zip(parameters, sequence)}


def _bind_parameter_value(value, bindings):
    if isinstance(value, Parameter):
        return bindings.get(value.name, value)
    if isinstance(value, Mapping):
        return {key: _bind_parameter_value(item, bindings) for key, item in value.items()}
    if isinstance(value, list):
        return [_bind_parameter_value(item, bindings) for item in value]
    if isinstance(value, tuple):
        return tuple(_bind_parameter_value(item, bindings) for item in value)
    return deepcopy(value)


def _required_n_qubits_from_gate(gate):
    gate_type = gate["type"]

    if gate_type == "unitary":
        if "n_qubits" in gate:
            return int(gate["n_qubits"])
        parameter = gate.get("parameter")
        shape = getattr(parameter, "shape", None)
        if shape is None:
            matrix = np.asarray(parameter)
            shape = matrix.shape
            ndim = matrix.ndim
        else:
            shape = tuple(int(dim) for dim in shape)
            ndim = len(shape)
        if ndim != 2 or shape[0] != shape[1]:
            raise ValueError("unitary 门参数必须是方阵")
        dim = shape[0]
        inferred = int(round(math.log2(dim))) if dim > 0 else 0
        if (1 << inferred) != dim:
            raise ValueError("unitary 门矩阵维度必须是 2 的幂")
        return inferred

    def extend_qubits(qubits, values):
        if isinstance(values, (list, tuple, set)):
            qubits.extend(int(qubit) for qubit in values)
        else:
            qubits.append(int(values))

    explicit_qubits = []
    if "target_qubit" in gate:
        explicit_qubits.append(int(gate["target_qubit"]))
    controls = gate.get("control_qubits")
    if controls is not None:
        extend_qubits(explicit_qubits, controls)
    for key in ("qubit_1", "qubit_2"):
        if key in gate:
            explicit_qubits.append(int(gate[key]))
    for key in ("qubits", "targets"):
        values = gate.get(key)
        if values is not None:
            extend_qubits(explicit_qubits, values)

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
        return max(explicit_qubits) + 1
    if gate_type in ["cnot", "cx", "cz", "cy", "crx", "cry", "crz"]:
        return max(explicit_qubits) + 1
    if gate_type in ["toffoli", "ccnot"]:
        return max(explicit_qubits) + 1
    if gate_type in ["identity", "I"]:
        return gate["n_qubits"]
    if explicit_qubits:
        return max(explicit_qubits) + 1
    return gate["target_qubit"] + 1


def _infer_n_qubits_from_gates(gates):
    if not gates:
        raise ValueError("未提供 n_qubits 且没有输入量子门，无法自动推断总量子比特数")
    return max(_required_n_qubits_from_gate(gate) for gate in gates)


def _single_gate_symbol(gate_type):
    symbols = {
        "pauli_x": "X",
        "X": "X",
        "pauli_y": "Y",
        "Y": "Y",
        "pauli_z": "Z",
        "Z": "Z",
        "hadamard": "H",
        "H": "H",
        "s_gate": "S",
        "S": "S",
        "t_gate": "T",
        "T": "T",
        "rx": "Rx",
        "ry": "Ry",
        "rz": "Rz",
        "u2": "U2",
        "u3": "U3",
        "identity": "I",
        "I": "I",
        "unitary": "U",
    }
    return symbols.get(gate_type)


def _controlled_target_symbol(gate_type):
    symbols = {
        "cnot": "X",
        "cx": "X",
        "cy": "Y",
        "cz": "Z",
        "crx": "Rx",
        "cry": "Ry",
        "crz": "Rz",
        "toffoli": "X",
        "ccnot": "X",
    }
    return symbols.get(gate_type)


def _token(symbol):
    if symbol is None:
        return "─?─"
    if len(symbol) == 1:
        return f"─{symbol}─"
    if len(symbol) == 2:
        return f"{symbol}─"
    return symbol[:3]


def _fallback_symbol(gate_type):
    s = str(gate_type).upper() if gate_type is not None else "?"
    if not s:
        return "?"
    return s[:2] if len(s) > 1 else s


_CELL_WIDTH = 9


def _wire_cell():
    return "─" * _CELL_WIDTH


def _blank_cell():
    return " " * _CELL_WIDTH


def _vertical_cell():
    mid = _CELL_WIDTH // 2
    return " " * mid + "│" + " " * (_CELL_WIDTH - mid - 1)


def _symbol_cell(symbol):
    if symbol is None:
        symbol = "?"
    symbol = str(symbol)
    inner_width = _CELL_WIDTH - 2
    if len(symbol) > inner_width:
        symbol = symbol[:inner_width]
    left = (inner_width - len(symbol)) // 2
    right = inner_width - len(symbol) - left
    return "─" + ("─" * left) + symbol + ("─" * right) + "─"


def _format_angle_value(value):
    if value is None:
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)

    # 优先显示常见 pi 分数，便于阅读。
    for den in (1, 2, 3, 4, 6, 8, 12, 16):
        for num in range(-16, 17):
            if num == 0:
                continue
            target = math.pi * num / den
            if abs(x - target) < 1e-10:
                if den == 1:
                    return "pi" if num == 1 else ("-pi" if num == -1 else f"{num}pi")
                if num == 1:
                    return f"pi/{den}"
                if num == -1:
                    return f"-pi/{den}"
                return f"{num}pi/{den}"

    return f"{x:.3f}"


def _rotation_angle_label(gate):
    gate_type = gate.get("type")
    if gate_type in {"rx", "ry", "rz", "crx", "cry", "crz", "rzz"}:
        return f"θ={_format_angle_value(gate.get('parameter'))}"
    return None


def _angle_row_index_for_gate(gate, n_qubits):
    if n_qubits <= 1:
        return None

    gate_type = gate.get("type")
    if gate_type in {"rzz", "swap"}:
        q1 = int(gate["qubit_1"])
        q2 = int(gate["qubit_2"])
        lo, hi = min(q1, q2), max(q1, q2)
        return (lo + hi - 1) // 2

    controls = [int(q) for q in gate.get("control_qubits", [])]
    if controls and "target_qubit" in gate:
        target = int(gate["target_qubit"])
        lo, hi = min(controls + [target]), max(controls + [target])
        return (lo + hi - 1) // 2

    if "target_qubit" in gate:
        target = int(gate["target_qubit"])
        return target - 1 if target > 0 else 0

    return None


def _angle_cell(label):
    if not label:
        return _blank_cell()
    if len(label) > _CELL_WIDTH:
        label = label[:_CELL_WIDTH]
    return label.center(_CELL_WIDTH)


def _gate_to_column(gate, n_qubits):
    qubit_col = [_wire_cell()] * n_qubits
    between_col = [_blank_cell()] * max(0, n_qubits - 1)
    angle_col = [_blank_cell()] * max(0, n_qubits - 1)
    gate_type = gate["type"]

    if gate_type in ["identity", "I"]:
        qubit_col = [_symbol_cell("I")] * n_qubits
        return qubit_col, between_col, angle_col

    if gate_type == "unitary":
        gate_n_qubits = min(int(gate.get("n_qubits", n_qubits)), n_qubits)
        for q in range(gate_n_qubits):
            qubit_col[q] = _symbol_cell("U")
        for q in range(max(0, gate_n_qubits - 1)):
            between_col[q] = _vertical_cell()
        return qubit_col, between_col, angle_col

    if gate_type == "swap":
        q1 = int(gate["qubit_1"])
        q2 = int(gate["qubit_2"])
        lo, hi = min(q1, q2), max(q1, q2)
        qubit_col[q1] = _symbol_cell("x")
        qubit_col[q2] = _symbol_cell("x")
        for q in range(lo, hi):
            between_col[q] = _vertical_cell()
        return qubit_col, between_col, angle_col

    if gate_type == "rzz":
        q1 = int(gate["qubit_1"])
        q2 = int(gate["qubit_2"])
        lo, hi = min(q1, q2), max(q1, q2)
        qubit_col[q1] = _symbol_cell("ZZ")
        qubit_col[q2] = _symbol_cell("ZZ")
        for q in range(lo, hi):
            between_col[q] = _vertical_cell()
        label = _rotation_angle_label(gate)
        row_idx = _angle_row_index_for_gate(gate, n_qubits)
        if label is not None and row_idx is not None:
            angle_col[row_idx] = _angle_cell(label)
        return qubit_col, between_col, angle_col

    controls = [int(q) for q in gate.get("control_qubits", [])]

    if controls:
        target = int(gate["target_qubit"])
        involved = controls + [target]
        lo, hi = min(involved), max(involved)
        for q in range(lo, hi):
            between_col[q] = _vertical_cell()
        for c in controls:
            qubit_col[c] = _symbol_cell("●")
        symbol = _controlled_target_symbol(gate_type)
        if symbol is None:
            symbol = _fallback_symbol(gate_type)
        qubit_col[target] = _symbol_cell(symbol)

        label = _rotation_angle_label(gate)
        row_idx = _angle_row_index_for_gate(gate, n_qubits)
        if label is not None and row_idx is not None:
            angle_col[row_idx] = _angle_cell(label)
        return qubit_col, between_col, angle_col

    if "target_qubit" in gate:
        target = int(gate["target_qubit"])
        symbol = _single_gate_symbol(gate_type)
        if symbol is None:
            symbol = _fallback_symbol(gate_type)
        qubit_col[target] = _symbol_cell(symbol)

        label = _rotation_angle_label(gate)
        row_idx = _angle_row_index_for_gate(gate, n_qubits)
        if label is not None and row_idx is not None:
            angle_col[row_idx] = _angle_cell(label)
        return qubit_col, between_col, angle_col

    return qubit_col, between_col, angle_col


def _circuit_to_ascii(circuit):
    prefix_width = len(f"q{circuit.n_qubits - 1}:")
    qubit_rows = [[f"q{i}:".ljust(prefix_width)] for i in range(circuit.n_qubits)]
    between_rows = [[" " * prefix_width] for _ in range(max(0, circuit.n_qubits - 1))]
    angle_rows = [[" " * prefix_width] for _ in range(max(0, circuit.n_qubits - 1))]

    if not circuit.gates:
        for i in range(circuit.n_qubits):
            qubit_rows[i].append(_wire_cell())
            if i < circuit.n_qubits - 1:
                between_rows[i].append(_blank_cell())
    else:
        for gate in circuit.gates:
            qubit_col, between_col, angle_col = _gate_to_column(gate, circuit.n_qubits)
            for i in range(circuit.n_qubits):
                qubit_rows[i].append(qubit_col[i])
                if i < circuit.n_qubits - 1:
                    between_rows[i].append(between_col[i])
                    angle_rows[i].append(angle_col[i])

    lines = []
    if any(any(cell.strip() for cell in row[1:]) for row in angle_rows):
        for i in range(len(angle_rows)):
            lines.append(" ".join(angle_rows[i]))

    for i in range(circuit.n_qubits):
        lines.append(" ".join(qubit_rows[i]))
        if i < circuit.n_qubits - 1:
            lines.append(" ".join(between_rows[i]))

    return "\n".join(lines)


class Circuit:
    """量子电路类：支持门序构建、拼接和矩阵生成。"""

    def __init__(self, *gates, n_qubits=None, backend=None):
        self.gates = list(gates)
        self.n_qubits = _infer_n_qubits_from_gates(self.gates) if n_qubits is None else n_qubits
        self._backend = backend

    def __add__(self, other):
        if not isinstance(other, Circuit):
            return NotImplemented
        if self.n_qubits != other.n_qubits:
            raise ValueError(
                f"Cannot compose circuits with different n_qubits: {self.n_qubits} != {other.n_qubits}"
            )
        backend = self._backend if self._backend is not None else other._backend
        return Circuit(*self.gates, *other.gates, n_qubits=self.n_qubits, backend=backend)

    def append(self, gate):
        self.gates.append(gate)
        return self

    def extend(self, *gates):
        self.gates.extend(gates)
        return self

    @property
    def parameters(self):
        """Return unbound symbolic parameters in first-use order."""
        return _collect_parameters(self.gates)

    @property
    def backend(self):
        return self._backend

    def bind_backend(self, backend):
        self._backend = backend
        return self

    def bind_parameters(self, values, *, inplace: bool = False, allow_partial: bool = False):
        """Bind symbolic ``Parameter`` placeholders to numeric values.

        ``values`` can be either a mapping keyed by parameter name/``Parameter``
        or a sequence ordered like ``self.parameters``.
        """
        parameters = self.parameters
        bindings = _normalize_parameter_bindings(values, parameters)
        missing = [parameter.name for parameter in parameters if parameter.name not in bindings]
        if missing and not allow_partial:
            raise ValueError(f"Missing parameter binding(s): {', '.join(missing)}")

        gates = [_bind_parameter_value(gate, bindings) for gate in self.gates]
        if inplace:
            self.gates = gates
            return self
        return Circuit(*gates, n_qubits=self.n_qubits, backend=self._backend)

    def unitary(self, backend=None):
        parameters = self.parameters
        if parameters:
            names = ", ".join(parameter.name for parameter in parameters)
            raise ValueError(f"Circuit has unbound parameter(s): {names}; call bind_parameters(...) first")

        backend = backend or self._backend
        if not self.gates:
            return identity(self.n_qubits) if backend is None else backend.eye(1 << self.n_qubits)

        gate_qubits = _infer_n_qubits_from_gates(self.gates)

        if gate_qubits > self.n_qubits:
            raise ValueError(f"量子门的量子比特数量超出总量子比特数: {gate_qubits} > {self.n_qubits}")

        circuit_matrix = identity(self.n_qubits) if backend is None else backend.eye(1 << self.n_qubits)
        for gate in self.gates:
            gm = gate_to_matrix(gate, self.n_qubits, backend=backend)
            if backend is None:
                circuit_matrix = np.matmul(gm, circuit_matrix)
            else:
                circuit_matrix = backend.matmul(gm, circuit_matrix)
        return circuit_matrix

    def matrix(self, backend=None):
        return self.unitary(backend=backend)

    def show(self, file=None):
        """在终端打印量子线路 ASCII 图，并返回该字符串。"""
        stream = sys.stdout if file is None else file
        diagram = _circuit_to_ascii(self)
        print(diagram, file=stream)
        return diagram

    def __len__(self):
        return len(self.gates)

    def __iter__(self):
        return iter(self.gates)

    def __repr__(self):
        backend_name = None if self._backend is None else self._backend.name
        return f"Circuit(n_qubits={self.n_qubits}, gates={self.gates}, backend={backend_name})"


def circuit(*gates, n_qubits=1, backend=None):
    return Circuit(*gates, n_qubits=n_qubits, backend=backend).unitary(backend=backend)


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


def rzz(theta, qubit_1=0, qubit_2=1):
    return {"type": "rzz", "qubit_1": qubit_1, "qubit_2": qubit_2, "parameter": theta}


def toffoli(target_qubit=2, control_qubits=(0, 1)):
    return {"type": "toffoli", "target_qubit": target_qubit, "control_qubits": list(control_qubits)}


ccnot = toffoli


def u3(theta, phi, lam, target_qubit=0):
    return {"type": "u3", "target_qubit": target_qubit, "parameter": [theta, phi, lam]}


def u2(phi, lam, target_qubit=0):
    return {"type": "u2", "target_qubit": target_qubit, "parameter": [phi, lam]}


__all__ = [
    "Circuit",
    "Parameter",
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
    "rzz",
    "toffoli",
    "ccnot",
    "u3",
    "u2",
]
