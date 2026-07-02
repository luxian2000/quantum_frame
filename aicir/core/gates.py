"""
aicir/core/gates.py

实现量子门矩阵构造逻辑。
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from ..gates import canonical_gate_name
from ..ir.operation import normalize_gate

_CDTYPE = np.complex64

KET_0 = np.array([[1.0 + 0.0j], [0.0 + 0.0j]], dtype=_CDTYPE)
KET_1 = np.array([[0.0 + 0.0j], [1.0 + 0.0j]], dtype=_CDTYPE)

DENSITY_0 = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]], dtype=_CDTYPE)
DENSITY_1 = np.array([[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]], dtype=_CDTYPE)
IDENTITY_2 = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]], dtype=_CDTYPE)


def _controlled_from_base_backend(base_single, target_qubit: int, control_qubits: Iterable[int], control_states: Iterable[int], backend):
    control_qubits = list(control_qubits)
    control_states = list(control_states)
    if len(control_states) != len(control_qubits):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")

    all_qubits = [target_qubit] + control_qubits
    n_qubits = max(all_qubits) + 1
    base_backend = backend.cast(base_single)
    delta = base_backend - backend.cast(IDENTITY_2)

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
        torch is not None
        and isinstance(identity_backend, torch.Tensor)
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


def _unitary_parameter_matrix(value, backend=None):
    if torch is not None and isinstance(value, torch.Tensor):
        if backend is not None and hasattr(backend, "_device"):
            return backend.cast(value)
        return value.detach().cpu().numpy().astype(_CDTYPE)
    matrix = np.asarray(value, dtype=_CDTYPE)
    return backend.cast(matrix) if backend is not None else matrix


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


def _normalized_control_data(gate):
    controls = list(gate["control_qubits"])
    control_states = list(gate.get("control_states", [1] * len(controls)))
    if len(control_states) != len(controls):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")
    return controls, control_states


def _cx(target_qubit, control_qubits, control_states):
    return _controlled_from_base(_pauli_x(), target_qubit, control_qubits, control_states)


def _swap(qubit_1=0, qubit_2=1):
    return matrix_product(
        _cx(qubit_1, [qubit_2], [1]),
        _cx(qubit_2, [qubit_1], [1]),
        _cx(qubit_1, [qubit_2], [1]),
    )


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


def _rxx(theta, qubit_1=0, qubit_2=1):
    _ = (qubit_1, qubit_2)
    t = float(theta)
    cos = math.cos(t / 2.0)
    sin = math.sin(t / 2.0)
    neg_i_sin = -1j * sin
    return np.array(
        [
            [cos, 0.0 + 0.0j, 0.0 + 0.0j, neg_i_sin],
            [0.0 + 0.0j, cos, neg_i_sin, 0.0 + 0.0j],
            [0.0 + 0.0j, neg_i_sin, cos, 0.0 + 0.0j],
            [neg_i_sin, 0.0 + 0.0j, 0.0 + 0.0j, cos],
        ],
        dtype=_CDTYPE,
    )


def _single_excitation(theta, qubit_1=0, qubit_2=1):
    _ = (qubit_1, qubit_2)
    t = float(theta)
    c = math.cos(t / 2.0)
    s = math.sin(t / 2.0)
    return np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, c, -s, 0.0 + 0.0j],
            [0.0 + 0.0j, s, c, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
        ],
        dtype=_CDTYPE,
    )


def _contains_torch_tensor(value) -> bool:
    if torch is None:
        return False
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_torch_tensor(item) for item in value)
    return False


def _first_torch_tensor(value):
    if torch is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_torch_tensor(item)
            if found is not None:
                return found
    return None


def _torch_complex_dtype(dtype):
    return torch.complex128 if dtype in (torch.float64, torch.complex128) else torch.complex64


def _torch_real_dtype(dtype):
    return torch.float64 if dtype == torch.complex128 else torch.float32


def _torch_angle(value, backend):
    ref = _first_torch_tensor(value)
    backend_dtype = getattr(backend, "_dtype", torch.complex64)
    device = getattr(backend, "_device", ref.device if ref is not None else None)
    real_dtype = _torch_real_dtype(backend_dtype)
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device)
        return torch.real(tensor) if torch.is_complex(tensor) else tensor.to(dtype=real_dtype)
    return torch.tensor(float(value), dtype=real_dtype, device=device)


def _torch_complex(real, imag=None, complex_dtype=None):
    if imag is None:
        imag = torch.zeros_like(real)
    out = torch.complex(real, imag)
    return out.to(dtype=complex_dtype or _torch_complex_dtype(out.dtype))


def _torch_expi(angle, complex_dtype=None):
    """Return ``exp(i * angle)`` for a real tensor as ``cos + i*sin``.

    Equivalent to ``torch.exp(_torch_complex(0, angle))`` but built from real
    ``cos``/``sin``. Ascend NPU has no complex64 ``exp`` kernel, so the direct
    complex-exp form raises there; this real-valued form runs on CPU/CUDA/NPU
    alike and is autograd-friendly.
    """
    return _torch_complex(torch.cos(angle), torch.sin(angle), complex_dtype)


def _torch_base_matrix(entries, dtype, device):
    rows = []
    for row in entries:
        rows.append(
            torch.stack(
                [
                    item if isinstance(item, torch.Tensor) else torch.tensor(item, dtype=dtype, device=device)
                    for item in row
                ]
            )
        )
    return torch.stack(rows).to(dtype=dtype, device=device)


def _single_qubit_base_for_gate_backend(gate, backend):
    gate = normalize_gate(gate)
    parameter = gate.get("parameter", None)
    if not _contains_torch_tensor(parameter):
        return _single_qubit_base_for_gate(gate)

    gate_type = canonical_gate_name(gate["type"])
    ref = _first_torch_tensor(parameter)
    backend_dtype = getattr(backend, "_dtype", torch.complex64)
    dtype = _torch_complex_dtype(backend_dtype)
    device = getattr(backend, "_device", ref.device if ref is not None else None)
    zero = torch.zeros((), dtype=_torch_real_dtype(dtype), device=device)
    one = torch.ones((), dtype=_torch_real_dtype(dtype), device=device)

    if gate_type in ["rx", "crx"]:
        t = _torch_angle(parameter, backend)
        cos = torch.cos(t / 2.0)
        sin = torch.sin(t / 2.0)
        return _torch_base_matrix(
            [
                [_torch_complex(cos, complex_dtype=dtype), _torch_complex(zero, -sin, dtype)],
                [_torch_complex(zero, -sin, dtype), _torch_complex(cos, complex_dtype=dtype)],
            ],
            dtype,
            device,
        )
    if gate_type in ["ry", "cry"]:
        t = _torch_angle(parameter, backend)
        cos = torch.cos(t / 2.0)
        sin = torch.sin(t / 2.0)
        return _torch_base_matrix(
            [
                [_torch_complex(cos, complex_dtype=dtype), _torch_complex(-sin, complex_dtype=dtype)],
                [_torch_complex(sin, complex_dtype=dtype), _torch_complex(cos, complex_dtype=dtype)],
            ],
            dtype,
            device,
        )
    if gate_type in ["rz", "crz"]:
        t = _torch_angle(parameter, backend)
        exp_neg = _torch_expi(-t / 2.0, dtype)
        exp_pos = _torch_expi(t / 2.0, dtype)
        return _torch_base_matrix(
            [[exp_neg, _torch_complex(zero, complex_dtype=dtype)], [_torch_complex(zero, complex_dtype=dtype), exp_pos]],
            dtype,
            device,
        )
    if gate_type == "u3":
        theta = _torch_angle(parameter[0], backend)
        phi = _torch_angle(parameter[1], backend)
        lam = _torch_angle(parameter[2], backend)
        cos = torch.cos(theta / 2.0)
        sin = torch.sin(theta / 2.0)
        # exp(i*x) * r == complex(r*cos x, r*sin x). Build each phased entry
        # directly from real parts instead of multiplying complex tensors: NPU
        # has no complex64 mul kernel (aclnnMul) and its backward would also do
        # complex add. Real construction is NPU-safe and autograd-correct.
        return _torch_base_matrix(
            [
                [_torch_complex(cos, complex_dtype=dtype),
                 _torch_complex(-sin * torch.cos(lam), -sin * torch.sin(lam), dtype)],
                [_torch_complex(sin * torch.cos(phi), sin * torch.sin(phi), dtype),
                 _torch_complex(cos * torch.cos(phi + lam), cos * torch.sin(phi + lam), dtype)],
            ],
            dtype,
            device,
        )
    if gate_type == "u2":
        phi = _torch_angle(parameter[0], backend)
        lam = _torch_angle(parameter[1], backend)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        # Same real-part construction as u3 (no complex multiply); the fixed
        # cos = sin = 1/sqrt(2) of u2 fold into the real scale factor.
        return _torch_base_matrix(
            [
                [_torch_complex(one * inv_sqrt2, complex_dtype=dtype),
                 _torch_complex(-inv_sqrt2 * torch.cos(lam), -inv_sqrt2 * torch.sin(lam), dtype)],
                [_torch_complex(inv_sqrt2 * torch.cos(phi), inv_sqrt2 * torch.sin(phi), dtype),
                 _torch_complex(inv_sqrt2 * torch.cos(phi + lam), inv_sqrt2 * torch.sin(phi + lam), dtype)],
            ],
            dtype,
            device,
        )

    return _single_qubit_base_for_gate(gate)


def _rzz_backend(theta, backend):
    if not _contains_torch_tensor(theta):
        return backend.cast(_rzz(theta))

    ref = _first_torch_tensor(theta)
    backend_dtype = getattr(backend, "_dtype", torch.complex64)
    dtype = _torch_complex_dtype(backend_dtype)
    device = getattr(backend, "_device", ref.device if ref is not None else None)
    zero = torch.zeros((), dtype=_torch_real_dtype(dtype), device=device)
    t = _torch_angle(theta, backend)
    # Each diagonal phase must be a *fresh* complex tensor: reusing one
    # grad-bearing complex tensor across cells makes autograd accumulate its
    # gradient with a complex add, which Ascend NPU's aclnnAdd cannot do for
    # complex64. Distinct tensors push the accumulation onto the real angle t
    # (real adds, which NPU supports). ``z`` is constant (no grad) so reuse is
    # fine. See _single_qubit_base_for_gate_backend for the same rule.
    z = _torch_complex(zero, complex_dtype=dtype)
    return _torch_base_matrix(
        [
            [_torch_expi(-t / 2.0, dtype), z, z, z],
            [z, _torch_expi(t / 2.0, dtype), z, z],
            [z, z, _torch_expi(t / 2.0, dtype), z],
            [z, z, z, _torch_expi(-t / 2.0, dtype)],
        ],
        dtype,
        device,
    )


def _rxx_backend(theta, backend):
    if not _contains_torch_tensor(theta):
        return backend.cast(_rxx(theta))

    ref = _first_torch_tensor(theta)
    backend_dtype = getattr(backend, "_dtype", torch.complex64)
    dtype = _torch_complex_dtype(backend_dtype)
    device = getattr(backend, "_device", ref.device if ref is not None else None)
    zero = torch.zeros((), dtype=_torch_real_dtype(dtype), device=device)
    t = _torch_angle(theta, backend)
    cos = torch.cos(t / 2.0)
    sin = torch.sin(t / 2.0)
    # Build each grad-bearing cell as a fresh complex tensor (see _rzz_backend):
    # reusing one complex tensor across cells would force a complex64 add in
    # autograd's gradient accumulation, which Ascend NPU lacks. The real
    # ``cos``/``sin`` may be shared (their accumulation is a real add). ``z`` is
    # constant (no grad), so it is safe to reuse.
    def cell_c():
        return _torch_complex(cos, complex_dtype=dtype)

    def cell_nis():
        return _torch_complex(zero, -sin, dtype)

    z = _torch_complex(zero, complex_dtype=dtype)
    return _torch_base_matrix(
        [
            [cell_c(), z, z, cell_nis()],
            [z, cell_c(), cell_nis(), z],
            [z, cell_nis(), cell_c(), z],
            [cell_nis(), z, z, cell_c()],
        ],
        dtype,
        device,
    )


def _double_excitation(theta, q1=0, q2=1, q3=2, q4=3):
    _ = (q1, q2, q3, q4)
    t = float(theta)
    c = math.cos(t / 2.0)
    s = math.sin(t / 2.0)
    m = np.eye(16, dtype=_CDTYPE)
    m[3, 3] = c
    m[3, 12] = -s
    m[12, 3] = s
    m[12, 12] = c
    return m


def _single_excitation_backend(theta, backend):
    if not _contains_torch_tensor(theta):
        return backend.cast(_single_excitation(theta))

    ref = _first_torch_tensor(theta)
    backend_dtype = getattr(backend, "_dtype", torch.complex64)
    dtype = _torch_complex_dtype(backend_dtype)
    device = getattr(backend, "_device", ref.device if ref is not None else None)
    zero = torch.zeros((), dtype=_torch_real_dtype(dtype), device=device)
    one_r = torch.ones((), dtype=_torch_real_dtype(dtype), device=device)
    t = _torch_angle(theta, backend)
    cos = torch.cos(t / 2.0)
    sin = torch.sin(t / 2.0)
    neg_sin = -sin
    # 每个含梯度复数 cell 必须是新张量（见 _rxx_backend 说明）；实数 cos/sin 可共享，
    # 常量 z/one 可重用。
    z = _torch_complex(zero, complex_dtype=dtype)
    one = _torch_complex(one_r, complex_dtype=dtype)
    return _torch_base_matrix(
        [
            [one, z, z, z],
            [z, _torch_complex(cos, complex_dtype=dtype), _torch_complex(neg_sin, complex_dtype=dtype), z],
            [z, _torch_complex(sin, complex_dtype=dtype), _torch_complex(cos, complex_dtype=dtype), z],
            [z, z, z, one],
        ],
        dtype,
        device,
    )


def _double_excitation_backend(theta, backend):
    if not _contains_torch_tensor(theta):
        return backend.cast(_double_excitation(theta))

    ref = _first_torch_tensor(theta)
    backend_dtype = getattr(backend, "_dtype", torch.complex64)
    dtype = _torch_complex_dtype(backend_dtype)
    device = getattr(backend, "_device", ref.device if ref is not None else None)
    zero = torch.zeros((), dtype=_torch_real_dtype(dtype), device=device)
    one_r = torch.ones((), dtype=_torch_real_dtype(dtype), device=device)
    t = _torch_angle(theta, backend)
    cos = torch.cos(t / 2.0)
    sin = torch.sin(t / 2.0)
    neg_sin = -sin
    z = _torch_complex(zero, complex_dtype=dtype)
    one = _torch_complex(one_r, complex_dtype=dtype)
    # 16×16 单位阵，仅 (3,3),(3,12),(12,3),(12,12) 为含梯度新张量（fresh cell 规则）。
    special = {
        (3, 3): lambda: _torch_complex(cos, complex_dtype=dtype),
        (3, 12): lambda: _torch_complex(neg_sin, complex_dtype=dtype),
        (12, 3): lambda: _torch_complex(sin, complex_dtype=dtype),
        (12, 12): lambda: _torch_complex(cos, complex_dtype=dtype),
    }
    rows = []
    for i in range(16):
        row = []
        for j in range(16):
            if (i, j) in special:
                row.append(special[(i, j)]())
            elif i == j:
                row.append(one)
            else:
                row.append(z)
        rows.append(row)
    return _torch_base_matrix(rows, dtype, device)


def _basis_bits(index: int, n_qubits: int):
    return [(index >> (n_qubits - 1 - qubit)) & 1 for qubit in range(n_qubits)]


def _bits_to_index(bits):
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


def _expand_local_matrix_to_full(local_matrix, axes, n_qubits: int, backend=None):
    axes = [int(axis) for axis in axes]
    if len(set(axes)) != len(axes):
        raise ValueError("局部门作用的量子比特不能重复")
    if any(axis < 0 or axis >= n_qubits for axis in axes):
        raise ValueError("局部门作用的量子比特索引超出范围")

    dim = 1 << int(n_qubits)
    dim_local = 1 << len(axes)
    if tuple(local_matrix.shape) != (dim_local, dim_local):
        raise ValueError(
            f"局部门矩阵维度 {tuple(local_matrix.shape)} 与作用量子比特数量 {len(axes)} 不一致"
        )

    if torch is not None and isinstance(local_matrix, torch.Tensor):
        zero = local_matrix.new_tensor(0.0 + 0.0j)
        rows = []
        for row_index in range(dim):
            row_bits = _basis_bits(row_index, n_qubits)
            row_entries = []
            for col_index in range(dim):
                col_bits = _basis_bits(col_index, n_qubits)
                if any(row_bits[axis] != col_bits[axis] for axis in range(n_qubits) if axis not in axes):
                    row_entries.append(zero)
                    continue
                local_row = _bits_to_index(row_bits[axis] for axis in axes)
                local_col = _bits_to_index(col_bits[axis] for axis in axes)
                row_entries.append(local_matrix[local_row, local_col])
            rows.append(torch.stack(row_entries))
        return torch.stack(rows)

    local_np = np.asarray(local_matrix, dtype=_CDTYPE)
    full = np.zeros((dim, dim), dtype=_CDTYPE)
    for row_index in range(dim):
        row_bits = _basis_bits(row_index, n_qubits)
        for col_index in range(dim):
            col_bits = _basis_bits(col_index, n_qubits)
            if any(row_bits[axis] != col_bits[axis] for axis in range(n_qubits) if axis not in axes):
                continue
            local_row = _bits_to_index(row_bits[axis] for axis in axes)
            local_col = _bits_to_index(col_bits[axis] for axis in axes)
            full[row_index, col_index] = local_np[local_row, local_col]

    return backend.cast(full) if backend is not None else full


def _inverse_permutation(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def _permute_tensor(tensor, perm):
    if torch is not None and isinstance(tensor, torch.Tensor):
        return tensor.permute(perm)
    return np.transpose(tensor, perm)


def _contiguous_if_torch(tensor):
    return tensor.contiguous() if torch is not None and isinstance(tensor, torch.Tensor) else tensor


def _should_use_flat_local_apply(backend, n_qubits: int) -> bool:
    device = getattr(backend, "_device", None)
    return bool(getattr(device, "type", None) == "npu" and int(n_qubits) > 8)


def _parameter_cache_key(value):
    if _contains_torch_tensor(value):
        return None
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(_parameter_cache_key(v) for v in value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return repr(value)


def _cast_local_matrix(backend, matrix, cache_key=None):
    if hasattr(backend, "cast_local_matrix"):
        return backend.cast_local_matrix(matrix, cache_key=cache_key)

    if cache_key is not None:
        cache = getattr(backend, "_local_matrix_cache", None)
        if cache is None:
            cache = {}
            setattr(backend, "_local_matrix_cache", cache)
        if cache_key in cache:
            return cache[cache_key]
        casted = backend.cast(matrix)
        cache[cache_key] = casted
        return casted

    return backend.cast(matrix)


def _apply_local_matrix_to_state(state, local_matrix, axes, n_qubits, backend):
    axes = [int(axis) for axis in axes]
    if len(set(axes)) != len(axes):
        raise ValueError("局部门作用的量子比特不能重复")
    if any(axis < 0 or axis >= n_qubits for axis in axes):
        raise ValueError("局部门作用的量子比特索引超出范围")

    dim_local = 1 << len(axes)
    if local_matrix.shape != (dim_local, dim_local):
        raise ValueError(
            f"局部门矩阵维度 {local_matrix.shape} 与作用量子比特数量 {len(axes)} 不一致"
        )

    if _should_use_flat_local_apply(backend, n_qubits):
        return _apply_local_matrix_to_state_flat(state, local_matrix, axes, n_qubits, backend)

    # 将态向量整形为「目标比特轴 + 合并后的空闲段」的分组张量，而非按比特展开
    # 成 (2,)*n 的高阶张量。后者在 20+ 量子比特时秩高达 n，超出昇腾 NPU ACL
    # 算子最多 8 维的限制（aclnnInplaceCopy 报错）。这里把相邻的非目标比特合并
    # 成单个维度，使工作张量的秩至多为 2*len(axes)+1（Toffoli 也仅 7 维）。
    target_set = set(axes)
    group_shape = []          # 分组后各维度大小（行优先，轴 0 为最高位）
    target_dim_index = {}     # 目标比特 -> 其在 group_shape 中的维度下标
    gap_dim_indices = []      # 合并后的空闲段维度下标
    pending_gap = 1

    def _flush_gap():
        nonlocal pending_gap
        if pending_gap > 1:
            gap_dim_indices.append(len(group_shape))
            group_shape.append(pending_gap)
            pending_gap = 1

    for qubit in range(n_qubits):
        if qubit in target_set:
            _flush_gap()
            target_dim_index[qubit] = len(group_shape)
            group_shape.append(2)
        else:
            pending_gap *= 2
    _flush_gap()

    # 前置目标轴需保持 axes 给定的顺序（决定局部矩阵的基排序），其后接空闲段。
    perm = [target_dim_index[axis] for axis in axes] + gap_dim_indices
    inv_perm = _inverse_permutation(perm)

    psi = state.reshape(group_shape)
    moved = _contiguous_if_torch(_permute_tensor(psi, perm))
    flat = moved.reshape(dim_local, -1)
    if hasattr(backend, "apply_local_matrix"):
        updated = backend.apply_local_matrix(local_matrix, flat)
    else:
        updated = backend.matmul(local_matrix, flat)
    restored = updated.reshape([2] * len(axes) + [group_shape[g] for g in gap_dim_indices])
    restored = _contiguous_if_torch(_permute_tensor(restored, inv_perm))
    return restored.reshape(1 << n_qubits, 1)


def _flat_local_state_indices(axes, n_qubits: int):
    axes = tuple(int(axis) for axis in axes)
    dim = 1 << int(n_qubits)
    basis = np.arange(dim, dtype=np.int64)
    base_mask = np.ones(dim, dtype=bool)
    for axis in axes:
        base_mask &= ((basis >> (int(n_qubits) - 1 - axis)) & 1) == 0
    base_indices = basis[base_mask]

    rows = []
    for local_index in range(1 << len(axes)):
        offset = 0
        for local_pos, axis in enumerate(axes):
            bit = (local_index >> (len(axes) - 1 - local_pos)) & 1
            offset |= bit << (int(n_qubits) - 1 - axis)
        rows.append(base_indices | np.int64(offset))
    return np.stack(rows, axis=0)


def _backend_index_tensor(indices, reference):
    if torch is not None and isinstance(reference, torch.Tensor):
        return torch.as_tensor(indices, dtype=torch.long, device=reference.device)
    return indices


def _is_npu_complex_tensor(value, backend) -> bool:
    if torch is None or not isinstance(value, torch.Tensor):
        return False
    device = getattr(backend, "_device", getattr(value, "device", None))
    return bool(getattr(device, "type", None) == "npu" and torch.is_complex(value))


def _apply_local_matrix_to_state_flat(state, local_matrix, axes, n_qubits, backend):
    axes = [int(axis) for axis in axes]
    dim_local = 1 << len(axes)
    if local_matrix.shape != (dim_local, dim_local):
        raise ValueError(
            f"local gate matrix shape {local_matrix.shape} does not match {len(axes)} target qubit(s)"
        )

    flat = state.reshape(-1)
    indices = _backend_index_tensor(_flat_local_state_indices(axes, n_qubits), flat)
    npu_complex = _is_npu_complex_tensor(flat, backend)

    # NPU autograd 安全路径：flat 和 local_matrix 各只在图中出现一次，
    # backward 全程 float32，避免 aclnnAdd(DT_COMPLEX64)。
    if npu_complex and hasattr(backend, "apply_flat_gate"):
        return backend.apply_flat_gate(flat, local_matrix, indices).reshape(1 << n_qubits, 1)

    if npu_complex:
        gathered = torch.complex(torch.real(flat)[indices], torch.imag(flat)[indices])
    else:
        gathered = flat[indices]
    if hasattr(backend, "apply_local_matrix"):
        updated = backend.apply_local_matrix(local_matrix, gathered)
    else:
        updated = backend.matmul(local_matrix, gathered)

    if torch is not None and isinstance(flat, torch.Tensor):
        if npu_complex:
            out_real = torch.empty_like(torch.real(flat))
            out_imag = torch.empty_like(torch.imag(flat))
            out_real[indices.reshape(-1)] = torch.real(updated).reshape(-1)
            out_imag[indices.reshape(-1)] = torch.imag(updated).reshape(-1)
            out = torch.complex(out_real, out_imag)
        else:
            out = torch.empty_like(flat)
            out[indices.reshape(-1)] = updated.reshape(-1)
    else:
        out = np.empty_like(flat)
        out[indices.reshape(-1)] = np.asarray(updated).reshape(-1)
    return out.reshape(1 << n_qubits, 1)


def _single_qubit_base_for_gate(gate):
    gate = normalize_gate(gate)
    gate_type = canonical_gate_name(gate["type"])
    gate_parameter = gate.get("parameter", None)

    if gate_type in ["pauli_x", "cx", "toffoli"]:
        return np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=_CDTYPE)
    if gate_type in ["pauli_y", "cy"]:
        return np.array([[0.0 + 0.0j, -1j], [1j, 0.0 + 0.0j]], dtype=_CDTYPE)
    if gate_type in ["pauli_z", "cz"]:
        return np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]], dtype=_CDTYPE)
    if gate_type == "hadamard":
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        return np.array(
            [[sqrt2_inv + 0.0j, sqrt2_inv + 0.0j], [sqrt2_inv + 0.0j, -sqrt2_inv + 0.0j]],
            dtype=_CDTYPE,
        )
    if gate_type == "s_gate":
        return np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1j]], dtype=_CDTYPE)
    if gate_type == "t_gate":
        return np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, np.exp(1j * math.pi / 4.0)]], dtype=_CDTYPE)
    if gate_type in ["rx", "crx"]:
        t = float(gate_parameter)
        cos = math.cos(t / 2.0)
        sin = math.sin(t / 2.0)
        return np.array([[cos, -1j * sin], [-1j * sin, cos]], dtype=_CDTYPE)
    if gate_type in ["ry", "cry"]:
        t = float(gate_parameter)
        cos = math.cos(t / 2.0)
        sin = math.sin(t / 2.0)
        return np.array([[cos, -sin], [sin, cos]], dtype=_CDTYPE)
    if gate_type in ["rz", "crz"]:
        t = float(gate_parameter)
        return np.array([[np.exp(-1j * t / 2.0), 0.0 + 0.0j], [0.0 + 0.0j, np.exp(1j * t / 2.0)]], dtype=_CDTYPE)
    if gate_type == "u3":
        theta, phi, lam = float(gate_parameter[0]), float(gate_parameter[1]), float(gate_parameter[2])
        cos = math.cos(theta / 2.0)
        sin = math.sin(theta / 2.0)
        return np.array(
            [[cos, -np.exp(1j * lam) * sin], [np.exp(1j * phi) * sin, np.exp(1j * (phi + lam)) * cos]],
            dtype=_CDTYPE,
        )
    if gate_type == "u2":
        phi, lam = float(gate_parameter[0]), float(gate_parameter[1])
        cos = math.cos(math.pi / 4.0)
        sin = math.sin(math.pi / 4.0)
        return np.array(
            [[cos, -np.exp(1j * lam) * sin], [np.exp(1j * phi) * sin, np.exp(1j * (phi + lam)) * cos]],
            dtype=_CDTYPE,
        )
    return None


def _controlled_local_from_base(base, control_states):
    """把 ``base``（作用于 ``k`` 个目标比特的 ``2^k × 2^k`` 局部矩阵）包裹为受控门局部矩阵。

    当全部控制比特命中 ``control_states`` 时对目标施加 ``base``，否则为恒等。目标比特可
    多于一个（``k >= 1``）：单比特底门（``k=1``，如 ``cx`` 的 X 底门）与多比特底门
    （如受控 ``swap`` 的 4x4 底门）统一处理。
    """
    control_states = [int(state) for state in control_states]
    for state in control_states:
        if state not in (0, 1):
            raise ValueError("control_states 只能包含 0 或 1")
    n_controls = len(control_states)
    base_dim = int(base.shape[0])
    n_targets = base_dim.bit_length() - 1
    if (1 << n_targets) != base_dim:
        raise ValueError("controlled 底门维度必须是 2 的幂")
    dim = 1 << (n_controls + n_targets)
    control_index = 0
    for state in control_states:
        control_index = (control_index << 1) | state
    block_indices = [(control_index << n_targets) | target for target in range(base_dim)]

    if torch is not None and isinstance(base, torch.Tensor):
        zero = base.new_tensor(0.0 + 0.0j)
        one = base.new_tensor(1.0 + 0.0j)
        rows = []
        for row in range(dim):
            row_entries = []
            for col in range(dim):
                if row in block_indices and col in block_indices:
                    row_entries.append(base[block_indices.index(row), block_indices.index(col)])
                elif row == col:
                    row_entries.append(one)
                else:
                    row_entries.append(zero)
            rows.append(torch.stack(row_entries))
        return torch.stack(rows)

    local = np.eye(dim, dtype=_CDTYPE)
    local[np.ix_(block_indices, block_indices)] = np.asarray(base, dtype=_CDTYPE)
    return local


_MULTI_TARGET_CONTROLLED = {"cx", "cy", "cz", "crx", "cry", "crz"}


def _multi_target_subgates(gate):
    """多目标受控门按目标展开为逐目标的单目标门列表；无需展开时返回 ``None``。

    ``cx([t1, t2], [controls])`` 携带 ``qubits`` 键；展开后的各单目标门彼此
    对易，故展开顺序不影响结果。单目标门（``target_qubit`` 形式）返回 ``None``。
    """
    if canonical_gate_name(gate["type"]) not in _MULTI_TARGET_CONTROLLED:
        return None
    targets = gate.get("qubits")
    if targets is None:
        return None
    subgates = []
    for target in targets:
        subgate = {key: value for key, value in gate.items() if key != "qubits"}
        subgate["target_qubit"] = int(target)
        subgates.append(subgate)
    return subgates


_CONTROLLED_BASE_GATE = {
    "cx": "pauli_x",
    "cy": "pauli_y",
    "cz": "pauli_z",
    "crx": "rx",
    "cry": "ry",
    "crz": "rz",
    "toffoli": "pauli_x",
}


def _gate_axes(gate):
    """门作用比特轴序：``qubits`` > ``qubit_1``/``qubit_2`` > ``target_qubit``。"""
    if gate.get("qubits") is not None:
        return [int(q) for q in gate["qubits"]]
    if gate.get("qubit_1") is not None and gate.get("qubit_2") is not None:
        return [int(gate["qubit_1"]), int(gate["qubit_2"])]
    if gate.get("target_qubit") is not None:
        return [int(gate["target_qubit"])]
    raise ValueError("门缺少 qubits/qubit_1/qubit_2/target_qubit，无法定位作用比特")


def _gate_local_matrix(gate, gate_type, backend):
    """门局部矩阵的**唯一来源**（Approach A）：返回 ``(local, axes, cache_key)``。

    受控门取底门局部矩阵再由 ``_controlled_local_from_base`` 按本门实例的 ``control_states``
    包裹，底门有两种：内置受控类型（``_CONTROLLED_BASE_GATE`` 映射到不同门名，如
    ``cx→pauli_x``）；或**任意携带 ``control_qubits`` 的门**（底门即该门自身的非受控矩阵，
    要求为单比特 2x2），从而支持受控自定义门。非受控门直接取注册表局部矩阵。无局部矩阵
    （未注册门 / ``measure`` 等）返回 ``(None, None, None)``。
    """
    from ..gates import gate_matrix as _registry_gate_matrix

    parameter = gate.get("parameter")

    if gate.get("control_qubits"):
        controls, control_states = _normalized_control_data(gate)
        base_name = _CONTROLLED_BASE_GATE.get(gate_type, gate_type)
        base = _registry_gate_matrix(base_name, parameter, backend)
        target_axes = _gate_axes(gate)
        if base is None:
            return None, None, None
        base_dim = int(base.shape[0])
        if (base_dim & (base_dim - 1)) != 0 or (1 << len(target_axes)) != base_dim:
            # 底门未注册/非 2 的幂，或目标比特数与底门维度不符。
            return None, None, None
        local = _controlled_local_from_base(base, control_states)
        axes = controls + target_axes
        cache_key = None if _contains_torch_tensor(parameter) else (
            "ctrl",
            gate_type,
            len(controls),
            tuple(int(state) for state in control_states),
            _parameter_cache_key(parameter),
        )
        return local, axes, cache_key

    local = _registry_gate_matrix(gate_type, parameter, backend)
    if local is None:
        return None, None, None
    cache_key = None if _contains_torch_tensor(parameter) else (
        "local",
        gate_type,
        _parameter_cache_key(parameter),
    )
    return local, _gate_axes(gate), cache_key


def apply_gate_to_state(gate, state, n_qubits: int, backend):
    """
    直接将局部门作用到态向量，避免构造 2^n × 2^n 全局矩阵。

    局部矩阵统一取自 ``_gate_local_matrix``（注册表单一来源）。若门类型无法局部
    展开则返回 None，调用方可回退到 gate_to_matrix + apply_unitary。
    """
    gate = normalize_gate(gate)
    gate_type = canonical_gate_name(gate["type"])

    subgates = _multi_target_subgates(gate)
    if subgates is not None:
        for subgate in subgates:
            state = apply_gate_to_state(subgate, state, n_qubits, backend)
        return state

    if gate_type == "identity":
        return state

    if gate_type == "unitary":
        matrix = _unitary_parameter_matrix(gate.get("parameter"), backend)
        shape = tuple(int(dim) for dim in matrix.shape)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("unitary 门参数必须是方阵")
        dim = shape[0]
        inferred = int(round(math.log2(dim))) if dim > 0 else 0
        if (1 << inferred) != dim:
            raise ValueError("unitary 门矩阵维度必须是 2 的幂")
        gate_qubits = int(gate.get("n_qubits", inferred))
        if (1 << gate_qubits) != dim:
            raise ValueError("unitary 门的 n_qubits 与矩阵维度不一致")
        if gate_qubits > n_qubits:
            raise ValueError(f"量子门的量子比特数量超出总量子比特数: {gate_qubits} > {n_qubits}")
        return _apply_local_matrix_to_state(
            state,
            matrix,
            list(range(gate_qubits)),
            n_qubits,
            backend,
        )

    local, axes, cache_key = _gate_local_matrix(gate, gate_type, backend)
    if local is None:
        return None
    return _apply_local_matrix_to_state(
        state,
        _cast_local_matrix(backend, local, cache_key=cache_key),
        axes,
        n_qubits,
        backend,
    )


def _local_target_qubits(gate):
    """自定义门回退路径的目标比特列表（``qubits`` 优先，退而 ``target_qubit``）。"""
    if gate.get("qubits") is not None:
        return [int(q) for q in gate["qubits"]]
    if gate.get("target_qubit") is not None:
        return [int(gate["target_qubit"])]
    raise ValueError("门缺少 qubits/target_qubit，无法定位作用比特")


def gate_to_matrix(gate, cir_qubits=1, backend=None):
    gate = normalize_gate(gate)
    gate_type = canonical_gate_name(gate["type"])
    gate_parameter = gate.get("parameter", None)

    subgates = _multi_target_subgates(gate)
    if subgates is not None:
        result = None
        for subgate in subgates:
            sub_matrix = gate_to_matrix(subgate, cir_qubits, backend=backend)
            if result is None:
                result = sub_matrix
            elif backend is None:
                result = np.matmul(sub_matrix, result)
            else:
                result = backend.matmul(sub_matrix, result)
        return result

    if gate_type == "unitary":
        matrix = _unitary_parameter_matrix(gate_parameter, backend)
        shape = tuple(int(dim) for dim in matrix.shape)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("unitary 门参数必须是方阵")

        dim = shape[0]
        inferred = int(round(math.log2(dim))) if dim > 0 else 0
        if (1 << inferred) != dim:
            raise ValueError("unitary 门矩阵维度必须是 2 的幂")

        gate_qubits = gate.get("n_qubits", inferred)
        if (1 << int(gate_qubits)) != dim:
            raise ValueError("unitary 门的 n_qubits 与矩阵维度不一致")

        gate_matrix = matrix
        if gate_qubits < cir_qubits:
            for _ in range(gate_qubits, cir_qubits):
                if backend is None:
                    gate_matrix = np.kron(gate_matrix, IDENTITY_2)
                else:
                    gate_matrix = backend.kron(gate_matrix, backend.cast(IDENTITY_2))
        elif gate_qubits > cir_qubits:
            raise ValueError(f"量子门的量子比特数量超出总量子比特数: {gate_qubits} > {cir_qubits}")
        return gate_matrix

    if gate_type == "identity":
        return identity(cir_qubits) if backend is None else backend.eye(1 << cir_qubits)

    # Approach A：非 unitary/identity 门统一经 _gate_local_matrix（注册表单一来源）
    # 取局部矩阵，再嵌入到整线路空间；受控与自定义门同此一条路径。
    local, axes, _ = _gate_local_matrix(gate, gate_type, backend)
    if local is None:
        raise ValueError(f"不支持的门类型: {gate_type}")
    return _expand_local_matrix_to_full(local, axes, int(cir_qubits), backend=backend)


def gate_tensors(gate, backend):
    """把门转为张量列表 ``[(matrix, axes), ...]``（TN 构建单一来源）。

    - 多目标受控门（``cx([t1,t2],[c])`` 等）展开为逐目标项；
    - ``identity``/``measure`` 返回 ``[]``（不产生节点）；
    - ``unitary`` 自定义门用其参数矩阵；其余走 ``_gate_local_matrix``；
    - 无法局部展开的门（未注册/含未绑定参数）抛 ``ValueError``。
    ``matrix`` 为 ``2^k×2^k`` 后端张量，``axes`` 为其作用比特轴序（controls+targets）。
    """
    gate = normalize_gate(gate)
    subgates = _multi_target_subgates(gate)
    if subgates is not None:
        result = []
        for subgate in subgates:
            result.extend(gate_tensors(subgate, backend))
        return result

    gate_type = canonical_gate_name(gate["type"])
    if gate_type in ("identity", "measure"):
        return []

    if gate_type == "unitary":
        matrix = _unitary_parameter_matrix(gate.get("parameter"), backend)
        return [(matrix, _local_target_qubits(gate))]

    local, axes, _ = _gate_local_matrix(gate, gate_type, backend)
    if local is None:
        raise ValueError(f"门 {gate_type!r} 无法转为张量（未注册/含未绑定参数/measure）")
    return [(local, list(axes))]


# ---------------------------------------------------------------------------
# GateSpec.matrix 局部矩阵构造器（NEXT.md §7）。复用上面的局部矩阵原语，于本
# 模块导入时附加到注册表（避免 gates ↔ core 循环导入）。仅不受控门有局部矩阵；
# 受控门/measure/reset 不在此抽象内。
# ---------------------------------------------------------------------------


def _local_single_builder(name):
    def build(params, backend):
        gate = {"type": name, "target_qubit": 0, "parameter": params}
        if backend is None:
            return _single_qubit_base_for_gate(gate)
        return _single_qubit_base_for_gate_backend(gate, backend)

    return build


def _local_swap_builder(params, backend):
    return _swap(0, 1)


def _local_rzz_builder(params, backend):
    return _rzz(params) if backend is None else _rzz_backend(params, backend)


def _local_rxx_builder(params, backend):
    return _rxx(params) if backend is None else _rxx_backend(params, backend)


def _local_single_excitation_builder(params, backend):
    if backend is None:
        return _single_excitation(params, 0, 1)
    return _single_excitation_backend(params, backend)


def _local_double_excitation_builder(params, backend):
    if backend is None:
        return _double_excitation(params, 0, 1, 2, 3)
    return _double_excitation_backend(params, backend)


_LOCAL_MATRIX_BUILDERS = {
    "pauli_x": _local_single_builder("pauli_x"),
    "pauli_y": _local_single_builder("pauli_y"),
    "pauli_z": _local_single_builder("pauli_z"),
    "hadamard": _local_single_builder("hadamard"),
    "s_gate": _local_single_builder("s_gate"),
    "t_gate": _local_single_builder("t_gate"),
    "rx": _local_single_builder("rx"),
    "ry": _local_single_builder("ry"),
    "rz": _local_single_builder("rz"),
    "u2": _local_single_builder("u2"),
    "u3": _local_single_builder("u3"),
    "swap": _local_swap_builder,
    "rzz": _local_rzz_builder,
    "rxx": _local_rxx_builder,
    "single_excitation": _local_single_excitation_builder,
    "double_excitation": _local_double_excitation_builder,
}

from ..gates import set_gate_matrix as _set_gate_matrix

for _name, _builder in _LOCAL_MATRIX_BUILDERS.items():
    _set_gate_matrix(_name, _builder)
del _name, _builder
