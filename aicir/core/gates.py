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


def _swap_backend(qubit_1=0, qubit_2=1, backend=None):
    return backend.matrix_product(
        _controlled_from_base_backend(np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=_CDTYPE), qubit_1, [qubit_2], [1], backend),
        _controlled_from_base_backend(np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=_CDTYPE), qubit_2, [qubit_1], [1], backend),
        _controlled_from_base_backend(np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=_CDTYPE), qubit_1, [qubit_2], [1], backend),
    )


def _toffoli_backend(target_qubit=2, control_qubits=(0, 1), control_states=None, backend=None):
    control_qubits = list(control_qubits)
    if control_states is None:
        control_states = [1] * len(control_qubits)
    return _controlled_from_base_backend(
        np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=_CDTYPE),
        target_qubit,
        control_qubits,
        control_states,
        backend,
    )


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


def _normalized_control_data(gate):
    controls = list(gate["control_qubits"])
    control_states = list(gate.get("control_states", [1] * len(controls)))
    if len(control_states) != len(controls):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")
    return controls, control_states


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


def _toffoli(target_qubit=2, control_qubits=(0, 1), control_states=None):
    control_qubits = list(control_qubits)
    if control_states is None:
        control_states = [1] * len(control_qubits)
    return _controlled_from_base(_pauli_x(), target_qubit, control_qubits, control_states)


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

    rest_axes = [axis for axis in range(n_qubits) if axis not in axes]
    perm = axes + rest_axes
    inv_perm = _inverse_permutation(perm)

    psi = state.reshape([2] * n_qubits)
    moved = _contiguous_if_torch(_permute_tensor(psi, perm))
    flat = moved.reshape(dim_local, -1)
    if hasattr(backend, "apply_local_matrix"):
        updated = backend.apply_local_matrix(local_matrix, flat)
    else:
        updated = backend.matmul(local_matrix, flat)
    restored = updated.reshape([2] * len(axes) + [2] * len(rest_axes))
    restored = _contiguous_if_torch(_permute_tensor(restored, inv_perm))
    return restored.reshape(1 << n_qubits, 1)


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


def _controlled_local_from_base(base_single, control_states):
    control_states = [int(state) for state in control_states]
    n_controls = len(control_states)
    dim = 1 << (n_controls + 1)
    if torch is not None and isinstance(base_single, torch.Tensor):
        rows = []
        control_index = 0
        for state in control_states:
            if state not in (0, 1):
                raise ValueError("control_states 只能包含 0 或 1")
            control_index = (control_index << 1) | state
        block_indices = [(control_index << 1) | target_state for target_state in (0, 1)]
        zero = base_single.new_tensor(0.0 + 0.0j)
        one = base_single.new_tensor(1.0 + 0.0j)
        for row in range(dim):
            row_entries = []
            for col in range(dim):
                if row in block_indices and col in block_indices:
                    row_entries.append(base_single[block_indices.index(row), block_indices.index(col)])
                elif row == col:
                    row_entries.append(one)
                else:
                    row_entries.append(zero)
            rows.append(torch.stack(row_entries))
        return torch.stack(rows)

    local = np.eye(dim, dtype=_CDTYPE)
    control_index = 0
    for state in control_states:
        if state not in (0, 1):
            raise ValueError("control_states 只能包含 0 或 1")
        control_index = (control_index << 1) | state

    block_indices = [(control_index << 1) | target_state for target_state in (0, 1)]
    local[np.ix_(block_indices, block_indices)] = np.asarray(base_single, dtype=_CDTYPE)
    return local


def apply_gate_to_state(gate, state, n_qubits: int, backend):
    """
    直接将局部门作用到态向量，避免构造 2^n × 2^n 全局矩阵。

    返回后端原生态向量；若门类型无法局部展开则返回 None，调用方可回退到
    gate_to_matrix + apply_unitary。
    """
    gate = normalize_gate(gate)
    gate_type = canonical_gate_name(gate["type"])

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

    if gate_type in [
        "pauli_x",
        "pauli_y",
        "pauli_z",
        "hadamard",
        "s_gate",
        "t_gate",
        "rx",
        "ry",
        "rz",
        "u3",
        "u2",
    ]:
        base = _single_qubit_base_for_gate_backend(gate, backend)
        parameter = gate.get("parameter")
        cache_key = None if _contains_torch_tensor(parameter) else ("single", gate_type, _parameter_cache_key(parameter))
        return _apply_local_matrix_to_state(
            state,
            _cast_local_matrix(backend, base, cache_key=cache_key),
            [gate["target_qubit"]],
            n_qubits,
            backend,
        )

    if gate_type in ["cx", "cy", "cz", "crx", "cry", "crz"]:
        controls, control_states = _normalized_control_data(gate)
        base = _single_qubit_base_for_gate_backend(gate, backend)
        local = _controlled_local_from_base(base, control_states)
        parameter = gate.get("parameter")
        cache_key = None if _contains_torch_tensor(parameter) else (
            "controlled",
            gate_type,
            len(controls),
            tuple(int(state) for state in control_states),
            _parameter_cache_key(parameter),
        )
        return _apply_local_matrix_to_state(
            state,
            _cast_local_matrix(backend, local, cache_key=cache_key),
            controls + [gate["target_qubit"]],
            n_qubits,
            backend,
        )

    if gate_type == "toffoli":
        controls, control_states = _normalized_control_data(gate)
        base = _single_qubit_base_for_gate(gate)
        local = _controlled_local_from_base(base, control_states)
        cache_key = ("controlled", gate_type, len(controls), tuple(control_states), None)
        return _apply_local_matrix_to_state(
            state,
            _cast_local_matrix(backend, local, cache_key=cache_key),
            controls + [gate["target_qubit"]],
            n_qubits,
            backend,
        )

    if gate_type == "swap":
        local = np.array(
            [
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            ],
            dtype=_CDTYPE,
        )
        cache_key = ("swap",)
        return _apply_local_matrix_to_state(
            state,
            _cast_local_matrix(backend, local, cache_key=cache_key),
            [gate["qubit_1"], gate["qubit_2"]],
            n_qubits,
            backend,
        )

    if gate_type in {"rzz", "rxx"}:
        parameter = gate.get("parameter")
        local = _rzz_backend(parameter, backend) if gate_type == "rzz" else _rxx_backend(parameter, backend)
        cache_key = None if _contains_torch_tensor(parameter) else (gate_type, _parameter_cache_key(parameter))
        return _apply_local_matrix_to_state(
            state,
            _cast_local_matrix(backend, local, cache_key=cache_key),
            [gate["qubit_1"], gate["qubit_2"]],
            n_qubits,
            backend,
        )

    return None


def gate_to_matrix(gate, cir_qubits=1, backend=None):
    gate = normalize_gate(gate)
    gate_type = canonical_gate_name(gate["type"])
    gate_parameter = gate.get("parameter", None)

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

    if backend is None:
        if gate_type == "pauli_x":
            gate_qubits = gate["target_qubit"] + 1
            gate_matrix = _pauli_x(gate["target_qubit"])
        elif gate_type == "pauli_y":
            gate_qubits = gate["target_qubit"] + 1
            gate_matrix = _pauli_y(gate["target_qubit"])
        elif gate_type == "pauli_z":
            gate_qubits = gate["target_qubit"] + 1
            gate_matrix = _pauli_z(gate["target_qubit"])
        elif gate_type == "hadamard":
            gate_qubits = gate["target_qubit"] + 1
            gate_matrix = _hadamard(gate["target_qubit"])
        elif gate_type == "s_gate":
            gate_qubits = gate["target_qubit"] + 1
            gate_matrix = _s_gate(gate["target_qubit"])
        elif gate_type == "t_gate":
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
        elif gate_type == "cx":
            controls, control_states = _normalized_control_data(gate)
            gate_qubits = max(gate["target_qubit"], max(controls)) + 1
            gate_matrix = _cx(gate["target_qubit"], controls, control_states)
        elif gate_type == "cy":
            controls, control_states = _normalized_control_data(gate)
            gate_qubits = max(gate["target_qubit"], max(controls)) + 1
            gate_matrix = _cy(gate["target_qubit"], controls, control_states)
        elif gate_type == "cz":
            controls, control_states = _normalized_control_data(gate)
            gate_qubits = max(gate["target_qubit"], max(controls)) + 1
            gate_matrix = _cz(gate["target_qubit"], controls, control_states)
        elif gate_type == "crx":
            controls, control_states = _normalized_control_data(gate)
            gate_qubits = max(gate["target_qubit"], max(controls)) + 1
            gate_matrix = _crx(gate_parameter, gate["target_qubit"], controls, control_states)
        elif gate_type == "cry":
            controls, control_states = _normalized_control_data(gate)
            gate_qubits = max(gate["target_qubit"], max(controls)) + 1
            gate_matrix = _cry(gate_parameter, gate["target_qubit"], controls, control_states)
        elif gate_type == "crz":
            controls, control_states = _normalized_control_data(gate)
            gate_qubits = max(gate["target_qubit"], max(controls)) + 1
            gate_matrix = _crz(gate_parameter, gate["target_qubit"], controls, control_states)
        elif gate_type == "swap":
            gate_qubits = max(gate["qubit_1"], gate["qubit_2"]) + 1
            gate_matrix = _swap(gate["qubit_1"], gate["qubit_2"])
        elif gate_type == "toffoli":
            controls, control_states = _normalized_control_data(gate)
            gate_qubits = max([gate["target_qubit"]] + controls) + 1
            gate_matrix = _toffoli(gate["target_qubit"], controls, control_states)
        elif gate_type == "identity":
            return identity(cir_qubits)
        elif gate_type in {"rzz", "rxx"}:
            local = _rzz(gate_parameter) if gate_type == "rzz" else _rxx(gate_parameter)
            return _expand_local_matrix_to_full(
                local,
                [gate["qubit_1"], gate["qubit_2"]],
                int(cir_qubits),
            )
        else:
            raise ValueError(f"不支持的门类型: {gate_type}")
    else:
        if gate_type == "pauli_x":
            gate_qubits = gate["target_qubit"] + 1
            base = np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=_CDTYPE)
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type == "pauli_y":
            gate_qubits = gate["target_qubit"] + 1
            base = np.array([[0.0 + 0.0j, -1j], [1j, 0.0 + 0.0j]], dtype=_CDTYPE)
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type == "pauli_z":
            gate_qubits = gate["target_qubit"] + 1
            base = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]], dtype=_CDTYPE)
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type == "hadamard":
            gate_qubits = gate["target_qubit"] + 1
            sqrt2_inv = 1.0 / math.sqrt(2.0)
            base = np.array(
                [[sqrt2_inv + 0.0j, sqrt2_inv + 0.0j], [sqrt2_inv + 0.0j, -sqrt2_inv + 0.0j]],
                dtype=_CDTYPE,
            )
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type == "s_gate":
            gate_qubits = gate["target_qubit"] + 1
            base = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1j]], dtype=_CDTYPE)
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type == "t_gate":
            gate_qubits = gate["target_qubit"] + 1
            t_val = np.exp(1j * math.pi / 4.0)
            base = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, t_val]], dtype=_CDTYPE)
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type == "rx":
            gate_qubits = gate["target_qubit"] + 1
            base = _single_qubit_base_for_gate_backend(gate, backend)
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type == "ry":
            gate_qubits = gate["target_qubit"] + 1
            base = _single_qubit_base_for_gate_backend(gate, backend)
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type == "rz":
            gate_qubits = gate["target_qubit"] + 1
            base = _single_qubit_base_for_gate_backend(gate, backend)
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type == "u3":
            gate_qubits = gate["target_qubit"] + 1
            base = _single_qubit_base_for_gate_backend(gate, backend)
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type == "u2":
            gate_qubits = gate["target_qubit"] + 1
            base = _single_qubit_base_for_gate_backend(gate, backend)
            gate_matrix = _single_qubit_from_base_backend(base, gate["target_qubit"], backend)
        elif gate_type == "cx":
            controls, control_states = _normalized_control_data(gate)
            gate_qubits = max(gate["target_qubit"], max(controls)) + 1
            base = np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=_CDTYPE)
            gate_matrix = _controlled_from_base_backend(base, gate["target_qubit"], controls, control_states, backend)
        elif gate_type == "cy":
            controls, control_states = _normalized_control_data(gate)
            gate_qubits = max(gate["target_qubit"], max(controls)) + 1
            base = np.array([[0.0 + 0.0j, -1j], [1j, 0.0 + 0.0j]], dtype=_CDTYPE)
            gate_matrix = _controlled_from_base_backend(base, gate["target_qubit"], controls, control_states, backend)
        elif gate_type == "cz":
            controls, control_states = _normalized_control_data(gate)
            gate_qubits = max(gate["target_qubit"], max(controls)) + 1
            base = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]], dtype=_CDTYPE)
            gate_matrix = _controlled_from_base_backend(base, gate["target_qubit"], controls, control_states, backend)
        elif gate_type == "crx":
            controls, control_states = _normalized_control_data(gate)
            gate_qubits = max(gate["target_qubit"], max(controls)) + 1
            base = _single_qubit_base_for_gate_backend(gate, backend)
            gate_matrix = _controlled_from_base_backend(base, gate["target_qubit"], controls, control_states, backend)
        elif gate_type == "cry":
            controls, control_states = _normalized_control_data(gate)
            gate_qubits = max(gate["target_qubit"], max(controls)) + 1
            base = _single_qubit_base_for_gate_backend(gate, backend)
            gate_matrix = _controlled_from_base_backend(base, gate["target_qubit"], controls, control_states, backend)
        elif gate_type == "crz":
            controls, control_states = _normalized_control_data(gate)
            gate_qubits = max(gate["target_qubit"], max(controls)) + 1
            base = _single_qubit_base_for_gate_backend(gate, backend)
            gate_matrix = _controlled_from_base_backend(base, gate["target_qubit"], controls, control_states, backend)
        elif gate_type == "swap":
            gate_qubits = max(gate["qubit_1"], gate["qubit_2"]) + 1
            gate_matrix = _swap_backend(gate["qubit_1"], gate["qubit_2"], backend)
        elif gate_type == "toffoli":
            controls, control_states = _normalized_control_data(gate)
            gate_qubits = max([gate["target_qubit"]] + controls) + 1
            gate_matrix = _toffoli_backend(gate["target_qubit"], controls, control_states, backend)
        elif gate_type == "identity":
            return backend.eye(1 << cir_qubits)
        elif gate_type in {"rzz", "rxx"}:
            local = _rzz_backend(gate_parameter, backend) if gate_type == "rzz" else _rxx_backend(gate_parameter, backend)
            return _expand_local_matrix_to_full(
                local,
                [gate["qubit_1"], gate["qubit_2"]],
                int(cir_qubits),
                backend=backend,
            )
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
