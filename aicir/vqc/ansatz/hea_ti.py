"""Trapped-ion hardware-efficient ansatz (HEA-TI).

This module implements the ansatz described in
"Hardware-efficient variational quantum algorithm in trapped-ion quantum
computer" (arXiv:2407.03116):

* General HEA-TI: per layer, local ``Rx Ry Rx`` rotations on every ion followed
  by global evolution under a transverse-field Ising Hamiltonian (TFIM).
* Charge-conserving/symmetry HEA-TI: per layer, local ``Rz`` rotations followed
  by global evolution under the effective XY Hamiltonian.

The global evolution is represented as the repository's native ``unitary`` gate,
so the returned ansatz is a regular :class:`aicir.core.circuit.Circuit`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ...core.circuit import Circuit, Parameter, rx, ry, rz


_CDTYPE = np.complex64
_I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=_CDTYPE)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=_CDTYPE)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=_CDTYPE)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=_CDTYPE)

_GENERAL_VARIANTS = {"general", "tfim", "tfim_su2"}
_SYMMETRY_VARIANTS = {"symmetry", "symmetric", "charge_conserving", "chemistry", "xy"}


class _ParameterStream:
    def __init__(self, values: Sequence[Any] | None, prefix: str) -> None:
        self._values = values
        self._prefix = prefix
        self._index = 0

    def next(self) -> Any:
        index = self._index
        self._index += 1
        if self._values is None:
            return Parameter(f"{self._prefix}_{index}")
        if index >= len(self._values):
            raise ValueError(f"Expected at least {self._index} parameter value(s), got {len(self._values)}")
        return self._values[index]

    def finish(self) -> None:
        if self._values is not None and self._index != len(self._values):
            raise ValueError(f"Expected {self._index} parameter value(s), got {len(self._values)}")


def _flatten_parameters(parameters: Sequence[Any] | None) -> list[Any] | None:
    if parameters is None:
        return None
    if isinstance(parameters, (str, bytes)):
        raise TypeError("parameters must be a non-string sequence")
    if hasattr(parameters, "reshape"):
        flat = parameters.reshape(-1)
        return [flat[index] for index in range(len(flat))]
    return list(parameters)


def _validate_n_qubits(n_qubits: int) -> int:
    value = int(n_qubits)
    if value <= 0:
        raise ValueError(f"n_qubits must be positive, got {n_qubits}")
    return value


def _validate_layers(layers: int) -> int:
    value = int(layers)
    if value < 0:
        raise ValueError(f"layers must be non-negative, got {layers}")
    return value


def _normalize_variant(variant: str) -> str:
    key = str(variant).lower()
    if key in _GENERAL_VARIANTS:
        return "general"
    if key in _SYMMETRY_VARIANTS:
        return "symmetry"
    allowed = ", ".join(sorted(_GENERAL_VARIANTS | _SYMMETRY_VARIANTS))
    raise ValueError(f"Unsupported HEA-TI variant: {variant}. Supported variants: {allowed}")


def _normalize_evolution_times(
    layers: int,
    *,
    evolution_time: float,
    evolution_times: Sequence[float] | None,
) -> list[float]:
    if evolution_times is None:
        return [float(evolution_time)] * layers
    if isinstance(evolution_times, (str, bytes)):
        raise TypeError("evolution_times must be a non-string sequence")
    values = [float(value) for value in evolution_times]
    if len(values) != layers:
        raise ValueError(f"Expected {layers} evolution time(s), got {len(values)}")
    return values


def power_law_couplings(
    n_qubits: int,
    *,
    j0: float = 1.0,
    alpha: float = 1.5,
    dtype: Any = float,
) -> np.ndarray:
    """Return trapped-ion power-law couplings ``J_ij = j0 / |i-j|^alpha``."""

    n_qubits = _validate_n_qubits(n_qubits)
    alpha_value = float(alpha)
    if alpha_value < 0.0:
        raise ValueError("alpha must be non-negative")

    matrix = np.zeros((n_qubits, n_qubits), dtype=dtype)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            coupling = float(j0) / (abs(i - j) ** alpha_value)
            matrix[i, j] = coupling
            matrix[j, i] = coupling
    return matrix


def _normalize_couplings(
    n_qubits: int,
    couplings: np.ndarray | Sequence[Sequence[float]] | None,
    *,
    j0: float,
    alpha: float,
) -> np.ndarray:
    if couplings is None:
        return power_law_couplings(n_qubits, j0=j0, alpha=alpha, dtype=float)

    array = np.asarray(couplings, dtype=float)
    if array.shape != (n_qubits, n_qubits):
        raise ValueError(f"couplings must have shape {(n_qubits, n_qubits)}, got {array.shape}")
    if not np.allclose(array, array.T, atol=1e-12):
        raise ValueError("couplings must be symmetric")
    return array


def _kron_all(matrices: Sequence[np.ndarray]) -> np.ndarray:
    result = np.asarray(matrices[0], dtype=np.complex128)
    for matrix in matrices[1:]:
        result = np.kron(result, np.asarray(matrix, dtype=np.complex128))
    return result


def _one_body(pauli: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    return _kron_all([pauli if index == qubit else _I for index in range(n_qubits)])


def _two_body(pauli_a: np.ndarray, qubit_a: int, pauli_b: np.ndarray, qubit_b: int, n_qubits: int) -> np.ndarray:
    matrices = []
    for index in range(n_qubits):
        if index == qubit_a:
            matrices.append(pauli_a)
        elif index == qubit_b:
            matrices.append(pauli_b)
        else:
            matrices.append(_I)
    return _kron_all(matrices)


def trapped_ion_hamiltonian(
    n_qubits: int,
    *,
    kind: str = "tfim",
    couplings: np.ndarray | Sequence[Sequence[float]] | None = None,
    j0: float = 1.0,
    alpha: float = 1.5,
    transverse_field: float = 1.0,
    dtype: Any = _CDTYPE,
) -> np.ndarray:
    """Build the dense trapped-ion global Hamiltonian used by HEA-TI.

    ``kind="tfim"`` builds ``sum_ij J_ij X_i X_j + B sum_i Z_i``.
    ``kind="xy"`` builds ``1/2 sum_ij J_ij (X_i X_j + Y_i Y_j)``, equivalent
    to ``sum_ij J_ij (sigma+_i sigma-_j + sigma-_i sigma+_j)``.
    """

    n_qubits = _validate_n_qubits(n_qubits)
    key = str(kind).lower()
    if key not in {"tfim", "xy"}:
        raise ValueError("kind must be 'tfim' or 'xy'")

    dim = 1 << n_qubits
    hamiltonian = np.zeros((dim, dim), dtype=np.complex128)
    coupling_matrix = _normalize_couplings(n_qubits, couplings, j0=j0, alpha=alpha)

    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            jij = coupling_matrix[i, j]
            if jij == 0.0:
                continue
            if key == "tfim":
                hamiltonian += jij * _two_body(_X, i, _X, j, n_qubits)
            else:
                hamiltonian += 0.5 * jij * (
                    _two_body(_X, i, _X, j, n_qubits)
                    + _two_body(_Y, i, _Y, j, n_qubits)
                )

    if key == "tfim" and transverse_field != 0.0:
        for qubit in range(n_qubits):
            hamiltonian += float(transverse_field) * _one_body(_Z, qubit, n_qubits)

    return hamiltonian.astype(dtype, copy=False)


def global_evolution_unitary(
    hamiltonian: np.ndarray,
    evolution_time: float,
    *,
    dtype: Any = _CDTYPE,
) -> np.ndarray:
    """Return ``exp(-i H t)`` for a Hermitian dense Hamiltonian."""

    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hamiltonian must be a square matrix")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-8):
        raise ValueError("hamiltonian must be Hermitian")

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    phases = np.exp(-1.0j * float(evolution_time) * eigenvalues)
    unitary = (eigenvectors * phases) @ eigenvectors.conj().T
    return unitary.astype(dtype, copy=False)


def hea_ti_parameter_count(
    n_qubits: int,
    layers: int = 1,
    *,
    variant: str = "general",
    include_evolution_times: bool = False,
) -> int:
    """Return the number of HEA-TI variational parameters.

    By default this counts single-qubit rotation parameters only, matching the
    parameters generated by :func:`hea_ti_ansatz`. Set
    ``include_evolution_times=True`` to count the layer evolution times from the
    paper's resource formula for the general TFIM variant.
    """

    n_qubits = _validate_n_qubits(n_qubits)
    layers = _validate_layers(layers)
    normalized = _normalize_variant(variant)
    per_layer = 3 * n_qubits if normalized == "general" else n_qubits
    total = layers * per_layer
    if include_evolution_times:
        total += layers
    return total


def _append_general_rotations(gates: list[dict[str, Any]], n_qubits: int, params: _ParameterStream) -> None:
    for qubit in range(n_qubits):
        gates.append(rx(params.next(), qubit))
        gates.append(ry(params.next(), qubit))
        gates.append(rx(params.next(), qubit))


def _append_symmetry_rotations(gates: list[dict[str, Any]], n_qubits: int, params: _ParameterStream) -> None:
    for qubit in range(n_qubits):
        gates.append(rz(params.next(), qubit))


def _global_unitary_gate(unitary: np.ndarray, n_qubits: int, *, variant: str, evolution_time: float) -> dict[str, Any]:
    return {
        "type": "unitary",
        "parameter": unitary,
        "n_qubits": n_qubits,
        "label": f"HEA-TI-{variant}",
        "evolution_time": float(evolution_time),
    }


def hea_ti_ansatz(
    n_qubits: int,
    layers: int = 1,
    *,
    variant: str = "general",
    evolution_time: float = 0.4,
    evolution_times: Sequence[float] | None = None,
    couplings: np.ndarray | Sequence[Sequence[float]] | None = None,
    j0: float = 1.0,
    alpha: float = 1.5,
    transverse_field: float | None = None,
    rotation_first: bool = True,
    parameter_prefix: str = "theta",
    parameters: Sequence[Any] | None = None,
    backend: Any = None,
    dtype: Any = _CDTYPE,
) -> Circuit:
    """Build an HEA-TI ansatz circuit.

    Args:
        n_qubits: Number of trapped-ion qubits.
        layers: Number of HEA-TI layers ``D``.
        variant: ``"general"``/``"tfim"`` for TFIM + ``Rx Ry Rx`` layers, or
            ``"symmetry"``/``"chemistry"``/``"xy"`` for charge-conserving XY +
            ``Rz`` layers.
        evolution_time: Scalar global evolution time used for every layer when
            ``evolution_times`` is not provided. The paper uses ``0.4`` for the
            chemistry simulations.
        evolution_times: Optional per-layer global evolution times.
        couplings: Optional symmetric ``J_ij`` matrix. Defaults to the trapped
            ion power law ``J_ij = j0 / |i-j|^alpha``.
        j0: Power-law coupling scale.
        alpha: Power-law exponent. The paper uses ``alpha=1.5``.
        transverse_field: TFIM magnetic field ``B``. Defaults to ``1`` for the
            general variant and ``0`` for the symmetry variant.
        rotation_first: If true, each layer is rotations then global evolution;
            otherwise global evolution then rotations.
        parameter_prefix: Prefix for generated symbolic rotation parameters.
        parameters: Optional flat sequence of rotation parameter values.
        backend: Optional backend bound to the returned circuit.
        dtype: Complex dtype used for dense Hamiltonian/unitary matrices.

    Returns:
        A native ``Circuit`` containing local rotation gates and global unitary
        gates.
    """

    n_qubits = _validate_n_qubits(n_qubits)
    layers = _validate_layers(layers)
    normalized = _normalize_variant(variant)
    times = _normalize_evolution_times(layers, evolution_time=evolution_time, evolution_times=evolution_times)
    params = _ParameterStream(_flatten_parameters(parameters), parameter_prefix)

    if normalized == "general":
        hamiltonian_kind = "tfim"
        field = 1.0 if transverse_field is None else float(transverse_field)
        append_rotations = _append_general_rotations
    else:
        hamiltonian_kind = "xy"
        field = 0.0 if transverse_field is None else float(transverse_field)
        append_rotations = _append_symmetry_rotations

    hamiltonian = trapped_ion_hamiltonian(
        n_qubits,
        kind=hamiltonian_kind,
        couplings=couplings,
        j0=j0,
        alpha=alpha,
        transverse_field=field,
        dtype=dtype,
    )
    unitary_cache: dict[float, np.ndarray] = {}
    gates: list[dict[str, Any]] = []

    for time in times:
        if time not in unitary_cache:
            unitary_cache[time] = global_evolution_unitary(hamiltonian, time, dtype=dtype)
        global_gate = _global_unitary_gate(unitary_cache[time], n_qubits, variant=normalized, evolution_time=time)

        if rotation_first:
            append_rotations(gates, n_qubits, params)
            gates.append(global_gate)
        else:
            gates.append(global_gate)
            append_rotations(gates, n_qubits, params)

    params.finish()
    return Circuit(*gates, n_qubits=n_qubits, backend=backend)


hea_ti = hea_ti_ansatz


__all__ = [
    "global_evolution_unitary",
    "hea_ti",
    "hea_ti_ansatz",
    "hea_ti_parameter_count",
    "power_law_couplings",
    "trapped_ion_hamiltonian",
]
