"""伴随微分（adjoint differentiation）：``<O>`` 对 ansatz 线路的反向传播梯度。"""

from __future__ import annotations

import numpy as np

from ...gates import gate_generator, parametric_pauli_gates
from ...ir import (
    circuit_instructions,
    instruction_controls,
    instruction_name,
    instruction_parameter,
    instruction_qubits,
)

_CDTYPE = np.complex64

# Single-parameter gates differentiated by the adjoint method, mapped to the
# (Hermitian) generator G in U = exp(-i θ G / 2). Sourced from the GateSpec
# registry (NEXT.md §7) so registering a new Pauli-rotation gate with a
# ``generator`` makes it adjoint-differentiable without editing this module.
# Single-Pauli generators (X/Y/Z, incl. controlled rotations) use the local
# Pauli-matrix path; two-qubit generators (ZZ/XX) have bespoke matrices below.
_AD_PAULI_GENERATOR = {
    name: gen
    for name in parametric_pauli_gates()
    if (gen := gate_generator(name)) in ("X", "Y", "Z")
}
_AD_DIFFERENTIABLE = set(parametric_pauli_gates())

_SWAP_LOCAL = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=_CDTYPE
)


def _pauli_local(label: str) -> np.ndarray:
    if label == "X":
        return np.array([[0, 1], [1, 0]], dtype=_CDTYPE)
    if label == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=_CDTYPE)
    if label == "Z":
        return np.array([[1, 0], [0, -1]], dtype=_CDTYPE)
    raise ValueError(f"unknown Pauli label {label!r}")


def _controlled_generator_local(pauli: np.ndarray, control_states) -> np.ndarray:
    """Generator |1…1><1…1|_controls ⊗ pauli as a (zeroed) local block matrix."""
    cs = [int(s) for s in control_states]
    dim = 1 << (len(cs) + 1)
    local = np.zeros((dim, dim), dtype=_CDTYPE)
    control_index = 0
    for state in cs:
        control_index = (control_index << 1) | state
    block = [(control_index << 1) | target for target in (0, 1)]
    local[np.ix_(block, block)] = pauli
    return local


def _ad_gate_local_matrix_and_axes(gate, backend):
    """Return ``(local_matrix, axes)`` for a gate's unitary (numpy local matrix).

    Mirrors the dispatch used by ``apply_gate_to_state`` so the forward and
    backward (dagger) passes are exact inverses by construction.
    """
    from aicir.core.gates import (
        _controlled_local_from_base,
        _normalized_control_data,
        _rxx,
        _rzz,
        _single_qubit_base_for_gate,
        _unitary_parameter_matrix,
    )

    gate_type = instruction_name(gate)
    qubits = [int(qubit) for qubit in instruction_qubits(gate)]
    if gate_type in ("identity", "I"):
        return None, None
    if gate_type == "swap":
        return _SWAP_LOCAL, qubits
    if gate_type == "rzz":
        return np.asarray(_rzz(instruction_parameter(gate)), dtype=_CDTYPE), qubits
    if gate_type == "rxx":
        return np.asarray(_rxx(instruction_parameter(gate)), dtype=_CDTYPE), qubits
    if gate_type == "unitary":
        matrix = np.asarray(_unitary_parameter_matrix(instruction_parameter(gate), backend=None), dtype=_CDTYPE)
        gate_qubits = int(round(np.log2(matrix.shape[0])))
        return matrix, list(range(gate_qubits))

    base = _single_qubit_base_for_gate(gate)
    if base is None:
        raise ValueError(f"adjoint differentiation does not support gate type {gate_type!r}")
    if instruction_controls(gate):
        controls, control_states = _normalized_control_data(gate)
        local = _controlled_local_from_base(base, control_states)
        return np.asarray(local, dtype=_CDTYPE), controls + qubits
    return np.asarray(base, dtype=_CDTYPE), qubits


def _ad_generator_local_and_axes(gate):
    """Return ``(generator_matrix, axes)`` for a differentiable gate."""
    from aicir.core.gates import _normalized_control_data

    gate_type = instruction_name(gate)
    qubits = [int(qubit) for qubit in instruction_qubits(gate)]
    if gate_type == "rzz":
        zz = np.diag([1.0, -1.0, -1.0, 1.0]).astype(_CDTYPE)
        return zz, qubits
    if gate_type == "rxx":
        xx = np.array(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=_CDTYPE,
        )
        return xx, qubits
    pauli = _pauli_local(_AD_PAULI_GENERATOR[gate_type])
    if gate_type in ("crx", "cry", "crz"):
        controls, control_states = _normalized_control_data(gate)
        return _controlled_generator_local(pauli, control_states), controls + qubits
    return pauli, qubits


def _ad_apply(backend, matrix, axes, n_qubits, state):
    from aicir.core.gates import _apply_local_matrix_to_state

    return _apply_local_matrix_to_state(state, backend.cast(matrix), axes, n_qubits, backend)


def ad(circuit, observable, *, backend=None, return_value: bool = False):
    """Adjoint-differentiation gradient of ``<O>`` over an ansatz circuit.

    Reverse-mode differentiation specialised for noiseless state-vector
    simulation. It propagates a "lambda state" backwards through the circuit
    and reads each gradient off

        ∂<O>/∂θ_k = 2 Re[<λ_k| ∂U_k/∂θ_k |ψ_{k-1}>]  =  Im[<λ_k| G_k |ψ_k>],

    where ``U_k = exp(-i θ_k G_k / 2)``. The whole gradient costs one forward
    pass plus one backward pass — ``O(P)`` gate applications for ``P``
    parameters with only ``O(1)`` extra state storage — versus ``O(P^2)`` for
    the parameter-shift rule. State-vector simulators only (no noise / mixed
    states).

    Unlike :func:`psr`/:func:`fd`, which differentiate a black-box scalar
    function, ``ad`` is structure-aware: it differentiates each single-angle
    Pauli-rotation gate (``rx``, ``ry``, ``rz``, ``crx``, ``cry``, ``crz``,
    ``rzz``, ``rxx``) in the circuit, returning one gradient per such gate in order of
    appearance. Other gates (``H``, ``cx``, ``u3``, ``unitary``, …) are applied
    but not differentiated (use :func:`psr` for those).

    Args:
        circuit: A fully-bound ``Circuit`` (no unbound ``Parameter`` symbols).
        observable: Hermitian operator ``O`` — a ``(2^n, 2^n)`` matrix
            (numpy/backend tensor) or any object exposing ``to_matrix(backend)``
            (e.g. ``Hamiltonian``).
        backend: Computation backend. Defaults to ``NumpyBackend``.
        return_value: If ``True``, also return the expectation value ``<O>``.

    Returns:
        ``np.ndarray`` of gradients (one per differentiable gate). If
        ``return_value`` is ``True``, returns ``(grad, expectation)``.
    """
    from aicir.backends.numpy_backend import NumpyBackend

    bk = backend if backend is not None else NumpyBackend()

    unbound = getattr(circuit, "parameters", ())
    if unbound:
        names = ", ".join(parameter.name for parameter in unbound)
        raise ValueError(f"circuit has unbound parameter(s): {names}; call bind_parameters(...) first")

    n_qubits = int(circuit.n_qubits)
    gates = list(circuit_instructions(circuit))

    if hasattr(observable, "to_matrix"):
        operator = observable.to_matrix(bk)
    else:
        operator = bk.cast(observable)

    # Forward pass: |ψ> = U_P ... U_1 |0>, caching each gate's local operator.
    psi = bk.zeros_state(n_qubits)
    cached = []
    for gate in gates:
        matrix, axes = _ad_gate_local_matrix_and_axes(gate, bk)
        cached.append((matrix, axes))
        if matrix is not None:
            psi = _ad_apply(bk, matrix, axes, n_qubits, psi)

    lam = bk.matmul(operator, psi)  # |λ_P> = O|ψ>
    expectation = float(np.real(np.asarray(bk.to_numpy(bk.inner_product(psi, lam)))))

    phi = psi  # walks backward as |ψ_k>
    grads_reversed: list[float] = []
    for gate, (matrix, axes) in zip(reversed(gates), reversed(cached)):
        if instruction_name(gate) in _AD_DIFFERENTIABLE:
            gen, gen_axes = _ad_generator_local_and_axes(gate)
            g_phi = _ad_apply(bk, gen, gen_axes, n_qubits, phi)  # G_k |ψ_k>
            overlap = bk.to_numpy(bk.inner_product(lam, g_phi))  # <λ_k| G_k |ψ_k>
            grads_reversed.append(float(np.imag(np.asarray(overlap))))
        if matrix is not None:
            dagger = bk.dagger(bk.cast(matrix))
            phi = _ad_apply(bk, dagger, axes, n_qubits, phi)  # |ψ_{k-1}>
            lam = _ad_apply(bk, dagger, axes, n_qubits, lam)  # |λ_{k-1}>

    grad = np.array(list(reversed(grads_reversed)), dtype=float)
    if return_value:
        return grad, expectation
    return grad
