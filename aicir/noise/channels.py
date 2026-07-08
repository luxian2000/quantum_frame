"""Common noise channels embedded through Kraus operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

import numpy as np

from .base import NoiseChannel

_I2 = np.array([[1, 0], [0, 1]], dtype=np.complex64)
_X2 = np.array([[0, 1], [1, 0]], dtype=np.complex64)
_Y2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
_Z2 = np.array([[1, 0], [0, -1]], dtype=np.complex64)


def _embed_single_qubit(op_2x2: np.ndarray, target_qubit: int, n_qubits: int, backend):
    mats = []
    for q in range(n_qubits):
        mats.append(backend.cast(op_2x2 if q == target_qubit else _I2))
    return backend.tensor_product(*mats)


def _validate_probability(value: float, name: str) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1]")


def _validate_target_qubits(target_qubits: Sequence[int], n_qubits: int) -> tuple[int, ...]:
    targets = tuple(int(q) for q in target_qubits)
    if not targets:
        raise ValueError("target_qubits must not be empty")
    if len(set(targets)) != len(targets):
        raise ValueError("target_qubits must be unique")
    for qubit in targets:
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError(f"target_qubit={qubit} out of range [0, {n_qubits})")
    return targets


def _index_to_bits(index: int, n_qubits: int) -> list[int]:
    return [(index >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]


def _bits_to_index(bits: Sequence[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _embed_operator(op: np.ndarray, target_qubits: Sequence[int], n_qubits: int, backend):
    targets = _validate_target_qubits(target_qubits, n_qubits)
    op = np.asarray(op, dtype=np.complex64)
    target_dim = 1 << len(targets)
    if op.shape != (target_dim, target_dim):
        raise ValueError(
            f"operator shape {op.shape} does not match {len(targets)} target qubits "
            f"({target_dim}, {target_dim})"
        )

    dim = 1 << n_qubits
    embedded = np.zeros((dim, dim), dtype=np.complex64)
    target_positions = {qubit: i for i, qubit in enumerate(targets)}
    for in_index in range(dim):
        in_bits = _index_to_bits(in_index, n_qubits)
        local_in_bits = [in_bits[q] for q in targets]
        local_in = _bits_to_index(local_in_bits)
        for local_out in range(target_dim):
            amp = op[local_out, local_in]
            if amp == 0:
                continue
            out_bits = list(in_bits)
            local_out_bits = _index_to_bits(local_out, len(targets))
            for qubit, target_pos in target_positions.items():
                out_bits[qubit] = local_out_bits[target_pos]
            embedded[_bits_to_index(out_bits), in_index] += amp
    return backend.cast(embedded)


def _pauli_products(n_targets: int) -> Iterable[np.ndarray]:
    if n_targets <= 0:
        return
    single = (_I2, _X2, _Y2, _Z2)
    products = [np.array([[1]], dtype=np.complex64)]
    for _ in range(n_targets):
        products = [np.kron(prefix, op).astype(np.complex64) for prefix in products for op in single]
    yield from products


_PAULI_BY_NAME = {
    "i": _I2,
    "id": _I2,
    "identity": _I2,
    "x": _X2,
    "y": _Y2,
    "z": _Z2,
}


def _pauli_operator(label: str) -> np.ndarray:
    key = str(label).strip().lower()
    if key not in _PAULI_BY_NAME:
        raise ValueError(f"unsupported Pauli label: {label!r}")
    return _PAULI_BY_NAME[key]


@dataclass
class _SingleQubitChannel(NoiseChannel):
    target_qubit: int

    def _validate_target(self, n_qubits: int) -> None:
        if self.target_qubit < 0 or self.target_qubit >= n_qubits:
            raise ValueError(f"target_qubit={self.target_qubit} out of range [0, {n_qubits})")


@dataclass
class DepolarizingChannel(_SingleQubitChannel):
    """Single-qubit depolarizing channel with parameter p in [0, 1]."""

    p: float

    @property
    def name(self) -> str:
        return "depolarizing"

    def kraus_operators(self, n_qubits: int, backend) -> List[object]:
        self._validate_target(n_qubits)
        _validate_probability(float(self.p), "depolarizing p")

        p = float(self.p)
        k0 = np.sqrt(1.0 - p) * _I2
        k1 = np.sqrt(p / 3.0) * _X2
        k2 = np.sqrt(p / 3.0) * _Y2
        k3 = np.sqrt(p / 3.0) * _Z2
        return [
            _embed_single_qubit(k0, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(k1, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(k2, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(k3, self.target_qubit, n_qubits, backend),
        ]


@dataclass
class BitFlipChannel(_SingleQubitChannel):
    """Single-qubit bit-flip channel with parameter p in [0, 1]."""

    p: float

    @property
    def name(self) -> str:
        return "bit_flip"

    def kraus_operators(self, n_qubits: int, backend) -> List[object]:
        self._validate_target(n_qubits)
        _validate_probability(float(self.p), "bit flip p")

        p = float(self.p)
        k0 = np.sqrt(1.0 - p) * _I2
        k1 = np.sqrt(p) * _X2
        return [
            _embed_single_qubit(k0, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(k1, self.target_qubit, n_qubits, backend),
        ]


@dataclass
class PhaseFlipChannel(_SingleQubitChannel):
    """Single-qubit phase-flip channel with parameter p in [0, 1]."""

    p: float

    @property
    def name(self) -> str:
        return "phase_flip"

    def kraus_operators(self, n_qubits: int, backend) -> List[object]:
        self._validate_target(n_qubits)
        _validate_probability(float(self.p), "phase flip p")

        p = float(self.p)
        k0 = np.sqrt(1.0 - p) * _I2
        k1 = np.sqrt(p) * _Z2
        return [
            _embed_single_qubit(k0, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(k1, self.target_qubit, n_qubits, backend),
        ]


@dataclass
class AmplitudeDampingChannel(_SingleQubitChannel):
    """Single-qubit amplitude-damping channel with gamma in [0, 1]."""

    gamma: float

    @property
    def name(self) -> str:
        return "amplitude_damping"

    def kraus_operators(self, n_qubits: int, backend) -> List[object]:
        self._validate_target(n_qubits)
        _validate_probability(float(self.gamma), "amplitude damping gamma")

        g = float(self.gamma)
        k0 = np.array([[1, 0], [0, np.sqrt(1.0 - g)]], dtype=np.complex64)
        k1 = np.array([[0, np.sqrt(g)], [0, 0]], dtype=np.complex64)
        return [
            _embed_single_qubit(k0, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(k1, self.target_qubit, n_qubits, backend),
        ]


@dataclass
class ResetChannel(_SingleQubitChannel):
    """Probabilistic reset of a target qubit to ``|0>``."""

    p: float = 1.0

    @property
    def name(self) -> str:
        return "reset"

    def kraus_operators(self, n_qubits: int, backend) -> List[object]:
        self._validate_target(n_qubits)
        p = float(self.p)
        _validate_probability(p, "reset p")
        k_identity = np.sqrt(1.0 - p) * _I2
        k_reset_0 = np.sqrt(p) * np.array([[1, 0], [0, 0]], dtype=np.complex64)
        k_reset_1 = np.sqrt(p) * np.array([[0, 1], [0, 0]], dtype=np.complex64)
        return [
            _embed_single_qubit(k_identity, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(k_reset_0, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(k_reset_1, self.target_qubit, n_qubits, backend),
        ]


@dataclass
class ErasureChannel(_SingleQubitChannel):
    """Fixed-Hilbert-space erasure proxy.

    With probability ``p``, replaces the target qubit by the diagonal state
    ``diag(1 - erase_to, erase_to)``. This is not a flagged erasure channel;
    no extra erasure state is added to the Hilbert space.
    """

    p: float
    erase_to: float = 0.5

    @property
    def name(self) -> str:
        return "erasure"

    def kraus_operators(self, n_qubits: int, backend) -> List[object]:
        self._validate_target(n_qubits)
        p = float(self.p)
        erase_to = float(self.erase_to)
        _validate_probability(p, "erasure p")
        _validate_probability(erase_to, "erase_to")
        ground = 1.0 - erase_to
        k_identity = np.sqrt(1.0 - p) * _I2
        replacement_ops = [
            np.sqrt(p * ground) * np.array([[1, 0], [0, 0]], dtype=np.complex64),
            np.sqrt(p * ground) * np.array([[0, 1], [0, 0]], dtype=np.complex64),
            np.sqrt(p * erase_to) * np.array([[0, 0], [1, 0]], dtype=np.complex64),
            np.sqrt(p * erase_to) * np.array([[0, 0], [0, 1]], dtype=np.complex64),
        ]
        return [
            _embed_single_qubit(k_identity, self.target_qubit, n_qubits, backend),
            *[_embed_single_qubit(op, self.target_qubit, n_qubits, backend) for op in replacement_ops],
        ]


@dataclass
class ReadoutErrorChannel(_SingleQubitChannel):
    """Asymmetric readout-confusion proxy represented as a quantum channel.

    ``p01`` is the probability that a true 0 is reported as 1; ``p10`` is the
    probability that a true 1 is reported as 0. In the current density-matrix
    pipeline this is a pre-measurement proxy, not a separate classical result
    postprocessor.
    """

    p01: float
    p10: float

    @property
    def name(self) -> str:
        return "readout_error"

    def kraus_operators(self, n_qubits: int, backend) -> List[object]:
        self._validate_target(n_qubits)
        p01, p10 = float(self.p01), float(self.p10)
        _validate_probability(p01, "p01")
        _validate_probability(p10, "p10")
        k_keep = np.array([[np.sqrt(1.0 - p01), 0], [0, np.sqrt(1.0 - p10)]], dtype=np.complex64)
        k_01 = np.array([[0, 0], [np.sqrt(p01), 0]], dtype=np.complex64)
        k_10 = np.array([[0, np.sqrt(p10)], [0, 0]], dtype=np.complex64)
        return [
            _embed_single_qubit(k_keep, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(k_01, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(k_10, self.target_qubit, n_qubits, backend),
        ]


@dataclass
class PauliChannel(_SingleQubitChannel):
    """Single-qubit Pauli channel with independent X/Y/Z probabilities."""

    px: float
    py: float
    pz: float

    @property
    def name(self) -> str:
        return "pauli"

    def kraus_operators(self, n_qubits: int, backend) -> List[object]:
        self._validate_target(n_qubits)
        px, py, pz = float(self.px), float(self.py), float(self.pz)
        for value, name in ((px, "px"), (py, "py"), (pz, "pz")):
            _validate_probability(value, name)
        p_identity = 1.0 - px - py - pz
        if p_identity < -1e-12:
            raise ValueError("px + py + pz must be <= 1")
        p_identity = max(0.0, p_identity)
        return [
            _embed_single_qubit(np.sqrt(p_identity) * _I2, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(np.sqrt(px) * _X2, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(np.sqrt(py) * _Y2, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(np.sqrt(pz) * _Z2, self.target_qubit, n_qubits, backend),
        ]


@dataclass
class PhaseDampingChannel(_SingleQubitChannel):
    """Single-qubit pure dephasing channel with gamma in [0, 1]."""

    gamma: float

    @property
    def name(self) -> str:
        return "phase_damping"

    def kraus_operators(self, n_qubits: int, backend) -> List[object]:
        self._validate_target(n_qubits)
        gamma = float(self.gamma)
        _validate_probability(gamma, "phase damping gamma")
        k0 = np.array([[1, 0], [0, np.sqrt(1.0 - gamma)]], dtype=np.complex64)
        k1 = np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=np.complex64)
        return [
            _embed_single_qubit(k0, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(k1, self.target_qubit, n_qubits, backend),
        ]


@dataclass
class GeneralizedAmplitudeDampingChannel(_SingleQubitChannel):
    """Finite-temperature amplitude damping channel.

    ``p_excited`` is the environment excited-state population. ``p_excited=0``
    reduces to ordinary zero-temperature amplitude damping.
    """

    gamma: float
    p_excited: float

    @property
    def name(self) -> str:
        return "generalized_amplitude_damping"

    def kraus_operators(self, n_qubits: int, backend) -> List[object]:
        self._validate_target(n_qubits)
        gamma = float(self.gamma)
        p_excited = float(self.p_excited)
        _validate_probability(gamma, "generalized amplitude damping gamma")
        _validate_probability(p_excited, "p_excited")
        p_ground = 1.0 - p_excited
        k0 = np.sqrt(p_ground) * np.array([[1, 0], [0, np.sqrt(1.0 - gamma)]], dtype=np.complex64)
        k1 = np.sqrt(p_ground) * np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex64)
        k2 = np.sqrt(p_excited) * np.array([[np.sqrt(1.0 - gamma), 0], [0, 1]], dtype=np.complex64)
        k3 = np.sqrt(p_excited) * np.array([[0, 0], [np.sqrt(gamma), 0]], dtype=np.complex64)
        return [
            _embed_single_qubit(k0, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(k1, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(k2, self.target_qubit, n_qubits, backend),
            _embed_single_qubit(k3, self.target_qubit, n_qubits, backend),
        ]


@dataclass
class TwoQubitDepolarizingChannel(NoiseChannel):
    """Two-qubit depolarizing channel ``E(rho) = (1-p)rho + p I/4``."""

    qubit_1: int
    qubit_2: int
    p: float

    @property
    def name(self) -> str:
        return "two_qubit_depolarizing"

    def kraus_operators(self, n_qubits: int, backend) -> List[object]:
        targets = _validate_target_qubits((self.qubit_1, self.qubit_2), n_qubits)
        p = float(self.p)
        _validate_probability(p, "two-qubit depolarizing p")
        identity = np.eye(4, dtype=np.complex64)
        kraus = [_embed_operator(np.sqrt(1.0 - 15.0 * p / 16.0) * identity, targets, n_qubits, backend)]
        scale = np.sqrt(p / 16.0)
        for product in list(_pauli_products(2))[1:]:
            kraus.append(_embed_operator(scale * product, targets, n_qubits, backend))
        return kraus


@dataclass
class CorrelatedTwoQubitPauliChannel(NoiseChannel):
    """Two-qubit Pauli channel with explicitly correlated error probabilities."""

    qubit_1: int
    qubit_2: int
    probabilities: Mapping[tuple[str, str] | str, float]

    @property
    def name(self) -> str:
        return "correlated_two_qubit_pauli"

    def kraus_operators(self, n_qubits: int, backend) -> List[object]:
        targets = _validate_target_qubits((self.qubit_1, self.qubit_2), n_qubits)
        total = 0.0
        terms: list[tuple[np.ndarray, float]] = []
        for labels, probability in self.probabilities.items():
            if isinstance(labels, str):
                compact = labels.strip().lower()
                if len(compact) != 2:
                    raise ValueError("string Pauli labels must have length 2, e.g. 'xz'")
                pair = (compact[0], compact[1])
            else:
                pair = tuple(labels)
                if len(pair) != 2:
                    raise ValueError("Pauli label tuples must have length 2")
            p = float(probability)
            _validate_probability(p, f"probability for {pair}")
            total += p
            terms.append((np.kron(_pauli_operator(pair[0]), _pauli_operator(pair[1])).astype(np.complex64), p))

        if total > 1.0 + 1e-12:
            raise ValueError("correlated Pauli probabilities must sum to <= 1")
        identity_probability = max(0.0, 1.0 - total)
        kraus = [_embed_operator(np.sqrt(identity_probability) * np.eye(4, dtype=np.complex64), targets, n_qubits, backend)]
        for op, p in terms:
            if p > 0.0:
                kraus.append(_embed_operator(np.sqrt(p) * op, targets, n_qubits, backend))
        return kraus


@dataclass
class ThermalRelaxationChannel(_SingleQubitChannel):
    """Single-qubit thermal relaxation derived from T1/T2 and gate duration."""

    t1: float
    t2: float
    gate_time: float
    excited_population: float = 0.0

    @property
    def name(self) -> str:
        return "thermal_relaxation"

    def kraus_operators(self, n_qubits: int, backend) -> List[object]:
        self._validate_target(n_qubits)
        t1 = float(self.t1)
        t2 = float(self.t2)
        gate_time = float(self.gate_time)
        excited_population = float(self.excited_population)
        if t1 <= 0.0:
            raise ValueError("t1 must be positive")
        if t2 <= 0.0:
            raise ValueError("t2 must be positive")
        if gate_time < 0.0:
            raise ValueError("gate_time must be non-negative")
        _validate_probability(excited_population, "excited_population")

        gamma = 1.0 - np.exp(-gate_time / t1)
        t_phi_inv = max(0.0, (1.0 / t2) - (1.0 / (2.0 * t1)))
        phase_gamma = 1.0 - np.exp(-gate_time * t_phi_inv)
        gad = GeneralizedAmplitudeDampingChannel(
            self.target_qubit,
            gamma=float(gamma),
            p_excited=excited_population,
        ).kraus_operators(n_qubits, backend)
        phase = PhaseDampingChannel(
            self.target_qubit,
            gamma=float(phase_gamma),
        ).kraus_operators(n_qubits, backend)
        return [backend.matmul(k_phase, k_gad) for k_phase in phase for k_gad in gad]


@dataclass
class KrausChannel(NoiseChannel):
    """Custom Kraus channel.

    If ``target_qubits`` is ``None``, each operator must already be embedded in
    the full Hilbert space. Otherwise, operators are embedded on the requested
    target qubits using the repository's qubit ordering convention.
    """

    kraus_ops: Sequence[object]
    target_qubits: Sequence[int] | None = None
    channel_name: str = "kraus"

    @property
    def name(self) -> str:
        return self.channel_name

    def kraus_operators(self, n_qubits: int, backend) -> List[object]:
        if not self.kraus_ops:
            raise ValueError("kraus_ops must not be empty")
        dim = 1 << n_qubits
        operators = [np.asarray(op, dtype=np.complex64) for op in self.kraus_ops]
        if self.target_qubits is None:
            for op in operators:
                if op.shape != (dim, dim):
                    raise ValueError(f"full-system Kraus operator shape must be ({dim}, {dim})")
            return [backend.cast(op) for op in operators]
        targets = _validate_target_qubits(self.target_qubits, n_qubits)
        return [_embed_operator(op, targets, n_qubits, backend) for op in operators]


__all__ = [
    "AmplitudeDampingChannel",
    "BitFlipChannel",
    "CorrelatedTwoQubitPauliChannel",
    "DepolarizingChannel",
    "ErasureChannel",
    "GeneralizedAmplitudeDampingChannel",
    "KrausChannel",
    "PauliChannel",
    "PhaseDampingChannel",
    "PhaseFlipChannel",
    "ReadoutErrorChannel",
    "ResetChannel",
    "ThermalRelaxationChannel",
    "TwoQubitDepolarizingChannel",
]
