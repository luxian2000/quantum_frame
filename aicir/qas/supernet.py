"""Supernet-based QAS for variational quantum algorithms.

This module implements the QAS workflow from Du et al., npj Quantum
Information 2022, in an aicir-native form:

1. set up a supernet-indexed ansatz pool,
2. share weights by layer-level single-qubit gate layout,
3. optimize sampled architectures in one stage,
4. rank sampled architectures with trained shared weights,
5. fine-tune the selected ansatz with independent parameters.

The differentiable path intentionally uses aicir ``Circuit`` objects,
``GPUBackend`` (or ``NPUBackend`` on Ascend), and aicir gate matrix/state
evolution helpers. No external quantum SDK is used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import inspect
import io
import math
import random
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from ..backends.gpu_backend import GPUBackend
from ..backends.npu_backend import NPUBackend
from ..operators import Hamiltonian
from ..core.circuit import Circuit, cx, hadamard, rx, ry, rz, rzz
from ..core.gates import apply_gate_to_state, gate_to_matrix
from ..ir import circuit_instructions
from ..qml.deriv import psr


ObjectiveFn = Callable[..., torch.Tensor | float]
# Trainable-parameter address inside the supernet. The leading tag ("sq" / "tq")
# separates single-qubit from two-qubit parameters so the two weight-sharing
# signatures never collide; the layout tuple is the per-layer gate layout that
# the weight-sharing rule keys on; param_index keeps multi-angle gates
# addressable (the scoped gate set only uses single-angle rotations, so it is
# always 0 here, but the structure generalizes to u2/u3).
ParameterKey = tuple[str, int, int, tuple[str, ...], int, str, int]

# Sentinel for "no two-qubit gate on this pair" (replaces the old boolean False).
_NO_TWO_QUBIT = "none"


@dataclass(frozen=True)
class GateSpec:
    """Builder + arity metadata for one searchable gate.

    ``n_params`` is the number of trainable angles the gate owns: 0 for fixed
    gates (the identity placeholder ``i`` and ``cx``) and 1 for the rotations
    ``rx/ry/rz`` and the parameterized two-qubit ``rzz``. ``builder`` turns a
    parameter list plus a qubit tuple into an aicir gate dict, or returns
    ``None`` for a no-op (used by ``i`` so that no gate is emitted at all).
    """

    name: str
    n_params: int
    arity: int
    builder: Callable[[Sequence[Any], Sequence[int]], dict[str, Any] | None]


# Scoped single-qubit gate set: identity placeholder + the three Pauli rotations.
_SINGLE_QUBIT_GATES: dict[str, GateSpec] = {
    "i": GateSpec("i", 0, 1, lambda params, qubits: None),
    "h": GateSpec("h", 0, 1, lambda params, qubits: hadamard(int(qubits[0]))),
    "rx": GateSpec("rx", 1, 1, lambda params, qubits: rx(params[0], target_qubit=int(qubits[0]))),
    "ry": GateSpec("ry", 1, 1, lambda params, qubits: ry(params[0], target_qubit=int(qubits[0]))),
    "rz": GateSpec("rz", 1, 1, lambda params, qubits: rz(params[0], target_qubit=int(qubits[0]))),
}

# Scoped two-qubit gate set: fixed CNOT entangler + trainable ZZ rotation.
_TWO_QUBIT_GATES: dict[str, GateSpec] = {
    "cx": GateSpec(
        "cx",
        0,
        2,
        lambda params, qubits: cx(target_qubit=int(qubits[1]), control_qubits=[int(qubits[0])]),
    ),
    "rzz": GateSpec(
        "rzz",
        1,
        2,
        lambda params, qubits: rzz(params[0], qubit_1=int(qubits[0]), qubit_2=int(qubits[1])),
    ),
}


def _normalize_single_gate(name: str) -> str:
    gate = str(name).strip().lower()
    if gate not in _SINGLE_QUBIT_GATES:
        raise ValueError(
            f"supernet single-qubit gates are {tuple(_SINGLE_QUBIT_GATES)}; got {name!r}"
        )
    return gate


def _normalize_two_qubit_gate(name: str) -> str:
    gate = str(name).strip().lower()
    if gate not in _TWO_QUBIT_GATES:
        raise ValueError(
            f"supernet two-qubit gates are {tuple(_TWO_QUBIT_GATES)}; got {name!r}"
        )
    return gate


def _normalize_two_qubit_choice(value: Any) -> str:
    """Normalize a per-pair choice; accepts ``bool`` for backward compatibility.

    Historically a layer's two-qubit structure was a ``tuple[bool, ...]`` mask
    over CNOT pairs. ``True`` now maps to ``"cx"`` and ``False`` to ``"none"`` so
    existing call sites and tests keep working unchanged.
    """
    if isinstance(value, bool):
        return "cx" if value else _NO_TWO_QUBIT
    name = str(value).strip().lower()
    if name in ("", _NO_TWO_QUBIT, "i", "identity"):
        return _NO_TWO_QUBIT
    return _normalize_two_qubit_gate(name)


@dataclass(frozen=True)
class NoiseConfig:
    """Placeholder for future noisy differentiable QAS support."""

    enabled: bool = False
    description: str = "Differentiable noisy supernet QAS is not implemented."


@dataclass
class SupernetConfig:
    n_qubits: int = 3
    layers: int = 3
    single_qubit_gates: tuple[str, ...] = ("i", "h", "rx", "ry", "rz")
    two_qubit_gates: tuple[str, ...] = ("cx", "rzz")
    two_qubit_pairs: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))
    search_single_qubit_gates: bool = True
    search_two_qubit_gates: bool = True
    supernet_num: int = 1
    supernet_steps: int = 100
    ranking_num: int = 50
    finetune_steps: int = 20
    learning_rate: float = 0.05
    finetune_learning_rate: float = 0.03
    seed: int = 42
    device: str = "cpu"
    task: str = "classification"
    log_interval: int = 0
    use_parameter_shift: bool = False
    track_best_validation: bool = True
    ranking_strategy: str = "random"
    use_evolutionary_ranking: bool = False
    noise_mode: str = "none"
    noise_config: NoiseConfig | None = None


@dataclass(frozen=True)
class LayerArchitecture:
    single_qubit_gates: tuple[str, ...]
    two_qubit_choices: tuple[str, ...]

    def __post_init__(self) -> None:
        single = tuple(_normalize_single_gate(gate) for gate in self.single_qubit_gates)
        choices = tuple(_normalize_two_qubit_choice(choice) for choice in self.two_qubit_choices)
        object.__setattr__(self, "single_qubit_gates", single)
        object.__setattr__(self, "two_qubit_choices", choices)

    @property
    def two_qubit_mask(self) -> tuple[bool, ...]:
        """Backward-compatible view: ``True`` where a two-qubit gate is present."""
        return tuple(choice != _NO_TWO_QUBIT for choice in self.two_qubit_choices)


@dataclass(frozen=True)
class Architecture:
    layers: tuple[LayerArchitecture, ...]


@dataclass
class SupernetResult:
    best_architecture: Architecture
    best_circuit: Circuit
    best_score: float
    best_supernet_id: int
    ranking_records: list[dict[str, Any]]
    supernet_log: list[dict[str, Any]]
    finetune_log: list[dict[str, Any]]
    final_metrics: dict[str, Any]
    config: SupernetConfig


def _make_backend(device: str | torch.device | None) -> GPUBackend:
    """Pick the torch-based backend matching the requested device.

    Ascend NPU lacks complex64 kernels (matmul/kron/conj/...), so NPU devices
    must use ``NPUBackend``, whose overrides decompose those ops into real
    arithmetic. CPU/CUDA use the plain ``GPUBackend``. On a machine without a
    real NPU, ``NPUBackend`` transparently falls back to CPU and behaves like
    ``GPUBackend``.
    """
    if str(device).lower().startswith("npu"):
        return NPUBackend(device=device)
    return GPUBackend(device=device)


def _as_torch_scalar(value: torch.Tensor | float, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.reshape(())
    return torch.tensor(float(value), dtype=torch.float32, device=device)


def _float_value(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(()))
    return float(value)


def _unique_tensors(tensors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    seen: set[int] = set()
    unique: list[torch.Tensor] = []
    for tensor in tensors:
        ident = id(tensor)
        if ident not in seen:
            seen.add(ident)
            unique.append(tensor)
    return unique


def _circuit_diagram(circuit: Circuit) -> str:
    stream = io.StringIO()
    circuit.show(file=stream)
    return stream.getvalue().rstrip()


class Supernet:
    """One-stage supernet QAS for VQA ansatz selection."""

    def __init__(self, config: SupernetConfig | None = None):
        config = config or SupernetConfig()
        normalized_single = tuple(_normalize_single_gate(gate) for gate in config.single_qubit_gates)
        normalized_two = tuple(_normalize_two_qubit_gate(gate) for gate in config.two_qubit_gates)
        config = replace(config, single_qubit_gates=normalized_single, two_qubit_gates=normalized_two)
        self.config = config
        self._validate_config()

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self._rng = random.Random(config.seed)
        self._np_rng = np.random.default_rng(config.seed)

        self.backend = _make_backend(config.device)
        self.device = self.backend._device
        # Index alphabet for the per-pair two-qubit choice: "none" first so the
        # absent option keeps index 0, then the configured two-qubit gates.
        self._two_qubit_choice_alphabet: tuple[str, ...] = (_NO_TWO_QUBIT,) + tuple(config.two_qubit_gates)
        self._single_layouts = self._build_single_layouts()
        self._two_qubit_layouts = self._build_two_qubit_layouts()
        self._readout_index_cache: dict[int, torch.Tensor] = {}
        self._hamiltonian_cache: dict[int, torch.Tensor] = {}
        self._pauli_expectation_cache: dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]]] = {}

        self.shared_parameters: dict[ParameterKey, torch.nn.Parameter] = {}
        self._supernet_parameter_lists: list[list[torch.nn.Parameter]] = []
        self._best_shared_parameter_values: dict[ParameterKey, torch.Tensor] | None = None
        self._best_validation_accuracy = -math.inf
        self._best_validation_loss = math.inf
        self._initialize_shared_parameters()
        self._optimizers = [
            torch.optim.Adam(parameters, lr=config.learning_rate)
            for parameters in self._supernet_parameter_lists
        ]

    def _validate_config(self) -> None:
        cfg = self.config
        if str(cfg.noise_mode).strip().lower() != "none":
            raise NotImplementedError("Noisy differentiable supernet is not implemented yet; use noise_mode='none'.")
        if cfg.noise_config is not None and cfg.noise_config.enabled:
            raise NotImplementedError("Noisy differentiable supernet is not implemented yet.")
        ranking_strategy = str(cfg.ranking_strategy).strip().lower()
        if ranking_strategy not in {"random", "evolutionary"}:
            raise ValueError("ranking_strategy must be 'random' or 'evolutionary'")
        if cfg.n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if cfg.layers <= 0:
            raise ValueError("layers must be positive")
        if cfg.supernet_num <= 0:
            raise ValueError("supernet_num must be positive")
        if cfg.supernet_steps < 0:
            raise ValueError("supernet_steps must be non-negative")
        if cfg.ranking_num <= 0:
            raise ValueError("ranking_num must be positive")
        if cfg.finetune_steps < 0:
            raise ValueError("finetune_steps must be non-negative")
        if cfg.learning_rate <= 0 or cfg.finetune_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if not cfg.single_qubit_gates:
            raise ValueError("single_qubit_gates must be non-empty")
        for control, target in cfg.two_qubit_pairs:
            if not (0 <= int(control) < cfg.n_qubits and 0 <= int(target) < cfg.n_qubits):
                raise ValueError("two_qubit_pairs contain a qubit outside [0, n_qubits)")
            if int(control) == int(target):
                raise ValueError("CNOT control and target must be different")

    def _build_single_layouts(self) -> list[tuple[str, ...]]:
        cfg = self.config
        if not cfg.search_single_qubit_gates:
            return [tuple(cfg.single_qubit_gates[0] for _ in range(cfg.n_qubits))]
        return [tuple(layout) for layout in product(cfg.single_qubit_gates, repeat=cfg.n_qubits)]

    def _build_two_qubit_layouts(self) -> list[tuple[str, ...]]:
        width = len(self.config.two_qubit_pairs)
        if width == 0:
            return [()]
        if not self.config.search_two_qubit_gates:
            fixed = self.config.two_qubit_gates[0] if self.config.two_qubit_gates else _NO_TWO_QUBIT
            return [tuple(fixed for _ in range(width))]
        return [tuple(choice) for choice in product(self._two_qubit_choice_alphabet, repeat=width)]

    def _new_shared_parameter(self) -> torch.nn.Parameter:
        value = float(self._np_rng.uniform(-0.05, 0.05))
        return torch.nn.Parameter(torch.tensor(value, dtype=torch.float32, device=self.device))

    def _initialize_shared_parameters(self) -> None:
        cfg = self.config
        for supernet_id in range(cfg.supernet_num):
            parameters: list[torch.nn.Parameter] = []
            for layer_id in range(cfg.layers):
                # Single-qubit angles: shared across ansatze whose single-qubit
                # layout of this layer is identical (the paper's rule). Fixed
                # gates (the identity placeholder) own zero parameters.
                for layout in self._single_layouts:
                    for qubit_id, gate_type in enumerate(layout):
                        spec = _SINGLE_QUBIT_GATES[gate_type]
                        for param_index in range(spec.n_params):
                            key = self.single_parameter_key(
                                supernet_id, layer_id, layout, qubit_id, gate_type, param_index
                            )
                            if key in self.shared_parameters:
                                continue
                            tensor = self._new_shared_parameter()
                            self.shared_parameters[key] = tensor
                            parameters.append(tensor)
                # Two-qubit angles (e.g. rzz): the same indexing principle applied
                # to the second category — shared across ansatze whose two-qubit
                # layout of this layer is identical. cx owns zero parameters.
                for layout in self._two_qubit_layouts:
                    for pair_index, choice in enumerate(layout):
                        if choice == _NO_TWO_QUBIT:
                            continue
                        spec = _TWO_QUBIT_GATES[choice]
                        for param_index in range(spec.n_params):
                            key = self.two_qubit_parameter_key(
                                supernet_id, layer_id, layout, pair_index, choice, param_index
                            )
                            if key in self.shared_parameters:
                                continue
                            tensor = self._new_shared_parameter()
                            self.shared_parameters[key] = tensor
                            parameters.append(tensor)
            self._supernet_parameter_lists.append(parameters)

    def single_parameter_key(
        self,
        supernet_id: int,
        layer_id: int,
        single_layout: Sequence[str],
        qubit_id: int,
        gate_type: str,
        param_index: int = 0,
    ) -> ParameterKey:
        return (
            "sq",
            int(supernet_id),
            int(layer_id),
            tuple(_normalize_single_gate(gate) for gate in single_layout),
            int(qubit_id),
            _normalize_single_gate(gate_type),
            int(param_index),
        )

    def two_qubit_parameter_key(
        self,
        supernet_id: int,
        layer_id: int,
        two_qubit_layout: Sequence[str],
        pair_index: int,
        gate_type: str,
        param_index: int = 0,
    ) -> ParameterKey:
        return (
            "tq",
            int(supernet_id),
            int(layer_id),
            tuple(_normalize_two_qubit_choice(choice) for choice in two_qubit_layout),
            int(pair_index),
            _normalize_two_qubit_gate(gate_type),
            int(param_index),
        )

    def layer_search_space_size(self) -> int:
        return len(self._single_layouts) * len(self._two_qubit_layouts)

    def logical_search_space_size(self) -> int:
        return self.layer_search_space_size() ** self.config.layers

    def sample_architecture(self) -> Architecture:
        layers: list[LayerArchitecture] = []
        for _ in range(self.config.layers):
            single_layout = self._rng.choice(self._single_layouts)
            two_layout = self._rng.choice(self._two_qubit_layouts)
            layers.append(LayerArchitecture(single_layout, two_layout))
        return Architecture(tuple(layers))

    def encode_architecture(self, architecture: Architecture) -> tuple[tuple[int, ...], ...]:
        encoded: list[tuple[int, ...]] = []
        for layer in architecture.layers:
            single_indices = tuple(
                self.config.single_qubit_gates.index(_normalize_single_gate(gate))
                for gate in layer.single_qubit_gates
            )
            two_indices = tuple(
                self._two_qubit_choice_alphabet.index(_normalize_two_qubit_choice(choice))
                for choice in layer.two_qubit_choices
            )
            encoded.append(single_indices + two_indices)
        return tuple(encoded)

    def decode_architecture(self, indices: Sequence[Sequence[int]] | Sequence[int]) -> Architecture:
        width = self.config.n_qubits + len(self.config.two_qubit_pairs)
        if len(indices) == self.config.layers and all(hasattr(item, "__len__") for item in indices):
            layer_indices = [tuple(int(v) for v in item) for item in indices]  # type: ignore[arg-type]
        else:
            flat = tuple(int(v) for v in indices)  # type: ignore[arg-type]
            if len(flat) != width * self.config.layers:
                raise ValueError("flat architecture index list has the wrong length")
            layer_indices = [flat[i : i + width] for i in range(0, len(flat), width)]

        layers: list[LayerArchitecture] = []
        for raw in layer_indices:
            if len(raw) != width:
                raise ValueError("layer architecture index list has the wrong length")
            single = tuple(self.config.single_qubit_gates[int(idx)] for idx in raw[: self.config.n_qubits])
            choices = tuple(self._two_qubit_choice_alphabet[int(v)] for v in raw[self.config.n_qubits :])
            layers.append(LayerArchitecture(single, choices))
        return Architecture(tuple(layers))

    def cnot_count(self, architecture: Architecture) -> int:
        """Number of CNOT (``cx``) gates — the paper's noise-relevant metric."""
        return sum(
            1 for layer in architecture.layers for choice in layer.two_qubit_choices if choice == "cx"
        )

    def two_qubit_count(self, architecture: Architecture) -> int:
        """Number of two-qubit gates of any type (``cx`` and ``rzz``)."""
        return sum(
            1 for layer in architecture.layers for choice in layer.two_qubit_choices if choice != _NO_TWO_QUBIT
        )

    def build_circuit(
        self,
        architecture: Architecture,
        supernet_id: int = 0,
        parameters: Mapping[ParameterKey, torch.Tensor | float] | None = None,
    ) -> tuple[Circuit, list[ParameterKey], list[torch.Tensor]]:
        if len(architecture.layers) != self.config.layers:
            raise ValueError("architecture layer count does not match config.layers")

        parameter_source = self.shared_parameters if parameters is None else parameters
        gates: list[dict[str, Any]] = []
        active_keys: list[ParameterKey] = []
        active_tensors: list[torch.Tensor] = []

        for layer_id, layer in enumerate(architecture.layers):
            if len(layer.single_qubit_gates) != self.config.n_qubits:
                raise ValueError("single_qubit_gates length does not match n_qubits")
            if len(layer.two_qubit_choices) != len(self.config.two_qubit_pairs):
                raise ValueError("two_qubit_choices length does not match two_qubit_pairs")

            single_layout = tuple(_normalize_single_gate(gate) for gate in layer.single_qubit_gates)
            for qubit_id, gate_type in enumerate(single_layout):
                spec = _SINGLE_QUBIT_GATES[gate_type]
                params: list[torch.Tensor | float] = []
                for param_index in range(spec.n_params):
                    key = self.single_parameter_key(
                        supernet_id, layer_id, single_layout, qubit_id, gate_type, param_index
                    )
                    theta = parameter_source[key]
                    params.append(theta)
                    active_keys.append(key)
                    if isinstance(theta, torch.Tensor):
                        active_tensors.append(theta)
                gate = spec.builder(params, (qubit_id,))
                if gate is not None:
                    gates.append(gate)

            two_layout = tuple(_normalize_two_qubit_choice(choice) for choice in layer.two_qubit_choices)
            for pair_index, choice in enumerate(two_layout):
                if choice == _NO_TWO_QUBIT:
                    continue
                spec = _TWO_QUBIT_GATES[choice]
                control, target = self.config.two_qubit_pairs[pair_index]
                params = []
                for param_index in range(spec.n_params):
                    key = self.two_qubit_parameter_key(
                        supernet_id, layer_id, two_layout, pair_index, choice, param_index
                    )
                    theta = parameter_source[key]
                    params.append(theta)
                    active_keys.append(key)
                    if isinstance(theta, torch.Tensor):
                        active_tensors.append(theta)
                gate = spec.builder(params, (int(control), int(target)))
                if gate is not None:
                    gates.append(gate)

        return Circuit(*gates, n_qubits=self.config.n_qubits, backend=self.backend), active_keys, active_tensors

    def _simulate_gates(self, gates: Sequence[dict[str, Any]], initial_state: torch.Tensor | None = None) -> torch.Tensor:
        state = self.backend.zeros_state(self.config.n_qubits) if initial_state is None else initial_state
        for gate in gates:
            evolved = apply_gate_to_state(gate, state, self.config.n_qubits, self.backend)
            if evolved is None:
                matrix = gate_to_matrix(gate, cir_qubits=self.config.n_qubits, backend=self.backend)
                evolved = self.backend.apply_unitary(state, matrix)
            state = evolved
        return state

    def simulate_state(self, circuit: Circuit, initial_state: torch.Tensor | None = None) -> torch.Tensor:
        return self._simulate_gates(circuit_instructions(circuit), initial_state=initial_state)

    def _readout_indices(self, qubit: int = 0) -> torch.Tensor:
        qubit = int(qubit)
        if qubit not in self._readout_index_cache:
            indices = [
                index
                for index in range(1 << self.config.n_qubits)
                if ((index >> (self.config.n_qubits - qubit - 1)) & 1) == 1
            ]
            self._readout_index_cache[qubit] = torch.tensor(indices, dtype=torch.long, device=self.device)
        return self._readout_index_cache[qubit]

    def _probability_qubit_one(self, state: torch.Tensor, qubit: int = 0) -> torch.Tensor:
        probs = self.backend.measure_probs(state)
        return probs.index_select(0, self._readout_indices(qubit)).sum()

    def _probability_qubit_zero(self, state: torch.Tensor, qubit: int = 0) -> torch.Tensor:
        return torch.as_tensor(1.0, dtype=torch.float32, device=self.device) - self._probability_qubit_one(state, qubit)

    def _classification_predictions(self, circuit: Circuit, dataset: dict[str, tuple[torch.Tensor, torch.Tensor]], split: str) -> torch.Tensor:
        features, _ = dataset.get(split, dataset["train"])
        outputs: list[torch.Tensor] = []
        for row in features:
            encoding = [ry(row[qubit], target_qubit=qubit) for qubit in range(self.config.n_qubits)]
            state = self._simulate_gates(encoding)
            state = self.simulate_state(circuit, initial_state=state)
            outputs.append(self._probability_qubit_zero(state, qubit=self.config.n_qubits - 1))
        return torch.stack(outputs)

    def _classification_loss(self, circuit: Circuit, dataset: dict[str, tuple[torch.Tensor, torch.Tensor]], split: str) -> torch.Tensor:
        predictions = self._classification_predictions(circuit, dataset, split)
        _, labels = dataset.get(split, dataset["train"])
        return torch.mean((predictions - labels) ** 2)

    def _classification_accuracy(self, circuit: Circuit, dataset: dict[str, tuple[torch.Tensor, torch.Tensor]], split: str) -> float:
        with torch.no_grad():
            predictions = self._classification_predictions(circuit, dataset, split)
            _, labels = dataset.get(split, dataset["train"])
            pred_labels = (predictions >= 0.5).to(dtype=labels.dtype)
            return float((pred_labels == labels).float().mean().detach().cpu())

    def _hamiltonian_matrix(self, hamiltonian: Hamiltonian | np.ndarray | torch.Tensor | None) -> torch.Tensor:
        if hamiltonian is None:
            hamiltonian = h2_hamiltonian()
        cache_key = id(hamiltonian)
        if cache_key in self._hamiltonian_cache:
            return self._hamiltonian_cache[cache_key]
        if isinstance(hamiltonian, Hamiltonian):
            matrix = hamiltonian.to_matrix(self.backend)
        elif isinstance(hamiltonian, torch.Tensor):
            matrix = hamiltonian.to(dtype=self.backend._dtype, device=self.device)
        else:
            matrix = self.backend.cast(np.asarray(hamiltonian, dtype=np.complex64))
        self._hamiltonian_cache[cache_key] = matrix
        return matrix

    def _pauli_term_cache(
        self,
        hamiltonian: Hamiltonian,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]]:
        cache_key = id(hamiltonian)
        if cache_key in self._pauli_expectation_cache:
            return self._pauli_expectation_cache[cache_key]

        n_qubits = int(hamiltonian.n_qubits)
        dim = 1 << n_qubits
        cached_terms: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]] = []
        for term in hamiltonian._terms:
            mapped_indices: list[int] = []
            phase_real: list[float] = []
            phase_imag: list[float] = []
            labels = term.qubit_labels
            for index in range(dim):
                mapped = index
                phase = 1.0 + 0.0j
                for qubit, label in enumerate(labels):
                    if label == "I":
                        continue
                    bit_shift = n_qubits - qubit - 1
                    bit = (index >> bit_shift) & 1
                    if label == "X":
                        mapped ^= 1 << bit_shift
                    elif label == "Y":
                        mapped ^= 1 << bit_shift
                        phase *= 1.0j if bit == 0 else -1.0j
                    elif label == "Z":
                        if bit:
                            phase *= -1.0
                mapped_indices.append(mapped)
                phase_real.append(float(phase.real))
                phase_imag.append(float(phase.imag))

            coefficient = complex(term.coefficient)
            cached_terms.append(
                (
                    torch.tensor(mapped_indices, dtype=torch.long, device=self.device),
                    torch.tensor(phase_real, dtype=torch.float32, device=self.device),
                    torch.tensor(phase_imag, dtype=torch.float32, device=self.device),
                    float(coefficient.real),
                    float(coefficient.imag),
                )
            )
        self._pauli_expectation_cache[cache_key] = cached_terms
        return cached_terms

    def _hamiltonian_expectation(self, state: torch.Tensor, hamiltonian: Hamiltonian | np.ndarray | torch.Tensor | None) -> torch.Tensor:
        if hamiltonian is None:
            hamiltonian = h2_hamiltonian()
        if isinstance(hamiltonian, Hamiltonian):
            # 态向量可能是 (2^n, 1) 列向量；展平成一维，避免与一维 phase/index
            # 向量做广播（(2^n,1)*(2^n,) -> (2^n,2^n)）导致能量被放大 2^n 倍。
            state = state.reshape(-1)
            state_real = torch.real(state)
            state_imag = torch.imag(state)
            energy = torch.zeros((), dtype=torch.float32, device=self.device)
            for mapped_indices, phase_real, phase_imag, coefficient_real, coefficient_imag in self._pauli_term_cache(hamiltonian):
                mapped_real = state_real.index_select(0, mapped_indices)
                mapped_imag = state_imag.index_select(0, mapped_indices)
                overlap_real = mapped_real * state_real + mapped_imag * state_imag
                overlap_imag = mapped_real * state_imag - mapped_imag * state_real
                phased_real = overlap_real * phase_real - overlap_imag * phase_imag
                phased_imag = overlap_real * phase_imag + overlap_imag * phase_real
                term_real = phased_real.sum()
                term_imag = phased_imag.sum()
                energy = energy + coefficient_real * term_real - coefficient_imag * term_imag
            return energy
        return self.backend.expectation_sv(state, self._hamiltonian_matrix(hamiltonian))

    def _h2_energy(self, circuit: Circuit, hamiltonian: Hamiltonian | np.ndarray | torch.Tensor | None) -> torch.Tensor:
        state = self.simulate_state(circuit)
        return self._hamiltonian_expectation(state, hamiltonian)

    def _external_objective_loss(
        self,
        objective: ObjectiveFn,
        *,
        architecture: Architecture,
        circuit: Circuit,
        supernet_id: int,
        active_parameter_keys: list[ParameterKey],
        active_parameter_tensors: list[torch.Tensor],
        dataset: Any,
        hamiltonian: Any,
        split: str,
    ) -> torch.Tensor:
        kwargs = {
            "qas": self,
            "architecture": architecture,
            "circuit": circuit,
            "backend": self.backend,
            "supernet_id": supernet_id,
            "active_parameter_keys": active_parameter_keys,
            "active_parameter_tensors": active_parameter_tensors,
            "dataset": dataset,
            "hamiltonian": hamiltonian,
            "split": split,
            "simulate_state": self.simulate_state,
        }

        signature = inspect.signature(objective)
        parameters = signature.parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
            value = objective(**kwargs)
        else:
            filtered = {
                name: kwargs[name]
                for name, param in parameters.items()
                if name in kwargs
                and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            if filtered:
                value = objective(**filtered)
            else:
                value = objective(circuit)
        return _as_torch_scalar(value, self.device)

    def _loss(
        self,
        architecture: Architecture,
        supernet_id: int,
        objective: ObjectiveFn | str | None,
        dataset: Any = None,
        hamiltonian: Any = None,
        *,
        split: str = "train",
        parameters: Mapping[ParameterKey, torch.Tensor | float] | None = None,
    ) -> tuple[torch.Tensor, Circuit, list[ParameterKey], list[torch.Tensor]]:
        circuit, active_keys, active_tensors = self.build_circuit(
            architecture,
            supernet_id=supernet_id,
            parameters=parameters,
        )

        task = self.config.task.lower()
        objective_name = objective.lower() if isinstance(objective, str) else None
        if objective_name in {"classification", "binary_classification"} or (
            objective is None and task in {"classification", "binary_classification"}
        ):
            loss = self._classification_loss(circuit, dataset, split)
        elif objective_name in {"h2", "h2_vqe", "vqe"} or (
            objective is None and task in {"h2", "h2_vqe", "vqe"}
        ):
            loss = self._h2_energy(circuit, hamiltonian)
        elif callable(objective):
            loss = self._external_objective_loss(
                objective,
                architecture=architecture,
                circuit=circuit,
                supernet_id=supernet_id,
                active_parameter_keys=active_keys,
                active_parameter_tensors=active_tensors,
                dataset=dataset,
                hamiltonian=hamiltonian,
                split=split,
            )
        else:
            raise ValueError("objective must be callable, a known objective name, or None")
        return _as_torch_scalar(loss, self.device), circuit, active_keys, active_tensors

    def select_supernet(
        self,
        architecture: Architecture,
        objective: ObjectiveFn | str | None,
        dataset: Any = None,
        hamiltonian: Any = None,
        *,
        split: str = "train",
    ) -> tuple[int, list[float]]:
        losses: list[float] = []
        with torch.no_grad():
            for supernet_id in range(self.config.supernet_num):
                loss, _, _, _ = self._loss(
                    architecture,
                    supernet_id,
                    objective,
                    dataset,
                    hamiltonian,
                    split=split,
                )
                losses.append(_float_value(loss))
        selected = min(range(len(losses)), key=losses.__getitem__)
        return selected, losses

    def _snapshot_shared_parameters(self) -> dict[ParameterKey, torch.Tensor]:
        return {key: value.detach().clone() for key, value in self.shared_parameters.items()}

    def _restore_shared_parameters(self, values: Mapping[ParameterKey, torch.Tensor]) -> None:
        with torch.no_grad():
            for key, value in values.items():
                self.shared_parameters[key].copy_(value)

    def _track_validation_best(
        self,
        architecture: Architecture,
        supernet_id: int,
        objective: ObjectiveFn | str | None,
        dataset: Any,
        hamiltonian: Any,
    ) -> dict[str, float]:
        loss, circuit, _, _ = self._loss(
            architecture,
            supernet_id,
            objective,
            dataset,
            hamiltonian,
            split="validation",
        )
        validation_loss = _float_value(loss)
        validation_accuracy = self._classification_accuracy(circuit, dataset, "validation")
        if (
            validation_accuracy > self._best_validation_accuracy
            or (
                math.isclose(validation_accuracy, self._best_validation_accuracy)
                and validation_loss < self._best_validation_loss
            )
        ):
            self._best_validation_accuracy = validation_accuracy
            self._best_validation_loss = validation_loss
            self._best_shared_parameter_values = self._snapshot_shared_parameters()
        return {
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "best_validation_loss": self._best_validation_loss,
            "best_validation_accuracy": self._best_validation_accuracy,
        }

    def _grad_norm(self, parameters: Sequence[torch.Tensor]) -> float:
        total = 0.0
        for parameter in parameters:
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            total += float(torch.sum(grad * grad).detach().cpu())
        return math.sqrt(total)

    def _parameter_shift_update(
        self,
        parameters: Sequence[torch.Tensor],
        optimizer: torch.optim.Optimizer,
        loss_closure: Callable[[], torch.Tensor],
    ) -> float:
        # Delegate the parameter-shift rule to aicir.qml.deriv.psr so the
        # gradient definition lives in one place. The active gate angles are
        # optimized as torch Parameters (read by ``loss_closure`` via object
        # identity), so we expose a black-box objective that writes a candidate
        # angle vector into those tensors and returns the resulting scalar loss.
        # psr then shifts one coordinate by ±π/2 at a time (the standard
        # Pauli-rotation rule, coefficient 0.5).
        active = _unique_tensors(parameters)
        optimizer.zero_grad(set_to_none=True)

        base_values = np.array([_float_value(parameter) for parameter in active], dtype=float)

        def _write_values(values: np.ndarray) -> None:
            with torch.no_grad():
                for parameter, value in zip(active, np.asarray(values).reshape(-1)):
                    parameter.copy_(
                        torch.as_tensor(
                            float(value), dtype=parameter.dtype, device=parameter.device
                        ).reshape_as(parameter)
                    )

        def objective(theta: np.ndarray) -> float:
            _write_values(theta)
            return _float_value(loss_closure())

        gradients = np.asarray(psr(objective, base_values), dtype=float).reshape(-1)

        # psr leaves the tensors at the last shifted point; restore the base
        # angles before recording the gradient for the optimizer step.
        _write_values(base_values)
        for parameter, grad in zip(active, gradients):
            parameter.grad = torch.as_tensor(
                float(grad), dtype=parameter.dtype, device=parameter.device
            ).reshape_as(parameter)

        grad_norm = self._grad_norm(active)
        optimizer.step()
        return grad_norm

    def optimize_supernet(
        self,
        objective: ObjectiveFn | str | None,
        dataset: Any = None,
        hamiltonian: Any = None,
    ) -> list[dict[str, Any]]:
        log: list[dict[str, Any]] = []
        for step in range(self.config.supernet_steps):
            architecture = self.sample_architecture()
            selected_id, candidate_losses = self.select_supernet(
                architecture,
                objective,
                dataset,
                hamiltonian,
                split="train",
            )
            optimizer = self._optimizers[selected_id]

            def loss_closure() -> torch.Tensor:
                loss_value, _, _, _ = self._loss(
                    architecture,
                    selected_id,
                    objective,
                    dataset,
                    hamiltonian,
                    split="train",
                )
                return loss_value

            loss, _, active_keys, active_tensors = self._loss(
                architecture,
                selected_id,
                objective,
                dataset,
                hamiltonian,
                split="train",
            )
            active_tensors = _unique_tensors(active_tensors)
            if self.config.use_parameter_shift:
                grad_norm = self._parameter_shift_update(active_tensors, optimizer, loss_closure)
            else:
                optimizer.zero_grad(set_to_none=True)
                if loss.requires_grad:
                    loss.backward()
                grad_norm = self._grad_norm(active_tensors)
                optimizer.step()

            record = {
                "step": step,
                "architecture": architecture,
                "architecture_indices": self.encode_architecture(architecture),
                "selected_supernet_id": selected_id,
                "candidate_losses": candidate_losses,
                "loss": _float_value(loss),
                "gradient_norm": grad_norm,
                "active_parameter_count": len(active_keys),
                "cnot_count": self.cnot_count(architecture),
                "two_qubit_count": self.two_qubit_count(architecture),
            }
            if self.config.track_best_validation and self.config.task.lower() in {"classification", "binary_classification"}:
                record.update(self._track_validation_best(architecture, selected_id, objective, dataset, hamiltonian))
            log.append(record)
            if self.config.log_interval and (step + 1) % self.config.log_interval == 0:
                print(
                    f"[supernet] step={step + 1} supernet={selected_id} "
                    f"loss={record['loss']:.6f} cnot={record['cnot_count']}"
                )
        return log

    def rank_architectures(
        self,
        objective: ObjectiveFn | str | None,
        dataset: Any = None,
        hamiltonian: Any = None,
        *,
        candidates: Sequence[Architecture] | None = None,
        split: str = "validation",
    ) -> list[dict[str, Any]]:
        strategy = "evolutionary" if self.config.use_evolutionary_ranking else self.config.ranking_strategy.lower()
        if strategy == "evolutionary":
            raise NotImplementedError(
                "Evolutionary ranking from the Supplementary Information is not implemented yet; "
                "use ranking_strategy='random'."
            )
        if strategy != "random":
            raise ValueError("ranking_strategy must be 'random' or 'evolutionary'")
        if candidates is None:
            candidates = [self.sample_architecture() for _ in range(self.config.ranking_num)]
        if not candidates:
            raise ValueError("ranking requires at least one candidate architecture")

        records: list[dict[str, Any]] = []
        for architecture in candidates:
            selected_id, losses = self.select_supernet(
                architecture,
                objective,
                dataset,
                hamiltonian,
                split=split,
            )
            records.append(
                {
                    "architecture": architecture,
                    "architecture_indices": self.encode_architecture(architecture),
                    "selected_supernet_id": selected_id,
                    "score": losses[selected_id],
                    "candidate_losses": losses,
                    "cnot_count": self.cnot_count(architecture),
                    "two_qubit_count": self.two_qubit_count(architecture),
                }
            )
        records.sort(key=lambda item: item["score"])
        for rank, record in enumerate(records, start=1):
            record["rank"] = rank
        return records

    def _finetune_parameters(self, architecture: Architecture, supernet_id: int) -> dict[ParameterKey, torch.nn.Parameter]:
        _, active_keys, active_tensors = self.build_circuit(architecture, supernet_id=supernet_id)
        parameters: dict[ParameterKey, torch.nn.Parameter] = {}
        for key, tensor in zip(active_keys, active_tensors):
            if key not in parameters:
                parameters[key] = torch.nn.Parameter(tensor.detach().clone())
        return parameters

    def finetune_architecture(
        self,
        architecture: Architecture,
        supernet_id: int,
        objective: ObjectiveFn | str | None,
        dataset: Any = None,
        hamiltonian: Any = None,
    ) -> tuple[Circuit, dict[ParameterKey, torch.nn.Parameter], list[dict[str, Any]], float]:
        parameters = self._finetune_parameters(architecture, supernet_id)
        optimizer = torch.optim.Adam(list(parameters.values()), lr=self.config.finetune_learning_rate)
        log: list[dict[str, Any]] = []
        best_loss = math.inf
        best_values = {key: value.detach().clone() for key, value in parameters.items()}

        def loss_closure() -> torch.Tensor:
            loss_value, _, _, _ = self._loss(
                architecture,
                supernet_id,
                objective,
                dataset,
                hamiltonian,
                split="train",
                parameters=parameters,
            )
            return loss_value

        if self.config.finetune_steps == 0:
            loss = loss_closure()
            best_loss = _float_value(loss)

        for step in range(self.config.finetune_steps):
            loss = loss_closure()
            active = list(parameters.values())
            if self.config.use_parameter_shift:
                grad_norm = self._parameter_shift_update(active, optimizer, loss_closure)
            else:
                optimizer.zero_grad(set_to_none=True)
                if loss.requires_grad:
                    loss.backward()
                grad_norm = self._grad_norm(active)
                optimizer.step()
            loss_float = _float_value(loss)
            if loss_float < best_loss:
                best_loss = loss_float
                best_values = {key: value.detach().clone() for key, value in parameters.items()}
            log.append(
                {
                    "step": step,
                    "loss": loss_float,
                    "gradient_norm": grad_norm,
                    "active_parameter_count": len(parameters),
                }
            )

        with torch.no_grad():
            for key, value in best_values.items():
                parameters[key].copy_(value)
        numeric_parameters = {key: float(value.detach().cpu()) for key, value in parameters.items()}
        circuit, _, _ = self.build_circuit(architecture, supernet_id=supernet_id, parameters=numeric_parameters)
        return circuit, parameters, log, float(best_loss)

    def _final_metrics(
        self,
        architecture: Architecture,
        circuit: Circuit,
        finetune_parameters: Mapping[ParameterKey, torch.Tensor | float],
        selected_supernet_id: int,
        objective: ObjectiveFn | str | None,
        dataset: Any,
        hamiltonian: Any,
        ranking_best_score: float,
    ) -> dict[str, Any]:
        task = self.config.task.lower()
        metrics: dict[str, Any] = {
            "selected_ansatz": architecture,
            "selected_cnot_count": self.cnot_count(architecture),
            "selected_two_qubit_count": self.two_qubit_count(architecture),
            "selected_circuit_ascii": _circuit_diagram(circuit),
        }
        if task in {"classification", "binary_classification"}:
            for split in ("train", "validation", "test"):
                loss, _, _, _ = self._loss(
                    architecture,
                    selected_supernet_id,
                    objective,
                    dataset,
                    hamiltonian,
                    split=split,
                    parameters=finetune_parameters,
                )
                metrics[f"{split}_loss"] = _float_value(loss)
                metrics[f"{split}_accuracy"] = self._classification_accuracy(circuit, dataset, split)
            metrics["qas_ranked_best_loss"] = float(ranking_best_score)
        elif task in {"h2", "h2_vqe", "vqe"}:
            energy, _, _, _ = self._loss(
                architecture,
                selected_supernet_id,
                objective,
                dataset,
                hamiltonian,
                parameters=finetune_parameters,
            )
            metrics["qas_ranked_best_energy"] = float(ranking_best_score)
            metrics["fine_tuned_energy"] = _float_value(energy)
            metrics["baseline_vqe_energy"] = self._run_fixed_h2_vqe_baseline(hamiltonian)
        else:
            loss, _, _, _ = self._loss(
                architecture,
                selected_supernet_id,
                objective,
                dataset,
                hamiltonian,
                parameters=finetune_parameters,
            )
            metrics["final_loss"] = _float_value(loss)
            metrics["qas_ranked_best_score"] = float(ranking_best_score)
        return metrics

    def _run_fixed_h2_vqe_baseline(self, hamiltonian: Any) -> float:
        # Conventional fixed VQE reference (Fig. 3a): every configured rotation
        # type on every qubit, then one fixed CNOT entangler per pair. The gate
        # specs carry arity, so multi-/zero-parameter gates are sized correctly.
        single_types = tuple(dict.fromkeys(self.config.single_qubit_gates))
        if self.config.two_qubit_gates:
            entangler = "cx" if "cx" in self.config.two_qubit_gates else self.config.two_qubit_gates[0]
            entangler_spec: GateSpec | None = _TWO_QUBIT_GATES[entangler]
        else:
            entangler_spec = None

        per_layer = sum(_SINGLE_QUBIT_GATES[g].n_params for g in single_types) * self.config.n_qubits
        if entangler_spec is not None:
            per_layer += entangler_spec.n_params * len(self.config.two_qubit_pairs)
        total = per_layer * self.config.layers

        parameters = [self._new_shared_parameter() for _ in range(total)]
        optimizer = torch.optim.Adam(parameters, lr=self.config.finetune_learning_rate) if parameters else None
        steps = max(1, self.config.finetune_steps)
        best_energy = math.inf

        def energy_closure() -> torch.Tensor:
            gates: list[dict[str, Any]] = []
            cursor = 0
            for _ in range(self.config.layers):
                for qubit in range(self.config.n_qubits):
                    for gate_type in single_types:
                        spec = _SINGLE_QUBIT_GATES[gate_type]
                        slot = parameters[cursor : cursor + spec.n_params]
                        cursor += spec.n_params
                        gate = spec.builder(slot, (qubit,))
                        if gate is not None:
                            gates.append(gate)
                if entangler_spec is not None:
                    for control, target in self.config.two_qubit_pairs:
                        slot = parameters[cursor : cursor + entangler_spec.n_params]
                        cursor += entangler_spec.n_params
                        gate = entangler_spec.builder(slot, (int(control), int(target)))
                        if gate is not None:
                            gates.append(gate)
            state = self._simulate_gates(gates)
            return self._hamiltonian_expectation(state, hamiltonian)

        for _ in range(steps):
            loss = energy_closure()
            best_energy = min(best_energy, _float_value(loss))
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        return float(best_energy)

    def train(
        self,
        objective: ObjectiveFn | str | None = None,
        dataset: Any = None,
        hamiltonian: Any = None,
    ) -> SupernetResult:
        task = self.config.task.lower()
        if task in {"classification", "binary_classification"}:
            dataset = prepare_classification_dataset(dataset, self.config, self.device)
            ranking_split = "validation"
        elif task in {"h2", "h2_vqe", "vqe"}:
            hamiltonian = h2_hamiltonian() if hamiltonian is None else hamiltonian
            ranking_split = "train"
        else:
            ranking_split = "validation"

        supernet_log = self.optimize_supernet(objective, dataset=dataset, hamiltonian=hamiltonian)
        if (
            self.config.track_best_validation
            and task in {"classification", "binary_classification"}
            and self._best_shared_parameter_values is not None
        ):
            self._restore_shared_parameters(self._best_shared_parameter_values)
        ranking_records = self.rank_architectures(
            objective,
            dataset=dataset,
            hamiltonian=hamiltonian,
            split=ranking_split,
        )
        best_record = ranking_records[0]
        best_architecture = best_record["architecture"]
        selected_supernet_id = int(best_record["selected_supernet_id"])
        best_circuit, finetune_parameters, finetune_log, finetune_score = self.finetune_architecture(
            best_architecture,
            selected_supernet_id,
            objective,
            dataset=dataset,
            hamiltonian=hamiltonian,
        )
        final_metrics = self._final_metrics(
            best_architecture,
            best_circuit,
            finetune_parameters,
            selected_supernet_id,
            objective,
            dataset,
            hamiltonian,
            ranking_best_score=float(best_record["score"]),
        )
        final_metrics["ranking_score_distribution"] = [float(record["score"]) for record in ranking_records]
        if task in {"classification", "binary_classification"} and self._best_validation_accuracy > -math.inf:
            final_metrics["best_supernet_validation_accuracy"] = float(self._best_validation_accuracy)
            final_metrics["best_supernet_validation_loss"] = float(self._best_validation_loss)

        if task in {"classification", "binary_classification"}:
            best_score = float(final_metrics["validation_loss"])
        elif task in {"h2", "h2_vqe", "vqe"}:
            best_score = float(final_metrics["fine_tuned_energy"])
        else:
            best_score = float(final_metrics.get("final_loss", finetune_score))

        return SupernetResult(
            best_architecture=best_architecture,
            best_circuit=best_circuit,
            best_score=best_score,
            best_supernet_id=selected_supernet_id,
            ranking_records=ranking_records,
            supernet_log=supernet_log,
            finetune_log=finetune_log,
            final_metrics=final_metrics,
            config=self.config,
        )


def prepare_classification_dataset(
    dataset: Any,
    config: SupernetConfig,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if dataset is None:
        x, y = _synthetic_quantum_kernel_dataset(config.seed, config.n_qubits)
        return {
            "train": _to_torch_pair(x[:100], y[:100], device),
            "validation": _to_torch_pair(x[100:200], y[100:200], device),
            "test": _to_torch_pair(x[200:300], y[200:300], device),
        }

    if not isinstance(dataset, Mapping):
        raise TypeError("classification dataset must be a mapping or None")

    if all(key in dataset for key in ("train", "validation", "test")):
        return {
            "train": _to_torch_pair(*dataset["train"], device),
            "validation": _to_torch_pair(*dataset["validation"], device),
            "test": _to_torch_pair(*dataset["test"], device),
        }
    required = ("x_train", "y_train", "x_validation", "y_validation", "x_test", "y_test")
    if all(key in dataset for key in required):
        return {
            "train": _to_torch_pair(dataset["x_train"], dataset["y_train"], device),
            "validation": _to_torch_pair(dataset["x_validation"], dataset["y_validation"], device),
            "test": _to_torch_pair(dataset["x_test"], dataset["y_test"], device),
        }
    raise ValueError("dataset must provide train/validation/test pairs or x_*/y_* arrays")


def _ry_matrix(theta: float) -> np.ndarray:
    half = float(theta) / 2.0
    return np.array(
        [[math.cos(half), -math.sin(half)], [math.sin(half), math.cos(half)]],
        dtype=np.complex64,
    )


def _apply_single_qubit_matrix_numpy(state: np.ndarray, matrix: np.ndarray, n_qubits: int, qubit: int) -> np.ndarray:
    tensor = state.reshape((2,) * n_qubits)
    moved = np.moveaxis(tensor, qubit, 0).reshape(2, -1)
    updated = matrix @ moved
    return np.moveaxis(updated.reshape((2,) + (2,) * (n_qubits - 1)), 0, qubit).reshape(-1)


def _apply_cnot_numpy(state: np.ndarray, n_qubits: int, control: int, target: int) -> np.ndarray:
    updated = np.zeros_like(state)
    for index, amplitude in enumerate(state):
        if ((index >> (n_qubits - control - 1)) & 1) == 1:
            index ^= 1 << (n_qubits - target - 1)
        updated[index] += amplitude
    return updated


def _synthetic_quantum_kernel_dataset(seed: int, n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    if n_qubits != 3:
        raise ValueError("The built-in Supplementary synthetic classification dataset requires n_qubits=3.")

    rng = np.random.default_rng(seed)
    theta_star = rng.uniform(0.0, 2.0 * math.pi, size=(3, 3))
    features: list[np.ndarray] = []
    labels: list[float] = []
    max_trials = 200_000

    for _ in range(max_trials):
        x = rng.uniform(-math.pi, math.pi, size=3).astype(np.float32)
        state = np.zeros(8, dtype=np.complex64)
        state[0] = 1.0
        for qubit in range(3):
            state = _apply_single_qubit_matrix_numpy(state, _ry_matrix(float(x[qubit])), 3, qubit)
        for layer in range(3):
            for qubit in range(3):
                state = _apply_single_qubit_matrix_numpy(state, _ry_matrix(float(theta_star[layer, qubit])), 3, qubit)
            state = _apply_cnot_numpy(state, 3, control=0, target=1)
            state = _apply_cnot_numpy(state, 3, control=1, target=2)

        prob_last_qubit_zero = 0.0
        for index, amplitude in enumerate(state):
            if ((index >> 0) & 1) == 0:
                prob_last_qubit_zero += float(np.real(np.conj(amplitude) * amplitude))
        if prob_last_qubit_zero >= 0.75:
            features.append(x)
            labels.append(1.0)
        elif prob_last_qubit_zero <= 0.25:
            features.append(x)
            labels.append(0.0)
        if len(features) == 300:
            return np.stack(features).astype(np.float32), np.asarray(labels, dtype=np.float32)

    raise RuntimeError("Could not generate 300 accepted synthetic classification samples.")


def _to_torch_pair(features: Any, labels: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device).reshape(-1)
    if x.ndim != 2:
        raise ValueError("features must be a 2D array")
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and labels must contain the same number of samples")
    return x, y


def h2_hamiltonian() -> Hamiltonian:
    """Return the four-qubit molecular hydrogen Hamiltonian from Supplementary Eq. (19)."""

    g = -0.042
    z = {
        0: 0.178,
        1: 0.178,
        2: -0.243,
        3: -0.243,
    }
    zz = {
        (0, 1): 0.171,
        (0, 2): 0.123,
        (0, 3): 0.168,
        (1, 2): 0.168,
        (1, 3): 0.123,
        (2, 3): 0.176,
    }
    quartic = {
        "YXXY": 0.045,
        "YYXX": -0.045,
        "XXYY": -0.045,
        "XYYX": 0.045,
    }

    terms = [("I", g, [0])]
    for qubit, coefficient in z.items():
        terms.append(("Z", coefficient, [qubit]))
    for (q0, q1), coefficient in zz.items():
        terms.append(("ZZ", coefficient, [q0, q1]))
    terms.extend([
        ("YXXY", quartic["YXXY"]),
        ("YYXX", quartic["YYXX"]),
        ("XXYY", quartic["XXYY"]),
        ("XYYX", quartic["XYYX"]),
    ])
    return Hamiltonian(n_qubits=4, terms=terms)


def train_supernet(
    objective: ObjectiveFn | str | None = None,
    config: SupernetConfig | None = None,
    dataset: Any = None,
    hamiltonian: Any = None,
) -> SupernetResult:
    return Supernet(config).train(objective, dataset=dataset, hamiltonian=hamiltonian)


def classification_supernet(config: SupernetConfig | None = None) -> SupernetResult:
    if config is None:
        config = SupernetConfig(
            n_qubits=3,
            layers=3,
            single_qubit_gates=("i", "h", "rx", "ry", "rz"),
            two_qubit_gates=("cx", "rzz"),
            two_qubit_pairs=((0, 1), (0, 2), (1, 2)),
            search_single_qubit_gates=True,
            search_two_qubit_gates=True,
            task="classification",
        )
    else:
        config = replace(config, task="classification")
    return train_supernet(None, config=config)


def h2_vqe_supernet(config: SupernetConfig | None = None) -> SupernetResult:
    if config is None:
        config = SupernetConfig(
            n_qubits=4,
            layers=3,
            single_qubit_gates=("i", "h", "rx", "ry", "rz"),
            two_qubit_gates=("cx", "rzz"),
            two_qubit_pairs=((0, 1), (1, 2), (2, 3)),
            search_single_qubit_gates=True,
            search_two_qubit_gates=True,
            supernet_steps=500,
            ranking_num=500,
            finetune_steps=50,
            task="h2_vqe",
        )
    else:
        config = replace(config, task="h2_vqe")
    if config.n_qubits != 4:
        raise ValueError("h2_vqe_supernet requires n_qubits=4")
    return train_supernet(None, config=config, hamiltonian=h2_hamiltonian())


def _default_two_qubit_pairs(n_qubits: int) -> tuple[tuple[int, int], ...]:
    """Nearest- + next-nearest-neighbour pairs on a linear chain of ``n_qubits``.

    A reasonable general-purpose entangler connectivity: every pair ``(i, j)``
    with ``1 <= j - i <= 2``. This matches the connectivity used by the LiH/H2O
    VQE demos.
    """
    return tuple(
        (i, j)
        for i in range(n_qubits)
        for j in range(i + 1, n_qubits)
        if j - i <= 2
    )


def supernet_qas(
    hamiltonian: Any,
    layers: int = 6,
    supernet_num: int = 5,
    supernet_steps: int = 250,
    finetune_steps: int = 250,
    *,
    n_qubits: int | None = None,
    ranking_num: int = 80,
    single_qubit_gates: tuple[str, ...] = ("i", "h", "rx", "ry", "rz"),
    two_qubit_gates: tuple[str, ...] = ("cx", "rzz"),
    two_qubit_pairs: tuple[tuple[int, int], ...] | None = None,
    learning_rate: float = 0.1,
    finetune_learning_rate: float = 0.05,
    seed: int = 2,
    device: str = "cpu",
    use_parameter_shift: bool = False,
    **config_overrides: Any,
) -> SupernetResult:
    """Search and fine-tune a ground-state-preparing ansatz for ``hamiltonian``.

    High-level, VQE-focused wrapper over :class:`Supernet` that exposes the knobs
    you usually tune directly, so callers do not have to assemble a full
    :class:`SupernetConfig`. It builds the weight-shared supernet, optimises and
    ranks the sampled ansatze, then fine-tunes the best one.

    The five primary parameters (``hamiltonian``, ``layers``, ``supernet_num``,
    ``supernet_steps``, ``finetune_steps``) are the levers usually swept; the rest
    have sensible VQE defaults. ``n_qubits`` and ``two_qubit_pairs`` are derived
    from the Hamiltonian when omitted. Any other :class:`SupernetConfig` field can
    still be set through ``config_overrides``.

    Args:
        hamiltonian: Target Hamiltonian (must expose ``n_qubits`` or pass it).
        layers: Ansatz depth ``L``.
        supernet_num: Number of weight-shared supernets ``W``.
        supernet_steps: Supernet optimisation steps.
        finetune_steps: Fine-tuning steps for the selected circuit.
        n_qubits: Qubit count; inferred from ``hamiltonian.n_qubits`` if ``None``.
        ranking_num: Number of sampled candidate ansatze to rank.
        single_qubit_gates: Single-qubit gate pool searched per site.
        two_qubit_gates: Two-qubit gate pool (e.g. ``cx``/``rzz``).
        two_qubit_pairs: Entangler connectivity; nearest + next-nearest by default.
        learning_rate: Adam learning rate for the supernet phase.
        finetune_learning_rate: Adam learning rate for fine-tuning.
        seed: Random seed.
        device: Torch device string (``"cpu"``, ``"cuda"``, ``"npu:0"`` ...).
        use_parameter_shift: Use parameter-shift gradients instead of autograd.
        **config_overrides: Any remaining :class:`SupernetConfig` field.

    Returns:
        SupernetResult: ``final_metrics["fine_tuned_energy"]`` and
        ``final_metrics["baseline_vqe_energy"]`` hold the key energies, and
        ``best_circuit`` is the selected, fine-tuned ansatz.
    """
    if n_qubits is None:
        n_qubits = getattr(hamiltonian, "n_qubits", None)
    if n_qubits is None:
        raise ValueError(
            "n_qubits could not be inferred from the Hamiltonian; pass n_qubits explicitly."
        )
    if two_qubit_pairs is None:
        two_qubit_pairs = _default_two_qubit_pairs(int(n_qubits))
    task = config_overrides.pop("task", "vqe")

    config = SupernetConfig(
        n_qubits=int(n_qubits),
        layers=layers,
        single_qubit_gates=single_qubit_gates,
        two_qubit_gates=two_qubit_gates,
        two_qubit_pairs=two_qubit_pairs,
        supernet_num=supernet_num,
        supernet_steps=supernet_steps,
        ranking_num=ranking_num,
        finetune_steps=finetune_steps,
        learning_rate=learning_rate,
        finetune_learning_rate=finetune_learning_rate,
        seed=seed,
        device=device,
        task=task,
        use_parameter_shift=use_parameter_shift,
        **config_overrides,
    )
    return Supernet(config).train(hamiltonian=hamiltonian)


__all__ = [
    "SupernetConfig",
    "LayerArchitecture",
    "Architecture",
    "SupernetResult",
    "Supernet",
    "train_supernet",
    "classification_supernet",
    "h2_vqe_supernet",
    "supernet_qas",
]
