"""Supernet-based QAS for variational quantum algorithms.

This module implements the QAS workflow from Du et al., npj Quantum
Information 2022, in an aicir-native form:

1. set up a supernet-indexed ansatz pool,
2. share weights by layer-level single-qubit gate layout,
3. optimize sampled architectures in one stage,
4. rank sampled architectures with trained shared weights,
5. fine-tune the selected ansatz with independent parameters.

The differentiable path intentionally uses aicir ``Circuit`` objects,
``TorchBackend``, and aicir gate matrix/state evolution helpers. No external
quantum SDK is used.
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

from ..channel.backends.torch_backend import TorchBackend
from ..channel.operators import Hamiltonian
from ..core.circuit import Circuit, cx, rx, ry, rz
from ..core.gates import apply_gate_to_state, gate_to_matrix


ObjectiveFn = Callable[..., torch.Tensor | float]
ParameterKey = tuple[int, int, tuple[str, ...], int, str]


@dataclass(frozen=True)
class NoiseConfig:
    """Placeholder for future noisy differentiable QAS support."""

    enabled: bool = False
    description: str = "Differentiable noisy supernet QAS is not implemented."


@dataclass
class VQAQASConfig:
    n_qubits: int = 3
    layers: int = 3
    single_qubit_gates: tuple[str, ...] = ("ry",)
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
    noise_config: NoiseConfig | None = None


@dataclass(frozen=True)
class LayerArchitecture:
    single_qubit_gates: tuple[str, ...]
    two_qubit_mask: tuple[bool, ...]


@dataclass(frozen=True)
class Architecture:
    layers: tuple[LayerArchitecture, ...]


@dataclass
class VQAQASResult:
    best_architecture: Architecture
    best_circuit: Circuit
    best_score: float
    ranking_records: list[dict[str, Any]]
    supernet_log: list[dict[str, Any]]
    finetune_log: list[dict[str, Any]]
    final_metrics: dict[str, Any]
    config: VQAQASConfig


_ROTATION_BUILDERS = {
    "rx": rx,
    "ry": ry,
    "rz": rz,
}


def _normalize_gate_name(name: str) -> str:
    gate = str(name).strip().lower()
    if gate not in _ROTATION_BUILDERS:
        raise ValueError("VQA_QAS supports trainable single-qubit gates: rx, ry, rz")
    return gate


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


class VQAQAS:
    """One-stage supernet QAS for VQA ansatz selection."""

    def __init__(self, config: VQAQASConfig | None = None):
        config = config or VQAQASConfig()
        normalized_gates = tuple(_normalize_gate_name(gate) for gate in config.single_qubit_gates)
        config = replace(config, single_qubit_gates=normalized_gates)
        self.config = config
        self._validate_config()

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self._rng = random.Random(config.seed)
        self._np_rng = np.random.default_rng(config.seed)

        self.backend = TorchBackend(device=config.device)
        self.device = self.backend._device
        self._single_layouts = self._build_single_layouts()
        self._two_qubit_masks = self._build_two_qubit_masks()
        self._readout_index_cache: dict[int, torch.Tensor] = {}
        self._hamiltonian_cache: dict[int, torch.Tensor] = {}

        self.shared_parameters: dict[ParameterKey, torch.nn.Parameter] = {}
        self._supernet_parameter_lists: list[list[torch.nn.Parameter]] = []
        self._initialize_shared_parameters()
        self._optimizers = [
            torch.optim.Adam(parameters, lr=config.learning_rate)
            for parameters in self._supernet_parameter_lists
        ]

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.noise_config is not None and cfg.noise_config.enabled:
            raise NotImplementedError("Noisy differentiable VQA_QAS is not implemented yet.")
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

    def _build_two_qubit_masks(self) -> list[tuple[bool, ...]]:
        width = len(self.config.two_qubit_pairs)
        if width == 0:
            return [()]
        if not self.config.search_two_qubit_gates:
            return [tuple(True for _ in range(width))]
        return [tuple(bool(bit) for bit in mask) for mask in product((False, True), repeat=width)]

    def _initialize_shared_parameters(self) -> None:
        cfg = self.config
        for supernet_id in range(cfg.supernet_num):
            parameters: list[torch.nn.Parameter] = []
            for layer_id in range(cfg.layers):
                for layout in self._single_layouts:
                    for qubit_id, gate_type in enumerate(layout):
                        key = self.parameter_key(supernet_id, layer_id, layout, qubit_id, gate_type)
                        value = float(self._np_rng.uniform(-0.05, 0.05))
                        tensor = torch.nn.Parameter(
                            torch.tensor(value, dtype=torch.float32, device=self.device)
                        )
                        self.shared_parameters[key] = tensor
                        parameters.append(tensor)
            self._supernet_parameter_lists.append(parameters)

    def parameter_key(
        self,
        supernet_id: int,
        layer_id: int,
        single_layout: Sequence[str],
        qubit_id: int,
        gate_type: str,
    ) -> ParameterKey:
        return (
            int(supernet_id),
            int(layer_id),
            tuple(_normalize_gate_name(gate) for gate in single_layout),
            int(qubit_id),
            _normalize_gate_name(gate_type),
        )

    def layer_search_space_size(self) -> int:
        return len(self._single_layouts) * len(self._two_qubit_masks)

    def logical_search_space_size(self) -> int:
        return self.layer_search_space_size() ** self.config.layers

    def sample_architecture(self) -> Architecture:
        layers: list[LayerArchitecture] = []
        for _ in range(self.config.layers):
            single_layout = self._rng.choice(self._single_layouts)
            two_mask = self._rng.choice(self._two_qubit_masks)
            layers.append(LayerArchitecture(single_layout, two_mask))
        return Architecture(tuple(layers))

    def encode_architecture(self, architecture: Architecture) -> tuple[tuple[int, ...], ...]:
        encoded: list[tuple[int, ...]] = []
        for layer in architecture.layers:
            single_indices = tuple(
                self.config.single_qubit_gates.index(_normalize_gate_name(gate))
                for gate in layer.single_qubit_gates
            )
            two_indices = tuple(1 if active else 0 for active in layer.two_qubit_mask)
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
            mask = tuple(bool(v) for v in raw[self.config.n_qubits :])
            layers.append(LayerArchitecture(single, mask))
        return Architecture(tuple(layers))

    def cnot_count(self, architecture: Architecture) -> int:
        return sum(1 for layer in architecture.layers for active in layer.two_qubit_mask if active)

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
            if len(layer.two_qubit_mask) != len(self.config.two_qubit_pairs):
                raise ValueError("two_qubit_mask length does not match two_qubit_pairs")

            layout = tuple(_normalize_gate_name(gate) for gate in layer.single_qubit_gates)
            for qubit_id, gate_type in enumerate(layout):
                key = self.parameter_key(supernet_id, layer_id, layout, qubit_id, gate_type)
                theta = parameter_source[key]
                gates.append(_ROTATION_BUILDERS[gate_type](theta, target_qubit=qubit_id))
                active_keys.append(key)
                if isinstance(theta, torch.Tensor):
                    active_tensors.append(theta)

            for active, (control, target) in zip(layer.two_qubit_mask, self.config.two_qubit_pairs):
                if active:
                    gates.append(cx(target_qubit=int(target), control_qubits=[int(control)]))

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
        return self._simulate_gates(circuit.gates, initial_state=initial_state)

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

    def _classification_predictions(self, circuit: Circuit, dataset: dict[str, tuple[torch.Tensor, torch.Tensor]], split: str) -> torch.Tensor:
        features, _ = dataset.get(split, dataset["train"])
        outputs: list[torch.Tensor] = []
        for row in features:
            encoding = [ry(row[qubit], target_qubit=qubit) for qubit in range(self.config.n_qubits)]
            state = self._simulate_gates(encoding)
            state = self.simulate_state(circuit, initial_state=state)
            outputs.append(self._probability_qubit_one(state, qubit=0))
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

    def _h2_energy(self, circuit: Circuit, hamiltonian: Hamiltonian | np.ndarray | torch.Tensor | None) -> torch.Tensor:
        state = self.simulate_state(circuit)
        return self.backend.expectation_sv(state, self._hamiltonian_matrix(hamiltonian))

    def _external_objective_loss(
        self,
        objective_fn: ObjectiveFn,
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

        signature = inspect.signature(objective_fn)
        parameters = signature.parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
            value = objective_fn(**kwargs)
        else:
            filtered = {
                name: kwargs[name]
                for name, param in parameters.items()
                if name in kwargs
                and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            if filtered:
                value = objective_fn(**filtered)
            else:
                value = objective_fn(circuit)
        return _as_torch_scalar(value, self.device)

    def _loss(
        self,
        architecture: Architecture,
        supernet_id: int,
        objective_fn: ObjectiveFn | str | None,
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
        objective_name = objective_fn.lower() if isinstance(objective_fn, str) else None
        if objective_name in {"classification", "binary_classification"} or (
            objective_fn is None and task in {"classification", "binary_classification"}
        ):
            loss = self._classification_loss(circuit, dataset, split)
        elif objective_name in {"h2", "h2_vqe", "vqe"} or (
            objective_fn is None and task in {"h2", "h2_vqe", "vqe"}
        ):
            loss = self._h2_energy(circuit, hamiltonian)
        elif callable(objective_fn):
            loss = self._external_objective_loss(
                objective_fn,
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
            raise ValueError("objective_fn must be callable, a known objective name, or None")
        return _as_torch_scalar(loss, self.device), circuit, active_keys, active_tensors

    def select_supernet(
        self,
        architecture: Architecture,
        objective_fn: ObjectiveFn | str | None,
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
                    objective_fn,
                    dataset,
                    hamiltonian,
                    split=split,
                )
                losses.append(_float_value(loss))
        selected = min(range(len(losses)), key=losses.__getitem__)
        return selected, losses

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
        active = _unique_tensors(parameters)
        optimizer.zero_grad(set_to_none=True)
        for parameter in active:
            with torch.no_grad():
                parameter.add_(math.pi / 2.0)
                plus = _float_value(loss_closure())
                parameter.add_(-math.pi)
                minus = _float_value(loss_closure())
                parameter.add_(math.pi / 2.0)
            grad = (plus - minus) / 2.0
            parameter.grad = torch.tensor(grad, dtype=parameter.dtype, device=parameter.device).reshape_as(parameter)
        grad_norm = self._grad_norm(active)
        optimizer.step()
        return grad_norm

    def optimize_supernet(
        self,
        objective_fn: ObjectiveFn | str | None,
        dataset: Any = None,
        hamiltonian: Any = None,
    ) -> list[dict[str, Any]]:
        log: list[dict[str, Any]] = []
        for step in range(self.config.supernet_steps):
            architecture = self.sample_architecture()
            selected_id, candidate_losses = self.select_supernet(
                architecture,
                objective_fn,
                dataset,
                hamiltonian,
                split="train",
            )
            optimizer = self._optimizers[selected_id]

            def loss_closure() -> torch.Tensor:
                loss_value, _, _, _ = self._loss(
                    architecture,
                    selected_id,
                    objective_fn,
                    dataset,
                    hamiltonian,
                    split="train",
                )
                return loss_value

            loss, _, active_keys, active_tensors = self._loss(
                architecture,
                selected_id,
                objective_fn,
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
            }
            log.append(record)
            if self.config.log_interval and (step + 1) % self.config.log_interval == 0:
                print(
                    f"[VQA_QAS] step={step + 1} supernet={selected_id} "
                    f"loss={record['loss']:.6f} cnot={record['cnot_count']}"
                )
        return log

    def rank_architectures(
        self,
        objective_fn: ObjectiveFn | str | None,
        dataset: Any = None,
        hamiltonian: Any = None,
        *,
        candidates: Sequence[Architecture] | None = None,
        split: str = "validation",
    ) -> list[dict[str, Any]]:
        if candidates is None:
            candidates = [self.sample_architecture() for _ in range(self.config.ranking_num)]
        if not candidates:
            raise ValueError("ranking requires at least one candidate architecture")

        records: list[dict[str, Any]] = []
        for architecture in candidates:
            selected_id, losses = self.select_supernet(
                architecture,
                objective_fn,
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
        objective_fn: ObjectiveFn | str | None,
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
                objective_fn,
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
        objective_fn: ObjectiveFn | str | None,
        dataset: Any,
        hamiltonian: Any,
        ranking_best_score: float,
    ) -> dict[str, Any]:
        task = self.config.task.lower()
        metrics: dict[str, Any] = {
            "selected_ansatz": architecture,
            "selected_cnot_count": self.cnot_count(architecture),
            "selected_circuit_ascii": _circuit_diagram(circuit),
        }
        if task in {"classification", "binary_classification"}:
            for split in ("train", "validation", "test"):
                loss, _, _, _ = self._loss(
                    architecture,
                    selected_supernet_id,
                    objective_fn,
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
                objective_fn,
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
                objective_fn,
                dataset,
                hamiltonian,
                parameters=finetune_parameters,
            )
            metrics["final_loss"] = _float_value(loss)
            metrics["qas_ranked_best_score"] = float(ranking_best_score)
        return metrics

    def _run_fixed_h2_vqe_baseline(self, hamiltonian: Any) -> float:
        hamiltonian_matrix = self._hamiltonian_matrix(hamiltonian)
        gate_types = tuple(dict.fromkeys(self.config.single_qubit_gates))
        parameters: list[torch.nn.Parameter] = []
        for _ in range(self.config.layers * self.config.n_qubits * len(gate_types)):
            value = float(self._np_rng.uniform(-0.05, 0.05))
            parameters.append(torch.nn.Parameter(torch.tensor(value, dtype=torch.float32, device=self.device)))
        optimizer = torch.optim.Adam(parameters, lr=self.config.finetune_learning_rate)
        steps = max(1, self.config.finetune_steps)
        best_energy = math.inf

        def energy_closure() -> torch.Tensor:
            gates: list[dict[str, Any]] = []
            cursor = 0
            for _ in range(self.config.layers):
                for qubit in range(self.config.n_qubits):
                    for gate_type in gate_types:
                        gates.append(_ROTATION_BUILDERS[gate_type](parameters[cursor], target_qubit=qubit))
                        cursor += 1
                for control, target in self.config.two_qubit_pairs:
                    gates.append(cx(target_qubit=int(target), control_qubits=[int(control)]))
            state = self._simulate_gates(gates)
            return self.backend.expectation_sv(state, hamiltonian_matrix)

        for _ in range(steps):
            loss = energy_closure()
            best_energy = min(best_energy, _float_value(loss))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        return float(best_energy)

    def train(
        self,
        objective_fn: ObjectiveFn | str | None = None,
        dataset: Any = None,
        hamiltonian: Any = None,
    ) -> VQAQASResult:
        task = self.config.task.lower()
        if task in {"classification", "binary_classification"}:
            dataset = prepare_classification_dataset(dataset, self.config, self.device)
            ranking_split = "validation"
        elif task in {"h2", "h2_vqe", "vqe"}:
            hamiltonian = h2_hamiltonian() if hamiltonian is None else hamiltonian
            ranking_split = "train"
        else:
            ranking_split = "validation"

        supernet_log = self.optimize_supernet(objective_fn, dataset=dataset, hamiltonian=hamiltonian)
        ranking_records = self.rank_architectures(
            objective_fn,
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
            objective_fn,
            dataset=dataset,
            hamiltonian=hamiltonian,
        )
        final_metrics = self._final_metrics(
            best_architecture,
            best_circuit,
            finetune_parameters,
            selected_supernet_id,
            objective_fn,
            dataset,
            hamiltonian,
            ranking_best_score=float(best_record["score"]),
        )

        if task in {"classification", "binary_classification"}:
            best_score = float(final_metrics["validation_loss"])
        elif task in {"h2", "h2_vqe", "vqe"}:
            best_score = float(final_metrics["fine_tuned_energy"])
        else:
            best_score = float(final_metrics.get("final_loss", finetune_score))

        return VQAQASResult(
            best_architecture=best_architecture,
            best_circuit=best_circuit,
            best_score=best_score,
            ranking_records=ranking_records,
            supernet_log=supernet_log,
            finetune_log=finetune_log,
            final_metrics=final_metrics,
            config=self.config,
        )


def prepare_classification_dataset(
    dataset: Any,
    config: VQAQASConfig,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if dataset is None:
        rng = np.random.default_rng(config.seed)
        total = 300
        x = rng.uniform(-math.pi, math.pi, size=(total, config.n_qubits)).astype(np.float32)
        signal = np.sin(x[:, 0])
        if config.n_qubits > 1:
            signal = signal + 0.5 * np.cos(x[:, 1])
        if config.n_qubits > 2:
            signal = signal - 0.25 * x[:, 2]
        y = (signal > np.median(signal)).astype(np.float32)
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


def _to_torch_pair(features: Any, labels: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device).reshape(-1)
    if x.ndim != 2:
        raise ValueError("features must be a 2D array")
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and labels must contain the same number of samples")
    return x, y


def h2_hamiltonian() -> Hamiltonian:
    """Return a fixed four-qubit H2 Hamiltonian at about 0.735 A bond length.

    Coefficients follow the common O'Malley et al. four-qubit Pauli expansion
    used in VQE examples. The term layout matches Eq. (6) in Du et al. 2022.
    """

    g = -0.810547980537326
    z = {
        0: 0.172183932619155,
        1: 0.172183932619155,
        2: -0.225753492224024,
        3: -0.225753492224024,
    }
    zz = {
        (0, 1): 0.168927538700879,
        (0, 2): 0.120546027942558,
        (0, 3): 0.165868975685246,
        (1, 2): 0.165868975685246,
        (1, 3): 0.120546027942558,
        (2, 3): 0.174349027774017,
    }
    quartic = {
        "YXXY": 0.0452327999460578,
        "YYXX": -0.0452327999460578,
        "XXYY": -0.0452327999460578,
        "XYYX": 0.0452327999460578,
    }

    hamiltonian = Hamiltonian(n_qubits=4).term(g, {"I": [0]})
    for qubit, coefficient in z.items():
        hamiltonian.term(coefficient, {"Z": [qubit]})
    for (q0, q1), coefficient in zz.items():
        hamiltonian.term(coefficient, {"Z": [q0, q1]})
    hamiltonian.term(quartic["YXXY"], {"Y": [0, 3], "X": [1, 2]})
    hamiltonian.term(quartic["YYXX"], {"Y": [0, 1], "X": [2, 3]})
    hamiltonian.term(quartic["XXYY"], {"X": [0, 1], "Y": [2, 3]})
    hamiltonian.term(quartic["XYYX"], {"X": [0, 3], "Y": [1, 2]})
    return hamiltonian


def train_vqa_qas(
    objective_fn: ObjectiveFn | str | None = None,
    config: VQAQASConfig | None = None,
    dataset: Any = None,
    hamiltonian: Any = None,
) -> VQAQASResult:
    return VQAQAS(config).train(objective_fn, dataset=dataset, hamiltonian=hamiltonian)


def vqa_qas(
    objective_fn: ObjectiveFn | str | None = None,
    config: VQAQASConfig | None = None,
    dataset: Any = None,
    hamiltonian: Any = None,
) -> VQAQASResult:
    return train_vqa_qas(objective_fn, config=config, dataset=dataset, hamiltonian=hamiltonian)


def classification_vqa_qas(config: VQAQASConfig | None = None) -> VQAQASResult:
    if config is None:
        config = VQAQASConfig(
            n_qubits=3,
            layers=3,
            single_qubit_gates=("ry",),
            two_qubit_pairs=((0, 1), (0, 2), (1, 2)),
            search_single_qubit_gates=True,
            search_two_qubit_gates=True,
            task="classification",
        )
    else:
        config = replace(config, task="classification")
    return train_vqa_qas(None, config=config)


def h2_vqe_qas(config: VQAQASConfig | None = None) -> VQAQASResult:
    if config is None:
        config = VQAQASConfig(
            n_qubits=4,
            layers=3,
            single_qubit_gates=("ry", "rz"),
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
        raise ValueError("h2_vqe_qas requires n_qubits=4")
    return train_vqa_qas(None, config=config, hamiltonian=h2_hamiltonian())


__all__ = [
    "VQAQASConfig",
    "LayerArchitecture",
    "Architecture",
    "VQAQASResult",
    "VQAQAS",
    "train_vqa_qas",
    "vqa_qas",
    "classification_vqa_qas",
    "h2_vqe_qas",
]
