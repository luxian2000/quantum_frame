"""Differentiable Quantum Architecture Search (DQAS).

This module implements the Monte Carlo DQAS loop from Zhang et al.,
"Differentiable Quantum Architecture Search" (2022): circuit structures are
sampled from independent categorical distributions parameterized by alpha,
while circuit angles are shared in a parameter pool theta[position, op, param].
The architecture logits are updated with the score-function estimator and the
circuit angles are updated by ordinary automatic differentiation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Sequence

import numpy as np
import torch

from aicir import (
    Circuit,
    GPUBackend,
    Hamiltonian,
    NPUBackend,
    cx,
    double_excitation,
    pauli_x,
    rx,
    ry,
    rz,
    single_excitation,
)
from ...core.gates import gate_to_matrix
from ..problems.hamiltonians import hamiltonian_matrix as pauli_hamiltonian_matrix


HamiltonianInput = np.ndarray | torch.Tensor | Hamiltonian | Sequence[tuple[float, str]]
GatePoolInput = str | Sequence[str] | set[str]
_GATE_ORDER = ("identity", "rx", "ry", "rz", "rzryrz", "cx", "excitation")


@dataclass(frozen=True)
class _Operation:
    kind: str
    qubits: tuple[int, ...]
    n_params: int
    label: str


@dataclass
class DQASConfig:
    n_qubits: int | None = None
    layers: int = 3
    gate_pool: GatePoolInput = "generic"
    two_qubit_pairs: tuple[tuple[int, int], ...] | None = None
    single_excitations: tuple[tuple[int, int], ...] = ()
    double_excitations: tuple[tuple[int, int, int, int], ...] = ()
    hf_occupied_qubits: tuple[int, ...] = ()
    pool: GatePoolInput | None = None
    operation_pool: GatePoolInput | None = None

    search_epochs: int = 100
    batch_size: int = 16
    theta_steps: int = 1
    finetune_steps: int = 20

    architecture_learning_rate: float = 0.05
    theta_learning_rate: float = 0.05
    finetune_learning_rate: float = 0.03
    baseline_momentum: float = 0.9

    device: str = "cpu"
    seed: int = 42
    log_interval: int = 0
    initial_state: Any = None


@dataclass
class DQASResult:
    circuit: Circuit
    parameters: dict[str, float]
    minimum_energy: float
    best_loss: float
    architecture_indices: tuple[int, ...]
    architecture_labels: tuple[str, ...]
    architecture_probabilities: np.ndarray
    search_log: list[dict[str, Any]]
    finetune_log: list[dict[str, Any]]
    config: DQASConfig


def _make_backend(device: str | torch.device | None) -> GPUBackend:
    if str(device).lower().startswith("npu"):
        return NPUBackend(device=device)
    return GPUBackend(device=device)


def _infer_n_qubits_from_matrix(matrix: np.ndarray) -> int:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hamiltonian matrix must be square")
    dim = int(matrix.shape[0])
    n_qubits = int(round(math.log2(dim))) if dim > 0 else 0
    if (1 << n_qubits) != dim:
        raise ValueError("hamiltonian matrix dimension must be a power of two")
    return n_qubits


def _hamiltonian_to_numpy(hamiltonian: HamiltonianInput, backend: GPUBackend) -> tuple[np.ndarray, int]:
    if isinstance(hamiltonian, Hamiltonian):
        matrix = backend.to_numpy(hamiltonian.to_matrix(backend))
    elif isinstance(hamiltonian, torch.Tensor):
        matrix = hamiltonian.detach().cpu().numpy()
    elif isinstance(hamiltonian, np.ndarray):
        matrix = hamiltonian
    else:
        matrix = pauli_hamiltonian_matrix(hamiltonian)
    matrix = np.asarray(matrix, dtype=np.complex64)
    return matrix, _infer_n_qubits_from_matrix(matrix)


class DifferentiableQAS:
    """DQAS trainer with independent categorical architecture variables."""

    def __init__(self, config: DQASConfig | None = None):
        self.config = DQASConfig() if config is None else config
        self._validate_config()

        torch.manual_seed(int(self.config.seed))
        np.random.seed(int(self.config.seed))

        self.backend = _make_backend(self.config.device)
        self.device = self.backend._device
        self.n_qubits = int(self.config.n_qubits)
        self.layers = int(self.config.layers)
        self._operations = self._build_operation_pool()
        self.operation_labels = tuple(op.label for op in self._operations)
        self.max_params = max(op.n_params for op in self._operations)

        self.alpha = torch.nn.Parameter(
            torch.zeros(
                self.layers,
                len(self._operations),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.theta = torch.nn.Parameter(
            torch.randn(
                self.layers,
                len(self._operations),
                self.max_params,
                dtype=torch.float32,
                device=self.device,
            )
        )

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.n_qubits is None:
            raise ValueError("DQASConfig.n_qubits must be set before constructing DifferentiableQAS")
        if int(cfg.n_qubits) <= 0:
            raise ValueError("n_qubits must be positive")
        if int(cfg.layers) <= 0:
            raise ValueError("layers must be positive")
        if int(cfg.search_epochs) < 0:
            raise ValueError("search_epochs must be non-negative")
        if int(cfg.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(cfg.theta_steps) < 0:
            raise ValueError("theta_steps must be non-negative")
        if int(cfg.finetune_steps) < 0:
            raise ValueError("finetune_steps must be non-negative")
        if cfg.architecture_learning_rate <= 0:
            raise ValueError("architecture_learning_rate must be positive")
        if cfg.theta_learning_rate <= 0:
            raise ValueError("theta_learning_rate must be positive")
        if cfg.finetune_learning_rate <= 0:
            raise ValueError("finetune_learning_rate must be positive")
        if not (0.0 <= float(cfg.baseline_momentum) < 1.0):
            raise ValueError("baseline_momentum must be in [0, 1)")
        self._normalized_gate_pool()
        self._normalized_two_qubit_pairs()
        self._normalized_single_excitations()
        self._normalized_double_excitations()
        self._normalized_hf_occupied_qubits()

    def _build_operation_pool(self) -> list[_Operation]:
        gate_names = self._normalized_gate_pool()
        pairs = self._normalized_two_qubit_pairs()
        operations: list[_Operation] = []
        for gate_name in gate_names:
            if gate_name == "identity":
                operations.append(_Operation("identity", (), 0, "identity"))
            elif gate_name in ("rx", "ry", "rz"):
                operations.extend(
                    _Operation(gate_name, (target,), 1, f"{gate_name}_{target}")
                    for target in range(self.n_qubits)
                )
            elif gate_name == "rzryrz":
                operations.extend(
                    _Operation("rzryrz", (target,), 3, f"rzryrz_{target}")
                    for target in range(self.n_qubits)
                )
            elif gate_name == "cx":
                operations.extend(
                    _Operation("cx", (control, target), 0, f"cx_{control}_to_{target}")
                    for control, target in pairs
                )
            elif gate_name == "excitation":
                operations.append(_Operation("identity", (), 0, "identity"))
                operations.extend(
                    _Operation("single_excitation", (i, j), 1, f"single_{i}_{j}")
                    for i, j in self._normalized_single_excitations()
                )
                operations.extend(
                    _Operation("double_excitation", qs, 1, "double_" + "_".join(map(str, qs)))
                    for qs in self._normalized_double_excitations()
                )
            else:
                raise ValueError(f"unsupported DQAS gate {gate_name!r}")
        if not operations:
            raise ValueError("gate_pool must expand to at least one operation")
        return operations

    def _configured_gate_pool(self) -> GatePoolInput:
        cfg = self.config
        provided: list[tuple[str, GatePoolInput]] = []
        if cfg.gate_pool != "generic":
            provided.append(("gate_pool", cfg.gate_pool))
        if cfg.pool is not None:
            provided.append(("pool", cfg.pool))
        if cfg.operation_pool is not None:
            provided.append(("operation_pool", cfg.operation_pool))
        if len(provided) > 1:
            names = ", ".join(name for name, _ in provided)
            raise ValueError(f"provide only one gate-pool field; got {names}")
        return provided[0][1] if provided else cfg.gate_pool

    def _normalized_gate_pool(self) -> tuple[str, ...]:
        pool = self._configured_gate_pool()
        if isinstance(pool, str):
            key = pool.strip().lower().replace("-", "_")
            if key == "generic":
                return ("identity", "rzryrz", "cx")
            if key == "excitation":
                return ("excitation",)
            names = (key,)
        elif isinstance(pool, set):
            lowered = {str(name).strip().lower().replace("-", "_") for name in pool}
            names = tuple(name for name in _GATE_ORDER if name in lowered)
            unknown = sorted(lowered.difference(_GATE_ORDER))
            if unknown:
                raise ValueError(f"unsupported DQAS gates: {', '.join(unknown)}")
        else:
            names = tuple(str(name).strip().lower().replace("-", "_") for name in pool)

        if not names:
            raise ValueError("gate_pool cannot be empty")
        unknown = [name for name in names if name not in _GATE_ORDER]
        if unknown:
            raise ValueError(f"unsupported DQAS gates: {', '.join(unknown)}")
        if len(set(names)) != len(names):
            raise ValueError("gate_pool contains duplicate gate names")
        return names

    def _normalized_two_qubit_pairs(self) -> tuple[tuple[int, int], ...]:
        n_qubits = int(self.config.n_qubits)
        pairs = self.config.two_qubit_pairs
        if pairs is None:
            return tuple(
                (control, target)
                for control in range(n_qubits)
                for target in range(n_qubits)
                if control != target
            )
        normalized: list[tuple[int, int]] = []
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError(f"two_qubit_pairs entry {pair!r} must contain exactly two qubits")
            control, target = int(pair[0]), int(pair[1])
            if control == target:
                raise ValueError("two_qubit_pairs cannot contain self-loops")
            if not (0 <= control < n_qubits and 0 <= target < n_qubits):
                raise ValueError(f"two_qubit pair {(control, target)} out of range for {n_qubits} qubits")
            normalized.append((control, target))
        if len(set(normalized)) != len(normalized):
            raise ValueError("two_qubit_pairs contains duplicate pairs")
        return tuple(normalized)

    def _normalized_single_excitations(self) -> tuple[tuple[int, int], ...]:
        n_qubits = int(self.config.n_qubits)
        normalized: list[tuple[int, int]] = []
        for i, j in self.config.single_excitations:
            i, j = int(i), int(j)
            if i == j:
                raise ValueError("single_excitations cannot contain self-excitations")
            if not (0 <= i < n_qubits and 0 <= j < n_qubits):
                raise ValueError(f"single excitation {(i, j)} out of range for {n_qubits} qubits")
            normalized.append((i, j))
        if len(set(normalized)) != len(normalized):
            raise ValueError("single_excitations contains duplicate entries")
        return tuple(normalized)

    def _normalized_double_excitations(self) -> tuple[tuple[int, int, int, int], ...]:
        n_qubits = int(self.config.n_qubits)
        normalized: list[tuple[int, int, int, int]] = []
        for group in self.config.double_excitations:
            if len(group) != 4:
                raise ValueError(f"double excitation {group!r} must contain exactly four qubits")
            qs = tuple(int(q) for q in group)
            if len(set(qs)) != 4:
                raise ValueError(f"double excitation {qs!r} must use four distinct qubits")
            if any(not (0 <= q < n_qubits) for q in qs):
                raise ValueError(f"double excitation {qs!r} out of range for {n_qubits} qubits")
            normalized.append(qs)
        if len(set(normalized)) != len(normalized):
            raise ValueError("double_excitations contains duplicate entries")
        return tuple(normalized)

    def _normalized_hf_occupied_qubits(self) -> tuple[int, ...]:
        n_qubits = int(self.config.n_qubits)
        occupied = tuple(int(q) for q in self.config.hf_occupied_qubits)
        if len(set(occupied)) != len(occupied):
            raise ValueError("hf_occupied_qubits contains duplicate qubits")
        for q in occupied:
            if not (0 <= q < n_qubits):
                raise ValueError(f"hf_occupied_qubit {q} out of range for {n_qubits} qubits")
        return occupied

    def backend_tensor(self, value: Any, *, dtype: str = "float") -> torch.Tensor:
        if dtype == "long":
            return torch.as_tensor(value, dtype=torch.long, device=self.device)
        if dtype == "float":
            return torch.as_tensor(value, dtype=torch.float32, device=self.device)
        raise ValueError("dtype must be 'float' or 'long'")

    def architecture_probabilities_tensor(self) -> torch.Tensor:
        return torch.softmax(self.alpha, dim=-1)

    def architecture_probabilities(self) -> np.ndarray:
        return self.architecture_probabilities_tensor().detach().cpu().numpy()

    def sample_architectures(self, batch_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch = int(self.config.batch_size if batch_size is None else batch_size)
        probs = self.architecture_probabilities_tensor()
        samples_by_layer = [
            torch.multinomial(probs[layer], num_samples=batch, replacement=True)
            for layer in range(self.layers)
        ]
        samples = torch.stack(samples_by_layer, dim=1)
        gathered = probs.unsqueeze(0).expand(batch, -1, -1).gather(2, samples.unsqueeze(-1)).squeeze(-1)
        log_probs = torch.log(gathered.clamp_min(1.0e-12)).sum(dim=1)
        return samples, log_probs

    def _score_function_alpha_gradient_tensor(
        self,
        samples: torch.Tensor,
        losses: torch.Tensor,
        *,
        baseline: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        detached_losses = losses.detach().reshape(-1)
        if baseline is None:
            centered = detached_losses - detached_losses.mean()
        else:
            centered = detached_losses - torch.as_tensor(baseline, dtype=torch.float32, device=self.device)
        probs = self.architecture_probabilities_tensor().detach()
        one_hot = torch.nn.functional.one_hot(samples, num_classes=len(self._operations)).to(torch.float32)
        return torch.mean(centered.reshape(-1, 1, 1) * (one_hot - probs.unsqueeze(0)), dim=0)

    def score_function_alpha_gradient(
        self,
        samples: torch.Tensor,
        losses: torch.Tensor,
        *,
        baseline: float | torch.Tensor | None = None,
    ) -> np.ndarray:
        return self._score_function_alpha_gradient_tensor(samples, losses, baseline=baseline).detach().cpu().numpy()

    def _identity_matrix(self) -> torch.Tensor:
        return self.backend.eye(1 << self.n_qubits)

    def _gate_matrix(self, gate: dict) -> torch.Tensor:
        return gate_to_matrix(gate, cir_qubits=self.n_qubits, backend=self.backend)

    def _operation_matrix(self, layer: int, op_index: int, *, detach_theta: bool) -> torch.Tensor:
        op = self._operations[int(op_index)]
        if op.kind == "identity":
            return self._identity_matrix()
        angles = self.theta[layer, int(op_index)]
        if detach_theta:
            angles = angles.detach()
        if op.kind == "rx":
            return self._gate_matrix(rx(angles[0], target_qubit=op.qubits[0]))
        if op.kind == "ry":
            return self._gate_matrix(ry(angles[0], target_qubit=op.qubits[0]))
        if op.kind == "rz":
            return self._gate_matrix(rz(angles[0], target_qubit=op.qubits[0]))
        if op.kind == "rzryrz":
            target = op.qubits[0]
            rz0 = self._gate_matrix(rz(angles[0], target_qubit=target))
            ry1 = self._gate_matrix(ry(angles[1], target_qubit=target))
            rz2 = self._gate_matrix(rz(angles[2], target_qubit=target))
            return self.backend.matmul(rz2, self.backend.matmul(ry1, rz0))
        if op.kind == "cx":
            control, target = op.qubits
            return self._gate_matrix(cx(target_qubit=target, control_qubits=[control]))
        if op.kind == "single_excitation":
            i, j = op.qubits
            return self._gate_matrix(single_excitation(angles[0], i, j))
        if op.kind == "double_excitation":
            return self._gate_matrix(double_excitation(angles[0], *op.qubits))
        raise ValueError(f"unknown operation kind {op.kind!r}")

    def _hf_state(self) -> torch.Tensor:
        state = self.backend.zeros_state(self.n_qubits)
        for q in self._normalized_hf_occupied_qubits():
            state = self.backend.apply_unitary(state, self._gate_matrix(pauli_x(target_qubit=q)))
        return state.reshape(-1)

    def _initial_state(self) -> torch.Tensor:
        if self.config.initial_state is not None:
            state = self.backend.cast(self.config.initial_state)
            return state.reshape(-1)
        if self.config.hf_occupied_qubits:
            return self._hf_state()
        return self.backend.zeros_state(self.n_qubits).reshape(-1)

    def _simulate_indices(self, indices: torch.Tensor, *, detach_theta: bool) -> torch.Tensor:
        state = self._initial_state()
        for layer, op_index in enumerate(indices.detach().cpu().numpy().astype(int).tolist()):
            matrix = self._operation_matrix(layer, op_index, detach_theta=detach_theta)
            state = self.backend.apply_unitary(state, matrix)
        return state.reshape(-1)

    def _loss(self, indices: torch.Tensor, hamiltonian_matrix: torch.Tensor, *, detach_theta: bool = False) -> torch.Tensor:
        state = self._simulate_indices(indices, detach_theta=detach_theta)
        value = self.backend.expectation_sv(state, hamiltonian_matrix)
        if isinstance(value, torch.Tensor):
            return torch.real(value).reshape(())
        return torch.tensor(float(value), dtype=torch.float32, device=self.device)

    def _batch_losses(self, samples: torch.Tensor, hamiltonian_matrix: torch.Tensor, *, detach_theta: bool = False) -> torch.Tensor:
        return torch.stack([self._loss(sample, hamiltonian_matrix, detach_theta=detach_theta) for sample in samples])

    @staticmethod
    def _float_value(value: torch.Tensor | float) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().reshape(()))
        return float(value)

    def _deterministic_indices(self) -> torch.Tensor:
        return self.alpha.detach().argmax(dim=-1)

    def _architecture_indices_tuple(self, indices: torch.Tensor) -> tuple[int, ...]:
        return tuple(int(value) for value in indices.detach().cpu().numpy().astype(int).tolist())

    def _architecture_labels_tuple(self, indices: torch.Tensor) -> tuple[str, ...]:
        return tuple(self.operation_labels[index] for index in self._architecture_indices_tuple(indices))

    def _build_circuit(self, indices: torch.Tensor) -> tuple[Circuit, dict[str, float]]:
        idx = self._architecture_indices_tuple(indices)
        theta_np = self.theta.detach().cpu().numpy()
        gates: list[dict[str, Any]] = []
        parameters: dict[str, float] = {}
        for q in self._normalized_hf_occupied_qubits():
            gates.append(pauli_x(target_qubit=q))
        for layer, op_index in enumerate(idx):
            op = self._operations[op_index]
            angles = theta_np[layer, op_index]
            if op.kind == "identity":
                continue
            if op.kind in ("rx", "ry", "rz"):
                target = op.qubits[0]
                parameters[f"theta_{layer}_{op_index}_{op.kind}"] = float(angles[0])
                if op.kind == "rx":
                    gates.append(rx(float(angles[0]), target_qubit=target))
                elif op.kind == "ry":
                    gates.append(ry(float(angles[0]), target_qubit=target))
                else:
                    gates.append(rz(float(angles[0]), target_qubit=target))
            elif op.kind == "rzryrz":
                target = op.qubits[0]
                parameters[f"theta_{layer}_{op_index}_rz0"] = float(angles[0])
                parameters[f"theta_{layer}_{op_index}_ry"] = float(angles[1])
                parameters[f"theta_{layer}_{op_index}_rz2"] = float(angles[2])
                gates.append(rz(float(angles[0]), target_qubit=target))
                gates.append(ry(float(angles[1]), target_qubit=target))
                gates.append(rz(float(angles[2]), target_qubit=target))
            elif op.kind == "cx":
                control, target = op.qubits
                gates.append(cx(target_qubit=target, control_qubits=[control]))
            elif op.kind == "single_excitation":
                i, j = op.qubits
                parameters[f"theta_{layer}_{op_index}"] = float(angles[0])
                gates.append(single_excitation(float(angles[0]), i, j))
            elif op.kind == "double_excitation":
                parameters[f"theta_{layer}_{op_index}"] = float(angles[0])
                gates.append(double_excitation(float(angles[0]), *op.qubits))
        return Circuit(*gates, n_qubits=self.n_qubits, backend=self.backend), parameters

    def _evaluate_deterministic(self, hamiltonian_matrix: torch.Tensor) -> tuple[torch.Tensor, float]:
        indices = self._deterministic_indices()
        with torch.no_grad():
            loss = self._loss(indices, hamiltonian_matrix)
        return indices, self._float_value(loss)

    def train(self, hamiltonian: HamiltonianInput) -> DQASResult:
        matrix_np, inferred_qubits = _hamiltonian_to_numpy(hamiltonian, self.backend)
        if inferred_qubits != self.n_qubits:
            raise ValueError(f"hamiltonian has {inferred_qubits} qubits, but config.n_qubits={self.n_qubits}")
        hamiltonian_matrix = self.backend.cast(matrix_np)

        theta_optimizer = torch.optim.Adam([self.theta], lr=float(self.config.theta_learning_rate))
        alpha_optimizer = torch.optim.Adam([self.alpha], lr=float(self.config.architecture_learning_rate))
        search_log: list[dict[str, Any]] = []

        best_indices, best_loss = self._evaluate_deterministic(hamiltonian_matrix)
        best_theta = self.theta.detach().clone()
        baseline: float | None = None

        for epoch in range(int(self.config.search_epochs)):
            theta_loss_float = best_loss
            for _ in range(int(self.config.theta_steps)):
                samples, _ = self.sample_architectures()
                theta_optimizer.zero_grad(set_to_none=True)
                theta_losses = self._batch_losses(samples, hamiltonian_matrix)
                theta_loss = theta_losses.mean()
                theta_loss_float = self._float_value(theta_loss)
                if theta_loss.requires_grad:
                    theta_loss.backward()
                    theta_optimizer.step()

            samples, _ = self.sample_architectures()
            with torch.no_grad():
                alpha_losses = self._batch_losses(samples, hamiltonian_matrix, detach_theta=True)
            batch_mean = self._float_value(alpha_losses.mean())
            if baseline is None:
                baseline = batch_mean
            else:
                momentum = float(self.config.baseline_momentum)
                baseline = momentum * baseline + (1.0 - momentum) * batch_mean

            alpha_optimizer.zero_grad(set_to_none=True)
            self.alpha.grad = self._score_function_alpha_gradient_tensor(samples, alpha_losses, baseline=baseline)
            alpha_optimizer.step()

            current_indices, current_loss = self._evaluate_deterministic(hamiltonian_matrix)
            if current_loss < best_loss:
                best_loss = current_loss
                best_indices = current_indices.detach().clone()
                best_theta = self.theta.detach().clone()

            search_log.append(
                {
                    "epoch": epoch,
                    "theta_loss": theta_loss_float,
                    "architecture_loss": batch_mean,
                    "baseline": baseline,
                    "deterministic_loss": current_loss,
                    "architecture_indices": self._architecture_indices_tuple(current_indices),
                }
            )
            if self.config.log_interval and (epoch + 1) % int(self.config.log_interval) == 0:
                print(f"[dqas] epoch={epoch + 1} loss={current_loss:.6f}")

        with torch.no_grad():
            self.theta.copy_(best_theta)

        finetune_log = self._finetune(best_indices, hamiltonian_matrix)
        final_indices, final_loss = self._evaluate_deterministic(hamiltonian_matrix)
        if final_loss < best_loss:
            best_loss = final_loss
            best_indices = final_indices

        circuit, parameters = self._build_circuit(best_indices)
        return DQASResult(
            circuit=circuit,
            parameters=parameters,
            minimum_energy=float(best_loss),
            best_loss=float(best_loss),
            architecture_indices=self._architecture_indices_tuple(best_indices),
            architecture_labels=self._architecture_labels_tuple(best_indices),
            architecture_probabilities=self.architecture_probabilities(),
            search_log=search_log,
            finetune_log=finetune_log,
            config=self.config,
        )

    def _finetune(self, indices: torch.Tensor, hamiltonian_matrix: torch.Tensor) -> list[dict[str, Any]]:
        steps = int(self.config.finetune_steps)
        if steps == 0:
            return []

        optimizer = torch.optim.Adam([self.theta], lr=float(self.config.finetune_learning_rate))
        log: list[dict[str, Any]] = []
        best_theta = self.theta.detach().clone()
        with torch.no_grad():
            best_loss = self._float_value(self._loss(indices, hamiltonian_matrix))

        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = self._loss(indices, hamiltonian_matrix)
            loss_before_step = self._float_value(loss)
            grad_norm = 0.0
            if loss.requires_grad:
                loss.backward()
                if self.theta.grad is not None:
                    grad_norm = float(torch.linalg.vector_norm(self.theta.grad.detach()).cpu())
                optimizer.step()
            with torch.no_grad():
                loss_after_step = self._float_value(self._loss(indices, hamiltonian_matrix))
            if loss_after_step < best_loss:
                best_loss = loss_after_step
                best_theta = self.theta.detach().clone()
            log.append(
                {
                    "step": step,
                    "loss": loss_after_step,
                    "loss_before_step": loss_before_step,
                    "gradient_norm": grad_norm,
                }
            )

        with torch.no_grad():
            self.theta.copy_(best_theta)
        return log


def train_dqas(hamiltonian: HamiltonianInput, config: DQASConfig | None = None) -> DQASResult:
    cfg = DQASConfig() if config is None else config
    probe_backend = _make_backend(cfg.device)
    _, inferred_qubits = _hamiltonian_to_numpy(hamiltonian, probe_backend)
    if cfg.n_qubits is None:
        cfg = replace(cfg, n_qubits=inferred_qubits)
    elif int(cfg.n_qubits) != inferred_qubits:
        raise ValueError(f"hamiltonian has {inferred_qubits} qubits, but config.n_qubits={cfg.n_qubits}")
    return DifferentiableQAS(cfg).train(hamiltonian)


def dqas(hamiltonian: HamiltonianInput, config: DQASConfig | None = None) -> DQASResult:
    return train_dqas(hamiltonian=hamiltonian, config=config)


__all__ = [
    "DQASConfig",
    "DQASResult",
    "DifferentiableQAS",
    "dqas",
    "train_dqas",
]
