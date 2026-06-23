"""QuantumDARTS-style differentiable QAS for variational algorithms.

本模块实现 Wu et al. (ICML 2023) 的 QuantumDARTS 宏观搜索流程：
使用 Gumbel-Softmax 在每个 qubit-layer 位置采样一个真实量子门，
同时用直通估计器更新架构权重，并交替优化旋转角参数。

输入 (Inputs)
- ``hamiltonian``: VQE 目标哈密顿量，可为矩阵、``Hamiltonian`` 或 Pauli 项列表。
- ``QDRATSConfig``: 搜索层数、温度、学习率、训练轮数和随机种子。

输出 (Outputs)
- ``QDRATSResult``: 离散化后的 ``aicir.Circuit``、最小能量、架构标签、
  架构概率、搜索日志和微调日志。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from aicir import Circuit, GPUBackend, Hamiltonian, NPUBackend, cx, ry, rz
from ...core.gates import gate_to_matrix
from ..core.config import QDRATSConfig
from ..problems.hamiltonians import hamiltonian_matrix as pauli_hamiltonian_matrix


HamiltonianInput = np.ndarray | torch.Tensor | Hamiltonian | Sequence[tuple[float, str]]


@dataclass
class QDRATSResult:
    circuit: Circuit
    parameters: dict[str, float]
    minimum_energy: float
    best_loss: float
    architecture_indices: tuple[tuple[int, ...], ...]
    architecture_labels: tuple[tuple[str, ...], ...]
    architecture_probabilities: np.ndarray
    search_log: list[dict[str, Any]]
    finetune_log: list[dict[str, Any]]
    config: QDRATSConfig


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


class QuantumDARTS:
    """Macro-search engine implementing the QuantumDARTS training loop."""

    def __init__(self, config: QDRATSConfig | None = None):
        self.config = QDRATSConfig() if config is None else config
        self._validate_config()

        torch.manual_seed(int(self.config.seed))
        np.random.seed(int(self.config.seed))

        self.backend = _make_backend(self.config.device)
        self.device = self.backend._device
        self.n_qubits = int(self.config.n_qubits)
        self.layers = int(self.config.layers)
        self.max_candidates = self.n_qubits + 1 if self.n_qubits > 1 else 2

        self._candidate_labels = tuple(
            self._build_candidate_labels_for_target(target)
            for target in range(self.n_qubits)
        )

        init_scale = 1.0e-2
        self.architecture_left = torch.nn.Parameter(
            init_scale
            * torch.randn(
                self.layers,
                self.n_qubits,
                int(self.config.hidden_dim),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.architecture_right = torch.nn.Parameter(
            init_scale
            * torch.randn(
                self.layers,
                self.n_qubits,
                int(self.config.hidden_dim),
                self.max_candidates,
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.theta = torch.nn.Parameter(
            torch.empty(
                self.layers,
                self.n_qubits,
                3,
                dtype=torch.float32,
                device=self.device,
            ).uniform_(-0.05, 0.05)
        )

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.n_qubits is None:
            raise ValueError("QDRATSConfig.n_qubits must be set before constructing QuantumDARTS")
        if int(cfg.n_qubits) <= 0:
            raise ValueError("n_qubits must be positive")
        if int(cfg.layers) <= 0:
            raise ValueError("layers must be positive")
        if int(cfg.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(cfg.search_epochs) < 0:
            raise ValueError("search_epochs must be non-negative")
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
        if cfg.temperature <= 0 or cfg.temperature_min <= 0:
            raise ValueError("temperature values must be positive")
        if cfg.temperature_decay <= 0:
            raise ValueError("temperature_decay must be positive")

    def _build_candidate_labels_for_target(self, target: int) -> tuple[str, ...]:
        labels = ["rzryrz", "identity"]
        labels.extend(f"cx_{control}_to_{target}" for control in range(self.n_qubits) if control != target)
        return tuple(labels)

    def candidate_labels_for_target(self, target: int) -> tuple[str, ...]:
        target = int(target)
        if target < 0 or target >= self.n_qubits:
            raise ValueError("target qubit is outside [0, n_qubits)")
        return self._candidate_labels[target]

    def _architecture_logits(self) -> torch.Tensor:
        return torch.einsum("lnh,lnhk->lnk", self.architecture_left, self.architecture_right)

    def _architecture_probabilities_tensor(self) -> torch.Tensor:
        return torch.softmax(self._architecture_logits(), dim=-1)

    def _sample_selection(self, temperature: float, *, hard: bool) -> torch.Tensor:
        logits = self._architecture_logits()
        flat = logits.reshape(-1, self.max_candidates)
        if self.config.use_gumbel_noise:
            sampled = F.gumbel_softmax(flat, tau=float(temperature), hard=hard, dim=-1)
        else:
            probs = torch.softmax(flat / float(temperature), dim=-1)
            if hard:
                indices = probs.argmax(dim=-1)
                one_hot = F.one_hot(indices, num_classes=self.max_candidates).to(dtype=probs.dtype)
                sampled = one_hot + probs - probs.detach()
            else:
                sampled = probs
        return sampled.reshape(self.layers, self.n_qubits, self.max_candidates)

    def _deterministic_indices(self) -> torch.Tensor:
        return self._architecture_logits().argmax(dim=-1)

    def _selection_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return F.one_hot(indices, num_classes=self.max_candidates).to(
            dtype=torch.float32,
            device=self.device,
        )

    def _identity_matrix(self) -> torch.Tensor:
        return self.backend.eye(1 << self.n_qubits)

    def _rzryrz_matrix(self, layer: int, target: int, *, detach_theta: bool) -> torch.Tensor:
        angles = self.theta[layer, target]
        if detach_theta:
            angles = angles.detach()
        rz0 = gate_to_matrix(rz(angles[0], target_qubit=target), cir_qubits=self.n_qubits, backend=self.backend)
        ry1 = gate_to_matrix(ry(angles[1], target_qubit=target), cir_qubits=self.n_qubits, backend=self.backend)
        rz2 = gate_to_matrix(rz(angles[2], target_qubit=target), cir_qubits=self.n_qubits, backend=self.backend)
        return self.backend.matmul(rz2, self.backend.matmul(ry1, rz0))

    def _candidate_matrix(self, layer: int, target: int, candidate_index: int, *, detach_theta: bool) -> torch.Tensor:
        if candidate_index == 0:
            return self._rzryrz_matrix(layer, target, detach_theta=detach_theta)
        if candidate_index == 1:
            return self._identity_matrix()
        controls = [q for q in range(self.n_qubits) if q != target]
        control = controls[candidate_index - 2]
        gate = cx(target_qubit=target, control_qubits=[control])
        return gate_to_matrix(gate, cir_qubits=self.n_qubits, backend=self.backend)

    def _mixed_gate_matrix(
        self,
        layer: int,
        target: int,
        weights: torch.Tensor,
        *,
        detach_theta: bool,
    ) -> torch.Tensor:
        matrices = [
            self._candidate_matrix(layer, target, index, detach_theta=detach_theta)
            for index in range(self.max_candidates)
        ]
        stacked = torch.stack(matrices, dim=0)
        complex_weights = weights.to(dtype=stacked.dtype).reshape(-1, 1, 1)
        return torch.sum(complex_weights * stacked, dim=0)

    def _initial_state(self) -> torch.Tensor:
        if self.config.initial_state is None:
            return self.backend.zeros_state(self.n_qubits)
        state = self.backend.cast(self.config.initial_state)
        return state.reshape(-1)

    def _simulate_selection(self, selection: torch.Tensor, *, detach_theta: bool) -> torch.Tensor:
        state = self._initial_state()
        for layer in range(self.layers):
            for target in range(self.n_qubits):
                matrix = self._mixed_gate_matrix(
                    layer,
                    target,
                    selection[layer, target],
                    detach_theta=detach_theta,
                )
                state = self.backend.apply_unitary(state, matrix)
        return state.reshape(-1)

    def _loss(
        self,
        selection: torch.Tensor,
        hamiltonian_matrix: torch.Tensor,
        *,
        detach_theta: bool = False,
    ) -> torch.Tensor:
        state = self._simulate_selection(selection, detach_theta=detach_theta)
        value = self.backend.expectation_sv(state, hamiltonian_matrix)
        if isinstance(value, torch.Tensor):
            return torch.real(value).reshape(())
        return torch.tensor(float(value), dtype=torch.float32, device=self.device)

    @staticmethod
    def _float_value(value: torch.Tensor | float) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().reshape(()))
        return float(value)

    def _architecture_indices_tuple(self, indices: torch.Tensor) -> tuple[tuple[int, ...], ...]:
        data = indices.detach().cpu().numpy().astype(int)
        return tuple(tuple(int(value) for value in row) for row in data)

    def _architecture_labels_tuple(self, indices: torch.Tensor) -> tuple[tuple[str, ...], ...]:
        idx = indices.detach().cpu().numpy().astype(int)
        labels: list[tuple[str, ...]] = []
        for layer in range(self.layers):
            layer_labels = []
            for target in range(self.n_qubits):
                layer_labels.append(self._candidate_labels[target][int(idx[layer, target])])
            labels.append(tuple(layer_labels))
        return tuple(labels)

    def _build_circuit(self, indices: torch.Tensor) -> tuple[Circuit, dict[str, float]]:
        idx = indices.detach().cpu().numpy().astype(int)
        theta_np = self.theta.detach().cpu().numpy()
        gates: list[dict[str, Any]] = []
        parameters: dict[str, float] = {}

        for layer in range(self.layers):
            for target in range(self.n_qubits):
                choice = int(idx[layer, target])
                if choice == 0:
                    values = theta_np[layer, target]
                    parameters[f"theta_{layer}_{target}_rz0"] = float(values[0])
                    parameters[f"theta_{layer}_{target}_ry"] = float(values[1])
                    parameters[f"theta_{layer}_{target}_rz2"] = float(values[2])
                    gates.append(rz(float(values[0]), target_qubit=target))
                    gates.append(ry(float(values[1]), target_qubit=target))
                    gates.append(rz(float(values[2]), target_qubit=target))
                elif choice == 1:
                    continue
                else:
                    controls = [q for q in range(self.n_qubits) if q != target]
                    control = controls[choice - 2]
                    gates.append(cx(target_qubit=target, control_qubits=[control]))

        return Circuit(*gates, n_qubits=self.n_qubits, backend=self.backend), parameters

    def _evaluate_deterministic(self, hamiltonian_matrix: torch.Tensor) -> tuple[torch.Tensor, float]:
        indices = self._deterministic_indices()
        selection = self._selection_from_indices(indices)
        with torch.no_grad():
            loss = self._loss(selection, hamiltonian_matrix)
        return indices, self._float_value(loss)

    def train(self, hamiltonian: HamiltonianInput) -> QDRATSResult:
        matrix_np, inferred_qubits = _hamiltonian_to_numpy(hamiltonian, self.backend)
        if inferred_qubits != self.n_qubits:
            raise ValueError(
                f"hamiltonian has {inferred_qubits} qubits, but config.n_qubits={self.n_qubits}"
            )
        hamiltonian_matrix = self.backend.cast(matrix_np)

        arch_optimizer = torch.optim.Adam(
            [self.architecture_left, self.architecture_right],
            lr=float(self.config.architecture_learning_rate),
        )
        theta_optimizer = torch.optim.Adam([self.theta], lr=float(self.config.theta_learning_rate))

        search_log: list[dict[str, Any]] = []
        temperature = float(self.config.temperature)
        best_indices, best_loss = self._evaluate_deterministic(hamiltonian_matrix)
        best_theta = self.theta.detach().clone()

        for epoch in range(int(self.config.search_epochs)):
            theta_loss_float = best_loss
            theta_selection = self._sample_selection(temperature, hard=True).detach()
            for _ in range(int(self.config.theta_steps)):
                theta_optimizer.zero_grad(set_to_none=True)
                theta_loss = self._loss(theta_selection, hamiltonian_matrix)
                theta_loss_float = self._float_value(theta_loss)
                if theta_loss.requires_grad:
                    theta_loss.backward()
                    theta_optimizer.step()

            arch_optimizer.zero_grad(set_to_none=True)
            arch_selection = self._sample_selection(temperature, hard=bool(self.config.hard_sampling))
            arch_loss = self._loss(arch_selection, hamiltonian_matrix, detach_theta=True)
            architecture_loss_float = self._float_value(arch_loss)
            if arch_loss.requires_grad:
                arch_loss.backward()
                arch_optimizer.step()

            current_indices, current_loss = self._evaluate_deterministic(hamiltonian_matrix)
            if current_loss < best_loss:
                best_loss = current_loss
                best_indices = current_indices.detach().clone()
                best_theta = self.theta.detach().clone()

            search_log.append(
                {
                    "epoch": epoch,
                    "temperature": temperature,
                    "theta_loss": theta_loss_float,
                    "architecture_loss": architecture_loss_float,
                    "deterministic_loss": current_loss,
                    "architecture_indices": self._architecture_indices_tuple(current_indices),
                }
            )

            if self.config.log_interval and (epoch + 1) % int(self.config.log_interval) == 0:
                print(
                    f"[qdrats] epoch={epoch + 1} "
                    f"loss={current_loss:.6f} temperature={temperature:.4f}"
                )
            temperature = max(
                float(self.config.temperature_min),
                temperature * float(self.config.temperature_decay),
            )

        with torch.no_grad():
            self.theta.copy_(best_theta)

        finetune_log = self._finetune(best_indices, hamiltonian_matrix)
        final_indices, final_loss = self._evaluate_deterministic(hamiltonian_matrix)
        if final_loss < best_loss:
            best_loss = final_loss
            best_indices = final_indices

        circuit, parameters = self._build_circuit(best_indices)
        probabilities = self._architecture_probabilities_tensor().detach().cpu().numpy()

        return QDRATSResult(
            circuit=circuit,
            parameters=parameters,
            minimum_energy=float(best_loss),
            best_loss=float(best_loss),
            architecture_indices=self._architecture_indices_tuple(best_indices),
            architecture_labels=self._architecture_labels_tuple(best_indices),
            architecture_probabilities=probabilities,
            search_log=search_log,
            finetune_log=finetune_log,
            config=self.config,
        )

    def _finetune(self, indices: torch.Tensor, hamiltonian_matrix: torch.Tensor) -> list[dict[str, Any]]:
        steps = int(self.config.finetune_steps)
        if steps == 0:
            return []

        selection = self._selection_from_indices(indices).detach()
        optimizer = torch.optim.Adam([self.theta], lr=float(self.config.finetune_learning_rate))
        log: list[dict[str, Any]] = []
        best_theta = self.theta.detach().clone()
        with torch.no_grad():
            best_loss = self._float_value(self._loss(selection, hamiltonian_matrix))

        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = self._loss(selection, hamiltonian_matrix)
            loss_before_step = self._float_value(loss)
            grad_norm = 0.0
            if loss.requires_grad:
                loss.backward()
                grad = self.theta.grad
                if grad is not None:
                    grad_norm = float(torch.linalg.vector_norm(grad.detach()).cpu())
                optimizer.step()

            with torch.no_grad():
                loss_after_step = self._float_value(self._loss(selection, hamiltonian_matrix))
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


def train_qdrats(
    hamiltonian: HamiltonianInput,
    config: QDRATSConfig | None = None,
) -> QDRATSResult:
    cfg = QDRATSConfig() if config is None else config
    probe_backend = _make_backend(cfg.device)
    _, inferred_qubits = _hamiltonian_to_numpy(hamiltonian, probe_backend)
    if cfg.n_qubits is None:
        cfg = replace(cfg, n_qubits=inferred_qubits)
    elif int(cfg.n_qubits) != inferred_qubits:
        raise ValueError(
            f"hamiltonian has {inferred_qubits} qubits, but config.n_qubits={cfg.n_qubits}"
        )
    return QuantumDARTS(cfg).train(hamiltonian)


def qdrats(
    hamiltonian: HamiltonianInput,
    config: QDRATSConfig | None = None,
) -> QDRATSResult:
    return train_qdrats(hamiltonian=hamiltonian, config=config)


__all__ = [
    "QDRATSConfig",
    "QDRATSResult",
    "QuantumDARTS",
    "qdrats",
    "train_qdrats",
]
