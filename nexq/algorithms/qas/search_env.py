"""Search environments for noise-adaptive QAS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...channel.backends.base import Backend
from ...channel.noise.model import NoiseModel
from ...core.circuit import Circuit
from ...core.gates import gate_to_matrix
from ...core.state import State
from ._utils import ensure_backend
from .evaluator import ArchitectureEvaluator
from ._types import ArchitectureSpec


@dataclass
class QASState:
    circuit: Circuit
    observation: np.ndarray
    step_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class NoisyQASEnv:
    """Minimal noisy QAS environment that scores appended-gate circuits through an evaluator."""

    def __init__(
        self,
        n_qubits: int,
        action_gates: Sequence[Dict[str, Any]],
        backend: Optional[Backend] = None,
        noise_model: Optional[NoiseModel] = None,
        evaluator: Optional[ArchitectureEvaluator] = None,
        max_steps: int = 20,
        gate_penalty: float = 0.01,
    ):
        self.n_qubits = int(n_qubits)
        self.backend = ensure_backend(backend)
        self.noise_model = noise_model
        self.max_steps = int(max_steps)
        self.gate_penalty = float(gate_penalty)
        self.evaluator = evaluator or ArchitectureEvaluator(backend=self.backend, noise_model=noise_model)
        self.action_gates = self._validate_actions(action_gates)
        self.circuit_gates: List[Dict[str, Any]] = []
        self.step_index = 0
        self.previous_score = 0.0

    def _validate_actions(self, action_gates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        checked: List[Dict[str, Any]] = []
        full_dim = 1 << self.n_qubits
        for gate in action_gates:
            matrix = gate_to_matrix(gate, cir_qubits=self.n_qubits, backend=self.backend)
            if np.shape(self.backend.to_numpy(matrix)) == (full_dim, full_dim):
                checked.append(dict(gate))
        if not checked:
            raise ValueError("action_gates must contain at least one valid gate")
        return checked

    def _build_circuit(self) -> Circuit:
        return Circuit(*self.circuit_gates, n_qubits=self.n_qubits, backend=self.backend)

    def _observation(self) -> np.ndarray:
        state = State.zero_state(self.n_qubits, backend=self.backend)
        for gate in self.circuit_gates:
            state = state.evolve(gate_to_matrix(gate, cir_qubits=self.n_qubits, backend=self.backend))
        probabilities = np.abs(state.to_numpy().reshape(-1)) ** 2
        return probabilities.astype(np.float32)

    def reset(self) -> QASState:
        self.circuit_gates = []
        self.step_index = 0
        self.previous_score = 0.0
        return QASState(circuit=self._build_circuit(), observation=self._observation(), step_index=0)

    def step(self, action: int) -> Tuple[QASState, float, bool, Dict[str, Any]]:
        if action < 0 or action >= len(self.action_gates):
            raise ValueError(f"Invalid action index: {action}")
        self.circuit_gates.append(dict(self.action_gates[action]))
        self.step_index += 1
        circuit = self._build_circuit()
        architecture = ArchitectureSpec(name=f"env_step_{self.step_index}", circuit=circuit)
        score = self.evaluator.evaluate(architecture)
        reward = score.weighted_score - self.previous_score - self.gate_penalty
        self.previous_score = score.weighted_score
        done = self.step_index >= self.max_steps
        state = QASState(
            circuit=circuit,
            observation=self._observation(),
            step_index=self.step_index,
            metadata={"weighted_score": score.weighted_score},
        )
        return state, float(reward), bool(done), {"architecture_score": score}


__all__ = ["QASState", "NoisyQASEnv"]
