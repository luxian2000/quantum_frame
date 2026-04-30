"""PPR-DQL state-synthesis implementation built on nexq primitives.

- Paper: "Quantum Architecture Search via Continual Reinforcement Learning"
- Reference: arXiv:2112.05779v1

This module turns the previous algorithm notes into executable code:
- environment state is represented with nexq State and Circuit objects
- observations are per-qubit X/Y/Z expectation values
- actions append nexq-supported gates to the current circuit
- learning uses DQN with experience replay and a target network
- policy reuse follows the PPR idea by mixing archived policies with the
  current learner during selected episodes
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ...core.circuit import Circuit
from ...core.gates import gate_to_matrix
from ...core.state import State


def _normalize_state_vector(vector: np.ndarray) -> np.ndarray:
    flat = np.asarray(vector, dtype=np.complex64).reshape(-1)
    norm = float(np.linalg.norm(flat))
    if norm <= 0.0:
        raise ValueError("目标态范数必须大于 0")
    return flat / norm


def _default_action_gates(n_qubits: int) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    for qubit_index in range(n_qubits):
        next_qubit = (qubit_index + 1) % n_qubits
        actions.append({"type": "rz", "target_qubit": qubit_index, "parameter": np.pi / 4.0})
        actions.append({"type": "pauli_x", "target_qubit": qubit_index})
        actions.append({"type": "pauli_y", "target_qubit": qubit_index})
        actions.append({"type": "pauli_z", "target_qubit": qubit_index})
        actions.append({"type": "hadamard", "target_qubit": qubit_index})
        if n_qubits > 1:
            actions.append(
                {
                    "type": "cx",
                    "target_qubit": next_qubit,
                    "control_qubits": [qubit_index],
                    "control_states": [1],
                }
            )
    return actions


def _default_observables(n_qubits: int) -> List[Dict[str, Any]]:
    observables: List[Dict[str, Any]] = []
    for qubit_index in range(n_qubits):
        observables.append({"type": "pauli_x", "target_qubit": qubit_index})
        observables.append({"type": "pauli_y", "target_qubit": qubit_index})
        observables.append({"type": "pauli_z", "target_qubit": qubit_index})
    return observables


def _fidelity(current: np.ndarray, target: np.ndarray) -> float:
    inner = np.vdot(current, target)
    return float(np.real(np.conj(inner) * inner))


def _validate_action_gates(action_gates: Sequence[Dict[str, Any]], n_qubits: int) -> List[Dict[str, Any]]:
    checked: List[Dict[str, Any]] = []
    for gate in action_gates:
        if gate.get("type") == "unitary":
            raise ValueError("action_gates 不能包含 unitary 门")
        gate_to_matrix(gate, cir_qubits=n_qubits, backend=None)
        checked.append(dict(gate))
    if not checked:
        raise ValueError("action_gates 不能为空")
    return checked


class _QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


@dataclass
class PPRDQLConfig:
    episode_num: int = 200
    max_steps_per_episode: int = 20
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 32
    replay_capacity: int = 10_000
    warmup_transitions: int = 64
    hidden_dim: int = 128
    target_update_interval: int = 10
    fidelity_threshold: float = 0.95
    gate_penalty: float = 0.01
    terminal_bonus: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.985
    policy_reuse_probability: float = 1.0
    policy_reuse_decay: float = 0.95
    temperature_init: float = 0.0
    temperature_step: float = 0.01
    action_gates: Optional[List[Dict[str, Any]]] = None
    seed: int = 42
    log_interval: int = 0


@dataclass
class PPRDQLPolicy:
    q_network: _QNetwork
    action_gates: List[Dict[str, Any]]
    n_qubits: int
    name: str = "policy"

    def clone(self, name: Optional[str] = None) -> "PPRDQLPolicy":
        copied = deepcopy(self.q_network)
        copied.eval()
        return PPRDQLPolicy(
            q_network=copied,
            action_gates=[dict(gate) for gate in self.action_gates],
            n_qubits=self.n_qubits,
            name=self.name if name is None else name,
        )


@dataclass
class PPRDQLResult:
    circuit: Circuit
    policy: PPRDQLPolicy
    best_fidelity: float
    episode_rewards: List[float]
    selected_policy_indices: List[int]


class _ReplayBuffer:
    def __init__(self, capacity: int):
        self._buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                1.0 if done else 0.0,
            )
        )

    def sample(self, batch_size: int, rng: random.Random):
        batch = rng.sample(list(self._buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.asarray(states, dtype=np.float32), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.asarray(next_states, dtype=np.float32), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)


class _PPRDQLEnv:
    def __init__(self, target_state: State, config: PPRDQLConfig):
        self.backend = target_state.backend
        self.n_qubits = target_state.n_qubits
        self.target = _normalize_state_vector(target_state.to_numpy())
        self.fidelity_threshold = float(config.fidelity_threshold)
        self.max_steps = int(config.max_steps_per_episode)
        self.gate_penalty = float(config.gate_penalty)
        self.terminal_bonus = float(config.terminal_bonus)
        self.action_gates = _validate_action_gates(
            _default_action_gates(self.n_qubits) if config.action_gates is None else config.action_gates,
            self.n_qubits,
        )
        self.observables = _default_observables(self.n_qubits)
        self.observable_matrices = [
            gate_to_matrix(observable, cir_qubits=self.n_qubits, backend=self.backend)
            for observable in self.observables
        ]
        self.circuit_gates: List[Dict[str, Any]] = []
        self.steps = 0
        self.prev_fidelity = 0.0

    @property
    def state_dim(self) -> int:
        return len(self.observables)

    @property
    def action_dim(self) -> int:
        return len(self.action_gates)

    def reset(self) -> np.ndarray:
        self.circuit_gates = []
        self.steps = 0
        self.prev_fidelity = self._fidelity_of_current_state()
        return self._observation()

    def _build_state(self) -> State:
        state = State.zero_state(self.n_qubits, backend=self.backend)
        for gate in self.circuit_gates:
            matrix = gate_to_matrix(gate, cir_qubits=self.n_qubits, backend=self.backend)
            state = state.evolve(matrix)
        return state

    def _observation(self) -> np.ndarray:
        state = self._build_state()
        values = [state.expectation(matrix) for matrix in self.observable_matrices]
        return np.asarray(values, dtype=np.float32)

    def _fidelity_of_current_state(self) -> float:
        current = _normalize_state_vector(self._build_state().to_numpy())
        return _fidelity(current, self.target)

    def build_circuit(self) -> Circuit:
        return Circuit(*self.circuit_gates, n_qubits=self.n_qubits, backend=self.backend)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if action < 0 or action >= self.action_dim:
            raise ValueError(f"非法 action 索引: {action}")

        self.circuit_gates.append(dict(self.action_gates[action]))
        self.steps += 1

        fidelity = self._fidelity_of_current_state()
        reward = fidelity - self.prev_fidelity - self.gate_penalty
        if fidelity >= self.fidelity_threshold:
            reward += self.terminal_bonus
        self.prev_fidelity = fidelity

        done = fidelity >= self.fidelity_threshold or self.steps >= self.max_steps
        info = {
            "fidelity": fidelity,
            "gate_count": len(self.circuit_gates),
            "circuit": self.build_circuit(),
        }
        return self._observation(), float(reward), bool(done), info


def _greedy_action(network: _QNetwork, state: np.ndarray) -> int:
    with torch.no_grad():
        q_values = network(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
    return int(torch.argmax(q_values, dim=1).item())


def _epsilon_greedy_action(
    network: _QNetwork,
    state: np.ndarray,
    action_dim: int,
    epsilon: float,
    rng: random.Random,
) -> int:
    if rng.random() < epsilon:
        return rng.randrange(action_dim)
    return _greedy_action(network, state)


def _softmax_probabilities(values: Sequence[float], temperature: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64)
    if scaled.size == 0:
        raise ValueError("softmax 输入不能为空")
    if abs(temperature) > 0.0:
        scaled = scaled * temperature
    scaled = scaled - np.max(scaled)
    weights = np.exp(scaled)
    total = np.sum(weights)
    if total <= 0.0 or not np.isfinite(total):
        return np.ones_like(weights) / float(len(weights))
    return weights / total


def _optimize_dqn_step(
    online_network: _QNetwork,
    target_network: _QNetwork,
    optimizer: torch.optim.Optimizer,
    replay_buffer: _ReplayBuffer,
    batch_size: int,
    gamma: float,
    rng: random.Random,
) -> Optional[float]:
    if len(replay_buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, rng)
    q_values = online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_values = target_network(next_states).max(dim=1).values
        targets = rewards + (1.0 - dones) * gamma * next_q_values
    loss = nn.functional.mse_loss(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def _rollout_greedy(env: _PPRDQLEnv, network: _QNetwork) -> Tuple[Circuit, float]:
    if env.prev_fidelity >= env.fidelity_threshold and len(env.circuit_gates) == 0:
        return env.build_circuit(), env.prev_fidelity

    state = env.reset()
    done = False
    last_info: Dict[str, Any] = {"fidelity": env.prev_fidelity, "circuit": env.build_circuit()}
    while not done:
        action = _greedy_action(network, state)
        state, _, done, last_info = env.step(action)
    return last_info["circuit"], float(last_info["fidelity"])


def train_ppr_dql(
    target_state: State,
    config: Optional[PPRDQLConfig] = None,
    policy_library: Optional[Sequence[PPRDQLPolicy]] = None,
) -> PPRDQLResult:
    cfg = PPRDQLConfig() if config is None else config

    if cfg.episode_num <= 0:
        raise ValueError("episode_num 必须是正整数")
    if cfg.max_steps_per_episode <= 0:
        raise ValueError("max_steps_per_episode 必须是正整数")
    if not 0.0 < cfg.fidelity_threshold <= 1.0:
        raise ValueError("fidelity_threshold 必须在 (0, 1] 区间")
    if cfg.batch_size <= 0 or cfg.replay_capacity <= 0:
        raise ValueError("batch_size 和 replay_capacity 必须是正整数")

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    env = _PPRDQLEnv(target_state=target_state, config=cfg)

    current_policy = _QNetwork(env.state_dim, env.action_dim, cfg.hidden_dim)
    target_policy = _QNetwork(env.state_dim, env.action_dim, cfg.hidden_dim)
    target_policy.load_state_dict(current_policy.state_dict())
    target_policy.eval()

    optimizer = torch.optim.Adam(current_policy.parameters(), lr=cfg.learning_rate)
    replay_buffer = _ReplayBuffer(cfg.replay_capacity)

    archived_policies: List[PPRDQLPolicy] = []
    for index, old_policy in enumerate(policy_library or []):
        if old_policy.n_qubits != env.n_qubits:
            raise ValueError("policy_library 中策略的 n_qubits 与目标态不一致")
        if len(old_policy.action_gates) != env.action_dim:
            raise ValueError("policy_library 中策略的动作空间与当前配置不一致")
        archived_policies.append(old_policy.clone(name=f"reused_{index}"))

    average_returns = [0.0 for _ in range(len(archived_policies) + 1)]
    selection_counts = [0 for _ in range(len(archived_policies) + 1)]
    temperature = float(cfg.temperature_init)
    epsilon = float(cfg.epsilon_start)

    best_circuit = Circuit(n_qubits=env.n_qubits, backend=env.backend)
    best_fidelity = env.reset() is not None and env.prev_fidelity or 0.0
    episode_rewards: List[float] = []
    selected_policy_indices: List[int] = []

    for episode_index in range(cfg.episode_num):
        probabilities = _softmax_probabilities(average_returns, temperature)
        selected_index = int(np.random.choice(len(probabilities), p=probabilities))
        reused_policy = archived_policies[selected_index - 1] if selected_index > 0 else None
        selected_policy_indices.append(selected_index)

        state = env.reset()
        done = False
        total_reward = 0.0
        reuse_probability = float(cfg.policy_reuse_probability)
        last_info: Dict[str, Any] = {"fidelity": env.prev_fidelity, "circuit": env.build_circuit()}

        while not done:
            use_archived = reused_policy is not None and rng.random() < reuse_probability
            if use_archived:
                action = _greedy_action(reused_policy.q_network, state)
            else:
                action = _epsilon_greedy_action(current_policy, state, env.action_dim, epsilon, rng)

            next_state, reward, done, last_info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

            if len(replay_buffer) >= max(cfg.batch_size, cfg.warmup_transitions):
                _optimize_dqn_step(
                    online_network=current_policy,
                    target_network=target_policy,
                    optimizer=optimizer,
                    replay_buffer=replay_buffer,
                    batch_size=cfg.batch_size,
                    gamma=cfg.gamma,
                    rng=rng,
                )

            if reused_policy is not None:
                reuse_probability *= cfg.policy_reuse_decay

        selection_counts[selected_index] += 1
        count = selection_counts[selected_index]
        average_returns[selected_index] += (total_reward - average_returns[selected_index]) / float(count)

        if float(last_info["fidelity"]) > best_fidelity:
            best_fidelity = float(last_info["fidelity"])
            best_circuit = last_info["circuit"]

        if (episode_index + 1) % cfg.target_update_interval == 0:
            target_policy.load_state_dict(current_policy.state_dict())

        episode_rewards.append(total_reward)
        temperature += cfg.temperature_step
        epsilon = max(cfg.epsilon_end, epsilon * cfg.epsilon_decay)

        if cfg.log_interval > 0 and (episode_index + 1) % cfg.log_interval == 0:
            print(
                f"[PPR-DQL] episode={episode_index + 1} reward={total_reward:.4f} "
                f"fidelity={float(last_info['fidelity']):.4f}"
            )

    greedy_circuit, greedy_fidelity = _rollout_greedy(env, current_policy)
    if greedy_fidelity >= best_fidelity:
        best_circuit = greedy_circuit
        best_fidelity = greedy_fidelity

    learned_policy = PPRDQLPolicy(
        q_network=deepcopy(current_policy).eval(),
        action_gates=[dict(gate) for gate in env.action_gates],
        n_qubits=env.n_qubits,
        name="ppr_dql",
    )
    return PPRDQLResult(
        circuit=best_circuit,
        policy=learned_policy,
        best_fidelity=best_fidelity,
        episode_rewards=episode_rewards,
        selected_policy_indices=selected_policy_indices,
    )


def ppr_dql_state_to_circuit(
    target_state: State,
    config: Optional[PPRDQLConfig] = None,
    policy_library: Optional[Sequence[PPRDQLPolicy]] = None,
) -> Circuit:
    return train_ppr_dql(target_state=target_state, config=config, policy_library=policy_library).circuit


__all__ = [
    "PPRDQLConfig",
    "PPRDQLPolicy",
    "PPRDQLResult",
    "ppr_dql_state_to_circuit",
    "train_ppr_dql",
]