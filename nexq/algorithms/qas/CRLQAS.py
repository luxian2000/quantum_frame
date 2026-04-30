"""Curriculum-based RL Quantum Architecture Search (CRLQAS).

This implementation follows the core ideas of CRLQAS in a nexq-native form:
- DDQN chooses circuit architecture actions (gate appends)
- Adam-SPSA refines all variational parameters after each append
- curriculum threshold controls success criterion during training
- random-halt sampling changes per-episode horizon
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
import math
import random
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ...channel.backends.numpy_backend import NumpyBackend
from ...channel.operators import Hamiltonian
from ...core.circuit import Circuit
from ...core.gates import gate_to_matrix
from ...core.state import State


@dataclass
class AdamSPSAConfig:
    iterations: int = 30
    a: float = 0.08
    alpha: float = 0.602
    c: float = 0.12
    gamma_sp: float = 0.101
    beta_1: float = 0.9
    beta_2: float = 0.999
    lam: float = 0.0
    epsilon: float = 1e-8


@dataclass
class CRLQASConfig:
    max_episodes: int = 300
    n_act: int = 12

    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.9995

    replay_capacity: int = 20_000
    batch_size: int = 64
    q_hidden_dim: int = 256
    q_learning_rate: float = 1e-3
    train_interval: int = 10
    target_update_interval: int = 200

    success_reward: float = 5.0
    failure_reward: float = -5.0
    reward_floor: float = -1.0

    curriculum_initial_threshold: float = 0.2
    curriculum_mu: float = -2.0
    curriculum_adjust_period: int = 500
    curriculum_delta: float = 0.2
    curriculum_kappa: float = 100.0
    curriculum_reset_patience: int = 40
    chemical_accuracy: float = 1.6e-3

    random_halt_p: float = 0.5

    # 可选：自定义动作门集合。None 时默认启用 core/gates 支持的全门集。
    action_gates: Optional[List[Dict[str, Any]]] = None

    adam_spsa: AdamSPSAConfig = field(default_factory=AdamSPSAConfig)
    seed: int = 42
    log_interval: int = 0


@dataclass
class CRLQASResult:
    circuit: Circuit
    parameters: List[float]
    minimum_energy: float
    curriculum_threshold: float
    episode_best_energies: List[float]
    q_network_state_dict: Dict[str, torch.Tensor]


class _QNet(nn.Module):
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


def _infer_n_qubits_from_hamiltonian_array(matrix: np.ndarray) -> int:
    mat = np.asarray(matrix, dtype=np.complex64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("哈密顿量矩阵必须是方阵")
    dim = int(mat.shape[0])
    n_qubits = int(round(math.log2(dim))) if dim > 0 else 0
    if (1 << n_qubits) != dim:
        raise ValueError("哈密顿量矩阵维度必须是 2 的幂")
    return n_qubits


def _resolve_hamiltonian_matrix(
    hamiltonian: np.ndarray | Hamiltonian,
    backend: NumpyBackend,
) -> Tuple[np.ndarray, int]:
    if isinstance(hamiltonian, Hamiltonian):
        n_qubits = int(hamiltonian.n_qubits)
        matrix = backend.to_numpy(hamiltonian.to_matrix(backend)).astype(np.complex64)
        return matrix, n_qubits
    matrix = np.asarray(hamiltonian, dtype=np.complex64)
    n_qubits = _infer_n_qubits_from_hamiltonian_array(matrix)
    return matrix, n_qubits


def _build_action_space(n_qubits: int) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []

    single_qubit_types = ["pauli_x", "pauli_y", "pauli_z", "hadamard", "s_gate", "t_gate"]
    for q in range(n_qubits):
        for gate_type in single_qubit_types:
            actions.append({"type": gate_type, "target_qubit": q})

        actions.append({"type": "rx", "target_qubit": q, "parameter": np.pi / 4.0})
        actions.append({"type": "ry", "target_qubit": q, "parameter": np.pi / 4.0})
        actions.append({"type": "rz", "target_qubit": q, "parameter": np.pi / 4.0})
        actions.append({"type": "u2", "target_qubit": q, "parameter": [np.pi / 4.0, np.pi / 4.0]})
        actions.append(
            {"type": "u3", "target_qubit": q, "parameter": [np.pi / 3.0, np.pi / 4.0, np.pi / 5.0]}
        )

    for ctrl in range(n_qubits):
        for tgt in range(n_qubits):
            if ctrl == tgt:
                continue
            base = {"target_qubit": tgt, "control_qubits": [ctrl], "control_states": [1]}
            actions.append({"type": "cx", **base})
            actions.append({"type": "cy", **base})
            actions.append({"type": "cz", **base})
            actions.append({"type": "crx", "parameter": np.pi / 4.0, **base})
            actions.append({"type": "cry", "parameter": np.pi / 4.0, **base})
            actions.append({"type": "crz", "parameter": np.pi / 4.0, **base})

    for q1 in range(n_qubits):
        for q2 in range(q1 + 1, n_qubits):
            actions.append({"type": "swap", "qubit_1": q1, "qubit_2": q2})
            actions.append({"type": "rzz", "qubit_1": q1, "qubit_2": q2, "parameter": np.pi / 4.0})

    if n_qubits >= 3:
        for tgt in range(n_qubits):
            ctrls = [q for q in range(n_qubits) if q != tgt]
            for i in range(len(ctrls)):
                for j in range(i + 1, len(ctrls)):
                    actions.append({"type": "toffoli", "target_qubit": tgt, "control_qubits": [ctrls[i], ctrls[j]]})

    actions.append({"type": "identity", "n_qubits": n_qubits})
    return actions


def _validate_action_space(action_gates: Sequence[Dict[str, Any]], n_qubits: int) -> List[Dict[str, Any]]:
    checked: List[Dict[str, Any]] = []
    full_dim = 1 << n_qubits
    backend = NumpyBackend()
    for gate in action_gates:
        candidate = dict(gate)
        try:
            gm_np = gate_to_matrix(candidate, cir_qubits=n_qubits, backend=None)
            gm_bk = gate_to_matrix(candidate, cir_qubits=n_qubits, backend=backend)
            if np.shape(gm_np) != (full_dim, full_dim):
                continue
            if np.shape(backend.to_numpy(gm_bk)) != (full_dim, full_dim):
                continue
            checked.append(candidate)
        except Exception:
            continue
    if not checked:
        raise ValueError("动作空间为空，无法训练")
    return checked


def _is_same_gate_signature(lhs: Dict[str, Any], rhs: Dict[str, Any]) -> bool:
    return (
        lhs.get("type") == rhs.get("type")
        and lhs.get("target_qubit") == rhs.get("target_qubit")
        and lhs.get("control_qubits") == rhs.get("control_qubits")
        and lhs.get("qubit_1") == rhs.get("qubit_1")
        and lhs.get("qubit_2") == rhs.get("qubit_2")
    )


def _involved_qubits(gate: Dict[str, Any], n_qubits: int) -> List[int]:
    qubits: List[int] = []
    if "target_qubit" in gate:
        qubits.append(int(gate["target_qubit"]))
    for c in gate.get("control_qubits", []) or []:
        qubits.append(int(c))
    if "qubit_1" in gate:
        qubits.append(int(gate["qubit_1"]))
    if "qubit_2" in gate:
        qubits.append(int(gate["qubit_2"]))
    if not qubits and gate.get("type") in {"identity", "I"}:
        qubits = list(range(n_qubits))
    return sorted(set([q for q in qubits if 0 <= q < n_qubits]))


def _gate_to_tensor_row(gate: Dict[str, Any], n_qubits: int) -> np.ndarray:
    width = (n_qubits + 3) * n_qubits
    row = np.zeros(width, dtype=np.float32)

    involved = _involved_qubits(gate, n_qubits)
    if not involved:
        return row

    has_parameter = "parameter" in gate
    is_multi = len(involved) > 1
    for q in involved:
        block_start = q * (n_qubits + 3)
        row[block_start + 0] = 1.0
        row[block_start + 1] = 1.0 if has_parameter else 0.0
        row[block_start + 2] = 1.0 if is_multi else 0.0
        for other in involved:
            if other == q:
                continue
            row[block_start + 3 + other] = 1.0
    return row


def _flatten_state_tensor(state_tensor: np.ndarray) -> np.ndarray:
    return state_tensor.reshape(-1).astype(np.float32)


def _build_state_from_gates(
    gates: Sequence[Dict[str, Any]],
    n_qubits: int,
    backend: NumpyBackend,
) -> State:
    state = State.zero_state(n_qubits, backend=backend)
    for gate in gates:
        matrix = gate_to_matrix(gate, cir_qubits=n_qubits, backend=backend)
        state = state.evolve(matrix)
    return state


def _energy_of_gates(
    gates: Sequence[Dict[str, Any]],
    hamiltonian_matrix: np.ndarray,
    n_qubits: int,
    backend: NumpyBackend,
) -> float:
    state = _build_state_from_gates(gates, n_qubits=n_qubits, backend=backend)
    h_backend = backend.cast(hamiltonian_matrix)
    return float(state.expectation(h_backend))


def _parameterized_gate_indices(gates: Sequence[Dict[str, Any]]) -> List[int]:
    indices: List[int] = []
    for index, gate in enumerate(gates):
        if "parameter" in gate:
            indices.append(index)
    return indices


def _parameter_size(param: Any) -> int:
    if np.isscalar(param):
        return 1
    arr = np.asarray(param, dtype=np.float32).reshape(-1)
    return int(arr.shape[0])


def _extract_parameters(gates: Sequence[Dict[str, Any]], param_indices: Sequence[int]) -> np.ndarray:
    values: List[float] = []
    for gate_index in param_indices:
        parameter = gates[gate_index].get("parameter")
        if np.isscalar(parameter):
            values.append(float(parameter))
        else:
            values.extend(np.asarray(parameter, dtype=np.float32).reshape(-1).tolist())
    return np.asarray(values, dtype=np.float32)


def _inject_parameters(
    gates: Sequence[Dict[str, Any]],
    param_indices: Sequence[int],
    parameters: np.ndarray,
) -> List[Dict[str, Any]]:
    updated = [dict(g) for g in gates]
    cursor = 0
    for gate_index in param_indices:
        original = gates[gate_index].get("parameter")
        size = _parameter_size(original)
        chunk = parameters[cursor: cursor + size]
        cursor += size
        if np.isscalar(original):
            updated[gate_index]["parameter"] = float(chunk[0])
        else:
            updated[gate_index]["parameter"] = [float(x) for x in chunk.tolist()]
    return updated


def _adam_spsa_optimize(
    gates: Sequence[Dict[str, Any]],
    hamiltonian_matrix: np.ndarray,
    n_qubits: int,
    backend: NumpyBackend,
    config: AdamSPSAConfig,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], float, np.ndarray]:
    param_indices = _parameterized_gate_indices(gates)
    if not param_indices:
        energy = _energy_of_gates(gates, hamiltonian_matrix=hamiltonian_matrix, n_qubits=n_qubits, backend=backend)
        return [dict(g) for g in gates], energy, np.zeros(0, dtype=np.float32)

    theta = _extract_parameters(gates, param_indices)
    m = np.zeros_like(theta, dtype=np.float32)
    v = np.zeros_like(theta, dtype=np.float32)

    best_theta = theta.copy()
    best_energy = float("inf")

    for k in range(1, config.iterations + 1):
        a_k = config.a / ((k + 1) ** config.alpha)
        c_k = config.c / ((k + 1) ** config.gamma_sp)
        beta_1_k = config.beta_1 / ((k + 1) ** config.lam)

        delta = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=theta.shape[0])

        theta_plus = theta + c_k * delta
        theta_minus = theta - c_k * delta
        gates_plus = _inject_parameters(gates, param_indices, theta_plus)
        gates_minus = _inject_parameters(gates, param_indices, theta_minus)

        e_plus = _energy_of_gates(gates_plus, hamiltonian_matrix=hamiltonian_matrix, n_qubits=n_qubits, backend=backend)
        e_minus = _energy_of_gates(gates_minus, hamiltonian_matrix=hamiltonian_matrix, n_qubits=n_qubits, backend=backend)

        grad = (e_plus - e_minus) / (2.0 * c_k * delta)
        grad = grad.astype(np.float32)

        m = beta_1_k * m + (1.0 - beta_1_k) * grad
        v = config.beta_2 * v + (1.0 - config.beta_2) * (grad * grad)

        m_hat = m / (1.0 - beta_1_k + config.epsilon)
        v_hat = v / (1.0 - config.beta_2 + config.epsilon)
        adjusted_grad = m_hat / (np.sqrt(v_hat) + config.epsilon)

        theta = theta - a_k * adjusted_grad

        current_gates = _inject_parameters(gates, param_indices, theta)
        current_energy = _energy_of_gates(
            current_gates,
            hamiltonian_matrix=hamiltonian_matrix,
            n_qubits=n_qubits,
            backend=backend,
        )
        if current_energy < best_energy:
            best_energy = current_energy
            best_theta = theta.copy()

    best_gates = _inject_parameters(gates, param_indices, best_theta)
    return best_gates, float(best_energy), best_theta


def _build_illegal_action_indices(
    action_space: Sequence[Dict[str, Any]],
    last_gate: Optional[Dict[str, Any]],
) -> List[int]:
    if last_gate is None:
        return []
    illegal: List[int] = []
    for index, action in enumerate(action_space):
        if _is_same_gate_signature(action, last_gate):
            illegal.append(index)
    return illegal


def _select_action(
    q_online: _QNet,
    state: np.ndarray,
    action_dim: int,
    epsilon: float,
    illegal_indices: Sequence[int],
    rng: random.Random,
) -> int:
    legal = [i for i in range(action_dim) if i not in illegal_indices]
    if not legal:
        legal = list(range(action_dim))

    if rng.random() < epsilon:
        return int(rng.choice(legal))

    with torch.no_grad():
        q_values = q_online(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0)
        q_np = q_values.detach().cpu().numpy()
    q_np = q_np.copy()
    for idx in illegal_indices:
        q_np[idx] = -np.inf
    best = int(np.argmax(q_np))
    if not np.isfinite(q_np[best]):
        return int(rng.choice(legal))
    return best


class _ReplayBuffer:
    def __init__(self, capacity: int):
        self._buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
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


def _train_ddqn_step(
    q_online: _QNet,
    q_target: _QNet,
    optimizer: torch.optim.Optimizer,
    replay_buffer: _ReplayBuffer,
    batch_size: int,
    gamma: float,
    rng: random.Random,
) -> Optional[float]:
    if len(replay_buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, rng)

    q_current = q_online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_actions = torch.argmax(q_online(next_states), dim=1)
        next_q = q_target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = rewards + (1.0 - dones) * gamma * next_q

    loss = nn.functional.mse_loss(q_current, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_crlqas(
    hamiltonian: np.ndarray | Hamiltonian,
    config: Optional[CRLQASConfig] = None,
) -> CRLQASResult:
    cfg = CRLQASConfig() if config is None else config

    if cfg.max_episodes <= 0:
        raise ValueError("max_episodes 必须是正整数")
    if cfg.n_act <= 0:
        raise ValueError("n_act 必须是正整数")
    if not (0.0 < cfg.random_halt_p <= 1.0):
        raise ValueError("random_halt_p 必须在 (0, 1]")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")

    backend = NumpyBackend()
    h_matrix, n_qubits = _resolve_hamiltonian_matrix(hamiltonian, backend=backend)

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng_py = random.Random(cfg.seed)
    rng_np = np.random.default_rng(cfg.seed)

    action_space = _build_action_space(n_qubits)
    if cfg.action_gates is not None:
        action_space = _validate_action_space(cfg.action_gates, n_qubits)
    else:
        action_space = _validate_action_space(action_space, n_qubits)
    action_dim = len(action_space)
    state_width = (n_qubits + 3) * n_qubits
    state_dim = cfg.n_act * state_width

    q_online = _QNet(state_dim=state_dim, action_dim=action_dim, hidden_dim=cfg.q_hidden_dim)
    q_target = _QNet(state_dim=state_dim, action_dim=action_dim, hidden_dim=cfg.q_hidden_dim)
    q_target.load_state_dict(q_online.state_dict())
    q_target.eval()

    optimizer = torch.optim.Adam(q_online.parameters(), lr=cfg.q_learning_rate)
    replay_buffer = _ReplayBuffer(cfg.replay_capacity)

    xi_cur = float(cfg.curriculum_initial_threshold)
    xi_best = float(cfg.curriculum_initial_threshold)
    epsilon = float(cfg.epsilon_start)

    global_step = 0
    failed_streak = 0

    best_energy = float("inf")
    best_gates: List[Dict[str, Any]] = []
    best_params = np.zeros(0, dtype=np.float32)
    episode_best_energies: List[float] = []

    for episode_index in range(cfg.max_episodes):
        sampled = int(rng_np.negative_binomial(cfg.n_act, cfg.random_halt_p))
        horizon = max(1, min(cfg.n_act, sampled))

        state_tensor = np.zeros((cfg.n_act, state_width), dtype=np.float32)
        state = _flatten_state_tensor(state_tensor)

        gates: List[Dict[str, Any]] = []
        param_vector = np.zeros(0, dtype=np.float32)
        prev_cost: Optional[float] = None
        episode_success = False
        episode_min = float("inf")

        for t in range(horizon):
            last_gate = gates[-1] if gates else None
            illegal_actions = _build_illegal_action_indices(action_space, last_gate)
            action_index = _select_action(
                q_online=q_online,
                state=state,
                action_dim=action_dim,
                epsilon=epsilon,
                illegal_indices=illegal_actions,
                rng=rng_py,
            )

            gate = dict(action_space[action_index])
            if "parameter" in gate:
                if np.isscalar(gate["parameter"]):
                    gate["parameter"] = float(rng_np.uniform(-np.pi, np.pi))
                else:
                    size = len(np.asarray(gate["parameter"]).reshape(-1))
                    gate["parameter"] = [float(x) for x in rng_np.uniform(-np.pi, np.pi, size=size).tolist()]
            gates.append(gate)

            row_index = min(len(gates) - 1, cfg.n_act - 1)
            state_tensor[row_index] = _gate_to_tensor_row(gate, n_qubits=n_qubits)
            next_state = _flatten_state_tensor(state_tensor)

            gates, cost, param_vector = _adam_spsa_optimize(
                gates,
                hamiltonian_matrix=h_matrix,
                n_qubits=n_qubits,
                backend=backend,
                config=cfg.adam_spsa,
                rng=rng_np,
            )

            episode_min = min(episode_min, cost)
            if cost < xi_best:
                xi_best = cost

            done = False
            if cost < xi_cur:
                reward = cfg.success_reward
                done = True
                episode_success = True
            elif t >= horizon - 1 and cost >= xi_cur:
                reward = cfg.failure_reward
                done = True
            else:
                c_prev = cost if prev_cost is None else prev_cost
                denominator = c_prev - min(xi_best, cfg.curriculum_mu)
                if abs(denominator) < 1e-8:
                    shaped = cfg.reward_floor
                else:
                    shaped = ((c_prev - cost) / denominator) - 1.0
                    shaped = max(shaped, cfg.reward_floor)
                reward = shaped

            replay_buffer.push(state, action_index, reward, next_state, done)

            global_step += 1
            if global_step % cfg.train_interval == 0:
                _train_ddqn_step(
                    q_online=q_online,
                    q_target=q_target,
                    optimizer=optimizer,
                    replay_buffer=replay_buffer,
                    batch_size=cfg.batch_size,
                    gamma=cfg.gamma,
                    rng=rng_py,
                )

            if global_step % cfg.target_update_interval == 0:
                q_target.load_state_dict(q_online.state_dict())

            prev_cost = cost
            state = next_state

            if cost < best_energy:
                best_energy = cost
                best_gates = [dict(g) for g in gates]
                best_params = param_vector.copy()

            if done:
                break

        episode_best_energies.append(episode_min)

        if (episode_index + 1) % cfg.curriculum_adjust_period == 0:
            xi_cur = abs(cfg.curriculum_mu - xi_best)

        if episode_success:
            xi_cur = xi_cur - (cfg.curriculum_delta / max(cfg.curriculum_kappa, 1.0))
            failed_streak = 0
        else:
            failed_streak += 1
            if failed_streak >= cfg.curriculum_reset_patience:
                xi_cur = abs(cfg.curriculum_mu - xi_best) + cfg.curriculum_delta
                failed_streak = 0

        xi_cur = max(cfg.chemical_accuracy, xi_cur)
        epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)

        if cfg.log_interval > 0 and (episode_index + 1) % cfg.log_interval == 0:
            print(
                f"[CRLQAS] episode={episode_index + 1} "
                f"best_energy={best_energy:.6f} xi_cur={xi_cur:.6f} epsilon={epsilon:.4f}"
            )

    if not best_gates:
        best_gates = []
        best_energy = _energy_of_gates(best_gates, hamiltonian_matrix=h_matrix, n_qubits=n_qubits, backend=backend)

    result_circuit = Circuit(*best_gates, n_qubits=n_qubits, backend=backend)
    return CRLQASResult(
        circuit=result_circuit,
        parameters=[float(x) for x in best_params.tolist()],
        minimum_energy=float(best_energy),
        curriculum_threshold=float(xi_cur),
        episode_best_energies=episode_best_energies,
        q_network_state_dict={k: v.detach().cpu().clone() for k, v in q_online.state_dict().items()},
    )


def crlqas(
    hamiltonian: np.ndarray | Hamiltonian,
    config: Optional[CRLQASConfig] = None,
) -> Tuple[Circuit, float]:
    result = train_crlqas(hamiltonian=hamiltonian, config=config)
    return result.circuit, result.minimum_energy


__all__ = [
    "AdamSPSAConfig",
    "CRLQASConfig",
    "CRLQASResult",
    "crlqas",
    "train_crlqas",
]
