"""Trust Region-based PPO with Rollback for Quantum Architecture Search.

X. Zhu and X. Hou, ‘Quantum architecture search via truly proximal policy optimization’, 
Sci Rep, vol. 13, no. 1, p. 5157, Mar. 2023, doi: 10.1038/s41598-023-32349-2.


- 从 |0...0> 初态开始
- 动作为追加一个门到当前线路
- 输入为目标密度矩阵和保真度阈值 epsilon
- 输出为优化后的策略参数 theta 与运行策略得到的电路
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from ...channel.backends.numpy_backend import NumpyBackend
from ...core.circuit import Circuit
from ...core.gates import gate_to_matrix
from ...core.state import State


@dataclass
class PPORollbackConfig:
    """ 超参数配置 """

    episode_num: int = 200
    max_steps_per_episode: int = 20
    update_timestep: int = 64
    epoch_num: int = 4  # K
    epsilon_clip: float = 0.2  # C
    rollback_alpha: float = -0.3  # alpha
    kl_threshold: float = 0.03  # delta
    gamma: float = 0.99
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 0.002
    hidden_dim: int = 256
    gate_penalty: float = 0.01
    seed: int = 42
    # 可选：外部注入动作集合（None 则自动生成全量门）
    action_gates: Optional[List[Dict[str, object]]] = None
    # 到达 epsilon 时给予的额外奖励
    terminal_bonus: float = 0.0
    # 每 N 个 episode 打印一次进度（0 = 关闭）
    log_interval: int = 0
    # 可选：使用上一阶段训练得到的策略参数进行热启动
    init_theta: Optional[Dict[str, torch.Tensor]] = None


class _PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(state)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return logits, value


def _infer_n_qubits_from_density(target_density_matrix: np.ndarray) -> int:
    rho = np.asarray(target_density_matrix, dtype=np.complex64)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("目标密度矩阵必须是方阵")
    dim = int(rho.shape[0])
    n_qubits = int(round(np.log2(dim))) if dim > 0 else 0
    if (1 << n_qubits) != dim:
        raise ValueError("目标密度矩阵维度必须是 2 的幂")
    return n_qubits


def _all_supported_action_gates(n_qubits: int) -> List[Dict[str, object]]:
    """构造离散动作空间，覆盖 nexq/core/gates 的主要门类型。"""
    actions: List[Dict[str, object]] = []

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

    checked: List[Dict[str, object]] = []
    full_dim = 1 << n_qubits
    backend = NumpyBackend()
    for gate in actions:
        try:
            gm_np = gate_to_matrix(gate, cir_qubits=n_qubits, backend=None)
            gm_bk = gate_to_matrix(gate, cir_qubits=n_qubits, backend=backend)
            if np.shape(gm_np) != (full_dim, full_dim):
                continue
            if np.shape(backend.to_numpy(gm_bk)) != (full_dim, full_dim):
                continue
            checked.append(gate)
        except Exception:
            continue
    return checked


class _QASEnv:
    """从 |0> 态出发的量子架构搜索环境。"""

    def __init__(
        self,
        target_rho: np.ndarray,
        epsilon: float,
        max_steps: int,
        gate_penalty: float,
        action_gates: Optional[List[Dict[str, object]]] = None,
        terminal_bonus: float = 0.0,
    ):
        self.backend = NumpyBackend()
        self.target_rho = np.asarray(target_rho, dtype=np.complex64)
        self.n_qubits = _infer_n_qubits_from_density(self.target_rho)
        self.epsilon = float(epsilon)
        self.max_steps = int(max_steps)
        self.gate_penalty = float(gate_penalty)
        self.terminal_bonus = float(terminal_bonus)

        if action_gates is not None:
            # 外部注入时同样做维度校验
            full_dim = 1 << self.n_qubits
            bk = self.backend
            checked: List[Dict[str, object]] = []
            for gate in action_gates:
                try:
                    gm = gate_to_matrix(gate, cir_qubits=self.n_qubits, backend=bk)
                    if np.shape(bk.to_numpy(gm)) == (full_dim, full_dim):
                        checked.append(dict(gate))
                except Exception:
                    continue
            self.action_gates: List[Dict[str, object]] = checked
        else:
            self.action_gates = _all_supported_action_gates(self.n_qubits)
        if not self.action_gates:
            raise ValueError("动作空间为空，无法训练")

        self.circuit_gates: List[Dict[str, object]] = []
        self.steps = 0
        self.prev_fidelity = 0.0

    def reset(self) -> np.ndarray:
        self.circuit_gates = []
        self.steps = 0
        self.prev_fidelity = self._fidelity()
        return self._state_feature()

    def _build_state(self) -> State:
        state = State.zero_state(self.n_qubits, backend=self.backend)
        for gate in self.circuit_gates:
            gm = gate_to_matrix(gate, cir_qubits=self.n_qubits, backend=self.backend)
            state = state.evolve(gm)
        return state

    def _current_density(self) -> np.ndarray:
        psi = self._build_state().to_numpy().astype(np.complex64).reshape(-1, 1)
        return psi @ psi.conj().T

    def _state_feature(self) -> np.ndarray:
        rho = self._current_density()
        feature = np.concatenate([rho.real.reshape(-1), rho.imag.reshape(-1)], axis=0)
        return feature.astype(np.float32)

    def _fidelity(self) -> float:
        psi = self._build_state().to_numpy().astype(np.complex64).reshape(-1, 1)
        val = (psi.conj().T @ self.target_rho @ psi).item()
        return float(np.real(val))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        gate = dict(self.action_gates[action])
        self.circuit_gates.append(gate)
        self.steps += 1

        fid = self._fidelity()
        reward = (fid - self.prev_fidelity) - self.gate_penalty
        self.prev_fidelity = fid

        done = (fid >= self.epsilon) or (self.steps >= self.max_steps)
        if fid >= self.epsilon:
            reward += self.terminal_bonus
        info = {
            "fidelity": fid,
            "gate_count": len(self.circuit_gates),
            "circuit": Circuit(*self.circuit_gates, n_qubits=self.n_qubits, backend=self.backend),
        }
        return self._state_feature(), float(reward), bool(done), info


def _discounted_returns(rewards: Sequence[float], gamma: float) -> torch.Tensor:
    returns: List[float] = []
    running = 0.0
    for r in reversed(rewards):
        running = float(r) + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def _select_action(policy: _PolicyValueNet, state: np.ndarray) -> Tuple[int, float]:
    state_t = torch.from_numpy(state).unsqueeze(0)
    with torch.no_grad():
        logits, _ = policy(state_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
    return int(action.item()), float(log_prob.item())


def _run_policy_to_circuit(env: _QASEnv, policy: _PolicyValueNet) -> Circuit:
    state = env.reset()
    done = False
    final_info: Dict[str, object] = {}
    while not done:
        state_t = torch.from_numpy(state).unsqueeze(0)
        with torch.no_grad():
            logits, _ = policy(state_t)
            action = int(torch.argmax(logits, dim=-1).item())
        state, _, done, final_info = env.step(action)
    circuit_obj = final_info.get("circuit")
    if not isinstance(circuit_obj, Circuit):
        raise RuntimeError("策略运行失败：未获得有效电路")
    return cast(Circuit, circuit_obj)


def ppo_rb_qas(
    target_density_matrix: np.ndarray,
    epsilon: float,
    config: Optional[PPORollbackConfig] = None,
) -> Tuple[Dict[str, torch.Tensor], Circuit]:
    """Algorithm 3: Trust Region-based PPO with Rollback for QAS.

    Args:
        target_density_matrix: 目标量子态密度矩阵。
        epsilon: 保真度阈值。
        config: 训练超参数（不传则使用默认值）。

    Returns:
        (theta, circuit)
        - theta: 优化后的策略网络参数
        - circuit: 运行优化策略得到的量子电路
    """
    cfg = PPORollbackConfig() if config is None else config

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = _QASEnv(
        target_rho=np.asarray(target_density_matrix, dtype=np.complex64),
        epsilon=epsilon,
        max_steps=cfg.max_steps_per_episode,
        gate_penalty=cfg.gate_penalty,
        action_gates=cfg.action_gates,
        terminal_bonus=cfg.terminal_bonus,
    )

    state_dim = env.reset().shape[0]
    action_dim = len(env.action_gates)

    policy = _PolicyValueNet(state_dim=state_dim, action_dim=action_dim, hidden_dim=cfg.hidden_dim)
    old_policy = _PolicyValueNet(state_dim=state_dim, action_dim=action_dim, hidden_dim=cfg.hidden_dim)

    if cfg.init_theta is not None:
        with torch.no_grad():
            current = policy.state_dict()
            loaded: Dict[str, torch.Tensor] = {}
            for k, v in cfg.init_theta.items():
                if k in current and tuple(current[k].shape) == tuple(v.shape):
                    loaded[k] = v.detach().cpu().to(dtype=current[k].dtype)
            if loaded:
                current.update(loaded)
                policy.load_state_dict(current)

    old_policy.load_state_dict(policy.state_dict())

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)

    trajectory_states: List[np.ndarray] = []
    trajectory_actions: List[int] = []
    trajectory_log_probs_old: List[float] = []
    trajectory_rewards: List[float] = []

    t = 0
    best_fidelity = 0.0
    best_circuit: Optional[Circuit] = None
    for _episode in range(cfg.episode_num):
        state = env.reset()

        for _step in range(cfg.max_steps_per_episode):
            t += 1

            action, log_prob_old = _select_action(old_policy, state)
            next_state, reward, done, _info = env.step(action)

            trajectory_states.append(state)
            trajectory_actions.append(action)
            trajectory_log_probs_old.append(log_prob_old)
            trajectory_rewards.append(reward)

            ep_fid = float(_info.get("fidelity") or 0.0)  # type: ignore[arg-type]
            if ep_fid > best_fidelity:
                best_fidelity = ep_fid
                ep_circuit = _info.get("circuit")
                if isinstance(ep_circuit, Circuit):
                    best_circuit = ep_circuit

            state = next_state

            if t == cfg.update_timestep:
                returns = _discounted_returns(trajectory_rewards, gamma=cfg.gamma)
                states_t = torch.tensor(np.asarray(trajectory_states, dtype=np.float32), dtype=torch.float32)
                actions_t = torch.tensor(trajectory_actions, dtype=torch.long)
                old_log_probs_t = torch.tensor(trajectory_log_probs_old, dtype=torch.float32)

                with torch.no_grad():
                    old_logits, _ = old_policy(states_t)
                    old_probs = torch.softmax(old_logits, dim=-1)
                    old_log_probs_all = torch.log(old_probs + 1e-8)

                for _k in range(cfg.epoch_num):
                    logits, values = policy(states_t)
                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(actions_t)
                    entropy = dist.entropy()

                    ratio = torch.exp(new_log_probs - old_log_probs_t)
                    advantages = returns - values

                    new_probs = torch.softmax(logits, dim=-1)
                    new_log_probs_all = torch.log(new_probs + 1e-8)
                    kl = torch.sum(old_probs * (old_log_probs_all - new_log_probs_all), dim=-1)

                    unclipped = ratio * advantages
                    clipped_ratio = torch.clamp(ratio, 1.0 - cfg.epsilon_clip, 1.0 + cfg.epsilon_clip)
                    clipped = clipped_ratio * advantages
                    ppo_surr = torch.minimum(unclipped, clipped)

                    rollback_surr = -cfg.rollback_alpha * advantages
                    surr = torch.where(kl >= cfg.kl_threshold, rollback_surr, ppo_surr)

                    # Gradient descent should maximize surrogate objective, so use -surr.
                    loss = torch.mean(
                        -surr
                        + cfg.value_loss_coef * (values - returns) ** 2
                        - cfg.entropy_coef * entropy
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                old_policy.load_state_dict(policy.state_dict())

                trajectory_states.clear()
                trajectory_actions.clear()
                trajectory_log_probs_old.clear()
                trajectory_rewards.clear()
                t = 0

            if done:
                break

        if cfg.log_interval > 0 and (_episode + 1) % cfg.log_interval == 0:
            print(f"  episode {_episode + 1:4d}/{cfg.episode_num}  best_fidelity={best_fidelity:.4f}")

    # 优先返回训练中百实限住的最佳线路；若未找到则回退到随机采样推演
    if best_circuit is None:
        best_circuit = _run_policy_to_circuit(env, policy)
    theta = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    return theta, best_circuit


__all__ = ["PPORollbackConfig", "ppo_rb_qas"]