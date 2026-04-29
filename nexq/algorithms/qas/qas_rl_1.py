"""Reinforcement-learning based quantum architecture search using nexq.

This module is designed from the paper:
"Quantum Architecture Search via Deep Reinforcement Learning"
arXiv:2104.07715v1
https://github.com/qdevpsi3/quantum-arch-search

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore[import-not-found]

from ...channel.backends.numpy_backend import NumpyBackend
from ...circuit.gates import gate_to_matrix
from ...circuit.model import Circuit

A2C: Any
DQN: Any
PPO: Any

try:
    import gymnasium as gym  # type: ignore[import-not-found]
    from gymnasium import spaces  # type: ignore[import-not-found]

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

    class _FallbackGym:
        class Env:
            pass

    class _FallbackSpaces:
        class Box:
            def __init__(self, *args, **kwargs):
                pass

        class Discrete:
            def __init__(self, *args, **kwargs):
                pass

    gym = _FallbackGym()
    spaces = _FallbackSpaces()

try:
    from stable_baselines3 import A2C, DQN, PPO  # type: ignore[import-not-found]

    SB3_AVAILABLE = True
except ImportError:
    A2C = DQN = PPO = None
    SB3_AVAILABLE = False


def get_default_gates(n_qubits: int) -> List[Dict[str, Any]]:
    """Build the default action gate set for architecture search."""
    gates: List[Dict[str, Any]] = []
    for idx in range(n_qubits):
        next_qubit = (idx + 1) % n_qubits
        gates.append({"type": "rz", "target_qubit": idx, "parameter": np.pi / 4})
        gates.append({"type": "pauli_x", "target_qubit": idx})
        gates.append({"type": "pauli_y", "target_qubit": idx})
        gates.append({"type": "pauli_z", "target_qubit": idx})
        gates.append({"type": "hadamard", "target_qubit": idx})
        gates.append(
            {
                "type": "cnot",
                "target_qubit": next_qubit,
                "control_qubits": [idx],
                "control_states": [1],
            }
        )
    return gates


def get_default_observables(n_qubits: int) -> List[Dict[str, Any]]:
    """Build default X/Y/Z observables for each qubit."""
    observables: List[Dict[str, Any]] = []
    for q in range(n_qubits):
        observables.append({"type": "pauli_x", "target_qubit": q})
        observables.append({"type": "pauli_y", "target_qubit": q})
        observables.append({"type": "pauli_z", "target_qubit": q})
    return observables


def get_bell_state() -> np.ndarray:
    """Return Bell state |Phi+>."""
    state = np.zeros(4, dtype=np.complex128)
    state[0] = 1.0 / np.sqrt(2)
    state[-1] = 1.0 / np.sqrt(2)
    return state


def get_ghz_state(n_qubits: int = 3) -> np.ndarray:
    """Return n-qubit GHZ state."""
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[0] = 1.0 / np.sqrt(2)
    state[-1] = 1.0 / np.sqrt(2)
    return state


class QuantumArchSearchEnvCore:
    """Core environment implementation for quantum architecture search."""

    def __init__(
        self,
        target: np.ndarray,
        n_qubits: int,
        state_observables: List[Dict[str, Any]],
        action_gates: List[Dict[str, Any]],
        fidelity_threshold: float = 0.95,
        reward_penalty: float = 0.01,
        max_timesteps: int = 20,
        backend: Optional[NumpyBackend] = None,
    ) -> None:
        if len(target) != 2**n_qubits:
            raise ValueError(
                f"target length must be 2**n_qubits ({2**n_qubits}), got {len(target)}"
            )

        target_norm = np.linalg.norm(target)
        if target_norm <= 0:
            raise ValueError("target state norm must be > 0")

        self.target = target / target_norm
        self.n_qubits = n_qubits
        self.state_observables = state_observables
        self.action_gates = action_gates
        self.fidelity_threshold = fidelity_threshold
        self.reward_penalty = reward_penalty
        self.max_timesteps = max_timesteps
        self.backend = backend if backend is not None else NumpyBackend()

        self.circuit_gates: List[Dict[str, Any]] = []
        self.current_timestep = 0

    @property
    def observation_shape(self) -> Tuple[int]:
        return (len(self.state_observables),)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.circuit_gates = []
        self.current_timestep = 0
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if action < 0 or action >= len(self.action_gates):
            raise ValueError(f"invalid action index: {action}")

        self.circuit_gates.append(self.action_gates[action])
        self.current_timestep += 1

        observation = self._get_obs()
        fidelity = self._get_fidelity()

        if fidelity > self.fidelity_threshold:
            reward = fidelity - self.reward_penalty
        else:
            reward = -self.reward_penalty

        done = fidelity > self.fidelity_threshold or self.current_timestep >= self.max_timesteps

        info = {
            "fidelity": float(fidelity),
            "timestep": self.current_timestep,
            "circuit": self.build_circuit(),
        }
        return observation, float(reward), bool(done), info

    def build_circuit(self) -> Circuit:
        return Circuit(*self.circuit_gates, n_qubits=self.n_qubits)

    def get_circuit_state(self) -> np.ndarray:
        state = np.zeros(2**self.n_qubits, dtype=np.complex128)
        state[0] = 1.0
        if not self.circuit_gates:
            return state

        for gate in self.circuit_gates:
            op = gate_to_matrix(gate, cir_qubits=self.n_qubits, backend=self.backend)
            op_np = op.numpy() if hasattr(op, "numpy") else np.asarray(op)
            state = op_np @ state
        return state

    def _get_obs(self) -> np.ndarray:
        state = self.get_circuit_state()
        observables: List[float] = []
        for obs in self.state_observables:
            operator = gate_to_matrix(obs, cir_qubits=self.n_qubits, backend=self.backend)
            operator_np = operator.numpy() if hasattr(operator, "numpy") else np.asarray(operator)
            expectation = np.real(np.conj(state) @ operator_np @ state)
            observables.append(float(expectation))
        return np.asarray(observables, dtype=np.float32)

    def _get_fidelity(self) -> float:
        state = self.get_circuit_state()
        inner = np.conj(state) @ self.target
        return float(np.real(np.abs(inner) ** 2))


class QuantumArchSearchGymEnv:
    """Gymnasium wrapper for stable-baselines3 training."""

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, core: QuantumArchSearchEnvCore):
        if not GYMNASIUM_AVAILABLE:
            raise RuntimeError("gymnasium is required. Install with: pip install gymnasium")
        self.core = core
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self.core.observation_shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(self.core.action_gates))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        observation = self.core.reset(seed=seed)
        return observation, {}

    def step(self, action: int):
        observation, reward, done, info = self.core.step(int(action))
        terminated = done
        truncated = False
        return observation, reward, terminated, truncated, info

    def render(self):
        return f"depth={len(self.core.circuit_gates)}, gates={self.core.circuit_gates}"


@dataclass
class TrainConfig:
    env_name: str = "basic2"
    algo: str = "ppo"
    total_timesteps: int = 10000
    fidelity_threshold: float = 0.95
    reward_penalty: float = 0.01
    max_timesteps: int = 20
    seed: Optional[int] = 42
    eval_episodes: int = 10
    save_path: Optional[str] = None


def create_core_env(config: TrainConfig) -> QuantumArchSearchEnvCore:
    """Create a preset QAS environment from the CLI config."""
    env_key = config.env_name.lower()
    if env_key == "basic2":
        target = get_bell_state()
        n_qubits = 2
    elif env_key == "basic3":
        target = get_ghz_state(3)
        n_qubits = 3
    else:
        raise ValueError(f"unsupported env_name: {config.env_name}. Use basic2 or basic3.")

    return QuantumArchSearchEnvCore(
        target=target,
        n_qubits=n_qubits,
        state_observables=get_default_observables(n_qubits),
        action_gates=get_default_gates(n_qubits),
        fidelity_threshold=config.fidelity_threshold,
        reward_penalty=config.reward_penalty,
        max_timesteps=config.max_timesteps,
    )


def build_model(algo: str, env: Any, seed: Optional[int]):
    """Create an SB3 model by algorithm name."""
    if not SB3_AVAILABLE:
        raise RuntimeError("stable-baselines3 is required. Install with: pip install stable-baselines3")

    key = algo.lower()
    if key == "ppo":
        return PPO("MlpPolicy", env, verbose=1, seed=seed)
    if key == "a2c":
        return A2C("MlpPolicy", env, verbose=1, seed=seed)
    if key == "dqn":
        return DQN("MlpPolicy", env, verbose=1, seed=seed)
    raise ValueError(f"unsupported algo: {algo}. Use ppo, a2c, or dqn.")


def evaluate_model(model: Any, env: Any, episodes: int) -> Dict[str, float]:
    """Run deterministic evaluation and return aggregate metrics."""
    rewards: List[float] = []
    fidelities: List[float] = []

    for _ in range(episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        ep_reward = 0.0
        ep_fidelity = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_fidelity = float(info["fidelity"])

        rewards.append(ep_reward)
        fidelities.append(ep_fidelity)

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_max": float(np.max(rewards)),
        "fidelity_mean": float(np.mean(fidelities)),
        "fidelity_max": float(np.max(fidelities)),
    }


def run_training(config: TrainConfig) -> Dict[str, float]:
    """Train and evaluate a reinforcement learning agent for QAS."""
    if not GYMNASIUM_AVAILABLE:
        raise RuntimeError(
            "gymnasium is required. Install with: pip install gymnasium"
        )
    if not SB3_AVAILABLE:
        raise RuntimeError(
            "stable-baselines3 is required. Install with: pip install stable-baselines3"
        )

    core = create_core_env(config)
    env = QuantumArchSearchGymEnv(core)
    model = build_model(config.algo, env, config.seed)

    model.learn(total_timesteps=config.total_timesteps)

    if config.save_path:
        save_path = Path(config.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))

    metrics = evaluate_model(model, env, episodes=config.eval_episodes)
    return metrics


def parse_args() -> TrainConfig:
    """Parse CLI args into TrainConfig."""
    parser = argparse.ArgumentParser(
        description="RL-based quantum architecture search using nexq."
    )
    parser.add_argument("--env", default="basic2", choices=["basic2", "basic3"], help="Environment preset")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "a2c", "dqn"], help="RL algorithm")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total training timesteps")
    parser.add_argument("--fidelity-threshold", type=float, default=0.95, help="Success fidelity threshold")
    parser.add_argument("--reward-penalty", type=float, default=0.01, help="Penalty per non-terminal step")
    parser.add_argument("--max-timesteps", type=int, default=20, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--save-path", default=None, help="Optional path for saving the trained model")

    args = parser.parse_args()

    return TrainConfig(
        env_name=args.env,
        algo=args.algo,
        total_timesteps=args.timesteps,
        fidelity_threshold=args.fidelity_threshold,
        reward_penalty=args.reward_penalty,
        max_timesteps=args.max_timesteps,
        seed=args.seed,
        eval_episodes=args.eval_episodes,
        save_path=args.save_path,
    )


def main() -> None:
    """CLI entrypoint."""
    config = parse_args()
    metrics = run_training(config)

    print("Training completed.")
    print(f"Env: {config.env_name}, Algo: {config.algo}, Timesteps: {config.total_timesteps}")
    print(
        "Evaluation -> "
        f"reward_mean={metrics['reward_mean']:.6f}, "
        f"reward_max={metrics['reward_max']:.6f}, "
        f"fidelity_mean={metrics['fidelity_mean']:.6f}, "
        f"fidelity_max={metrics['fidelity_max']:.6f}"
    )


if __name__ == "__main__":
    main()
