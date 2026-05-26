"""Small PPR-DQL demo with the default ion-trap noise model.

Run from the repository root:
    C:/ProgramData/anaconda3/python.exe nexq/algorithms/qas/demo/PPR_DQL_noise_adaptive_demo.py
"""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nexq.algorithms.qas import PPRDQLConfig, load_default_ion_trap_noise_config, train_ppr_dql
from nexq.algorithms.qas.multi_objective_reward import (
    ExpressibilityScore,
    HardwareEfficiencyScore,
    MultiObjectiveReward,
    NoiseRobustnessScore,
    RewardWeights,
    TrainabilityScore,
)
from nexq.channel.backends.numpy_backend import NumpyBackend
from nexq.core.circuit import Circuit
from nexq.core.state import State


def _prepare_target_state(backend: NumpyBackend) -> State:
    circuit = Circuit({"type": "pauli_x", "target_qubit": 0}, n_qubits=2, backend=backend)
    return State.zero_state(2, backend=backend).evolve(circuit.unitary(backend=backend))


def _make_config(*, noise_aware: bool) -> PPRDQLConfig:
    fast_reward = MultiObjectiveReward(
        weights=RewardWeights(
            expressibility=0.20,
            trainability=0.20,
            noise_robustness=0.40,
            hardware_efficiency=0.20,
        ),
        expressibility_score=ExpressibilityScore(n_samples=20),
        trainability_score=TrainabilityScore(n_samples=10),
        noise_robustness_score=NoiseRobustnessScore(n_samples=20),
        hardware_efficiency_score=HardwareEfficiencyScore(),
        fidelity_weight=0.2,
        penalty_weight=0.02,
    )

    return PPRDQLConfig(
        episode_num=6,
        max_steps_per_episode=1,
        batch_size=1,
        replay_capacity=32,
        warmup_transitions=1,
        target_update_interval=1,
        fidelity_threshold=0.99,
        gate_penalty=0.0,
        terminal_bonus=1.0,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1.0,
        action_gates=[{"type": "pauli_x", "target_qubit": 0}],
        use_noise_adaptive_reward=noise_aware,
        multi_objective_reward=fast_reward if noise_aware else None,
        multi_objective_reward_weight=0.5,
        seed=7,
        log_interval=0,
    )


def _summarize_run(label: str, result) -> None:
    rewards = [round(value, 4) for value in result.episode_rewards]
    print(f"\n[{label}]")
    print(f"best_fidelity: {result.best_fidelity:.6f}")
    print(f"episode_rewards: {rewards}")
    print(f"circuit_gates: {deepcopy(result.circuit.gates)}")


def main() -> None:
    backend = NumpyBackend()
    target_state = _prepare_target_state(backend)

    ion_config = load_default_ion_trap_noise_config()
    resolved = ion_config.resolved_parameters()
    noise_model = ion_config.build_noise_model(qubits=[0, 1])

    print("=== Default Ion-Trap Noise Parameters ===")
    for key in [
        "oneq_depol",
        "twoq_depol",
        "cross_talk",
        "meas_bitflip",
        "reset_bitflip",
        "T2",
        "oneq_gate_time",
        "twoq_gate_time",
    ]:
        print(f"{key}: {resolved[key]}")
    print(f"idle_oneq: {ion_config.idle_dephasing_probability(gate_family='oneq')}")
    print(f"idle_twoq: {ion_config.idle_dephasing_probability(gate_family='twoq')}")
    print(f"noise_rules: {len(noise_model.rules)}")
    print("target task: prepare X on qubit 0 from |00>")

    baseline = train_ppr_dql(target_state, config=_make_config(noise_aware=False))
    noise_aware = train_ppr_dql(target_state, config=_make_config(noise_aware=True))

    _summarize_run("baseline reward", baseline)
    _summarize_run("noise-aware reward", noise_aware)

    reward_delta = sum(noise_aware.episode_rewards) - sum(baseline.episode_rewards)
    print(f"\nreward_sum_delta(noise-aware - baseline): {reward_delta:.6f}")


if __name__ == "__main__":
    main()