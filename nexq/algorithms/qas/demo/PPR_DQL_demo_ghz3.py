"""Demo: prepare a 3-qubit GHZ state with PPR-DQL and export OpenQASM 3.0."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from nexq.algorithms.qas.PPR_DQL import PPRDQLConfig, train_ppr_dql
from nexq.channel.backends.numpy_backend import NumpyBackend
from nexq.core.gates import gate_to_matrix
from nexq.core.io.qasm import save_circuit_qasm3
from nexq.core.state import State


def build_ghz3_state(backend: NumpyBackend) -> State:
    vector = np.zeros(8, dtype=np.complex64)
    vector[0] = 1.0 / np.sqrt(2.0)
    vector[7] = 1.0 / np.sqrt(2.0)
    return State.from_array(vector, n_qubits=3, backend=backend)


def fidelity_of_circuit(circuit, target_state: State) -> float:
    state = State.zero_state(circuit.n_qubits, backend=target_state.backend)
    for gate in circuit.gates:
        matrix = gate_to_matrix(gate, cir_qubits=circuit.n_qubits, backend=target_state.backend)
        state = state.evolve(matrix)
    target = target_state.to_numpy().reshape(-1)
    current = state.to_numpy().reshape(-1)
    inner = np.vdot(current, target)
    return float(np.real(np.conj(inner) * inner))


def main() -> None:
    print("=" * 68)
    print("PPR-DQL Demo: Prepare 3-Qubit GHZ State")
    print("=" * 68)

    backend = NumpyBackend()
    target_state = build_ghz3_state(backend)
    action_gates = [
        {"type": "hadamard", "target_qubit": 0},
        {"type": "cx", "target_qubit": 1, "control_qubits": [0], "control_states": [1]},
        {"type": "cx", "target_qubit": 2, "control_qubits": [0], "control_states": [1]},
    ]
    config = PPRDQLConfig(
        episode_num=800,
        max_steps_per_episode=3,
        batch_size=16,
        replay_capacity=2048,
        warmup_transitions=16,
        target_update_interval=10,
        fidelity_threshold=0.99,
        gate_penalty=0.0,
        terminal_bonus=3.0,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        action_gates=action_gates,
        seed=42,
        log_interval=100,
    )

    print("[1/4] 开始训练 PPR-DQL...")
    result = train_ppr_dql(target_state=target_state, config=config)

    print("[2/4] 训练完成，评估线路...")
    fidelity = fidelity_of_circuit(result.circuit, target_state)
    print(f"最优线路保真度: {result.best_fidelity:.6f}")
    print(f"重放评估保真度: {fidelity:.6f}")
    print(f"线路门数: {len(result.circuit.gates)}")

    print("[3/4] 输出门序列...")
    for index, gate in enumerate(result.circuit.gates):
        print(f"  [{index:02d}] {gate}")

    print("[4/4] 导出 OpenQASM 3.0...")
    out_path = Path(__file__).parent / "ppr_dql_ghz3_circuit.qasm"
    save_circuit_qasm3(result.circuit, out_path)
    print(f"QASM 3.0 已保存到: {out_path}")


if __name__ == "__main__":
    main()