import unittest

import numpy as np

from nexq import NumpyBackend, State
from nexq.algorithms.qas import PPRDQLConfig, ppr_dql_state_to_circuit, train_ppr_dql


def _fidelity_from_circuit(circuit, target_state: State) -> float:
    evolved = State.zero_state(target_state.n_qubits, backend=target_state.backend).evolve(
        circuit.unitary(backend=target_state.backend)
    )
    target = target_state.to_numpy().reshape(-1)
    current = evolved.to_numpy().reshape(-1)
    inner = np.vdot(current, target)
    return float(np.real(np.conj(inner) * inner))


class TestPPRDQL(unittest.TestCase):
    def setUp(self):
        self.backend = NumpyBackend()
        self.target_state = State.from_array(np.array([0.0, 1.0], dtype=np.complex64), n_qubits=1, backend=self.backend)
        self.config = PPRDQLConfig(
            episode_num=6,
            max_steps_per_episode=1,
            batch_size=1,
            replay_capacity=16,
            warmup_transitions=1,
            target_update_interval=1,
            fidelity_threshold=0.99,
            gate_penalty=0.0,
            terminal_bonus=1.0,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay=1.0,
            action_gates=[{"type": "pauli_x", "target_qubit": 0}],
            seed=7,
        )

    def test_train_ppr_dql_returns_high_fidelity_circuit(self):
        result = train_ppr_dql(self.target_state, config=self.config)

        self.assertEqual(result.circuit.n_qubits, 1)
        self.assertEqual(len(result.circuit), 1)
        self.assertEqual(result.circuit.gates[0]["type"], "pauli_x")
        self.assertEqual(len(result.episode_rewards), self.config.episode_num)
        self.assertEqual(len(result.selected_policy_indices), self.config.episode_num)
        self.assertGreaterEqual(result.best_fidelity, 0.99)
        self.assertGreaterEqual(_fidelity_from_circuit(result.circuit, self.target_state), 0.99)

    def test_exported_helper_returns_same_preparation_circuit(self):
        circuit = ppr_dql_state_to_circuit(self.target_state, config=self.config)

        self.assertEqual(len(circuit), 1)
        self.assertEqual(circuit.gates[0]["type"], "pauli_x")
        self.assertGreaterEqual(_fidelity_from_circuit(circuit, self.target_state), 0.99)


if __name__ == "__main__":
    unittest.main()