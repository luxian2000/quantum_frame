import numpy as np
import pytest

from aicir.backends import NumpyBackend
from aicir.core.operators import Hamiltonian
from aicir.ir import instruction_name, instruction_parameter, instruction_qubits
from aicir.optimizer import NelderMead
from aicir.vqc import BasicQAOA


def _gate_summary(circuit):
    return [
        (instruction_name(gate), instruction_qubits(gate), instruction_parameter(gate))
        for gate in circuit.operations
    ]


def test_basic_qaoa_builds_canonical_gate_level_circuit_from_hamiltonian():
    hamiltonian = Hamiltonian(
        n_qubits=3,
        terms=[
            ("Z", [0], 0.5),
            ("ZZ", [0, 2], -1.25),
            ("III", 0.7),
        ],
    )
    qaoa = BasicQAOA(problem_hamiltonian=hamiltonian, p=1)

    circuit = qaoa.build_circuit(np.array([0.2, -0.3]))

    assert circuit.n_qubits == 3
    assert _gate_summary(circuit) == [
        ("hadamard", (0,), None),
        ("hadamard", (1,), None),
        ("hadamard", (2,), None),
        ("rz", (0,), 0.2),
        ("rzz", (0, 2), -0.5),
        ("rx", (0,), -0.6),
        ("rx", (1,), -0.6),
        ("rx", (2,), -0.6),
    ]


def test_basic_qaoa_accepts_non_diagonal_cost_hamiltonian():
    qaoa = BasicQAOA(problem_hamiltonian=Hamiltonian([("X", 1.0)]), p=1)

    circuit = qaoa.build_circuit(np.array([0.2, 0.1]))

    assert any(instruction_name(gate) == "rx" for gate in circuit.operations)


def test_basic_qaoa_trotterizes_non_diagonal_pauli_terms():
    hamiltonian = Hamiltonian(n_qubits=2, terms=[("X", [0], 0.5), ("YZ", [0, 1], -0.25)])
    qaoa = BasicQAOA(problem_hamiltonian=hamiltonian, p=1, trotter_steps=2, trotter_order=1)

    circuit = qaoa.build_circuit(np.array([0.4, 0.1]))
    summary = _gate_summary(circuit)

    assert qaoa.trotter_steps == 2
    assert qaoa.trotter_order == 1
    assert summary.count(("rx", (0,), 0.2)) == 3
    assert summary.count(("rz", (0,), -np.pi / 2.0)) == 2
    assert summary.count(("rz", (1,), -0.1)) == 2
    assert summary.count(("cx", (1,), None)) == 4
    assert summary[-2:] == [("rx", (0,), 0.2), ("rx", (1,), 0.2)]


def test_basic_qaoa_second_order_trotter_uses_symmetric_term_sequence():
    hamiltonian = Hamiltonian(n_qubits=2, terms=[("X", [0], 1.0), ("Z", [1], 2.0)])
    qaoa = BasicQAOA(problem_hamiltonian=hamiltonian, p=1, trotter_steps=2, trotter_order=2)

    circuit = qaoa.build_circuit(np.array([0.4, 0.3]))

    assert qaoa.trotter_steps == 2
    assert qaoa.trotter_order == 2
    assert _gate_summary(circuit) == [
        ("hadamard", (0,), None),
        ("hadamard", (1,), None),
        ("rx", (0,), 0.2),
        ("rz", (1,), 0.8),
        ("rx", (0,), 0.2),
        ("rx", (0,), 0.2),
        ("rz", (1,), 0.8),
        ("rx", (0,), 0.2),
        ("rx", (0,), 0.6),
        ("rx", (1,), 0.6),
    ]


def test_basic_qaoa_second_order_trotter_handles_identity_only_cost():
    qaoa = BasicQAOA(
        problem_hamiltonian=Hamiltonian(n_qubits=1, terms=[("I", [0], 2.0)]),
        p=1,
        trotter_order=2,
    )

    circuit = qaoa.build_circuit(np.array([0.4, 0.3]))

    assert _gate_summary(circuit) == [("hadamard", (0,), None), ("rx", (0,), 0.6)]


def test_basic_qaoa_rejects_unsupported_trotter_order():
    with pytest.raises(ValueError, match="trotter_order"):
        BasicQAOA(problem_hamiltonian=Hamiltonian([("X", 1.0)]), p=1, trotter_order=3)


def test_basic_qaoa_energy_uses_gate_level_exact_path_for_hamiltonian():
    hamiltonian = Hamiltonian(n_qubits=1, terms=[("Z", [0], 1.0), ("I", [0], 2.0)])
    qaoa = BasicQAOA(problem_hamiltonian=hamiltonian, p=1)

    assert np.isclose(qaoa.bitstring_energy("0"), 3.0)
    assert np.isclose(qaoa.bitstring_energy("1"), 1.0)
    assert np.isclose(qaoa.energy(np.array([0.0, 0.0])), 2.0, atol=1e-7)


def test_basic_qaoa_gate_level_energy_matches_dense_matrix_compatibility_path():
    hamiltonian = Hamiltonian(n_qubits=2, terms=[("Z", [0], 0.3), ("ZZ", [0, 1], -0.7)])
    matrix = NumpyBackend().to_numpy(hamiltonian.to_matrix(NumpyBackend()))
    params = np.array([0.4, -0.2])

    canonical = BasicQAOA(problem_hamiltonian=hamiltonian, p=1)
    dense = BasicQAOA(problem_hamiltonian=matrix, p=1)

    assert np.isclose(canonical.energy(params), dense.energy(params), atol=1e-7)


def test_basic_qaoa_non_diagonal_single_pauli_energy_matches_dense_matrix_path():
    hamiltonian = Hamiltonian(n_qubits=2, terms=[("XY", [0, 1], 0.7)])
    matrix = NumpyBackend().to_numpy(hamiltonian.to_matrix(NumpyBackend()))
    params = np.array([0.4, -0.2])

    trotterized = BasicQAOA(problem_hamiltonian=hamiltonian, p=1, trotter_steps=3, trotter_order=2)
    dense = BasicQAOA(problem_hamiltonian=matrix, p=1)

    assert np.isclose(trotterized.energy(params), dense.energy(params), atol=1e-6)


def test_basic_qaoa_samples_gate_level_circuit_counts():
    qaoa = BasicQAOA(problem_hamiltonian=Hamiltonian(n_qubits=1, terms=[("I", [0], 2.0)]), p=1)

    counts = qaoa.sample(np.array([0.0, 0.0]), shots=25, seed=123)
    sampled_energy = qaoa.energy(np.array([0.0, 0.0]), shots=25, seed=123)

    assert sum(counts.values()) == 25
    assert set(counts) <= {"0", "1"}
    assert np.isclose(sampled_energy, 2.0)


def test_basic_qaoa_rejects_shots_energy_for_non_diagonal_hamiltonian():
    qaoa = BasicQAOA(problem_hamiltonian=Hamiltonian([("X", 1.0)]), p=1)

    with pytest.raises(ValueError, match="non-diagonal"):
        qaoa.energy(np.array([0.0, 0.0]), shots=16, seed=123)


def test_basic_qaoa_run_accepts_black_box_optimizer():
    pytest.importorskip("scipy")
    qaoa = BasicQAOA(problem_hamiltonian=Hamiltonian(n_qubits=1, terms=[("Z", [0], 1.0)]), p=1)

    result = qaoa.run(
        init_params=np.array([0.1, 0.1]),
        optimizer=NelderMead(options={"maxiter": 80}),
    )

    assert result.optimizer_result is not None
    assert result.parameters.shape == (2,)
    assert len(result.energy_history) >= 1


def test_basic_qaoa_run_learning_rate_alias_matches_lr():
    hamiltonian = Hamiltonian(n_qubits=1, terms=[("Z", [0], 1.0)])
    init_params = np.array([0.1, 0.1])

    via_lr = BasicQAOA(problem_hamiltonian=hamiltonian, p=1).run(
        max_iters=3, lr=0.2, init_params=init_params
    )
    via_learning_rate = BasicQAOA(problem_hamiltonian=hamiltonian, p=1).run(
        max_iters=3, learning_rate=0.2, init_params=init_params
    )

    assert via_learning_rate.energy_history == via_lr.energy_history
    np.testing.assert_array_equal(via_learning_rate.parameters, via_lr.parameters)


def test_basic_qaoa_run_conflicting_lr_and_learning_rate_raises():
    qaoa = BasicQAOA(problem_hamiltonian=Hamiltonian(n_qubits=1, terms=[("Z", [0], 1.0)]), p=1)

    with pytest.raises(ValueError, match="learning_rate"):
        qaoa.run(max_iters=1, lr=0.3, learning_rate=0.2, init_params=np.array([0.1, 0.1]))
