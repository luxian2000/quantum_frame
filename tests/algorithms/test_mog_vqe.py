import numpy as np

from aicir.core.circuit import Circuit, cx
from aicir.qas.MoG_VQE import (
    MOGVQEConfig,
    block_hardware_efficient_ansatz,
    extract_blocks_from_circuit,
    run_mog_vqe,
)


def test_block_hardware_efficient_ansatz_builds_paper_style_blocks():
    individual = block_hardware_efficient_ansatz(n_qubits=3, layers=2, topology="linear")

    assert individual.n_qubits == 3
    assert len(individual.blocks) == 4
    assert individual.cnot_count == 4
    assert individual.parameter_count == 16

    circuit = individual.to_circuit(np.zeros(individual.parameter_count))

    assert circuit.n_qubits == 3
    assert sum(1 for gate in circuit.gates if gate["type"] == "cx") == 4


def test_extract_blocks_from_circuit_uses_existing_cx_topology():
    circuit = Circuit(cx(1, [0]), cx(2, [1]), n_qubits=3)

    individual = extract_blocks_from_circuit(circuit)

    assert individual.n_qubits == 3
    assert [(block.control, block.target) for block in individual.blocks] == [(0, 1), (1, 2)]


def test_run_mog_vqe_returns_modified_circuit_and_pareto_front():
    initial = block_hardware_efficient_ansatz(n_qubits=2, layers=1, topology="linear")
    config = MOGVQEConfig(
        population_size=4,
        generations=2,
        parameter_generations=1,
        parameter_population_size=2,
        seed=7,
        mutation_insert_weight=1.0,
        mutation_delete_weight=0.0,
        mutation_big_weight=0.0,
    )

    def energy(circuit):
        return -float(sum(1 for gate in circuit.gates if gate["type"] == "cx"))

    result = run_mog_vqe(initial, energy_evaluator=energy, config=config)

    assert result.best_circuit.n_qubits == 2
    assert result.best_individual.cnot_count >= initial.cnot_count
    assert result.pareto_front
    assert len(result.history) == config.generations + 1
