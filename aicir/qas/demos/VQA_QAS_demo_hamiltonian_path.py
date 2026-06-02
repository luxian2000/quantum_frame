"""Demo: VQA_QAS for the Hamiltonian-path problem (Lucas 2014, Eq. 56 variant).

The Hamiltonian-path formulation is the cycle formulation without the
wrap-around transition (we do not require the first and last vertices to be
joined). We encode it as the ground state of a diagonal Ising Hamiltonian and
let :mod:`aicir.qas.VQA_QAS` search a circuit preparing that ground state.

Instance: the 3-vertex path graph with edges {0-1, 1-2}. It admits the
Hamiltonian path 0 -> 1 -> 2 (energy 0) but NO Hamiltonian cycle, because
vertices 0 and 2 are not adjacent -- the demo also prints the cycle-formulation
optimum to make the contrast explicit. Uses N^2 = 9 qubits (no pinning, since a
path endpoint is not fixed in advance).

Run:
    python -m aicir.qas.demos.VQA_QAS_demo_hamiltonian_path
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from aicir.core.io.qasm import save_circuit_qasm3
from aicir.qas.demos._np_ising_utils import (
    assignment_to_order,
    bitstring_to_assignment,
    format_order,
    is_valid_ordering,
    ordering_qubo,
    qubo_to_hamiltonian,
    solve_ground_state_qas,
)

# 3-vertex path graph: a Hamiltonian path exists, a Hamiltonian cycle does not.
N_VERTICES = 3
EDGES = [(0, 1), (1, 2)]


def main() -> None:
    print("=" * 72)
    print("VQA_QAS Demo: Hamiltonian Path (Lucas 2014, Eq. 56 without wrap-around)")
    print("=" * 72)
    print(f"Graph: N={N_VERTICES} vertices, edges={EDGES} (path graph)")

    print("\n[1/4] Building the Ising Hamiltonian from the QUBO...")
    qubo = ordering_qubo(N_VERTICES, EDGES, cyclic=False, pin_first_vertex=False)
    hamiltonian, var_to_qubit = qubo_to_hamiltonian(qubo)
    n_qubits = len(var_to_qubit)
    print(f"  binary variables / qubits: {n_qubits} (= N^2)")
    print(f"  Hamiltonian terms: {len(hamiltonian)} (diagonal I/Z Ising model)")

    classical_energy, classical_assignment = qubo.brute_force_minimum()
    classical_order = assignment_to_order(classical_assignment, N_VERTICES)
    print(f"  brute-force optimum energy: {classical_energy:.6f}")
    print(f"  brute-force path: {format_order(classical_order, cyclic=False)}")

    # Contrast: the cycle formulation has no zero-energy solution on this graph.
    cycle_qubo = ordering_qubo(N_VERTICES, EDGES, cyclic=True, pin_first_vertex=False)
    cycle_energy, _ = cycle_qubo.brute_force_minimum()
    print(f"  (cycle formulation optimum energy: {cycle_energy:.6f} > 0 => no Hamiltonian cycle)")

    print("\n[2/4] Searching a circuit with VQA_QAS (task='vqe')...")
    solution = solve_ground_state_qas(
        hamiltonian,
        n_qubits,
        restarts=8,
        layers=2,
        supernet_steps=150,
        ranking_num=30,
        finetune_steps=250,
        base_seed=0,
    )

    print("\n[3/4] Decoding the optimised state...")
    assignment = bitstring_to_assignment(solution.best_index, var_to_qubit, n_qubits)
    order = assignment_to_order(assignment, N_VERTICES)
    valid = is_valid_ordering(order, EDGES, cyclic=False)
    print(f"  best VQA_QAS energy: {solution.energy:+.6f} (winning seed {solution.seed})")
    print(f"  measured ground-state probability: {float(solution.probabilities[solution.best_index]):.4f}")
    print(f"  decoded path: {format_order(order, cyclic=False)}")
    print(f"  valid Hamiltonian path: {valid}")
    print(f"  selected CNOT count: {solution.result.final_metrics['selected_cnot_count']}")
    print("  selected ansatz circuit:")
    print(solution.result.final_metrics["selected_circuit_ascii"])

    print("\n[4/4] Exporting OpenQASM 3.0...")
    out_path = Path(__file__).parent / "vqa_qas_hamiltonian_path_circuit.qasm"
    save_circuit_qasm3(solution.result.best_circuit, out_path)
    print(f"  QASM 3.0 saved to: {out_path}")


if __name__ == "__main__":
    main()
