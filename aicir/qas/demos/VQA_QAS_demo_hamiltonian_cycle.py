"""Demo: supernet for the Hamiltonian-cycle problem (Lucas 2014, Eq. 56).

We encode "does graph G admit a Hamiltonian cycle?" as the ground state of a
diagonal Ising Hamiltonian, then let :mod:`aicir.qas.supernet` search a circuit
that prepares that ground state. The optimal computational-basis state is
decoded back into the cycle ordering.

Instance: the 4-vertex cycle graph C4 (a square). Using the paper's spin
reduction (a cycle can always be rotated to start at vertex 0) the problem fits
in (N-1)^2 = 9 qubits. Expected solution energy is 0, recovering the cycle
0 -> 1 -> 2 -> 3 -> 0 (or an equivalent rotation/reflection).

Run:
    python -m aicir.qas.demos.VQA_QAS_demo_hamiltonian_cycle
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

# 4-vertex square: edges of C4.
N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
# Pinned reduction fixes vertex 0 at position 0.
PINNED = {(0, 0): 1}


def main() -> None:
    print("=" * 72)
    print("supernet Demo: Hamiltonian Cycle (Lucas 2014, Eq. 56)")
    print("=" * 72)
    print(f"Graph: N={N_VERTICES} vertices, edges={EDGES} (cycle graph C4)")

    print("\n[1/4] Building the Ising Hamiltonian from the QUBO...")
    qubo = ordering_qubo(N_VERTICES, EDGES, cyclic=True, pin_first_vertex=True)
    hamiltonian, var_to_qubit = qubo_to_hamiltonian(qubo)
    n_qubits = len(var_to_qubit)
    print(f"  binary variables / qubits: {n_qubits} (= (N-1)^2 after pinning vertex 0)")
    print(f"  Hamiltonian terms: {len(hamiltonian)} (diagonal I/Z Ising model)")

    classical_energy, classical_assignment = qubo.brute_force_minimum()
    classical_order = assignment_to_order({**classical_assignment, **PINNED}, N_VERTICES)
    print(f"  brute-force optimum energy: {classical_energy:.6f}")
    print(f"  brute-force cycle: {format_order(classical_order, cyclic=True)}")

    print("\n[2/4] Searching a circuit with supernet (task='vqe')...")
    solution = solve_ground_state_qas(
        hamiltonian,
        n_qubits,
        restarts=4,
        layers=2,
        supernet_steps=120,
        ranking_num=20,
        finetune_steps=150,
        base_seed=0,
    )

    print("\n[3/4] Decoding the optimised state...")
    assignment = bitstring_to_assignment(
        solution.best_index, var_to_qubit, n_qubits, pinned=PINNED
    )
    order = assignment_to_order(assignment, N_VERTICES)
    valid = is_valid_ordering(order, EDGES, cyclic=True)
    print(f"  best supernet energy: {solution.energy:+.6f} (winning seed {solution.seed})")
    print(f"  measured ground-state probability: {float(solution.probabilities[solution.best_index]):.4f}")
    print(f"  decoded cycle: {format_order(order, cyclic=True)}")
    print(f"  valid Hamiltonian cycle: {valid}")
    print(f"  selected CNOT count: {solution.result.final_metrics['selected_cnot_count']}")
    print("  selected ansatz circuit:")
    print(solution.result.final_metrics["selected_circuit_ascii"])

    print("\n[4/4] Exporting OpenQASM 3.0...")
    out_path = Path(__file__).parent / "vqa_qas_hamiltonian_cycle_circuit.qasm"
    save_circuit_qasm3(solution.result.best_circuit, out_path)
    print(f"  QASM 3.0 saved to: {out_path}")


if __name__ == "__main__":
    main()
