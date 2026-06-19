"""Demo: MoG-VQE on a small-molecule Hamiltonian (H2, 2-qubit model)."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from aicir.qas.algorithms.MoG_VQE import (
    MOGVQEConfig,
    block_hardware_efficient_ansatz,
    run_mog_vqe,
)
from aicir.core.operators import Hamiltonian
from aicir.core.io.qasm import save_circuit_qasm3


def build_h2_hamiltonian() -> Hamiltonian:
    """Build a common 2-qubit effective Hamiltonian for H2."""
    return Hamiltonian(n_qubits=2, terms=[
        ("II", -1.052373245772859),
        ("ZI", 0.39793742484318045),
        ("IZ", -0.39793742484318045),
        ("ZZ", -0.01128010425623538),
        ("XX", 0.18093119978423156),
    ])


def main() -> None:
    print("=" * 72)
    print("MoG-VQE Demo: H2 2-Qubit Hamiltonian")
    print("=" * 72)

    h2_hamiltonian = build_h2_hamiltonian()
    
    # 构造初始拓扑 (block-based HEA)
    initial_ansatz = block_hardware_efficient_ansatz(n_qubits=2, layers=1)

    # 算法参数配置 (轻量化演示配置)
    config = MOGVQEConfig(
        population_size=8,
        generations=10,
        mutation_insert_weight=2.0,
        mutation_delete_weight=1.0,
        mutation_big_weight=0.25,
        min_blocks=1,
        max_blocks=5,
        parameter_optimizer="separable_es",
        parameter_generations=10,
        parameter_population_size=8,
        seed=42,
    )

    print("[1/3] 训练 MoG-VQE...")
    result = run_mog_vqe(
        initial_ansatz=initial_ansatz,
        hamiltonian=h2_hamiltonian,
        config=config,
    )

    print("[2/3] 输出训练结果...")
    print(f"最优能量估计: {result.best_energy:.8f} Ha")
    print(f"最优线路块(block)数: {len(result.best_individual.blocks)}")
    print(f"最优线路 CNOT 数: {result.best_individual.cnot_count}")
    print(f"最优线路总门数: {len(result.best_circuit.gates)}")
    
    print("\n门序列（前 20 个）:")
    for index, gate in enumerate(result.best_circuit.gates[:20]):
        print(f"  [{index:02d}] {gate}")
    if len(result.best_circuit.gates) > 20:
        print(f"  ... 其余 {len(result.best_circuit.gates) - 20} 个门")
        
    print("\nPareto 前沿 (Energy, CNOT Count):")
    for cand in result.pareto_front:
        print(f"  Energy: {cand.energy:.6f}, CNOTs: {cand.cnot_count}")

    print("\n[3/3] 导出 OpenQASM 3.0...")
    out_path = Path(__file__).parent / "mog_vqe_h2_circuit.qasm"
    save_circuit_qasm3(result.best_circuit, out_path)
    print(f"QASM 3.0 已保存到: {out_path}")


if __name__ == "__main__":
    main()
