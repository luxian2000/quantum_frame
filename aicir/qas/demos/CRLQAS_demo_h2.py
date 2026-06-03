"""Demo: CRLQAS on a small-molecule Hamiltonian (H2, 2-qubit model)."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from aicir.qas.CRLQAS import AdamSPSAConfig, CRLQASConfig, train_crlqas
from aicir.channel.operators import Hamiltonian
from aicir.core.io.qasm import save_circuit_qasm3


def build_h2_hamiltonian() -> Hamiltonian:
    """Build a common 2-qubit effective Hamiltonian for H2.

    Coefficients correspond to a standard mapped H2 instance often used in VQE demos.
    """
    hamiltonian = Hamiltonian(n_qubits=2)
    hamiltonian.term(-1.052373245772859, {"I": [0, 1]})
    hamiltonian.term(0.39793742484318045, {"Z": [0]})
    hamiltonian.term(-0.39793742484318045, {"Z": [1]})
    hamiltonian.term(-0.01128010425623538, {"Z": [0, 1]})
    hamiltonian.term(0.18093119978423156, {"X": [0, 1]})
    return hamiltonian


def main() -> None:
    print("=" * 72)
    print("CRLQAS Demo: H2 2-Qubit Hamiltonian")
    print("=" * 72)

    h2_hamiltonian = build_h2_hamiltonian()
    config = CRLQASConfig(
        max_episodes=300,
        n_act=10,
        batch_size=32,
        replay_capacity=4000,
        train_interval=5,
        target_update_interval=100,
        curriculum_initial_threshold=0.2,
        curriculum_mu=-1.5,
        curriculum_delta=0.1,
        curriculum_kappa=80.0,
        curriculum_reset_patience=30,
        random_halt_p=0.5,
        chemical_accuracy=1.6e-3,
        adam_spsa=AdamSPSAConfig(
            iterations=20,
            a=0.06,
            c=0.10,
        ),
        seed=42,
        log_interval=50,
    )

    print("[1/3] 训练 CRLQAS...")
    result = train_crlqas(hamiltonian=h2_hamiltonian, config=config)

    print("[2/3] 输出训练结果...")
    print(f"最优能量估�? {result.minimum_energy:.8f} Ha")
    print(f"最终课程阈�? {result.curriculum_threshold:.8f}")
    print(f"最优线路门�? {len(result.circuit.gates)}")
    print("门序列（�?20 个）:")
    for index, gate in enumerate(result.circuit.gates[:20]):
        print(f"  [{index:02d}] {gate}")
    if len(result.circuit.gates) > 20:
        print(f"  ... 其余 {len(result.circuit.gates) - 20} 个门")

    print("[3/3] 导出 OpenQASM 3.0...")
    out_path = Path(__file__).parent / "crlqas_h2_circuit.qasm"
    save_circuit_qasm3(result.circuit, out_path)
    print(f"QASM 3.0 已保存到: {out_path}")


if __name__ == "__main__":
    main()
