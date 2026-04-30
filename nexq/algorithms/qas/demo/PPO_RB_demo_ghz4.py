"""nexq/algorithms/qas/PPO_RB_demo.py

演示：使用 Trust Region-based PPO with Rollback 制备 4 量子比特 GHZ 态。

GHZ 态：|GHZ> = (|0000> + |1111>) / sqrt(2)
目标输入为密度矩阵 rho_target = |GHZ><GHZ|。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from nexq.algorithms.qas.PPO_RB import PPORollbackConfig, ppo_rb_qas
from nexq.channel.backends.numpy_backend import NumpyBackend
from nexq.core.gates import gate_to_matrix
from nexq.core.io.qasm import circuit_to_qasm
from nexq.core.state import State


def build_ghz_density_4qubits() -> np.ndarray:
    """构造 4 比特 GHZ 目标密度矩阵。"""
    ghz = np.zeros((16, 1), dtype=np.complex64)
    ghz[0, 0] = 1.0 / np.sqrt(2.0)   # |0000>
    ghz[15, 0] = 1.0 / np.sqrt(2.0)  # |1111>
    return ghz @ ghz.conj().T


def circuit_density(circuit) -> np.ndarray:
    """将线路作用在 |0000> 后得到输出密度矩阵。"""
    backend = NumpyBackend()
    state = State.zero_state(circuit.n_qubits, backend=backend)
    for gate in circuit.gates:
        gm = gate_to_matrix(gate, cir_qubits=circuit.n_qubits, backend=backend)
        state = state.evolve(gm)
    psi = state.to_numpy().reshape(-1, 1).astype(np.complex64)
    return psi @ psi.conj().T


def fidelity_pure_density(rho_target: np.ndarray, rho_pred: np.ndarray) -> float:
    """纯态目标下的重叠保真度 Tr(rho_target * rho_pred)。"""
    return float(np.real(np.trace(rho_target @ rho_pred)))


def main() -> None:
    print("=" * 68)
    print("PPO-RB QAS Demo: Prepare 4-Qubit GHZ State")
    print("=" * 68)

    rho_target = build_ghz_density_4qubits()
    epsilon = 0.95

    # GHZ 经典线路: H(q0) -> CX(q0,q1) -> CX(q0,q2) -> CX(q0,q3)
    # 动作集合保留所有 H 与全部有序 CX 对（共 4+12=16 个），缩小搜索空间
    n = 4
    ghz_action_gates = [
        {"type": "hadamard", "target_qubit": q} for q in range(n)
    ] + [
        {"type": "cx", "target_qubit": tgt, "control_qubits": [ctrl], "control_states": [1]}
        for ctrl in range(n) for tgt in range(n) if ctrl != tgt
    ]

    config = PPORollbackConfig(
        # 与伪代码一致的核心超参数
        learning_rate=0.002,
        gamma=0.99,
        epsilon_clip=0.2,
        epoch_num=4,
        rollback_alpha=-0.3,
        kl_threshold=0.03,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        # 高成功率配置
        action_gates=ghz_action_gates,  # 限定 H + 全连接 CX
        terminal_bonus=3.0,             # 到达 GHZ 时额外奖励
        gate_penalty=0.005,             # 每步惩罚适当减小
        episode_num=1500,               # 4 比特搜索空间更大，适当增加轮数
        max_steps_per_episode=10,       # GHZ 只需 4 个门，预留余量
        update_timestep=128,
        hidden_dim=128,
        seed=42,
        log_interval=150,               # 每 150 个 episode 打印训练进度
    )

    print("[1/4] 开始训练 PPO-RB...")
    theta, circuit = ppo_rb_qas(rho_target, epsilon=epsilon, config=config)

    print("[2/4] 训练完成，输出策略参数摘要...")
    print(f"参数张量数量: {len(theta)}")
    total_params = sum(int(v.numel()) for v in theta.values())
    print(f"参数总数: {total_params}")

    print("[3/4] 评估最终线路与 GHZ 目标保真度...")
    rho_pred = circuit_density(circuit)
    fidelity = fidelity_pure_density(rho_target, rho_pred)

    print(f"线路量子比特数: {circuit.n_qubits}")
    print(f"线路门数: {len(circuit.gates)}")
    print(f"最终保真度: {fidelity:.6f}")

    print("门序列（前 20 个）:")
    for idx, gate in enumerate(circuit.gates[:20]):
        print(f"  [{idx:02d}] {gate}")
    if len(circuit.gates) > 20:
        print(f"  ... 其余 {len(circuit.gates) - 20} 个门")

    print("[4/4] 导出 QASM 3.0...")
    qasm_text = circuit_to_qasm(circuit, version="3.0")
    out_path = Path(__file__).parent / "ppo_rb_ghz4_circuit.qasm"
    out_path.write_text(qasm_text, encoding="utf-8")

    print(f"QASM 已保存: {out_path}")
    print("\nDemo 完成。")


if __name__ == "__main__":
    main()
