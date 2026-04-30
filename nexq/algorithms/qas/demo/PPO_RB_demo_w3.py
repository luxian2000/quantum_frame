"""nexq/algorithms/qas/PPO_RB_demo_w4.py

演示：使用 PPO_RB.py 中的强化学习方法制备 3 量子比特 W 态。

W 态：|W3⟩ = (|001⟩ + |010⟩ + |100⟩) / sqrt(3)
目标输入为密度矩阵 rho_target = |W3⟩⟨W3|。

说明：
- 为提高可达性，动作空间包含 X + Ry + CX + CRY。
- 受控门统一使用 control_states=[1]，可直接导出 QASM。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch

from nexq.algorithms.qas.PPO_RB import PPORollbackConfig, ppo_rb_qas
from nexq.channel.backends.numpy_backend import NumpyBackend
from nexq.core.circuit import Circuit
from nexq.core.gates import gate_to_matrix
from nexq.core.io.qasm import circuit_to_qasm
from nexq.core.state import State


def build_w3_density() -> np.ndarray:
    """构造 3 比特 W 态目标密度矩阵。"""
    n = 3
    w = np.zeros((1 << n, 1), dtype=np.complex64)
    amp = 1.0 / math.sqrt(n)
    for q in range(n):
        w[1 << q, 0] = amp  # indices: 1, 2, 4
    return w @ w.conj().T


def circuit_density(circuit: Circuit) -> np.ndarray:
    """将线路作用在 |000⟩ 后得到输出密度矩阵。"""
    backend = NumpyBackend()
    state = State.zero_state(circuit.n_qubits, backend=backend)
    for gate in circuit.gates:
        gm = gate_to_matrix(gate, cir_qubits=circuit.n_qubits, backend=backend)
        state = state.evolve(gm)
    psi = state.to_numpy().reshape(-1, 1).astype(np.complex64)
    return psi @ psi.conj().T


def fidelity_pure_density(rho_target: np.ndarray, rho_pred: np.ndarray) -> float:
    """纯态目标下的重叠保真度 Tr(rho_target · rho_pred)。"""
    return float(np.real(np.trace(rho_target @ rho_pred)))


def build_w3_action_gates(include_cry: bool = True) -> List[Dict[str, object]]:
    """构建 3 比特 W 态专用动作集合。"""
    n = 3

    # W_n 递归常用角度：2*asin(1/sqrt(k))，k=2..n
    w_angles = [2.0 * math.asin(math.sqrt(1.0 / k)) for k in range(2, n + 1)]
    # 额外加入经验有效角度，提升搜索覆盖。
    ry_angles = sorted(set(w_angles + [math.pi / 3.0]))
    cry_angles = sorted(set(w_angles + [math.pi / 3.0]))

    gates: List[Dict[str, object]] = []

    # X 门（3 个）
    for q in range(n):
        gates.append({"type": "pauli_x", "target_qubit": q})

    # Ry 门
    for q in range(n):
        for theta in ry_angles:
            gates.append({"type": "ry", "parameter": theta, "target_qubit": q})

    # 标准 CNOT ctrl=|1⟩（全连接有向对）
    for ctrl in range(n):
        for tgt in range(n):
            if ctrl != tgt:
                gates.append({"type": "cx", "target_qubit": tgt, "control_qubits": [ctrl], "control_states": [1]})

    if include_cry:
        # CRY(ctrl=|1⟩)（全连接有向对 × 多角度）
        for ctrl in range(n):
            for tgt in range(n):
                if ctrl != tgt:
                    for theta in cry_angles:
                        gates.append(
                            {
                                "type": "cry",
                                "parameter": theta,
                                "target_qubit": tgt,
                                "control_qubits": [ctrl],
                                "control_states": [1],
                            }
                        )

    return gates


def decompose_non_one_controls(circuit: Circuit) -> Circuit:
    """兼容函数：当前动作空间已只用 |1> 控制态，直接返回副本。"""
    return Circuit(*[dict(g) for g in circuit.gates], n_qubits=circuit.n_qubits, backend=circuit.backend)


def main() -> None:
    print("=" * 68)
    print("PPO-RB QAS Demo: Prepare 3-Qubit W State")
    print("=" * 68)

    rho_target = build_w3_density()
    # 分阶段动作空间：先在较小空间找到可行结构，再引入 CRY 提升上限。
    w3_action_gates_coarse = build_w3_action_gates(include_cry=False)
    w3_action_gates_full = build_w3_action_gates(include_cry=True)
    print(f"动作集合(coarse): {len(w3_action_gates_coarse)} 个门")
    print(f"动作集合(full):   {len(w3_action_gates_full)} 个门")

    # 课程学习：逐步提升目标阈值，并在阶段间热启动策略参数。
    # (epsilon, episodes, terminal_bonus, gate_penalty, seed, action_gates)
    curriculum: List[Tuple[float, int, float, float, int, List[Dict[str, object]]]] = [
        (0.80, 1200, 5.0, 0.0005, 101, w3_action_gates_coarse),
        (0.90, 1600, 7.0, 0.0005, 202, w3_action_gates_coarse),
        (0.95, 2200, 9.0, 0.0010, 303, w3_action_gates_full),
        (0.98, 2600, 12.0, 0.0010, 404, w3_action_gates_full),
    ]
    attempts_per_stage = 4

    print("[1/4] 开始课程学习训练 PPO-RB...")
    theta: Optional[Dict[str, torch.Tensor]] = None
    best_fidelity = -1.0
    best_circuit: Optional[Circuit] = None

    for stage_idx, (epsilon_stage, episodes, bonus, gate_penalty, seed, stage_action_gates) in enumerate(
        curriculum, start=1
    ):
        print(
            f"  阶段 {stage_idx}/{len(curriculum)}: "
            f"epsilon={epsilon_stage:.2f}, episodes={episodes}, terminal_bonus={bonus:.1f}, "
            f"actions={len(stage_action_gates)}"
        )
        stage_best_fid = -1.0
        stage_best_theta: Optional[Dict[str, torch.Tensor]] = None
        stage_best_circuit: Optional[Circuit] = None

        for attempt in range(attempts_per_stage):
            config = PPORollbackConfig(
                # PPO-RB 核心超参数
                learning_rate=0.002,
                gamma=0.99,
                epsilon_clip=0.2,
                epoch_num=4,
                rollback_alpha=-0.3,
                kl_threshold=0.03,
                value_loss_coef=0.5,
                entropy_coef=0.01,
                # 3 比特 W 态训练配置
                action_gates=stage_action_gates,
                terminal_bonus=bonus,
                gate_penalty=gate_penalty,
                episode_num=episodes,
                max_steps_per_episode=12,
                update_timestep=128,
                hidden_dim=128,
                seed=seed + attempt,
                log_interval=max(episodes // 8, 1),
                init_theta=theta,  # 阶段热启动
            )
            cand_theta, cand_circuit = ppo_rb_qas(rho_target, epsilon=epsilon_stage, config=config)
            cand_fid = fidelity_pure_density(rho_target, circuit_density(cand_circuit))
            print(f"    attempt {attempt + 1}/{attempts_per_stage}: fidelity={cand_fid:.6f}, gates={len(cand_circuit.gates)}")

            if cand_fid > stage_best_fid:
                stage_best_fid = cand_fid
                stage_best_theta = cand_theta
                stage_best_circuit = cand_circuit

        if stage_best_theta is not None:
            # 仅接受不退化结果；若退化则保留前一阶段参数。
            if stage_best_fid >= best_fidelity:
                theta = stage_best_theta
                best_fidelity = stage_best_fid
                best_circuit = stage_best_circuit
                print(f"    阶段接收: fidelity={stage_best_fid:.6f}")
            else:
                print(
                    f"    阶段回滚: stage_best={stage_best_fid:.6f} < global_best={best_fidelity:.6f}"
                )

    if theta is None or best_circuit is None:
        raise RuntimeError("课程训练未返回有效结果")

    circuit = best_circuit

    print("[2/4] 训练完成，输出策略参数摘要...")
    print(f"参数张量数量: {len(theta)}")
    total_params = sum(int(v.numel()) for v in theta.values())  # type: ignore[union-attr]
    print(f"参数总数: {total_params}")

    print("[3/4] 评估最终线路与 W3 目标保真度...")
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
    qasm_circuit = decompose_non_one_controls(circuit)
    qasm_str = circuit_to_qasm(qasm_circuit, version="3.0")
    out_path = Path(__file__).parent / "ppo_rb_w3_circuit.qasm"
    out_path.write_text(qasm_str, encoding="utf-8")
    print(f"QASM 已保存: {out_path}")

    print()
    print("Demo 完成。")


if __name__ == "__main__":
    main()
