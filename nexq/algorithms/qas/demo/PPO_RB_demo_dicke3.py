"""Demo: prepare a 3-qubit Dicke state with PPO-RB QAS.

This example uses `ppo_rb_qas` from `PPO_RB.py` to search a circuit that
prepares the Dicke state D(3,2):

    |D(3,2)> = (|011> + |101> + |110>) / sqrt(3)

The target is represented as a density matrix rho_target = |D><D|.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Make sure project root is available when running this file directly.
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nexq.algorithms.qas.PPO_RB import PPORollbackConfig, ppo_rb_qas
from nexq.channel.backends.numpy_backend import NumpyBackend
from nexq.core.circuit import Circuit
from nexq.core.gates import gate_to_matrix
from nexq.core.io.qasm import circuit_to_qasm
from nexq.core.state import State


def build_dicke_density(n_qubits: int = 3, excitations: int = 2) -> np.ndarray:
    """Build a pure Dicke state density matrix rho = |D(n,k)><D(n,k)|."""
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    if excitations < 0 or excitations > n_qubits:
        raise ValueError("excitations must be in [0, n_qubits]")

    dim = 1 << n_qubits
    vec = np.zeros((dim, 1), dtype=np.complex64)

    basis_indices = [idx for idx in range(dim) if int(bin(idx).count("1")) == excitations]
    if not basis_indices:
        raise ValueError("No basis states found for the requested Dicke state")

    amp = 1.0 / math.sqrt(len(basis_indices))
    for idx in basis_indices:
        vec[idx, 0] = amp
    return vec @ vec.conj().T


def circuit_density(circuit: Circuit) -> np.ndarray:
    """Apply circuit to |0...0> and return output density matrix."""
    backend = NumpyBackend()
    state = State.zero_state(circuit.n_qubits, backend=backend)
    for gate in circuit.gates:
        gm = gate_to_matrix(gate, cir_qubits=circuit.n_qubits, backend=backend)
        state = state.evolve(gm)
    psi = state.to_numpy().reshape(-1, 1).astype(np.complex64)
    return psi @ psi.conj().T


def fidelity_pure_density(rho_target: np.ndarray, rho_pred: np.ndarray) -> float:
    """Compute overlap fidelity Tr(rho_target @ rho_pred) for pure-state target."""
    return float(np.real(np.trace(rho_target @ rho_pred)))


def build_dicke3_action_gates() -> List[Dict[str, object]]:
    """Action set tuned for 3-qubit Dicke state search."""
    n = 3
    ry_angles = sorted(set([math.pi / 6.0, math.pi / 4.0, math.pi / 3.0, math.pi / 2.0]))
    cry_angles = sorted(set([2.0 * math.asin(math.sqrt(1.0 / k)) for k in (2, 3)] + [math.pi / 3.0]))

    gates: List[Dict[str, object]] = []

    for q in range(n):
        gates.append({"type": "pauli_x", "target_qubit": q})
        gates.append({"type": "hadamard", "target_qubit": q})

    for q in range(n):
        for theta in ry_angles:
            gates.append({"type": "ry", "parameter": theta, "target_qubit": q})

    for ctrl in range(n):
        for tgt in range(n):
            if ctrl == tgt:
                continue
            gates.append(
                {
                    "type": "cx",
                    "target_qubit": tgt,
                    "control_qubits": [ctrl],
                    "control_states": [1],
                }
            )
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


def main() -> None:
    print("=" * 68)
    print("PPO-RB QAS Demo: Prepare 3-Qubit Dicke State D(3,2)")
    print("=" * 68)

    rho_target = build_dicke_density(n_qubits=3, excitations=2)
    action_gates = build_dicke3_action_gates()
    print(f"action set size: {len(action_gates)}")

    # Curriculum training improves stability for sparse-target state synthesis.
    curriculum: List[Tuple[float, int, float, int]] = [
        (0.85, 1200, 6.0, 111),
        (0.93, 1600, 8.0, 222),
        (0.98, 2200, 10.0, 333),
    ]
    attempts_per_stage = 3

    print("[1/4] train PPO-RB with curriculum...")
    theta: Optional[Dict[str, torch.Tensor]] = None
    best_fidelity = -1.0
    best_circuit: Optional[Circuit] = None

    for stage_idx, (epsilon_stage, episodes, bonus, seed) in enumerate(curriculum, start=1):
        print(
            f"  stage {stage_idx}/{len(curriculum)}: "
            f"epsilon={epsilon_stage:.2f}, episodes={episodes}, terminal_bonus={bonus:.1f}"
        )

        stage_best_fid = -1.0
        stage_best_theta: Optional[Dict[str, torch.Tensor]] = None
        stage_best_circuit: Optional[Circuit] = None

        for attempt in range(attempts_per_stage):
            config = PPORollbackConfig(
                learning_rate=0.002,
                gamma=0.99,
                epsilon_clip=0.2,
                epoch_num=4,
                rollback_alpha=-0.3,
                kl_threshold=0.03,
                value_loss_coef=0.5,
                entropy_coef=0.01,
                action_gates=action_gates,
                terminal_bonus=bonus,
                gate_penalty=0.001,
                episode_num=episodes,
                max_steps_per_episode=10,
                update_timestep=128,
                hidden_dim=128,
                seed=seed + attempt,
                log_interval=max(episodes // 10, 1),
                init_theta=theta,
            )

            cand_theta, cand_circuit = ppo_rb_qas(rho_target, epsilon=epsilon_stage, config=config)
            cand_fid = fidelity_pure_density(rho_target, circuit_density(cand_circuit))
            print(
                f"    attempt {attempt + 1}/{attempts_per_stage}: "
                f"fidelity={cand_fid:.6f}, gates={len(cand_circuit.gates)}"
            )

            if cand_fid > stage_best_fid:
                stage_best_fid = cand_fid
                stage_best_theta = cand_theta
                stage_best_circuit = cand_circuit

        if stage_best_theta is not None and stage_best_circuit is not None:
            if stage_best_fid >= best_fidelity:
                theta = stage_best_theta
                best_fidelity = stage_best_fid
                best_circuit = stage_best_circuit
                print(f"    stage accepted: fidelity={stage_best_fid:.6f}")
            else:
                print(
                    f"    stage rollback: stage_best={stage_best_fid:.6f} "
                    f"< global_best={best_fidelity:.6f}"
                )

    if theta is None or best_circuit is None:
        raise RuntimeError("Training did not produce a valid Dicke(3,2) circuit")

    print("[2/4] training finished, summarize learned parameters...")
    print(f"parameter tensors: {len(theta)}")
    total_params = sum(int(v.numel()) for v in theta.values())
    print(f"total parameter count: {total_params}")

    print("[3/4] evaluate final circuit...")
    fidelity = fidelity_pure_density(rho_target, circuit_density(best_circuit))
    print(f"n_qubits: {best_circuit.n_qubits}")
    print(f"gate_count: {len(best_circuit.gates)}")
    print(f"final fidelity: {fidelity:.6f}")

    print("gate sequence (first 20):")
    for idx, gate in enumerate(best_circuit.gates[:20]):
        print(f"  [{idx:02d}] {gate}")
    if len(best_circuit.gates) > 20:
        print(f"  ... and {len(best_circuit.gates) - 20} more gates")

    print("[4/4] export QASM 3.0...")
    qasm_text = circuit_to_qasm(best_circuit, version="3.0")
    out_path = Path(__file__).parent / "ppo_rb_dicke3_circuit.qasm"
    out_path.write_text(qasm_text, encoding="utf-8")
    print(f"QASM saved: {out_path}")

    print("\nDemo done.")


if __name__ == "__main__":
    main()
