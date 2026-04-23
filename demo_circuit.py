#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_circuit.py

量子线路示例：使用仓库中的 `Circuit` 接口构造 Bell 态并打印态向量与测量概率。

用法:
    python demo_circuit.py
"""
import math
import numpy as np
import torch

# Circuit 层接口
from Circuit import (
    Circuit,
    hadamard,
    cnot,
    phi_0,
    rx,
    ry,
    rz,
    expectation,
)

# Core 层工具（偏迹、门矩阵、dagger）
from Core import partial_trace, gate_to_matrix, dagger


def pretty_state(psi):
    """返回态矢量和对应的测量概率向量（按计算基）。"""
    vec = psi.squeeze()
    probs = torch.abs(vec) ** 2
    probs = probs / probs.sum()
    return vec, probs


def sample_measurements(psi, shots=1024):
    """对态 `psi` 进行重复测量抽样，返回每个基态的频率向量。"""
    _, probs = pretty_state(psi)
    probs_cpu = probs.cpu().to(torch.float32)
    counts = torch.bincount(torch.multinomial(probs_cpu, num_samples=shots, replacement=True), minlength=int(probs_cpu.numel()))
    freqs = counts.float() / float(shots)
    return freqs


def run_bell_demo(n_qubits=2, shots=1024):
    """构造 Bell 态并打印矩阵、态矢量与测量分布。"""
    circ = Circuit(hadamard(0), cnot(1, [0]), n_qubits=n_qubits)
    print("Circuit:", circ)

    U = circ.unitary()
    print(f"Unitary matrix ({U.shape[0]}x{U.shape[1]}):")
    print(U.cpu().numpy())

    psi0 = phi_0(n_qubits)
    print("Initial state |00>:", psi0.cpu().numpy().reshape(-1))

    psi = torch.matmul(U, psi0)
    vec, probs = pretty_state(psi)

    print("Final state vector:", vec.cpu().numpy().reshape(-1))
    print("Probabilities:", probs.cpu().numpy().reshape(-1))

    print("Non-zero basis states:")
    for idx, p in enumerate(probs):
        if p.item() > 1e-8:
            print(f"  |{idx:0{n_qubits}b}>  prob={p.item():.6f}")

    # 采样测量
    freqs = sample_measurements(psi, shots=shots)
    print(f"Measurement sampling (shots={shots}):")
    for idx, f in enumerate(freqs.tolist()):
        if f > 1e-6:
            print(f"  |{idx:0{n_qubits}b}>  freq={f:.4f}")


def run_reduced_density_demo(psi, n_qubits=2):
    """演示对纯态做密度矩阵并对一个子系统做偏迹（reduced density matrix）。"""
    # 确保列向量形式
    psi_col = psi if psi.dim() == 2 and psi.shape[1] == 1 else psi.unsqueeze(1)
    rho = torch.matmul(psi_col, dagger(psi_col))
    print("Full density matrix (rho):")
    print(rho.cpu().numpy())

    # 保留第0号量子比特
    rho_red = partial_trace(rho, keep=[0], n_qubits=n_qubits)
    print("Reduced density matrix (keep=[0]):")
    print(rho_red.cpu().numpy())

    # 特征值（谱）用于判断纯态/混合态
    eigs = np.linalg.eigvals(rho_red.cpu().numpy())
    print("Eigenvalues of reduced density:", eigs)


def run_parameterized_demo(angle=math.pi / 4, n_qubits=2):
    """参数化门示例：计算某个可观测量的期望值。"""
    circ = Circuit(ry(angle, 0), hadamard(1), cnot(1, [0]), n_qubits=n_qubits)
    U = circ.unitary()
    psi0 = phi_0(n_qubits)
    psi = torch.matmul(U, psi0)

    # 计算 Z 在第0个比特上的伸展算符，并计算期望值
    Z_full = gate_to_matrix({'type': 'pauli_z', 'target_qubit': 0}, cir_qubits=n_qubits)
    exp_val = expectation(psi, Z_full)
    print(f"Expectation <Z_0> with angle={angle:.4f}: {exp_val}")


def main():
    # 运行一系列示例
    n_qubits = 2
    print("--- Bell state demo ---")
    run_bell_demo(n_qubits=n_qubits, shots=2048)

    # 复用上面的 Bell 态进行偏迹演示（重新构造以保证一致）
    circ = Circuit(hadamard(0), cnot(1, [0]), n_qubits=n_qubits)
    U = circ.unitary()
    psi = torch.matmul(U, phi_0(n_qubits))
    print("--- Reduced density demo ---")
    run_reduced_density_demo(psi, n_qubits=n_qubits)

    print("--- Parameterized gate demo ---")
    run_parameterized_demo(angle=math.pi / 3, n_qubits=n_qubits)


if __name__ == '__main__':
    main()
