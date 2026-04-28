"""
量子架构搜索环境 - 工具函数 (基于nexq库)

提供默认的量子门、观测量和目标态等工具函数。
"""

from typing import Dict, List, Optional
import numpy as np


def get_default_gates(n_qubits: int) -> List[Dict]:
    """
    获取默认的量子门集合。
    
    每个量子比特上包括：RZ, X, Y, Z, H 单量子门；
    以及与下一个量子比特的CNOT双量子门。
    
    Args:
        n_qubits: 量子比特数量
        
    Returns:
        门字典列表
    """
    gates = []
    for idx in range(n_qubits):
        next_qubit = (idx + 1) % n_qubits
        
        # 单量子门
        gates.append({"type": "rz", "target_qubit": idx, "parameter": np.pi / 4})
        gates.append({"type": "pauli_x", "target_qubit": idx})
        gates.append({"type": "pauli_y", "target_qubit": idx})
        gates.append({"type": "pauli_z", "target_qubit": idx})
        gates.append({"type": "hadamard", "target_qubit": idx})
        
        # 双量子门 (CNOT: 当控制比特为1时，作用NOT到目标比特)
        gates.append({
            "type": "cnot",
            "target_qubit": next_qubit,
            "control_qubits": [idx],
            "control_states": [1]  # 当控制比特处于|1⟩状态时触发
        })
    
    return gates


def get_default_observables(n_qubits: int) -> List[Dict]:
    """
    获取默认的观测量集合。
    
    对每个量子比特，观测X、Y、Z泡利算子。
    
    Args:
        n_qubits: 量子比特数量
        
    Returns:
        观测量字典列表
    """
    observables = []
    for qubit in range(n_qubits):
        observables.append({"type": "pauli_x", "target_qubit": qubit})
        observables.append({"type": "pauli_y", "target_qubit": qubit})
        observables.append({"type": "pauli_z", "target_qubit": qubit})
    
    return observables


def get_bell_state() -> np.ndarray:
    """
    获取Bell态（2量子比特纠缠态）。
    
    |Φ+⟩ = (|00⟩ + |11⟩) / √2
    
    Returns:
        状态向量 (复数数组，长度为4)
    """
    target = np.zeros(2**2, dtype=complex)
    target[0] = 1.0 / np.sqrt(2) + 0.0j
    target[-1] = 1.0 / np.sqrt(2) + 0.0j
    return target


def get_ghz_state(n_qubits: int = 3) -> np.ndarray:
    """
    获取GHZ态（多量子比特纠缠态）。
    
    |GHZ⟩ = (|00...0⟩ + |11...1⟩) / √2
    
    Args:
        n_qubits: 量子比特数量
        
    Returns:
        状态向量 (复数数组，长度为2^n_qubits)
    """
    target = np.zeros(2**n_qubits, dtype=complex)
    target[0] = 1.0 / np.sqrt(2) + 0.0j
    target[-1] = 1.0 / np.sqrt(2) + 0.0j
    return target


def get_w_state(n_qubits: int) -> np.ndarray:
    """
    获取W态（多量子比特叠加态）。
    
    |W⟩ = (|100...0⟩ + |010...0⟩ + ... + |00...01⟩) / √n
    
    Args:
        n_qubits: 量子比特数量
        
    Returns:
        状态向量 (复数数组，长度为2^n_qubits)
    """
    target = np.zeros(2**n_qubits, dtype=complex)
    # 在只有一个比特为1的位置设置幅度
    for i in range(n_qubits):
        target[1 << i] = 1.0 / np.sqrt(n_qubits) + 0.0j
    return target
