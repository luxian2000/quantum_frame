#!/usr/bin/env python3
"""
打印 demo 目录中保存的电路（JSON）内容及简单信息。
"""
import sys
from pathlib import Path
import numpy as np

# 确保能导入上层的包（nexq、demo 中的模块）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexq.circuit.io.json_io import load_circuit_json
from nexq.channel.backends.numpy_backend import NumpyBackend


def print_circuit(path: str):
    c = load_circuit_json(path)
    print(f"=== {path} ===")
    print(f"n_qubits: {c.n_qubits}")
    print(f"num_gates: {len(c.gates)}")
    print("Gates:")
    for i, g in enumerate(c.gates):
        print(f"  {i:02d}: {g}")
    try:
        U = c.unitary(backend=NumpyBackend())
        if hasattr(U, 'numpy'):
            U = U.numpy()
        else:
            U = np.asarray(U)
        print(f"Unitary shape: {U.shape}")
    except Exception as e:
        print(f"无法计算幺正矩阵: {e}")
    print()


if __name__ == '__main__':
    for fname in ['canonical_ghz.json', 'best_random_ghz.json']:
        p = Path(__file__).parent / fname
        if p.exists():
            print_circuit(str(p))
        else:
            print(f"文件不存在: {p}")
