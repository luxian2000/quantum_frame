"""
简单脚本：构建并在终端打印 3 比特 QFT（量子傅里叶变换）电路的 ASCII 图。

用法：
    python tests/print_qft_3.py
"""

import math
import sys
from pathlib import Path

# 确保直接运行此脚本时能导入项目中的 nexq 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexq import Circuit, hadamard, crz, swap


def build_qft_3():
    """构建 3 比特 QFT 电路（顺序为 q0, q1, q2）。"""
    # QFT(3) 局部实现：
    # H q0; CR(pi/2) q1->q0; CR(pi/4) q2->q0;
    # H q1; CR(pi/2) q2->q1;
    # H q2; swap q0 q2
    return Circuit(
        hadamard(0),
        crz(math.pi / 2, 0, [1]),
        crz(math.pi / 4, 0, [2]),
        hadamard(1),
        crz(math.pi / 2, 1, [2]),
        hadamard(2),
        swap(0, 2),
        n_qubits=3,
    )


if __name__ == "__main__":
    circ = build_qft_3()
    print("3-qubit QFT circuit (ASCII):")
    circ.show()
