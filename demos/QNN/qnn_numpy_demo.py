"""示例：使用 aicir 在 NumpyBackend 上构建和训练量子神经网络 (QNN)。

本脚本演示了如何构建一个参数化量子线路 (QNN)，
使用 NumpyBackend 计算其在特定可观测量上的期望值，
并使用参数平移规则 (psr) 优化参数。
"""

import numpy as np

from aicir import Circuit, NumpyBackend, State, cx, ry
from aicir.qml import psr


def build_qnn(theta: np.ndarray, n_qubits: int) -> Circuit:
    """构建一个简单的参数化量子神经网络 (QNN) 拟设。"""
    gates = []
    # 第 1 层：参数化旋转
    for i in range(n_qubits):
        gates.append(ry(float(theta[i]), i))
    
    # 第 2 层：纠缠
    for i in range(n_qubits - 1):
        gates.append(cx(i, [i + 1]))
        
    # 第 3 层：参数化旋转
    for i in range(n_qubits):
        gates.append(ry(float(theta[i + n_qubits]), i))
        
    return Circuit(*gates, n_qubits=n_qubits)


def main():
    n_qubits = 2
    n_params = n_qubits * 2
    n_steps = 50
    lr = 0.1
    
    # 初始化 NumpyBackend
    backend = NumpyBackend()
    
    # 定义可观测量：作用在第一个量子比特上的 Z
    # Z = [[1, 0], [0, -1]]
    # 因为这是一个双量子比特系统，我们需要作用在第一个量子比特上的 Z：Z ⊗ I
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    I = np.eye(2, dtype=np.complex128)
    ZI = np.kron(Z, I)
    
    def objective(theta: np.ndarray) -> float:
        """目标函数：QNN 的期望值。"""
        circuit = build_qnn(theta, n_qubits)
        # 演化零态并获取态向量
        sv = (
            State.zero_state(n_qubits, backend)
            .evolve(circuit.unitary(backend=backend))
            .to_numpy()
            .reshape(-1)
        )
        # 计算期望值：<psi | ZI | psi>
        return float(np.real(sv.conj() @ ZI @ sv))

    # 随机初始化参数
    rng = np.random.default_rng(42)
    theta = rng.uniform(-np.pi, np.pi, n_params)
    
    print(f"=== 基于 NumpyBackend 的 QNN 训练示例 ===")
    print(f"初始参数: {theta}")
    print(f"初始期望值: {objective(theta):.4f}")
    
    print("\n训练中...")
    for step in range(n_steps):
        # 使用参数平移规则计算梯度
        grad = psr(objective, theta)
        
        # 更新参数
        theta = theta - lr * grad
        
        if (step + 1) % 10 == 0:
            print(f"第 {step + 1:2d} 步 | 期望值: {objective(theta):.4f}")
            
    print("\n=== 训练完成 ===")
    print(f"最终参数: {theta}")
    print(f"最终期望值: {objective(theta):.4f}")


if __name__ == "__main__":
    main()
