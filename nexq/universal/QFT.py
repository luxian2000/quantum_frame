import numpy as np

# 假设本仓库量子态为 numpy 数组或兼容对象

def qft(state):
    """
    量子傅里叶变换（QFT）
    输入: state (np.ndarray) - 量子态向量，长度为2^n
    输出: QFT变换后的量子态 (np.ndarray)
    """
    state = np.asarray(state)
    n = int(np.log2(state.size))
    N = 2 ** n
    qft_matrix = np.zeros((N, N), dtype=complex)
    omega = np.exp(2j * np.pi / N)
    for i in range(N):
        for j in range(N):
            qft_matrix[i, j] = omega ** (i * j) / np.sqrt(N)
    return qft_matrix @ state

# 可选：提供一个类接口
class QFT:
    @staticmethod
    def apply(state):
        return qft(state)
