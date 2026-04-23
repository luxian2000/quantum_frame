# 注意：此文件依赖于 basic_torch.txt 转换后的 Ascend 兼容 MindSpore 版本。
# 请将 basic_torch.txt 转换后的代码（所有函数名保持不变，数据类型统一为 complex64，并适配 Ascend）保存为 'basic_mind.py' 并放在此文件同目录下。
from basic_mind import * # 导入所有转换后的函数和常量
import numpy as np # For np.pi
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Tensor, ops, context
from mindspore import value_and_grad # For automatic differentiation

# --- 设置运行环境 ---
# 指定在 Ascend 设备上运行
context.set_context(device_target="Ascend", device_id=0) # device_id 选择具体的 NPU ID

def phi_0(n_qubits=1):
    """
    生成全零态 |0...0> 的量子态向量
    参数:
    n_qubits (int): 量子比特的数量
    返回:
    Tensor: 形状为 (2^n_qubits, 1) 的量子态向量
    """
    # 从单个量子比特的 |0> 态开始
    state = KET_0
    # 通过张量积构造多量子比特的 |0...0> 态
    for _ in range(1, n_qubits):
        state = ops.kron(state, KET_0)
    return state

def expectation(state, hamiltonian):
    """
    计算量子态在给定哈密顿量下的期望值
    Args:
        state (Tensor): 量子态向量或者密度矩阵
                         如果是态矢量，形状应为 (2^N, 1) 或 (2^N,)
                         如果是密度矩阵，形状应为 (2^N, 2^N)
        hamiltonian (Tensor): 哈密顿量矩阵，形状应为 (2^N, 2^N)
    Returns:
        Tensor: 哈密顿量的期望值 <ψ|H|ψ> 或 Tr(ρH) (标量张量)
    """
    # 检查输入维度
    if state.ndim == 1:
        # 状态向量 |ψ>，形状 (2^N,) -> (2^N, 1)
        state = state.expand_dims(1) # Use expand_dims instead of unsqueeze
    if state.ndim == 2 and state.shape[1] == 1:
        # 纯态情况 |ψ>，计算 <ψ|H|ψ>
        # dagger(state) 是 (1, 2^N), hamiltonian 是 (2^N, 2^N), state 是 (2^N, 1)
        # 结果是 (1, 1) 的张量
        expectation = matrix_product(dagger(state), hamiltonian, state)
        # 取 (1,1) 张量中的标量值
        expectation = expectation.squeeze() # Or expectation[0, 0]
    elif state.ndim == 2 and state.shape[0] == state.shape[1]:
        # 混合态情况 ρ，计算 Tr(ρH)
        expectation = ops.trace(ops.matmul(state, hamiltonian))
    else:
        raise ValueError("state必须是态矢量(2^N,)或(2^N, 1)或密度矩阵(2^N, 2^N)")

    # 返回期望值的实部（理论上期望值应该是实数）
    # 如果 H 是厄米算符，<ψ|H|ψ> 或 Tr(ρH) 是实数
    # ops.real 对复数张量返回实部，梯度可以继续传播
    # 如果输入的 state 或 hamiltonian 是可微的，expectation 也是可微的
    # MindSpore 会自动处理复数张量的梯度（Wittrick规则或类似方法）
    return ops.real(expectation)

class Circuit:
    """量子电路类：支持门序构建、拼接和矩阵生成。"""

    def __init__(self, *gates, n_qubits):
        self.gates = list(gates)
        self.n_qubits = n_qubits

    def __add__(self, other):
        """Compose two circuits by concatenating gate order: self followed by other."""
        if not isinstance(other, Circuit):
            return NotImplemented
        if self.n_qubits != other.n_qubits:
            raise ValueError(
                f"Cannot compose circuits with different n_qubits: "
                f"{self.n_qubits} != {other.n_qubits}"
            )
        return Circuit(*self.gates, *other.gates, n_qubits=self.n_qubits)

    def append(self, gate):
        """Append one gate to the current circuit in-place."""
        self.gates.append(gate)
        return self

    def extend(self, *gates):
        """Append multiple gates to the current circuit in-place."""
        self.gates.extend(gates)
        return self

    def unitary(self):
        """Build the full circuit matrix from gate sequence."""
        if not self.gates:
            return IDENTITY_2

        gate_qubits = 0
        for gate in self.gates:
            gate_type = gate['type']
            if gate_type in ['pauli_x', 'X', 'pauli_y', 'Y', 'pauli_z', 'Z', 'hadamard', 'H', 's_gate', 'S', 't_gate', 'T', 'rx', 'ry', 'rz', 'u3', 'u2']:
                gate_qubits = max(gate_qubits, gate['target_qubit'] + 1)
            elif gate_type in ['cnot', 'cx', 'cz', 'cy', 'crx', 'cry', 'crz']:
                gate_qubits = max(gate_qubits, gate['target_qubit'] + 1, max(gate['control_qubits']) + 1)
            elif gate_type == 'toffoli':
                gate_qubits = max(gate_qubits, gate['target_qubit'] + 1, max(gate['control_qubits']) + 1)
            elif gate_type == 'swap':
                gate_qubits = max(gate_qubits, gate['qubit_1'] + 1, gate['qubit_2'] + 1)
            elif gate_type in ['identity', 'I']:
                gate_qubits = max(gate_qubits, gate['n_qubits'])
            else:
                gate_qubits = max(gate_qubits, gate['target_qubit'] + 1)

        if gate_qubits > self.n_qubits:
            raise ValueError(f"量子门的量子比特数量超出总量子比特数: {gate_qubits} > {self.n_qubits}")

        circuit_matrix = identity(self.n_qubits)
        for gate in self.gates:
            gate_matrix = gate_to_matrix(gate, self.n_qubits)
            circuit_matrix = ops.matmul(gate_matrix, circuit_matrix)
        return circuit_matrix

    def matrix(self):
        """Alias of unitary()."""
        return self.unitary()

    def __len__(self):
        return len(self.gates)

    def __iter__(self):
        return iter(self.gates)

    def __repr__(self):
        return f"Circuit(n_qubits={self.n_qubits}, gates={self.gates})"


def circuit(*gates, n_qubits=1):
    """兼容旧接口：内部已完全委托给 Circuit。"""
    return Circuit(*gates, n_qubits=n_qubits).unitary()

SINGLE_QUBIT_GATES = [
    'pauli_x', 'X',
    'pauli_y', 'Y', 
    'pauli_z', 'Z',
    'hadamard', 'H',
    's_gate', 'S',
    't_gate', 'T',
    'rx', 'ry', 'rz',
    'u3', 'u2'
]

TWO_QUBIT_GATES = [
    'cnot', 'cx',
    'cy', 'cz',
    'swap'
]

THREE_QUBIT_GATES = [
    'toffoli', 'ccnot'
]

def pauli_x(target_qubit=0):
    return {'type': 'pauli_x', 'target_qubit': target_qubit}

def pauli_y(target_qubit=0):
    return {'type': 'pauli_y', 'target_qubit': target_qubit}

def pauli_z(target_qubit=0):
    return {'type': 'pauli_z', 'target_qubit': target_qubit}

def hadamard(target_qubit=0):
    return {'type': 'hadamard', 'target_qubit': target_qubit}

def rx(theta, target_qubit=0):
    # 确保 theta 是实数类型，但矩阵元素是复数
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.float64) # 用于计算 cos/sin
    return {'type': 'rx', 'target_qubit': target_qubit, 'parameter': theta}

def ry(theta, target_qubit=0):
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.float64)
    return {'type': 'ry', 'target_qubit': target_qubit, 'parameter': theta}

def rz(theta, target_qubit=0):
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.float64)
    return {'type': 'rz', 'target_qubit': target_qubit, 'parameter': theta}

def s_gate(target_qubit=0):
    return {'type': 's_gate', 'target_qubit': target_qubit}

def t_gate(target_qubit=0):
    return {'type': 't_gate', 'target_qubit': target_qubit}

def cx(target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    return {'type': 'cx', 'target_qubit': target_qubit, 'control_qubits': control_qubits, 'control_states': control_states}

cnot = cx

def cy(target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    return {'type': 'cy', 'target_qubit': target_qubit, 'control_qubits': control_qubits, 'control_states': control_states}

def cz(target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    return {'type': 'cz', 'target_qubit': target_qubit, 'control_qubits': control_qubits, 'control_states': control_states}

def crx(theta, target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.float64)
    return {'type': 'crx', 'target_qubit': target_qubit, 'control_qubits': control_qubits, 'parameter': theta, 
            'control_states': control_states}

def cry(theta, target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.float64)
    return {'type': 'cry', 'target_qubit': target_qubit, 'control_qubits': control_qubits, 'parameter': theta, 
            'control_states': control_states}

def crz(theta, target_qubit, control_qubits, control_states=None):
    if control_states is None:
        control_states = [1] * len(control_qubits)
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.float64)
    return {'type': 'crz', 'target_qubit': target_qubit, 'control_qubits': control_qubits, 'parameter': theta, 
            'control_states': control_states}

def swap(qubit_1=0, qubit_2=1):
    return {'type': 'swap', 'qubit_1': qubit_1, 'qubit_2': qubit_2}

def toffoli(target_qubit=2, control_qubits=[0,1]):
    return {'type': 'toffoli', 'target_qubit': target_qubit, 'control_qubits': control_qubits}

ccnot = toffoli

def u3(theta, phi, lam, target_qubit=0):
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.float64)
    if not isinstance(phi, Tensor):
        phi = Tensor(phi, dtype=ms.float64)
    if not isinstance(lam, Tensor):
        lam = Tensor(lam, dtype=ms.float64)
    return {'type': 'u3', 'target_qubit': target_qubit, 'parameter': [theta, phi, lam]}

def u2(phi, lam, target_qubit=0):
    # Use np.pi
    return u3(Tensor(np.pi, dtype=ms.float64)/2, phi, lam, target_qubit)

