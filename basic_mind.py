import numpy as np
import math
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Tensor, ops, context
from mindspore import value_and_grad # For automatic differentiation

# --- 设置运行环境 ---
# 指定在 Ascend 设备上运行
context.set_context(device_target="Ascend", device_id=0) # device_id 选择具体的 NPU ID

def matrix_product(*matrices):
    """
    对多个矩阵进行连乘运算 (从左到右依次相乘)
    参数:
    *matrices: 可变数量的矩阵参数，每个矩阵都应该是MindSpore Tensor类型
    返回:
    Tensor: 所有输入矩阵从左到右连乘的结果
    异常:
    ValueError: 当没有输入矩阵或输入矩阵少于2个时抛出
    TypeError: 当输入参数不是Tensor类型时抛出
    """
    # 检查输入
    if len(matrices) == 0:
        raise ValueError("至少需要输入一个矩阵")
    if len(matrices) == 1:
        return matrices[0]
    # 验证所有输入都是Tensor类型
    for i, matrix in enumerate(matrices):
        if not isinstance(matrix, Tensor):
            raise TypeError(f"第{i+1}个参数必须是MindSpore Tensor类型，但得到了{type(matrix)}")
    # 依次进行矩阵乘法
    result = matrices[0]
    for i in range(1, len(matrices)):
        result = ops.matmul(result, matrices[i])
    return result

def tensor_product(*matrices):
    """
    计算多个矩阵的张量积（Kronecker积）
    参数:
    *matrices: 可变数量的矩阵参数，每个矩阵都应该是MindSpore Tensor类型
    返回:
    Tensor: 所有输入矩阵的张量积结果
    异常:
    ValueError: 当没有输入矩阵时抛出
    TypeError: 当输入参数不是Tensor类型时抛出
    """
    # 检查输入
    if len(matrices) == 0:
        raise ValueError("至少需要输入一个矩阵")
    # 验证所有输入都是Tensor类型
    for i, matrix in enumerate(matrices):
        if not isinstance(matrix, Tensor):
            raise TypeError(f"第{i+1}个参数必须是MindSpore Tensor类型，但得到了{type(matrix)}")
    # 如果只有一个矩阵，直接返回
    if len(matrices) == 1:
        return matrices[0]
    # 依次计算张量积
    result = matrices[0]
    for i in range(1, len(matrices)):
        result = ops.kron(result, matrices[i])
    return result

def dagger(matrix):
    """
    计算矩阵的共轭转置（dagger操作）
    参数:
    matrix: 输入的复数矩阵 (Tensor类型)
    返回:
    Tensor: 输入矩阵的共轭转置
    """
    # 先转置矩阵，再取共轭
    # MindSpore transpose 需要指定轴
    return ops.conj(ops.transpose(matrix, (1, 0))) # Assuming 2D matrix, swap rows and cols


def partial_trace(rho, keep, n_qubits=None):
    """对密度矩阵执行偏迹，返回保留子系统的约化密度矩阵。"""
    if not isinstance(rho, Tensor):
        raise TypeError(f"rho必须是MindSpore Tensor类型，但得到了{type(rho)}")
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho必须是方阵，形状为(2^N, 2^N)")

    dim = rho.shape[0]
    if n_qubits is None:
        n_qubits = int(math.log2(dim))
        if 2 ** n_qubits != dim:
            raise ValueError("无法从rho的维度推断量子比特数，请显式传入n_qubits")

    if isinstance(keep, int):
        keep = [keep]
    if not isinstance(keep, (list, tuple)):
        raise TypeError("keep必须是int、list或tuple")

    keep = sorted(set(int(k) for k in keep))
    if any(k < 0 or k >= n_qubits for k in keep):
        raise ValueError("keep中的量子比特索引超出范围")

    trace_out = [i for i in range(n_qubits) if i not in keep]
    if not trace_out:
        return rho

    reshaped = rho.reshape([2] * n_qubits + [2] * n_qubits)
    perm = keep + trace_out + [k + n_qubits for k in keep] + [t + n_qubits for t in trace_out]
    permuted = ops.transpose(reshaped, tuple(perm))

    d_keep = 2 ** len(keep)
    d_trace = 2 ** len(trace_out)
    permuted = permuted.reshape(d_keep, d_trace, d_keep, d_trace)
    reduced = mnp.einsum('abcb->ac', permuted)
    return reduced

# 量子比特基态 |0> 和 |1>
KET_0 = Tensor([[1.+0.j], [0.+0.j]], dtype=ms.complex64)  # |0>
KET_1 = Tensor([[0.+0.j], [1.+0.j]], dtype=ms.complex64)  # |1>
# --- 修改 KET_PLUS 和 KET_MINUS 的定义 ---
# 方案1: 使用 numpy 计算 sqrt(2) 的实数值，然后创建复数张量
sqrt2_val = np.sqrt(2.0)
sqrt2_tensor = Tensor(sqrt2_val, dtype=ms.complex64)
KET_PLUS = (KET_0 + KET_1) / sqrt2_tensor   # |+> = (|0> + |1>)/sqrt(2)
KET_MINUS = (KET_0 - KET_1) / sqrt2_tensor
# 对应的 bra 态 (行向量，ket的共轭转置)
BRA_0 = dagger(KET_0)  # <0|
BRA_1 = dagger(KET_1)  # <1|
BRA_PLUS = dagger(KET_PLUS)   # <+|
BRA_MINUS = dagger(KET_MINUS)
DENSITY_0 = Tensor([[1., 0.],
                    [0., 0.]], dtype=ms.complex64)  # |0><0|
DENSITY_1 = Tensor([[0., 0.],
                    [0., 1.]], dtype=ms.complex64)  # |1><1|
IDENTITY_2 = Tensor([[1., 0.],
                     [0., 1.]], dtype=ms.complex64)  # 单位矩阵 I

def identity(n_qubits=1):
    """Identity gate for n qubits."""
    dim = 2 ** n_qubits
    return ops.eye(dim, dtype=ms.complex64)

def _pauli_x(target_qubit=0):
    pauli_x = Tensor([[0.+0.j, 1.+0.j], [1.+0.j, 0.+0.j]], dtype=ms.complex64)
    if target_qubit > 0:
        pauli_x = ops.kron(identity(target_qubit), pauli_x)
    return pauli_x

def _pauli_y(target_qubit=0):
    pauli_y = Tensor([[0.+0.j, -1j], [1j, 0.+0.j]], dtype=ms.complex64)
    if target_qubit > 0:
        pauli_y = ops.kron(identity(target_qubit), pauli_y)
    return pauli_y

def _pauli_z(target_qubit=0):
    pauli_z = Tensor([[1.+0.j, 0.+0.j], [0.+0.j, -1.+0.j]], dtype=ms.complex64)
    if target_qubit > 0:
        pauli_z = ops.kron(identity(target_qubit), pauli_z)
    return pauli_z

def _hadamard(target_qubit=0):
    sqrt2_inv = 1.0 / mnp.sqrt(Tensor(2.0, dtype=ms.complex64))
    hadamard = Tensor([[sqrt2_inv, sqrt2_inv], [sqrt2_inv, -sqrt2_inv]], dtype=ms.complex64)
    if target_qubit > 0:
        hadamard = ops.kron(identity(target_qubit), hadamard)
    return hadamard

def _rx(theta, target_qubit=0):
    """Single qubit rotation around the X axis by angle theta."""
    # 确保 theta 是一个 Tensor
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.complex64)
    cos = ops.cos(theta / 2)
    sin = ops.sin(theta / 2)
    # 将 sin 和 cos 转换为 complex64 类型
    cos_c = ops.cast(cos, ms.complex64)
    sin_c = ops.cast(sin, ms.complex64)
    # 创建 -i*sin
    neg_i_sin = sin_c * Tensor(-1j, dtype=ms.complex64)
    # 修正：使用 ops.stack 构建 2x2 矩阵
    row1 = ops.stack([cos_c, neg_i_sin], axis=-1)  # Shape: [..., 2]
    row2 = ops.stack([neg_i_sin, cos_c], axis=-1)  # Shape: [..., 2]
    rx_matrix = ops.stack([row1, row2], axis=-2)   # Shape: [..., 2, 2]
    # ops.stack 会保留输入张量的批次维度（如果 theta 是标量，则无批次维度）
    # 如果 theta 是标量 [1.5]，则 rx_matrix 形状为 [1, 2, 2]
    # 如果 theta 是标量 1.5，则 rx_matrix 形状为 [2, 2]
    # 如果 theta 是标量，可能需要 squeeze 掉多余的维度
    if rx_matrix.ndim > 2 and rx_matrix.shape[0] == 1:
        rx_matrix = ops.squeeze(rx_matrix, axis=0)

    if target_qubit > 0:
        # 假设 identity(target_qubit) 返回一个合适的单位矩阵
        identity_block = identity(target_qubit) # 需要确保这个函数返回正确形状的复数单位矩阵
        rx_matrix = ops.kron(identity_block, rx_matrix)
    return rx_matrix

def _ry(theta, target_qubit=0):
    """Single qubit rotation around the Y axis by angle theta."""
    # 确保 theta 是一个 Tensor
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.complex64)

    cos = ops.cos(theta / 2)
    sin = ops.sin(theta / 2)
    # 将 sin 和 cos 转换为 complex64 类型
    cos_c = ops.cast(cos, ms.complex64)
    sin_c = ops.cast(sin, ms.complex64)

    # 创建 -sin(θ/2) 和 sin(θ/2)
    neg_sin_c = -sin_c
    sin_c_pos = sin_c # Just to make the matrix construction clear

    # 修正：使用 ops.stack 构建 2x2 矩阵
    # ry(θ) = [[cos(θ/2), -sin(θ/2)],
    #          [sin(θ/2),  cos(θ/2)]]
    row1 = ops.stack([cos_c, neg_sin_c], axis=-1)  # [cos(θ/2), -sin(θ/2)]
    row2 = ops.stack([sin_c_pos, cos_c], axis=-1)  # [sin(θ/2),  cos(θ/2)]
    ry_matrix = ops.stack([row1, row2], axis=-2)   # [[row1], [row2]]

    # 如果 theta 是标量， ops.stack 会保留维度，例如输入标量，输出形状为 [2, 2]
    # 如果 theta 是形状为 [N] 的张量，输出形状为 [N, 2, 2]
    # 如果 theta 是形状为 [1] 的张量，输出形状为 [1, 2, 2]，可能需要 squeeze
    # 但通常保留批次维度是更通用的做法，除非明确需要标量矩阵。
    # 如果输入是标量 1.5，则 theta 变成 [1.5] (shape [1])，导致 ry_matrix 为 [1, 2, 2]
    # 为了处理输入为标量的情况，可以检查并 squeeze
    if ry_matrix.ndim > 2 and ry_matrix.shape[0] == 1:
        ry_matrix = ops.squeeze(ry_matrix, axis=0)

    if target_qubit > 0:
        identity_block = identity(target_qubit)
        ry_matrix = ops.kron(identity_block, ry_matrix)

    return ry_matrix

def _rz(theta, target_qubit=0):
    """Single qubit rotation around the Z axis by angle theta."""
    # 确保 theta 是一个 Tensor
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.complex64)
    # 计算复数指数
    exp_neg = ops.exp(-1j * theta / 2) # Shape: [shape_of_theta]
    exp_pos = ops.exp(1j * theta / 2)  # Shape: [shape_of_theta]

    # 使用 ops.stack 构建 2x2 矩阵
    # rz(θ) = [[exp(-1j*θ/2), 0],
    #          [0,          exp(1j*θ/2)]]
    # 创建零元素（复数零）
    zero_elem = Tensor(0.+0.j, dtype=ms.complex64)
    # 如果 theta 是标量或有批次维度，zero_elem 需要广播到相同的形状
    # ops.stack 会自动处理广播
    row1 = ops.stack([exp_neg, zero_elem], axis=-1)  # Shape: [..., 2]
    row2 = ops.stack([zero_elem, exp_pos], axis=-1)  # Shape: [..., 2]
    rz_matrix = ops.stack([row1, row2], axis=-2)     # Shape: [..., 2, 2]

    # 如果 theta 是标量 (零维张量)，ops.stack 会正确处理
    # 如果 theta 是形状为 [N] 的一维张量，则输出是 [N, 2, 2]
    # 如果 theta 是形状为 [1] 的一维张量，则输出是 [1, 2, 2]，可能需要 squeeze
    # 为了处理输入为标量（零维张量）的情况，可以检查并 squeeze
    # 但是，对于零维输入 theta，ops.exp(theta/2) 仍然是零维，ops.stack([...], dim=-1) 会创建 [..., 2] 形状
    # ops.stack([[...], [...]], dim=-2) 会创建 [..., 2, 2] 形状
    # 如果输入是零维标量 theta，输出 rz_matrix 应该是 [2, 2]
    # 如果输入是 [1] 形状的 theta，输出 rz_matrix 是 [1, 2, 2]
    # 所以，如果第一个维度是 1，则 squeeze
    if rz_matrix.ndim > 2 and rz_matrix.shape[0] == 1:
        rz_matrix = ops.squeeze(rz_matrix, axis=0)

    if target_qubit > 0:
        identity_block = identity(target_qubit)
        rz_matrix = ops.kron(identity_block, rz_matrix)

    return rz_matrix

def _s_gate(target_qubit=0):
    s_gate = Tensor([[1.+0.j, 0.+0.j],
                     [0.+0.j, 1j]], dtype=ms.complex64)
    # 使用 ops.stack 重构矩阵（虽然元素是常数）
    row1 = ops.stack([Tensor(1.+0.j, dtype=ms.complex64), Tensor(0.+0.j, dtype=ms.complex64)], axis=-1)
    row2 = ops.stack([Tensor(0.+0.j, dtype=ms.complex64), Tensor(1j, dtype=ms.complex64)], axis=-1)
    s_gate_matrix = ops.stack([row1, row2], axis=-2)

    if target_qubit > 0:
        s_gate_matrix = ops.kron(identity(target_qubit), s_gate_matrix)
    return s_gate_matrix

def _t_gate(target_qubit=0):
    t_gate_val = ops.exp(1j * Tensor(np.pi, dtype=ms.complex64) / 4) # Use np.pi
    t_gate = Tensor([[1.+0.j, 0.+0.j],
                     [0.+0.j, t_gate_val]], dtype=ms.complex64)
    # 使用 ops.stack 重构矩阵
    one_elem = Tensor(1.+0.j, dtype=ms.complex64)
    zero_elem = Tensor(0.+0.j, dtype=ms.complex64)
    row1 = ops.stack([one_elem, zero_elem], axis=-1)
    row2 = ops.stack([zero_elem, t_gate_val], axis=-1)
    t_gate_matrix = ops.stack([row1, row2], axis=-2)

    if target_qubit > 0:
        t_gate_matrix = ops.kron(identity(target_qubit), t_gate_matrix)
    return t_gate_matrix

def _cx(target_qubit, control_qubits, control_states):
    """
    受控X门（Controlled X gate）
    参数:
    target_qubit: 目标量子比特的索引
    control_qubits: 控制量子比特的索引列表
    control_states: 控制量子比特的状态列表（0或1）
    返回:
    受控X门的矩阵表示
    """
    # 首先获取Pauli_X矩阵
    px_matrix = _pauli_x() - IDENTITY_2
    # 确定所需量子比特总数
    all_qubits = [target_qubit] + control_qubits
    n_qubits = max(all_qubits) + 1
    # 检查control_states长度是否与control_qubits匹配
    if len(control_states) != len(control_qubits):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")
    # 构建张量积矩阵
    matrices = []
    for qubit_index in range(n_qubits):
        if qubit_index == target_qubit:
            matrices.append(px_matrix)
        elif qubit_index in control_qubits:
            control_index = control_qubits.index(qubit_index)
            if control_states[control_index] == 1:
                matrices.append(DENSITY_1)
            else:
                matrices.append(DENSITY_0)
        else:
            matrices.append(IDENTITY_2)
    # 计算所有矩阵的张量积
    result_matrix = matrices[0]
    for i in range(1, len(matrices)):
        result_matrix = ops.kron(result_matrix, matrices[i])
    result_matrix = identity(n_qubits) + result_matrix
    return result_matrix

def _cy(target_qubit, control_qubits, control_states):
    """
    受控Y门（Controlled Y gate）
    参数:
    target_qubit: 目标量子比特的索引
    control_qubits: 控制量子比特的索引列表
    control_states: 控制量子比特的状态列表（0或1）
    返回:
    受控Y门的矩阵表示
    """
    # 首先获取Pauli_Y矩阵
    py_matrix = _pauli_y() - IDENTITY_2
    # 确定所需量子比特总数
    all_qubits = [target_qubit] + control_qubits
    n_qubits = max(all_qubits) + 1
    # 检查control_states长度是否与control_qubits匹配
    if len(control_states) != len(control_qubits):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")
    # 构建张量积矩阵
    matrices = []
    for qubit_index in range(n_qubits):
        if qubit_index == target_qubit:
            matrices.append(py_matrix)
        elif qubit_index in control_qubits:
            control_index = control_qubits.index(qubit_index)
            if control_states[control_index] == 1:
                matrices.append(DENSITY_1)
            else:
                matrices.append(DENSITY_0)
        else:
            matrices.append(IDENTITY_2)
    # 计算所有矩阵的张量积
    result_matrix = matrices[0]
    for i in range(1, len(matrices)):
        result_matrix = ops.kron(result_matrix, matrices[i])
    result_matrix = identity(n_qubits) + result_matrix
    return result_matrix

def _cz(target_qubit, control_qubits, control_states):
    """
    受控Z门（Controlled Z gate）
    参数:
    target_qubit: 目标量子比特的索引
    control_qubits: 控制量子比特的索引列表
    control_states: 控制量子比特的状态列表（0或1）
    返回:
    受控Y门的矩阵表示
    """
    # 首先获取Pauli_Z矩阵
    pz_matrix = _pauli_z() - IDENTITY_2
    # 确定所需量子比特总数
    all_qubits = [target_qubit] + control_qubits
    n_qubits = max(all_qubits) + 1
    # 检查control_states长度是否与control_qubits匹配
    if len(control_states) != len(control_qubits):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")
    # 构建张量积矩阵
    matrices = []
    for qubit_index in range(n_qubits):
        if qubit_index == target_qubit:
            matrices.append(pz_matrix)
        elif qubit_index in control_qubits:
            control_index = control_qubits.index(qubit_index)
            if control_states[control_index] == 1:
                matrices.append(DENSITY_1)
            else:
                matrices.append(DENSITY_0)
        else:
            matrices.append(IDENTITY_2)
    # 计算所有矩阵的张量积
    result_matrix = matrices[0]
    for i in range(1, len(matrices)):
        result_matrix = ops.kron(result_matrix, matrices[i])
    result_matrix = identity(n_qubits) + result_matrix
    return result_matrix

def _crx(theta, target_qubit, control_qubits, control_states):
    """
    受控rx门（Controlled rx gate）
    参数:
    theta: rx旋转角度
    target_qubit: 目标量子比特的索引
    control_qubits: 控制量子比特的索引列表
    control_states: 控制量子比特的状态列表（0或1）
    返回:
    受控rx门的矩阵表示
    """
    # 首先获取rx门矩阵 (注意：这里减去单位矩阵)
    rx_base = _rx(theta) - IDENTITY_2
    # 确定所需量子比特总数
    all_qubits = [target_qubit] + control_qubits
    n_qubits = max(all_qubits) + 1
    # 检查control_states长度是否与control_qubits匹配
    if len(control_states) != len(control_qubits):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")
    # 构建张量积矩阵
    matrices = []
    for qubit_index in range(n_qubits):
        if qubit_index == target_qubit:
            matrices.append(rx_base)
        elif qubit_index in control_qubits:
            control_index = control_qubits.index(qubit_index)
            if control_states[control_index] == 1:
                matrices.append(DENSITY_1)
            else:
                matrices.append(DENSITY_0)
        else:
            matrices.append(IDENTITY_2)
    # 计算所有矩阵的张量积
    result_matrix = matrices[0]
    for i in range(1, len(matrices)):
        result_matrix = ops.kron(result_matrix, matrices[i])
    # 加上单位矩阵得到完整的受控门 (注意：这里加上单位矩阵)
    result_matrix = identity(n_qubits) + result_matrix
    return result_matrix

def _cry(theta, target_qubit, control_qubits, control_states):
    """
    受控ry门（Controlled ry gate）
    参数:
    theta: ry旋转角度
    target_qubit: 目标量子比特的索引
    control_qubits: 控制量子比特的索引列表
    control_states: 控制量子比特的状态列表（0或1）
    返回:
    受控ry门的矩阵表示
    """
    # 首先获取ry门矩阵 (注意：这里减去单位矩阵)
    ry_base = _ry(theta) - IDENTITY_2
    # 确定所需量子比特总数
    all_qubits = [target_qubit] + control_qubits
    n_qubits = max(all_qubits) + 1
    # 检查control_states长度是否与control_qubits匹配
    if len(control_states) != len(control_qubits):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")
    # 构建张量积矩阵
    matrices = []
    for qubit_index in range(n_qubits):
        if qubit_index == target_qubit:
            matrices.append(ry_base)
        elif qubit_index in control_qubits:
            control_index = control_qubits.index(qubit_index)
            if control_states[control_index] == 1:
                matrices.append(DENSITY_1)
            else:
                matrices.append(DENSITY_0)
        else:
            matrices.append(IDENTITY_2)
    # 计算所有矩阵的张量积
    result_matrix = matrices[0]
    for i in range(1, len(matrices)):
        result_matrix = ops.kron(result_matrix, matrices[i])
    # 加上单位矩阵得到完整的受控门 (注意：这里加上单位矩阵)
    result_matrix = identity(n_qubits) + result_matrix
    return result_matrix

def _crz(theta, target_qubit, control_qubits, control_states):
    """
    受控rz门（Controlled rz gate）
    参数:
    theta: rz旋转角度
    target_qubit: 目标量子比特的索引
    control_qubits: 控制量子比特的索引列表
    control_states: 控制量子比特的状态列表（0或1）
    返回:
    受控ry门的矩阵表示
    """
    # 首先获取rz门矩阵 (注意：这里减去单位矩阵)
    rz_base = _rz(theta) - IDENTITY_2
    # 确定所需量子比特总数
    all_qubits = [target_qubit] + control_qubits
    n_qubits = max(all_qubits) + 1
    # 检查control_states长度是否与control_qubits匹配
    if len(control_states) != len(control_qubits):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")
    # 构建张量积矩阵
    matrices = []
    for qubit_index in range(n_qubits):
        if qubit_index == target_qubit:
            matrices.append(rz_base)
        elif qubit_index in control_qubits:
            control_index = control_qubits.index(qubit_index)
            if control_states[control_index] == 1:
                matrices.append(DENSITY_1)
            else:
                matrices.append(DENSITY_0)
        else:
            matrices.append(IDENTITY_2)
    # 计算所有矩阵的张量积
    result_matrix = matrices[0]
    for i in range(1, len(matrices)):
        result_matrix = ops.kron(result_matrix, matrices[i])
    # 加上单位矩阵得到完整的受控门 (注意：这里加上单位矩阵)
    result_matrix = identity(n_qubits) + result_matrix
    return result_matrix

def _swap(qubit_1=0, qubit_2=1):
    return matrix_product(_cx(qubit_1, [qubit_2], [1]), _cx(qubit_2, [qubit_1], [1]), _cx(qubit_1, [qubit_2], [1]))

def _toffoli(target_qubit=2, control_qubits=[0,1]):
    n_qubits = max(target_qubit, max(control_qubits)) + 1
    matrices_0 = [IDENTITY_2] * n_qubits
    matrices_0[control_qubits[0]] = DENSITY_1
    matrices_0[control_qubits[1]] = DENSITY_1
    matrices_0[target_qubit] = _pauli_x()
    result_0 = matrices_0[0]
    for i in range(1, n_qubits):
        result_0 = ops.kron(result_0, matrices_0[i])
    matrices_1 = [IDENTITY_2] * n_qubits
    matrices_1[control_qubits[0]] = DENSITY_0
    result_1 = matrices_1[0]
    for i in range(1, n_qubits):
        result_1 = ops.kron(result_1, matrices_1[i])
    matrices_2 = [IDENTITY_2] * n_qubits
    matrices_2[control_qubits[0]] = DENSITY_1
    matrices_2[control_qubits[1]] = DENSITY_0
    result_2 = matrices_2[0]
    for i in range(1, n_qubits):
        result_2 = ops.kron(result_2, matrices_2[i])
    return result_0 + result_1 + result_2

def _u3(theta, phi, lam, target_qubit=0):
    """General single-qubit rotation gate u3."""
    # 确保参数是 Tensor
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.complex64)
    if not isinstance(phi, Tensor):
        phi = Tensor(phi, dtype=ms.complex64)
    if not isinstance(lam, Tensor):
        lam = Tensor(lam, dtype=ms.complex64)

    cos = ops.cos(theta / 2)
    sin = ops.sin(theta / 2)
    exp_iphi = ops.exp(1j * phi)
    exp_ilam = ops.exp(1j * lam)
    exp_iphi_lam = ops.exp(1j * (phi + lam))

    # 使用 ops.stack 构建 u3 矩阵
    row1 = ops.stack([cos, -exp_ilam * sin], axis=-1)  # [cos, -exp(-i*lam) * sin]
    row2 = ops.stack([exp_iphi * sin, exp_iphi_lam * cos], axis=-1)  # [exp(i*phi) * sin, exp(i*(phi+lam)) * cos]
    u3_matrix = ops.stack([row1, row2], axis=-2)

    if target_qubit > 0:
        u3_matrix = ops.kron(identity(target_qubit), u3_matrix)
    return u3_matrix

def _u2(phi, lam, target_qubit=0):
    """Single-qubit rotation gate u2."""
    # Use np.pi
    return _u3(Tensor(np.pi, dtype=ms.complex64)/2, phi, lam, target_qubit)

def _rzz(theta):
    """
    RZZ门（控制Z旋转门）
    作用在两个量子比特上，实现条件相位旋转
    参数:
    theta: 旋转角度
    返回:
    Tensor: 4x4的RZZ门矩阵
    """
    # 确保 theta 是一个 Tensor
    if not isinstance(theta, Tensor):
        theta = Tensor(theta, dtype=ms.complex64)

    # RZZ门的矩阵形式:
    # [[exp(-1j*theta/2), 0, 0, 0],
    #  [0, exp(1j*theta/2), 0, 0],
    #  [0, 0, exp(1j*theta/2), 0],
    #  [0, 0, 0, exp(-1j*theta/2)]]
    exp_neg = ops.exp(-1j * theta / 2)
    exp_pos = ops.exp(1j * theta / 2)
    zero_elem = Tensor(0.+0.j, dtype=ms.complex64)

    # 使用 ops.stack 构建 4x4 矩阵
    row1 = ops.stack([exp_neg, zero_elem, zero_elem, zero_elem], axis=-1)
    row2 = ops.stack([zero_elem, exp_pos, zero_elem, zero_elem], axis=-1)
    row3 = ops.stack([zero_elem, zero_elem, exp_pos, zero_elem], axis=-1)
    row4 = ops.stack([zero_elem, zero_elem, zero_elem, exp_neg], axis=-1)
    rzz_matrix = ops.stack([row1, row2, row3, row4], axis=-2)

    return rzz_matrix

def gate_to_matrix(gate, cir_qubits=1):
    """
    将单个门的信息转换为矩阵
    参数:
    gate: 包含门信息的字典
    cir_qubits: 总量子比特数
    返回:
    Tensor: 门的矩阵表示
    支持的门类型和参数格式 (与 MindSpore 版本类似):
    """
    gate_type = gate['type']
    gate_parameter = gate.get('parameter', None)
    # 单量子比特门：
    if gate_type in ['pauli_x', 'X']:
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _pauli_x(gate['target_qubit'])
    elif gate_type in ['pauli_y', 'Y']:
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _pauli_y(gate['target_qubit'])
    elif gate_type in ['pauli_z', 'Z']:
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _pauli_z(gate['target_qubit'])
    elif gate_type in ['hadamard', 'H']:
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _hadamard(gate['target_qubit'])
    elif gate_type in ['s_gate', 'S']:
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _s_gate(gate['target_qubit'])
    elif gate_type in ['t_gate', 'T']:
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _t_gate(gate['target_qubit'])
    elif gate_type == 'rx':
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _rx(gate_parameter, gate['target_qubit'])
    elif gate_type == 'ry':
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _ry(gate_parameter, gate['target_qubit'])
    elif gate_type == 'rz':
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _rz(gate_parameter, gate['target_qubit'])
    elif gate_type == 'u3':
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _u3(gate_parameter[0], gate_parameter[1], gate_parameter[2], gate['target_qubit'])
    elif gate_type == 'u2':
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _u2(gate_parameter[0], gate_parameter[1],  gate['target_qubit'])
    # 两量子比特门：
    elif gate_type in ['cnot', 'cx']:
        gate_qubits = max(gate['target_qubit'], max(gate['control_qubits'])) + 1
        gate_matrix = _cx(target_qubit=gate['target_qubit'], control_qubits=gate['control_qubits'], control_states=gate['control_states'])
    elif gate_type == 'cy':
        gate_qubits = max(gate['target_qubit'], max(gate['control_qubits'])) + 1
        gate_matrix = _cy(target_qubit=gate['target_qubit'], control_qubits=gate['control_qubits'], control_states=gate['control_states'])
    elif gate_type == 'cz':
        gate_qubits = max(gate['target_qubit'], max(gate['control_qubits'])) + 1
        gate_matrix = _cz(target_qubit=gate['target_qubit'], control_qubits=gate['control_qubits'], control_states=gate['control_states'])
    elif gate_type == 'crx':
        gate_qubits = max(gate['target_qubit'], max(gate['control_qubits'])) + 1
        gate_matrix = _crx(gate_parameter, target_qubit=gate['target_qubit'], control_qubits=gate['control_qubits'],
                           control_states=gate['control_states'])
    elif gate_type == 'cry':
        gate_qubits = max(gate['target_qubit'], max(gate['control_qubits'])) + 1
        gate_matrix = _cry(gate_parameter, target_qubit=gate['target_qubit'], control_qubits=gate['control_qubits'],
                           control_states=gate['control_states'])
    elif gate_type == 'crz':
        gate_qubits = max(gate['target_qubit'], max(gate['control_qubits'])) + 1
        gate_matrix = _crz(gate_parameter, target_qubit=gate['target_qubit'], control_qubits=gate['control_qubits'],
                           control_states=gate['control_states'])
    elif gate_type == 'swap':
        gate_qubits = max(gate['qubit_1'], gate['qubit_2']) + 1
        gate_matrix = _swap(qubit_1=gate['qubit_1'], qubit_2=gate['qubit_2'])
    # 三量子比特门：
    elif gate_type == 'toffoli':
        gate_qubits = max(gate['target_qubit'], gate['control_qubits'][0], gate['control_qubits'][1]) + 1
        gate_matrix = _toffoli(target_qubit=gate['target_qubit'], control_qubits=gate['control_qubits'])
    # 恒等门
    elif gate_type in ['identity', 'I']:
        return identity(gate['n_qubits'])
    else:
        raise ValueError(f"不支持的门类型: {gate_type}")

    if gate_qubits < cir_qubits:
        # 扩展到总量子比特数
        for i in range(gate_qubits, cir_qubits):
            gate_matrix = ops.kron(gate_matrix, IDENTITY_2)
    elif gate_qubits > cir_qubits:
        raise ValueError(f"量子门的量子比特数量超出总量子比特数: {gate_qubits} > {cir_qubits}")
    return gate_matrix

