import math
import numpy as np
import torch
import torch.nn.functional as F

my_dtype = torch.complex64

xpu = torch.device(
    # "npu:0" if torch.npu.is_available() else
    "cuda:0" if torch.cuda.is_available() else 
    "cpu")

def Matrix_Product(*matrices):
    """
    对多个矩阵进行连乘运算 (从左到右依次相乘)
    参数:
    *matrices: 可变数量的矩阵参数，每个矩阵都应该是PyTorch Tensor类型
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
        if not isinstance(matrix, torch.Tensor):
            raise TypeError(f"第{i+1}个参数必须是PyTorch Tensor类型，但得到了{type(matrix)}")
    # 依次进行矩阵乘法
    result = matrices[0]
    for i in range(1, len(matrices)):
        result = torch.matmul(result, matrices[i])
    return result

def Tensor_Product(*matrices):
    """
    计算多个矩阵的张量积（Kronecker积）
    参数:
    *matrices: 可变数量的矩阵参数，每个矩阵都应该是PyTorch Tensor类型
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
        if not isinstance(matrix, torch.Tensor):
            raise TypeError(f"第{i+1}个参数必须是PyTorch Tensor类型，但得到了{type(matrix)}")
    # 如果只有一个矩阵，直接返回
    if len(matrices) == 1:
        return matrices[0]
    # 依次计算张量积
    result = matrices[0]
    for i in range(1, len(matrices)):
        result = torch.kron(result, matrices[i])
    return result

def Dagger(matrix):
    """
    计算矩阵的共轭转置（dagger操作）
    参数:
    matrix: 输入的复数矩阵 (Tensor类型)
    返回:
    Tensor: 输入矩阵的共轭转置
    """
    # 先转置矩阵，再取共轭
    return torch.conj(torch.transpose(matrix, -2, -1)) # -2, -1 for last two dims
    # Or: return torch.transpose(matrix, -2, -1).conj()

# 量子比特基态 |0> 和 |1>
KET_0 = torch.tensor([[1.+0.j], [0.+0.j]], dtype=my_dtype, device=xpu)  # |0>
KET_1 = torch.tensor([[0.+0.j], [1.+0.j]], dtype=my_dtype, device=xpu)  # |1>
KET_PLUS = (KET_0 + KET_1) / torch.sqrt(torch.tensor(2.0))   # |+> = (|0> + |1>)/sqrt(2)
KET_MINUS = (KET_0 - KET_1) / torch.sqrt(torch.tensor(2.0))
# 对应的 bra 态 (行向量，ket的共轭转置)
BRA_0 = Dagger(KET_0)  # <0|
BRA_1 = Dagger(KET_1)  # <1|
BRA_PLUS = Dagger(KET_PLUS)   # <+|
BRA_MINUS = Dagger(KET_MINUS)
DENSITY_0 = torch.tensor([[1., 0.],
                          [0., 0.]], dtype=my_dtype, device=xpu)  # |0><0|
DENSITY_1 = torch.tensor([[0., 0.],
                          [0., 1.]], dtype=my_dtype, device=xpu)  # |1><1|
IDENTITY_2 = torch.tensor([[1., 0.],
                           [0., 1.]], dtype=my_dtype, device=xpu)  # 单位矩阵 I

def IDENTITY(n_qubits=1):
    """Identity gate for n qubits."""
    dim = 2 ** n_qubits
    return torch.eye(dim, dtype=my_dtype, device=xpu)

def _PAULI_X(target_qubit=0):
    pauli_x = torch.tensor([[0.+0.j, 1.+0.j], [1.+0.j, 0.+0.j]], dtype=my_dtype, device=xpu)
    if target_qubit > 0:
        pauli_x = torch.kron(IDENTITY(target_qubit), pauli_x)
    return pauli_x

def _PAULI_Y(target_qubit=0):
    pauli_y = torch.tensor([[0.+0.j, -1j], [1j, 0.+0.j]], dtype=my_dtype, device=xpu)
    if target_qubit > 0:
        pauli_y = torch.kron(IDENTITY(target_qubit), pauli_y)
    return pauli_y

def _PAULI_Z(target_qubit=0):
    pauli_z = torch.tensor([[1.+0.j, 0.+0.j], [0.+0.j, -1.+0.j]], dtype=my_dtype, device=xpu)
    if target_qubit > 0:
        pauli_z = torch.kron(IDENTITY(target_qubit), pauli_z)
    return pauli_z

def _HADAMARD(target_qubit=0):
    sqrt2_inv = 1.0 / torch.sqrt(torch.tensor(2.0))
    hadamard = torch.tensor([[sqrt2_inv, sqrt2_inv], [sqrt2_inv, -sqrt2_inv]], dtype=my_dtype, device=xpu)
    if target_qubit > 0:
        hadamard = torch.kron(IDENTITY(target_qubit), hadamard)
    return hadamard

def _RX(theta, target_qubit=0):
    """Single qubit rotation around the X axis by angle theta."""
    # 确保 theta 是一个 Tensor
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=my_dtype, device=xpu)
    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)
    # 将 sin 和 cos 转换为 complex64 类型
    cos_c = cos.to(my_dtype)
    sin_c = sin.to(my_dtype)
    # 创建 -i*sin
    neg_i_sin = sin_c * torch.tensor(-1j, dtype=my_dtype, device=xpu)
    # 修正：使用 torch.stack 构建 2x2 矩阵
    row1 = torch.stack([cos_c, neg_i_sin], dim=-1)  # Shape: [..., 2]
    row2 = torch.stack([neg_i_sin, cos_c], dim=-1)  # Shape: [..., 2]
    rx_matrix = torch.stack([row1, row2], dim=-2)   # Shape: [..., 2, 2]
    # torch.stack 会保留输入张量的批次维度（如果 theta 是标量，则无批次维度）
    # 如果 theta 是标量 torch.tensor(1.5)，则 rx_matrix 形状为 [2, 2]
    # 如果 theta 是形状为 [N] 的张量，输出形状为 [N, 2, 2]
    # 如果 theta 是形状为 [1] 的张量，输出形状为 [1, 2, 2]，可能需要 squeeze
    # 为了处理输入为标量（零维张量）的情况，可以检查并 squeeze
    # 注意：如果 theta 是零维张量，torch.cos(theta/2) 也是零维，torch.stack 会报错
    # 因此，如果 theta 是标量，我们先将其变成一维
    # 但 torch.tensor(1.5) 是零维，torch.cos(torch.tensor(1.5)) 也是零维
    # torch.stack([zero_dim_tensor, ...], dim=-1) 会创建 [..., 2] 形状
    # torch.stack([[...], [...]], dim=-2) 会创建 [..., 2, 2] 形状
    # 如果输入是零维标量 theta，输出 rx_matrix 应该是 [2, 2]
    # torch.stack 处理零维输入是安全的，并且会创建正确的形状。
    # 如果 theta 是形状为 [1] 的一维张量，则输出是 [1, 2, 2]，需要 squeeze
    if rx_matrix.ndim > 2 and rx_matrix.shape[0] == 1:
        rx_matrix = torch.squeeze(rx_matrix, dim=0)
    if target_qubit > 0:
        # 假设 IDENTITY(target_qubit) 返回一个合适的单位矩阵
        identity_block = IDENTITY(target_qubit) # 需要确保这个函数返回正确形状的复数单位矩阵
        rx_matrix = torch.kron(identity_block, rx_matrix)
    return rx_matrix

def _RY(theta, target_qubit=0):
    """Single qubit rotation around the Y axis by angle theta."""
    # 确保 theta 是一个 Tensor
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=my_dtype, device=xpu)
    
    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)
    # 将 sin 和 cos 转换为 complex64 类型
    cos_c = cos.to(my_dtype)
    sin_c = sin.to(my_dtype)
    # 创建 -sin(θ/2) 和 sin(θ/2)
    neg_sin_c = -sin_c
    sin_c_pos = sin_c # Just to make the matrix construction clear

    # 修正：使用 torch.stack 构建 2x2 矩阵
    # RY(θ) = [[cos(θ/2), -sin(θ/2)],
    #          [sin(θ/2),  cos(θ/2)]]
    row1 = torch.stack([cos_c, neg_sin_c], dim=-1)  # [cos(θ/2), -sin(θ/2)]
    row2 = torch.stack([sin_c_pos, cos_c], dim=-1)  # [sin(θ/2),  cos(θ/2)]
    ry_matrix = torch.stack([row1, row2], dim=-2)   # [[row1], [row2]]

    # 如果 theta 是标量 torch.tensor(1.5)，则 ry_matrix 形状为 [2, 2]
    # 如果 theta 是形状为 [N] 的张量，输出形状为 [N, 2, 2]
    # 如果 theta 是形状为 [1] 的张量，输出形状为 [1, 2, 2]，可能需要 squeeze
    # 为了处理输入为标量（零维张量）的情况，可以检查并 squeeze
    if ry_matrix.ndim > 2 and ry_matrix.shape[0] == 1:
        ry_matrix = torch.squeeze(ry_matrix, dim=0)
    
    if target_qubit > 0:
        identity_block = IDENTITY(target_qubit)
        ry_matrix = torch.kron(identity_block, ry_matrix)

    return ry_matrix

def _RZ(theta, target_qubit=0):
    """Single qubit rotation around the Z axis by angle theta."""
    # 确保 theta 是一个 Tensor
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=my_dtype, device=xpu)
    # 计算复数指数
    exp_neg = torch.exp(-1j * theta / 2) # Shape: [shape_of_theta]
    exp_pos = torch.exp(1j * theta / 2)  # Shape: [shape_of_theta]

    # 使用 torch.stack 构建 2x2 矩阵
    # RZ(θ) = [[exp(-1j*θ/2), 0],
    #          [0,          exp(1j*θ/2)]]
    # 创建零元素（复数零）
    zero_elem = torch.tensor(0.+0.j, dtype=my_dtype, device=xpu)
    # 如果 theta 是标量或有批次维度，zero_elem 需要广播到相同的形状
    # torch.stack 会自动处理广播
    row1 = torch.stack([exp_neg, zero_elem], dim=-1)  # Shape: [..., 2]
    row2 = torch.stack([zero_elem, exp_pos], dim=-1)  # Shape: [..., 2]
    rz_matrix = torch.stack([row1, row2], dim=-2)     # Shape: [..., 2, 2]

    # 如果 theta 是标量 (零维张量)，torch.stack 会正确处理
    # 如果 theta 是形状为 [N] 的一维张量，则输出是 [N, 2, 2]
    # 如果 theta 是形状为 [1] 的一维张量，则输出是 [1, 2, 2]，可能需要 squeeze
    # 为了处理输入为标量（零维张量）的情况，可以检查并 squeeze
    # 但是，对于零维输入 theta，torch.exp(theta/2) 仍然是零维，torch.stack([...], dim=-1) 会创建 [..., 2] 形状
    # torch.stack([[...], [...]], dim=-2) 会创建 [..., 2, 2] 形状
    # 如果输入是零维标量 theta，输出 rz_matrix 应该是 [2, 2]
    # 如果输入是 [1] 形状的 theta，输出 rz_matrix 是 [1, 2, 2]
    # 所以，如果第一个维度是 1，则 squeeze
    if rz_matrix.ndim > 2 and rz_matrix.shape[0] == 1:
        rz_matrix = torch.squeeze(rz_matrix, dim=0)

    if target_qubit > 0:
        identity_block = IDENTITY(target_qubit)
        rz_matrix = torch.kron(identity_block, rz_matrix)

    return rz_matrix

def _S_GATE(target_qubit=0):
    s_gate = torch.tensor([[1.+0.j, 0.+0.j],
                           [0.+0.j, 1j]], dtype=my_dtype, device=xpu)
    # 使用 torch.stack 重构矩阵（虽然元素是常数）
    row1 = torch.stack([torch.tensor(1.+0.j, dtype=my_dtype, device=xpu), torch.tensor(0.+0.j, dtype=my_dtype, device=xpu)], dim=-1)
    row2 = torch.stack([torch.tensor(0.+0.j, dtype=my_dtype, device=xpu), torch.tensor(1j, dtype=my_dtype, device=xpu)], dim=-1)
    s_gate_matrix = torch.stack([row1, row2], dim=-2)

    if target_qubit > 0:
        s_gate_matrix = torch.kron(IDENTITY(target_qubit), s_gate_matrix)
    return s_gate_matrix

def _T_GATE(target_qubit=0):
    t_gate_val = torch.exp(1j * torch.tensor(math.pi) / 4) # Use torch.pi if available (PyTorch 2.0+)
    t_gate = torch.tensor([[1.+0.j, 0.+0.j],
                           [0.+0.j, t_gate_val]], dtype=my_dtype, device=xpu)
    # 使用 torch.stack 重构矩阵
    one_elem = torch.tensor(1.+0.j, dtype=my_dtype, device=xpu)
    zero_elem = torch.tensor(0.+0.j, dtype=my_dtype, device=xpu)
    row1 = torch.stack([one_elem, zero_elem], dim=-1)
    row2 = torch.stack([zero_elem, t_gate_val], dim=-1)
    t_gate_matrix = torch.stack([row1, row2], dim=-2)

    if target_qubit > 0:
        t_gate_matrix = torch.kron(IDENTITY(target_qubit), t_gate_matrix)
    return t_gate_matrix

def _CX(target_qubit, control_qubits, control_states):
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
    px_matrix = _PAULI_X() - IDENTITY_2
    # 确定所需量子比特总数
    all_qubits = [target_qubit] + control_qubits
    num_qubits = max(all_qubits) + 1
    # 检查control_states长度是否与control_qubits匹配
    if len(control_states) != len(control_qubits):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")
    # 构建张量积矩阵
    matrices = []
    for qubit_index in range(num_qubits):
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
        result_matrix = torch.kron(result_matrix, matrices[i])
    result_matrix = IDENTITY(num_qubits) + result_matrix
    return result_matrix

def _CY(target_qubit, control_qubits, control_states):
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
    py_matrix = _PAULI_Y() - IDENTITY_2
    # 确定所需量子比特总数
    all_qubits = [target_qubit] + control_qubits
    num_qubits = max(all_qubits) + 1
    # 检查control_states长度是否与control_qubits匹配
    if len(control_states) != len(control_qubits):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")
    # 构建张量积矩阵
    matrices = []
    for qubit_index in range(num_qubits):
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
        result_matrix = torch.kron(result_matrix, matrices[i])
    result_matrix = IDENTITY(num_qubits) + result_matrix
    return result_matrix

def _CZ(target_qubit, control_qubits, control_states):
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
    pz_matrix = _PAULI_Z() - IDENTITY_2
    # 确定所需量子比特总数
    all_qubits = [target_qubit] + control_qubits
    num_qubits = max(all_qubits) + 1
    # 检查control_states长度是否与control_qubits匹配
    if len(control_states) != len(control_qubits):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")
    # 构建张量积矩阵
    matrices = []
    for qubit_index in range(num_qubits):
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
        result_matrix = torch.kron(result_matrix, matrices[i])
    result_matrix = IDENTITY(num_qubits) + result_matrix
    return result_matrix

def _CRX(theta, target_qubit, control_qubits, control_states):
    """
    受控RX门（Controlled RX gate）
    参数:
    theta: RX旋转角度
    target_qubit: 目标量子比特的索引
    control_qubits: 控制量子比特的索引列表
    control_states: 控制量子比特的状态列表（0或1）
    返回:
    受控RX门的矩阵表示
    """
    # 首先获取RX门矩阵 (注意：这里减去单位矩阵)
    rx_base = _RX(theta) - IDENTITY_2
    # 确定所需量子比特总数
    all_qubits = [target_qubit] + control_qubits
    num_qubits = max(all_qubits) + 1
    # 检查control_states长度是否与control_qubits匹配
    if len(control_states) != len(control_qubits):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")
    # 构建张量积矩阵
    matrices = []
    for qubit_index in range(num_qubits):
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
        result_matrix = torch.kron(result_matrix, matrices[i])
    # 加上单位矩阵得到完整的受控门 (注意：这里加上单位矩阵)
    result_matrix = IDENTITY(num_qubits) + result_matrix
    return result_matrix

def _CRY(theta, target_qubit, control_qubits, control_states):
    """
    受控RY门（Controlled RY gate）
    参数:
    theta: RY旋转角度
    target_qubit: 目标量子比特的索引
    control_qubits: 控制量子比特的索引列表
    control_states: 控制量子比特的状态列表（0或1）
    返回:
    受控RY门的矩阵表示
    """
    # 首先获取RY门矩阵 (注意：这里减去单位矩阵)
    ry_base = _RY(theta) - IDENTITY_2
    # 确定所需量子比特总数
    all_qubits = [target_qubit] + control_qubits
    num_qubits = max(all_qubits) + 1
    # 检查control_states长度是否与control_qubits匹配
    if len(control_states) != len(control_qubits):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")
    # 构建张量积矩阵
    matrices = []
    for qubit_index in range(num_qubits):
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
        result_matrix = torch.kron(result_matrix, matrices[i])
    # 加上单位矩阵得到完整的受控门 (注意：这里加上单位矩阵)
    result_matrix = IDENTITY(num_qubits) + result_matrix
    return result_matrix

def _CRZ(theta, target_qubit, control_qubits, control_states):
    """
    受控RZ门（Controlled RZ gate）
    参数:
    theta: RZ旋转角度
    target_qubit: 目标量子比特的索引
    control_qubits: 控制量子比特的索引列表
    control_states: 控制量子比特的状态列表（0或1）
    返回:
    受控RY门的矩阵表示
    """
    # 首先获取RZ门矩阵 (注意：这里减去单位矩阵)
    rz_base = _RZ(theta) - IDENTITY_2
    # 确定所需量子比特总数
    all_qubits = [target_qubit] + control_qubits
    num_qubits = max(all_qubits) + 1
    # 检查control_states长度是否与control_qubits匹配
    if len(control_states) != len(control_qubits):
        raise ValueError("control_states的长度必须与control_qubits的长度相同")
    # 构建张量积矩阵
    matrices = []
    for qubit_index in range(num_qubits):
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
        result_matrix = torch.kron(result_matrix, matrices[i])
    # 加上单位矩阵得到完整的受控门 (注意：这里加上单位矩阵)
    result_matrix = IDENTITY(num_qubits) + result_matrix
    return result_matrix

def _SWAP(qubit_1=0, qubit_2=1):
    return Matrix_Product(_CX(qubit_1, [qubit_2], [1]), _CX(qubit_2, [qubit_1], [1]), _CX(qubit_1, [qubit_2], [1]))

def _TOFFOLI(target_qubit=2, control_qubits=[0,1]):
    num_qubits = max(target_qubit, max(control_qubits)) + 1
    matrices_0 = [IDENTITY_2] * num_qubits
    matrices_0[control_qubits[0]] = DENSITY_1
    matrices_0[control_qubits[1]] = DENSITY_1
    matrices_0[target_qubit] = _PAULI_X()
    result_0 = matrices_0[0]
    for i in range(1, num_qubits):
        result_0 = torch.kron(result_0, matrices_0[i])
    matrices_1 = [IDENTITY_2] * num_qubits
    matrices_1[control_qubits[0]] = DENSITY_0
    result_1 = matrices_1[0]
    for i in range(1, num_qubits):
        result_1 = torch.kron(result_1, matrices_1[i])
    matrices_2 = [IDENTITY_2] * num_qubits
    matrices_2[control_qubits[0]] = DENSITY_1
    matrices_2[control_qubits[1]] = DENSITY_0
    result_2 = matrices_2[0]
    for i in range(1, num_qubits):
        result_2 = torch.kron(result_2, matrices_2[i])
    return result_0 + result_1 + result_2

def _U3(theta, phi, lam, target_qubit=0):
    """General single-qubit rotation gate U3."""
    # 确保参数是 Tensor
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=torch.float64)
    if not isinstance(phi, torch.Tensor):
        phi = torch.tensor(phi, dtype=torch.float64)
    if not isinstance(lam, torch.Tensor):
        lam = torch.tensor(lam, dtype=torch.float64)

    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)
    exp_iphi = torch.exp(1j * phi)
    exp_ilam = torch.exp(1j * lam)
    exp_iphi_lam = torch.exp(1j * (phi + lam))

    # 使用 torch.stack 构建 U3 矩阵
    row1 = torch.stack([cos, -exp_ilam * sin], dim=-1)  # [cos, -exp(-i*lam) * sin]
    row2 = torch.stack([exp_iphi * sin, exp_iphi_lam * cos], dim=-1)  # [exp(i*phi) * sin, exp(i*(phi+lam)) * cos]
    u3_matrix = torch.stack([row1, row2], dim=-2)

    if target_qubit > 0:
        u3_matrix = torch.kron(IDENTITY(target_qubit), u3_matrix)
    return u3_matrix

def _U2(phi, lam, target_qubit=0):
    """Single-qubit rotation gate U2."""
    # Use torch.pi if available (PyTorch 2.0+), otherwise math.pi
    return _U3(torch.tensor(math.pi)/2, phi, lam, target_qubit)

def _RZZ(theta):
    """
    RZZ门（控制Z旋转门）
    作用在两个量子比特上，实现条件相位旋转
    参数:
    theta: 旋转角度
    返回:
    Tensor: 4x4的RZZ门矩阵
    """
    # 确保 theta 是一个 Tensor
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=torch.float64)

    # RZZ门的矩阵形式:
    # [[exp(-1j*theta/2), 0, 0, 0],
    #  [0, exp(1j*theta/2), 0, 0],
    #  [0, 0, exp(1j*theta/2), 0],
    #  [0, 0, 0, exp(-1j*theta/2)]]
    exp_neg = torch.exp(-1j * theta / 2)
    exp_pos = torch.exp(1j * theta / 2)
    zero_elem = torch.tensor(0.+0.j, dtype=my_dtype, device=xpu)

    # 使用 torch.stack 构建 4x4 矩阵
    row1 = torch.stack([exp_neg, zero_elem, zero_elem, zero_elem], dim=-1)
    row2 = torch.stack([zero_elem, exp_pos, zero_elem, zero_elem], dim=-1)
    row3 = torch.stack([zero_elem, zero_elem, exp_pos, zero_elem], dim=-1)
    row4 = torch.stack([zero_elem, zero_elem, zero_elem, exp_neg], dim=-1)
    rzz_matrix = torch.stack([row1, row2, row3, row4], dim=-2)

    return rzz_matrix

def Gate_To_Matrix(gate, cir_qubits=1):
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
    if gate_type in ['PAULI_X', 'X']:
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _PAULI_X(gate['target_qubit'])
    elif gate_type in ['PAULI_Y', 'Y']:
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _PAULI_Y(gate['target_qubit'])
    elif gate_type in ['PAULI_Z', 'Z']:
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _PAULI_Z(gate['target_qubit'])
    elif gate_type in ['HADAMARD', 'H']:
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _HADAMARD(gate['target_qubit'])
    elif gate_type in ['S_GATE', 'S']:
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _S_GATE(gate['target_qubit'])
    elif gate_type in ['T_GATE', 'T']:
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _T_GATE(gate['target_qubit'])
    elif gate_type == 'RX':
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _RX(gate_parameter, gate['target_qubit'])
    elif gate_type == 'RY':
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _RY(gate_parameter, gate['target_qubit'])
    elif gate_type == 'RZ':
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _RZ(gate_parameter, gate['target_qubit'])
    elif gate_type == 'U3':
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _U3(gate_parameter[0], gate_parameter[1], gate_parameter[2], gate['target_qubit'])
    elif gate_type == 'U2':
        gate_qubits = gate['target_qubit'] + 1
        gate_matrix = _U2(gate_parameter[0], gate_parameter[1],  gate['target_qubit'])
    # 两量子比特门：
    elif gate_type in ['CNOT', 'CX']:
        gate_qubits = max(gate['target_qubit'], max(gate['control_qubits'])) + 1
        gate_matrix = _CX(target_qubit=gate['target_qubit'], control_qubits=gate['control_qubits'], control_states=gate['control_states'])
    elif gate_type == 'CY':
        gate_qubits = max(gate['target_qubit'], max(gate['control_qubits'])) + 1
        gate_matrix = _CY(target_qubit=gate['target_qubit'], control_qubits=gate['control_qubits'], control_states=gate['control_states'])
    elif gate_type == 'CZ':
        gate_qubits = max(gate['target_qubit'], max(gate['control_qubits'])) + 1
        gate_matrix = _CZ(target_qubit=gate['target_qubit'], control_qubits=gate['control_qubits'], control_states=gate['control_states'])
    elif gate_type == 'CRX':
        gate_qubits = max(gate['target_qubit'], max(gate['control_qubits'])) + 1
        gate_matrix = _CRX(gate_parameter, target_qubit=gate['target_qubit'], control_qubits=gate['control_qubits'],
                           control_states=gate['control_states'])
    elif gate_type == 'CRY':
        gate_qubits = max(gate['target_qubit'], max(gate['control_qubits'])) + 1
        gate_matrix = _CRY(gate_parameter, target_qubit=gate['target_qubit'], control_qubits=gate['control_qubits'],
                           control_states=gate['control_states'])
    elif gate_type == 'CRZ':
        gate_qubits = max(gate['target_qubit'], max(gate['control_qubits'])) + 1
        gate_matrix = _CRZ(gate_parameter, target_qubit=gate['target_qubit'], control_qubits=gate['control_qubits'],
                           control_states=gate['control_states'])
    elif gate_type == 'SWAP':
        gate_qubits = max(gate['qubit_1'], gate['qubit_2']) + 1
        gate_matrix = _SWAP(qubit_1=gate['qubit_1'], qubit_2=gate['qubit_2'])
    # 三量子比特门：
    elif gate_type == 'TOFFOLI':
        gate_qubits = max(gate['target_qubit'], gate['control_qubits'][0], gate['control_qubits'][1]) + 1
        gate_matrix = _TOFFOLI(target_qubit=gate['target_qubit'], control_qubits=gate['control_qubits'])
    # 恒等门
    elif gate_type in ['IDENTITY', 'I']:
        return IDENTITY(gate['num_qubits'])
    else:
        raise ValueError(f"不支持的门类型: {gate_type}")

    if gate_qubits < cir_qubits:
        # 扩展到总量子比特数
        for i in range(gate_qubits, cir_qubits):
            gate_matrix = torch.kron(gate_matrix, IDENTITY_2)
    elif gate_qubits > cir_qubits:
        raise ValueError(f"量子门的量子比特数量超出总量子比特数: {gate_qubits} > {cir_qubits}")
    return gate_matrix

# --- 示例：使用 PyTorch 的自动微分 ---
if __name__ == "__main__":
    # 创建一个可求导的旋转角度
    theta = torch.tensor(1.5, requires_grad=True, dtype=my_dtype, device=xpu)
    # 创建一个简单的量子态 |psi> = |0>
    psi = KET_0.clone() # Clone to avoid modifying the global constant

    # 应用 RX(theta) 门
    rx_gate = _RZ(theta)
    # 假设这是一个简单的演化 U|psi>
    evolved_state = torch.matmul(rx_gate, psi)

    # 计算一个简单的实数输出，例如 <psi_out|Z|psi_out> (Z 是泡利Z算符)
    # 这里我们用 <0| evolved_state 来模拟一个简单的期望值计算
    bra_0 = Dagger(psi) # <0|
    overlap = torch.matmul(bra_0, evolved_state) # <0|RX(theta)|0>
    # 取模长平方 |<0|RX(theta)|0>|^2
    prob_0 = torch.abs(overlap)**2 # This is a real scalar

    print(f"Input theta: {theta.item()}")
    print(f"Probability of |0> after RX({theta.item()}): {prob_0.item()}")

    # 执行反向传播
    prob_0.backward()

    # 打印梯度
    print(f"Gradient of probability w.r.t. theta: {theta.grad.item()}")

    # --- 测试矩阵乘积和张量积 ---
    print("\n--- Matrix Product Test ---")
    I2 = IDENTITY(1) # 2x2 identity
    X = _PAULI_X()
    IX = Tensor_Product(I2, X) # I \otimes X
    XX = _PAULI_X(1) # X on qubit 1 (with I on qubit 0)
    print(f"I kron X:\n{IX}")
    print(f"X on qubit 1:\n{XX}")
    print(f"Are they equal? {torch.allclose(IX, XX)}")

    print("\n--- Gate Matrix Conversion Test ---")
    gate_info = {'type': 'RX', 'target_qubit': 0, 'parameter': torch.tensor(0.5, requires_grad=True)}
    rx_mat = Gate_To_Matrix(gate_info, cir_qubits=2) # Should expand to 2 qubits
    print(f"RX(0.5) gate matrix (2 qubits):\n{rx_mat}")
    print(f"Shape: {rx_mat.shape}")
    print(f"Requires grad: {rx_mat.requires_grad}") # Should be True if parameter requires grad

    # --- Test _RZ with stack ---
    print("\n--- _RZ with stack Test ---")
    rz_gate = _RZ(torch.tensor(0.7))
    print(f"RZ(0.7) gate matrix:\n{rz_gate}")

    # --- Test _S_GATE with stack ---
    print("\n--- _S_GATE with stack Test ---")
    s_gate = _S_GATE()
    print(f"S gate matrix:\n{s_gate}")

    # --- Test _T_GATE with stack ---
    print("\n--- _T_GATE with stack Test ---")
    t_gate = _T_GATE()
    print(f"T gate matrix:\n{t_gate}")

    # --- Test _U3 with stack ---
    print("\n--- _U3 with stack Test ---")
    u3_gate = _U3(torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3))
    print(f"U3(0.1, 0.2, 0.3) gate matrix:\n{u3_gate}")

    # --- Test RZZ with stack ---
    print("\n--- RZZ with stack Test ---")
    rzz_gate = RZZ(torch.tensor(0.8))
    print(f"RZZ(0.8) gate matrix:\n{rzz_gate}")

