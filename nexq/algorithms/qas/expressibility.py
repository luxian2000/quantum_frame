"""
nexq/algorithms/qas/expressibility.py

计算参数化量子电路的 expressibility 指标。

算法：计算电路参数化生成的保真度分布与 Haar 随机态保真度分布之间的 KL 散度。
"""

from __future__ import annotations

from copy import deepcopy
from typing import List, Optional

import numpy as np

from ...channel.backends.base import Backend
from ...circuit.model import Circuit
from ...circuit.state_vector import StateVector


# ─────────────────────────────────────────────────────────────────────────
# 参数处理工具
# ─────────────────────────────────────────────────────────────────────────

def _get_parametrized_gate_indices(circuit: Circuit) -> List[int]:
    """
    返回电路中所有包含 'parameter' 字段的门的索引。
    """
    indices = []
    for i, gate in enumerate(circuit.gates):
        if "parameter" in gate:
            indices.append(i)
    return indices


def _replace_circuit_parameters(circuit: Circuit, params: np.ndarray) -> Circuit:
    """
    根据参数向量创建一个新的电路副本，将所有参数化门的参数替换为新值。

    参数:
        circuit: nexq Circuit 对象
        params:  参数向量，长度应与参数化门的数量相匹配
                 （对于 u2/u3，可能需要多个参数值）

    返回:
        新的 Circuit 对象，所有参数化门的参数已更新
    """
    gates_copy = deepcopy(list(circuit.gates))
    param_idx = 0

    for i, gate in enumerate(gates_copy):
        if "parameter" not in gate:
            continue

        gate_type = gate["type"]

        # 根据门类型确定参数数量
        if gate_type in ("rx", "ry", "rz", "crx", "cry", "crz", "rzz"):
            # 单参数门
            gate["parameter"] = float(params[param_idx])
            param_idx += 1
        elif gate_type == "u2":
            # 两参数门
            gate["parameter"] = [float(params[param_idx]), float(params[param_idx + 1])]
            param_idx += 2
        elif gate_type == "u3":
            # 三参数门
            gate["parameter"] = [
                float(params[param_idx]),
                float(params[param_idx + 1]),
                float(params[param_idx + 2]),
            ]
            param_idx += 3

    return Circuit(*gates_copy, n_qubits=circuit.n_qubits)


def _compute_fidelity(sv1: StateVector, sv2: StateVector, backend: Backend) -> float:
    """
    计算两个量子态之间的保真度。
    
    Fidelity(|ψ1⟩, |ψ2⟩) = |⟨ψ1|ψ2⟩|²
    """
    # 使用后端的 inner_product 方法，它自动处理共轭
    data1 = sv1.data  # shape (2^n, 1)
    data2 = sv2.data  # shape (2^n, 1)

    # 计算内积 ⟨ψ1|ψ2⟩
    inner_prod = backend.inner_product(data1, data2)
    inner_prod_scalar = backend.to_numpy(inner_prod).item()

    # 保真度是内积模的平方
    fidelity = float(np.abs(inner_prod_scalar) ** 2)
    return fidelity


def _count_total_parameters(circuit: Circuit, param_indices: List[int]) -> int:
    """统计电路中所有参数化门的总参数数量。"""
    total_params = 0
    for idx in param_indices:
        gate = circuit.gates[idx]
        gate_type = gate["type"]
        if gate_type in ("rx", "ry", "rz", "crx", "cry", "crz", "rzz"):
            total_params += 1
        elif gate_type == "u2":
            total_params += 2
        elif gate_type == "u3":
            total_params += 3
    return total_params


# ─────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────


def KL_Haar_relative(
    cir: Circuit,
    samples: int = 1000,
    n_bins: int = 100,
    backend: Optional[Backend] = None,
) -> float:
    """
    计算参数化量子电路的相对 expressibility 指标（KL_Haar_relative）。
    
    算法流程：
    1. 随机采样参数向量，创建电路副本并运行得到末态
    2. 计算两个末态间的保真度
    3. 重复采样，构建保真度直方图
    4. 计算理论 Haar 分布
    5. 计算 KL 散度（KL_Haar）
    6. 计算相对 expressibility：Exp_1R = -ln(KL_Haar / Exp_1_idle)

    参数:
        cir:      nexq Circuit 对象（可参数化），量子比特数从 cir.n_qubits 读取
        samples:  采样次数（默认 1000），每次采样产生一对保真度值
        n_bins:   直方图 bin 数（默认 100），用于离散化保真度 [0, 1]
        backend:  计算后端；若为 None，使用默认 NumpyBackend

    返回:
        相对 expressibility 值（float），值越高表示电路越 expressive
    """
    # 使用默认后端
    if backend is None:
        from ...channel.backends.numpy_backend import NumpyBackend
        backend = NumpyBackend()

    n_qubits = cir.n_qubits
    dim = 2 ** n_qubits

    # ──────────── 第 1 步：确定参数化门和参数总数 ────────────
    param_indices = _get_parametrized_gate_indices(cir)
    if not param_indices:
        raise ValueError("电路中没有参数化门，无法计算 expressibility")

    # 计算总参数数
    total_params = _count_total_parameters(cir, param_indices)

    # ──────────── 第 2 步：初始态 ────────────────────────────
    zero_state = StateVector.zero_state(n_qubits, backend)

    # ──────────── 第 3 步：采样保真度 ──────────────────────
    fidelity_list = []
    np.random.seed(None)  # 使用随机种子

    for _ in range(samples):
        # 随机采样两组参数
        params1 = np.random.uniform(0, 2 * np.pi, total_params)
        params2 = np.random.uniform(0, 2 * np.pi, total_params)

        # 创建参数化电路副本并获取幺正矩阵
        cir1 = _replace_circuit_parameters(cir, params1)
        cir2 = _replace_circuit_parameters(cir, params2)

        U1 = backend.cast(backend.to_numpy(cir1.unitary(backend=backend)))
        U2 = backend.cast(backend.to_numpy(cir2.unitary(backend=backend)))

        # 计算末态：|ψ1⟩ = U1|0...0⟩，|ψ2⟩ = U2|0...0⟩
        sv1 = zero_state.evolve(U1)
        sv2 = zero_state.evolve(U2)

        # 计算保真度
        fidelity = _compute_fidelity(sv1, sv2, backend)
        fidelity_list.append(fidelity)

    fidelity_array = np.array(fidelity_list)

    # ──────────── 第 4 步：构建 PQC 保真度直方图 ────────────
    bin_width = 1.0 / n_bins
    p_pqc = np.zeros(n_bins)

    for f in fidelity_array:
        bin_idx = int(np.floor(f * n_bins))
        bin_idx = min(bin_idx, n_bins - 1)  # 边界处理：f=1.0 时落在最后一个 bin
        p_pqc[bin_idx] += 1

    # 归一化
    p_pqc = p_pqc / samples

    # ──────────── 第 5 步：计算理论 Haar 分布 ────────────
    # P_Haar(F) = (2^N - 1) * (1 - F)^(2^N - 2)
    # 离散化到 bin，在 bin i 中心计算理论值
    p_haar = np.zeros(n_bins)

    for i in range(n_bins):
        # bin i 的中心点
        f_center = (i + 0.5) * bin_width
        f_center = np.clip(f_center, 0, 1)

        # Haar 分布密度
        p_f = (dim - 1) * np.power(1 - f_center, dim - 2)

        # 按 bin 宽度积分近似（或简单使用密度值）
        p_haar[i] = p_f * bin_width

    # 归一化 Haar 分布
    p_haar = p_haar / np.sum(p_haar)

    # ──────────── 第 6 步：计算 KL 散度（KL_Haar） ────────────────
    # D_KL(P_PQC || P_Haar) = Σ_i P_PQC[i] * ln(P_PQC[i] / P_Haar[i])
    kl_haar = 0.0

    for i in range(n_bins):
        if p_pqc[i] > 0 and p_haar[i] > 0:
            kl_haar += p_pqc[i] * np.log(p_pqc[i] / p_haar[i])

    # ──────────── 第 7 步：计算相对 expressibility ────────────
    # 参考值：Exp_1_idle = (2^N - 1) * ln(n_bins)
    exp_1_idle = (dim - 1) * np.log(n_bins)

    # 处理边界情况
    if kl_haar <= 0 or kl_haar < 1e-15 or exp_1_idle <= 0:
        # 若 KL_Haar 接近 0，认为电路已达最大表达能力
        return float(1e10)

    # 计算相对 expressibility：Exp_1R = -ln(KL_Haar / Exp_1_idle)
    ratio = kl_haar / exp_1_idle
    kl_haar_relative = -np.log(ratio)

    return float(kl_haar_relative)


def MMD_relative(
    cir: Circuit,
    samples: int = 1000,
    sigma: float = 0.01,
    backend: Optional[Backend] = None,
) -> float:
    """
    基于 MMD 的量子线路表达能力估计。

    输入参数:
        cir: 参数化量子线路
        samples: 采样次数 M（默认 1000）
        sigma: 高斯核带宽（默认 0.01），必须为正数
        backend: 计算后端；若为 None，使用默认 NumpyBackend

    返回:
        Exp_2 = 1 - MMD，理论上越接近 1 表示越 expressive。
    """
    if samples <= 0:
        raise ValueError("samples 必须为正整数")
    if sigma <= 0:
        raise ValueError("sigma 必须为正数")

    if backend is None:
        from ...channel.backends.numpy_backend import NumpyBackend

        backend = NumpyBackend()

    n_qubits = cir.n_qubits
    dim = 1 << n_qubits

    param_indices = _get_parametrized_gate_indices(cir)
    if not param_indices:
        raise ValueError("电路中没有参数化门，无法计算 MMD_relative")
    total_params = _count_total_parameters(cir, param_indices)

    # |+>^N = (1/sqrt(2^N)) * sum_i |i>
    plus_state_data = np.ones(dim, dtype=np.complex64) / np.sqrt(dim)
    plus_state = StateVector.from_array(plus_state_data, n_qubits=n_qubits, backend=backend)

    # X: 从参数化电路诱导分布采样得到的概率向量
    x_samples = np.zeros((samples, dim), dtype=np.float64)
    np.random.seed(None)

    for i in range(samples):
        params = np.random.uniform(0, 2 * np.pi, total_params)
        cir_i = _replace_circuit_parameters(cir, params)
        U = backend.cast(backend.to_numpy(cir_i.unitary(backend=backend)))
        out_state = plus_state.evolve(U)
        probs = backend.to_numpy(out_state.probabilities()).reshape(-1)
        x_samples[i] = np.real(probs)

    # Y: 从 simplex 上均匀分布采样（Dirichlet(alpha=1)）
    expo = np.random.exponential(scale=1.0, size=(samples, dim))
    y_samples = expo / np.sum(expo, axis=1, keepdims=True)

    # 高斯核 k(x, y) = exp(-||x-y||^2 / (4*sigma^2))
    denom = 4.0 * (sigma ** 2)

    xx_dist2 = np.sum((x_samples[:, None, :] - x_samples[None, :, :]) ** 2, axis=2)
    yy_dist2 = np.sum((y_samples[:, None, :] - y_samples[None, :, :]) ** 2, axis=2)
    xy_dist2 = np.sum((x_samples[:, None, :] - y_samples[None, :, :]) ** 2, axis=2)

    term_xx = np.sum(np.exp(-xx_dist2 / denom))
    term_yy = np.sum(np.exp(-yy_dist2 / denom))
    term_xy = np.sum(np.exp(-xy_dist2 / denom))

    mmd = np.abs(term_xx + term_yy - 2.0 * term_xy) / (samples ** 2)
    exp_2 = 1.0 - mmd

    return float(exp_2)

