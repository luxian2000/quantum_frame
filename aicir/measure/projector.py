"""测量投影 / 重置 / 末端读出的后端无关纯函数（numpy 主机计算）。

约定：bit_order="msb"，qubit q 对应 flat index 第 (n-1-q) 位、reshape [2]*n 后第 q 轴。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..core.state import State

_H = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
_S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
_SDG = np.array([[1, 0], [0, -1j]], dtype=np.complex128)

# 设备常量缓存：{(id(backend), tag): 后端张量}。基变换 2x2 门与宇称掩码
# 按后端各 cast 一次（跨 shot 复用），避免每次操作重复 H2D 上传。
_DEVICE_CACHE: Dict[tuple, object] = {}


def _dev_const(backend, tag, build):
    """按 (backend, tag) 缓存 backend.cast(build()) 的设备常量。"""
    key = (id(backend), tag)
    value = _DEVICE_CACHE.get(key)
    if value is None:
        value = backend.cast(build())
        _DEVICE_CACHE[key] = value
    return value


def _dev_host(tag, build):
    """缓存主机侧 numpy float32 常量（密度路径 probabilities 为 numpy）。"""
    key = ("host", tag)
    value = _DEVICE_CACHE.get(key)
    if value is None:
        value = np.asarray(build(), dtype=np.float32)
        _DEVICE_CACHE[key] = value
    return value


def _dev_real(backend, tag, build):
    """按 (backend, tag) 缓存**实数**设备常量（torch 后端保持实 dtype）。

    与实数概率张量做逐元素乘/求和时须避免复数化——NPU 复数 reduce/加法
    kernel 缺失（aclnnAdd/DT_COMPLEX64），实数路径则完全受支持。
    """
    key = (id(backend), "real", tag)
    value = _DEVICE_CACHE.get(key)
    if value is None:
        arr = np.asarray(build(), dtype=np.float32)
        device = getattr(backend, "_device", None)
        if device is not None:
            import torch
            value = torch.as_tensor(arr, dtype=torch.float32, device=device)
        else:
            value = arr
        _DEVICE_CACHE[key] = value
    return value


def _apply_1q_sv(psi, n: int, q: int, u_dev, bk):
    """对纯态向量（后端张量, (2^n,1)）施加单比特门，设备侧完成。"""
    left = 1 << q
    right = 1 << (n - 1 - q)
    t = bk.reshape(psi, (left, 2, right))
    t = bk.tensordot(u_dev, t, ([1], [1]))      # (2, L, R)
    t = bk.transpose(t, [1, 0, 2])
    return bk.reshape(t, (1 << n, 1))


def _apply_1q_dm(rho, n: int, q: int, u_dev, u_conj_dev, bk):
    """对密度矩阵（后端张量, (2^n,2^n)）施加单比特门 U ρ U†，设备侧完成。

    工作张量秩恒为 6（NPU aclnnComplex 限 8 维，不可用 [2]*2n 整形）。
    """
    left = 1 << q
    right = 1 << (n - 1 - q)
    t = bk.reshape(rho, (left, 2, right, left, 2, right))
    t = bk.tensordot(u_dev, t, ([1], [1]))       # (2, L,R,L,2,R)
    t = bk.transpose(t, [1, 0, 2, 3, 4, 5])      # (L,2,R,L,2,R)
    t = bk.tensordot(u_conj_dev, t, ([1], [4]))  # (2, L,2,R,L,R)
    t = bk.transpose(t, [1, 2, 3, 4, 0, 5])      # (L,2,R,L,2,R)
    return bk.reshape(t, (1 << n, 1 << n))


def _basis_change_seq(basis: str, inverse: bool) -> List[np.ndarray]:
    """返回把单比特从 basis 旋到 Z 所需、按施加顺序排列的单比特门。

    X: 前向/逆向均为 H（自逆）。
    Y: 前向 Sdg 然后 H；逆向 H 然后 S。
    Z: 无操作（计算基）。
    """
    basis = basis.upper()
    if basis == "Z":
        return []
    if basis == "X":
        return [_H]
    if basis == "Y":
        return [_SDG, _H] if not inverse else [_H, _S]
    raise ValueError(f"未知 basis {basis!r}")


def pauli_basis_change(state: State, qubits: Sequence[int], basis: str, inverse: bool) -> State:
    """对指定量子比特施加 Pauli 基变换，将计算基旋转到目标基（或逆操作）。

    参数:
        state:   输入量子态（纯态或密度矩阵）
        qubits:  需要进行基变换的量子比特列表
        basis:   目标 Pauli 基，"X" / "Y" / "Z"
        inverse: False 表示旋到目标基；True 表示逆变换（旋回计算基）

    返回:
        变换后的新 State 对象
    """
    backend = state.backend
    n = state.n_qubits
    seq = _basis_change_seq(basis, inverse)
    if not seq:
        return state
    # 设备侧逐比特施加 2x2 门；门常量按 (backend, 矩阵) 各 cast 一次
    if state.is_density:
        rho = state.data
        for q in qubits:
            for U in seq:
                u = _dev_const(backend, ("u", id(U)), lambda U=U: U)
                uc = _dev_const(backend, ("uc", id(U)), lambda U=U: U.conj())
                rho = _apply_1q_dm(rho, n, int(q), u, uc, backend)
        return State(rho, n, backend)
    psi = state.data
    for q in qubits:
        for U in seq:
            u = _dev_const(backend, ("u", id(U)), lambda U=U: U)
            psi = _apply_1q_sv(psi, n, int(q), u, backend)
    return State(psi, n, backend, bit_order=state.bit_order)


def _parity_mask(n: int, qubits: Sequence[int]) -> int:
    """构造比特掩码：选中 qubits 列表对应的 flat-index 比特位（msb 约定）。"""
    m = 0
    for q in qubits:
        m |= 1 << (n - 1 - int(q))
    return m


def _parities(dim: int, mask: int) -> np.ndarray:
    """返回每个 flat index 在 mask 选中比特上的宇称（0=偶,1=奇）。"""
    idx = np.arange(dim, dtype=np.int64) & mask
    p = idx.copy()
    shift = 32
    while shift:
        p ^= p >> shift
        shift >>= 1
    return (p & 1).astype(np.int64)


def _parity_probs_rotated(rotated: State, qubits: Sequence[int]) -> Tuple[float, float]:
    """在已旋到 Z 基的态上按选中比特宇称分桶概率，返回 (p_plus, p_minus)。

    设备侧掩码点积求 p_plus，只下传一个标量；偶宇称指示掩码按
    (backend, n, 掩码位) 各 cast 一次（跨 shot 复用）。
    """
    n = rotated.n_qubits
    backend = rotated.backend
    mask_int = _parity_mask(n, qubits)
    # 实数掩码 × 实数概率：全程实 dtype（NPU 无复数 reduce/加法 kernel）。
    # probs 向量态为后端张量、密度态为 numpy——掩码按同侧缓存，避免跨设备混用
    probs = rotated.probabilities()
    if isinstance(probs, np.ndarray):
        even = _dev_host(("even", n, mask_int),
                         lambda: (_parities(1 << n, mask_int) == 0))
    else:
        even = _dev_real(backend, ("even", n, mask_int),
                         lambda: (_parities(1 << n, mask_int) == 0))
    weighted = probs.reshape(-1) * even
    p_plus = float(np.real(np.asarray(backend.to_numpy(weighted.sum()))))
    p_plus = min(max(p_plus, 0.0), 1.0)
    return p_plus, 1.0 - p_plus


def joint_parity_probs(state: State, qubits: Sequence[int], basis: str) -> Tuple[float, float]:
    """计算联合 Pauli 串 P 取本征值 +1 / -1 的概率（Born 规则）。

    做法：把指定比特旋到 Z 基后，按选中比特的宇称把概率分桶——
    偶宇称对应 P=+1，奇宇称对应 P=-1。
    返回 (p_plus, p_minus)。
    """
    rotated = pauli_basis_change(state, qubits, basis, inverse=False)
    return _parity_probs_rotated(rotated, qubits)


def _project_parity_rotated(rotated: State, qubits: Sequence[int], lam: int) -> State:
    """在已旋到 Z 的态上，投影到联合宇称 lam(±1) 子空间并归一化（保持子空间内相干）。

    设备侧掩码乘法 + 归一化；宇称保留掩码按 (backend, n, 掩码位, lam)
    各 cast 一次（跨 shot 复用）。
    """
    backend = rotated.backend
    n = rotated.n_qubits
    mask_int = _parity_mask(n, qubits)
    target = 0 if lam == 1 else 1
    keep = _dev_const(backend, ("pkeep", n, mask_int, target),
                      lambda: (_parities(1 << n, mask_int) == target).astype(np.float64))
    return _masked_normalize(rotated, keep, backend)


def measure_joint_pauli(state: State, qubits: Sequence[int], basis: str, rng) -> Tuple[State, int]:
    """非破坏性联合 Pauli 投影测量：返回 (坍缩后完整态, 本征值 lam∈{+1,-1})。

    实现的是真正的联合本征空间投影 Π_λ=(I+λP)/2，而非逐比特测量的乘积：
    单次两结果投影只坍缩 ±1 宇称子空间，保持子空间内部相干。
    """
    rotated = pauli_basis_change(state, qubits, basis, inverse=False)
    p_plus, _ = _parity_probs_rotated(rotated, qubits)
    lam = 1 if rng.random() < p_plus else -1
    projected = _project_parity_rotated(rotated, qubits, lam)
    restored = pauli_basis_change(projected, qubits, basis, inverse=True)
    return restored, lam


def _reset_dm(rho: np.ndarray, n: int, q: int) -> np.ndarray:
    """对密度矩阵施加重置信道 R_q(ρ)=K0 ρ K0† + K1 ρ K1†（K0=|0><0|, K1=|0><1|）。

    结果中目标比特恒处 |0>：仅在目标比特行/列均为 0 的子块上累加
    out[r,c] = rho[r,c] + rho[r1,c1]（r1/c1 为把 r/c 的目标比特置 1 后的 index）。
    向量化实现：整形为 (L,2,R, L,2,R) 秩-6 张量后对目标比特 0/0 与 1/1
    两个对角子块求和，无逐元素 Python 循环。
    """
    dim = 1 << n
    left = 1 << q
    right = 1 << (n - 1 - q)
    t = rho.reshape(left, 2, right, left, 2, right)
    out = np.zeros_like(t)
    out[:, 0, :, :, 0, :] = t[:, 0, :, :, 0, :] + t[:, 1, :, :, 1, :]
    return out.reshape(dim, dim)


def _reset_one(state: State, q: int) -> State:
    """对单个量子比特施加重置信道：

    - 密度矩阵输入：直接对 ρ 施加信道；
    - 纯态且目标比特与其余比特可分离（product）：结果仍为纯态 |0>_q ⊗ (rest)；
    - 纯态且目标比特纠缠：结果为混合态，升级为密度矩阵。
    """
    backend = state.backend
    n = state.n_qubits
    if state.is_density:
        rho = backend.to_numpy(state.data).reshape(1 << n, 1 << n).astype(np.complex128)
        return State(backend.cast(_reset_dm(rho, n, int(q))), n, backend)

    psi = backend.to_numpy(state.data).reshape([2] * n).astype(np.complex128)
    sl0 = [slice(None)] * n; sl0[q] = 0
    sl1 = [slice(None)] * n; sl1[q] = 1
    a = psi[tuple(sl0)].reshape(-1)
    b = psi[tuple(sl1)].reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)

    # 判断目标比特是否与其余比特可分离：q=0 切片 a 与 q=1 切片 b 是否平行
    parallel = False
    if na < 1e-12 or nb < 1e-12:
        parallel = True
    else:
        # 相对阈值：残差是 b 相对 a 的正交分量，可分离时仅为后端精度噪声
        # （complex64 约 1e-7·‖b‖），纠缠时为 O(‖b‖)，故按 ‖b‖ 归一化判定
        c = np.vdot(a, b) / (na * na)
        parallel = np.linalg.norm(b - c * a) < 1e-6 * nb

    if parallel:
        # 可分离：其余比特的纯态即非零切片归一化，目标比特重置为 |0>
        v = a if na >= nb else b
        v = v / np.linalg.norm(v)
        out = np.zeros([2] * n, dtype=np.complex128)
        out[tuple(sl0)] = v.reshape(out[tuple(sl0)].shape)
        return State(backend.cast(out.reshape(-1, 1)), n, backend, bit_order=state.bit_order)

    # 纠缠：构造 ρ=|ψ><ψ| 后施加信道，升级为密度矩阵
    flat = psi.reshape(-1, 1)
    rho = (flat @ flat.conj().T)
    rho = _reset_dm(rho, n, int(q))
    return State(backend.cast(rho), n, backend)


def reset_channel(state: State, qubits: Sequence[int]) -> State:
    """对一组量子比特依次施加重置信道 R_S(ρ)=|0><0|_S ⊗ Tr_S(ρ)。

    无需事先测量；逐比特处理，纠缠目标会把纯态升级为密度矩阵。
    """
    for q in qubits:
        state = _reset_one(state, int(q))
    return state


def _born_probs(state: State) -> np.ndarray:
    """计算基 Born 分布（裁剪负值并归一化的 numpy float64 向量）。

    经 State.probabilities 取概率：密度态在设备侧取对角线后只下传 2^n
    向量，避免为取对角线传输整个 (2^n,2^n) 密度矩阵。
    """
    backend = state.backend
    probs = np.asarray(backend.to_numpy(state.probabilities()), dtype=np.float64).reshape(-1)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    return probs / total if total > 0 else np.full_like(probs, 1.0 / probs.size)


def _born_sample_index(state: State, rng) -> int:
    """按 Born 规则从计算基分布中采样一个 flat index。"""
    probs = _born_probs(state)
    return int(rng.choice(probs.size, p=probs))


def _masked_normalize(state: State, keep_dev, backend) -> State:
    """设备侧掩码投影 + 归一化。keep_dev 为 (2^n,) 0/1 指示（后端张量）。

    向量态：psi ⊙ keep 后除以范数；密度态：行/列各乘一次掩码后按迹归一。
    归一化因子经标量下传后按复数常量乘回，全程不下传整态。
    """
    n = state.n_qubits
    dim = 1 << n
    if state.is_density:
        rho = backend.mul(state.data, backend.reshape(keep_dev, (dim, 1)))
        rho = backend.mul(rho, backend.reshape(keep_dev, (1, dim)))
        tr = float(np.real(np.asarray(backend.to_numpy(backend.trace(rho)))))
        if tr > 0:
            rho = backend.mul(rho, backend.cast(np.complex128(1.0 / tr)))
        return State(rho, n, backend)
    psi = backend.mul(state.data, backend.reshape(keep_dev, (dim, 1)))
    norm2 = float(np.real(np.asarray(backend.to_numpy(backend.abs_sq(psi).sum()))))
    if norm2 > 0:
        psi = backend.mul(psi, backend.cast(np.complex128(1.0 / np.sqrt(norm2))))
    return State(psi, n, backend, bit_order=state.bit_order)


def _project_subset_outcome(state: State, qubits: Sequence[int], bits: Sequence[int]) -> State:
    """把指定比特投影到给定 0/1 取值（其余比特保留），归一化（设备侧）。"""
    backend = state.backend
    n = state.n_qubits
    key = ("skeep", n, tuple(int(q) for q in qubits), tuple(int(b) for b in bits))

    def build():
        keep = np.ones(1 << n, dtype=bool)
        idx = np.arange(1 << n, dtype=np.int64)
        for q, bit in zip(qubits, bits):
            shift = n - 1 - int(q)
            keep &= (((idx >> shift) & 1) == int(bit))
        return keep.astype(np.float64)

    return _masked_normalize(state, _dev_const(backend, key, build), backend)


def sample_terminal_batch(state: State, measure_qubits: Sequence[int], shots: int, rng,
                          *, collapse: bool = True) -> Tuple[List[List[int]], Dict[tuple, State]]:
    """无中途随机源路径的末端 Z 基批量采样。

    与逐 shot 调用 terminal_z_measure 同分布：按 Born 规则一次性抽取 shots 个
    全寄存器计算基 index（分布只计算一次，O(2^n + shots)），读取各被测比特
    0/1 值；坍缩后完整态只按不同读出结果各构造一次（collapse=False 时不构造）。

    返回 (outcomes, posts)：
        outcomes: 长度 shots 的本征值列表（每项按 measure_qubits 顺序、取值 ±1）
        posts:    {本征值元组: 坍缩后完整态}；collapse=False 时为空字典
    """
    n = state.n_qubits
    probs = _born_probs(state)
    indices = rng.choice(probs.size, size=int(shots), p=probs)

    outcomes: List[List[int]] = []
    posts: Dict[tuple, State] = {}
    for x in indices:
        bits = [(int(x) >> (n - 1 - int(q))) & 1 for q in measure_qubits]
        eig = [1 if bit == 0 else -1 for bit in bits]
        outcomes.append(eig)
        key = tuple(eig)
        if collapse and key not in posts:
            posts[key] = _project_subset_outcome(state, measure_qubits, bits)
    return outcomes, posts


def terminal_z_measure(state: State, measure_qubits: Sequence[int], rng) -> Tuple[State, List[int]]:
    """对 measure_qubits 逐比特 Z 基测量（输入顺序保留）。

    返回 (坍缩后完整态, 本征值列表[按 measure_qubits 顺序, 取值 ±1])。
    实现：按 Born 规则采样全寄存器计算基 index，读取各被测比特的 0/1 值，
    对该子集比特模式做投影并归一化。本征值约定：比特 0 → +1，比特 1 → -1。
    """
    n = state.n_qubits
    x = _born_sample_index(state, rng)
    bits = [(x >> (n - 1 - int(q))) & 1 for q in measure_qubits]
    collapsed = _project_subset_outcome(state, measure_qubits, bits)
    eig = [1 if bit == 0 else -1 for bit in bits]
    return collapsed, eig
