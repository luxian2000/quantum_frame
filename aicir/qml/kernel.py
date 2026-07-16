"""量子核（qml 成熟化 #5）：BatchSV 整批演化 + 实/虚 gram。

量子核 ``K(x, z) = |<Φ(x)|Φ(z)>|²``，其中 ``Φ(x)`` 为特征映射线路末态。核矩阵
``K[i,j] = |<Φ(Xᵢ)|Φ(Zⱼ)>|²`` 可直接喂经典 SVM（预计算核）。

高效实现：用 :class:`~aicir.core.batch.BatchSV` **一次**批量演化全部 N 个特征态
（O(N) 次演化），再以实/虚分离矩阵乘算 gram（全实数 matmul，NPU 安全），替代
逐对重演化的 O(N²) 路径（对比 ``aicir.encoder.IQPEncoder.kernel_matrix``）。

``feature_map`` 为含符号 :class:`~aicir.core.circuit.Parameter` 的模板线路，**全部**
参数为数据驱动（首用序对应输入列）；参数门限 BatchSV 支持的
``rx/ry/rz/crx/cry/crz/rzz/rxx``。非线性特征映射（如 IQP 的 ``(π-xᵢ)(π-xⱼ)``）由
调用方预先算入输入角度矩阵。
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # torch 可选
    import torch
except ImportError:  # pragma: no cover
    torch = None

from ..core.batch import BatchSV
from ..core.circuit import Circuit, Parameter, cx, rx
from ..gates import canonical_gate_name
from ..ir import instruction_name

_BATCH_PARAM_GATES = {"rx", "ry", "rz", "crx", "cry", "crz", "rzz", "rxx"}


def angle_feature_map(n_qubits: int, *, entangle: bool = True) -> Circuit:
    """简单角度特征映射模板：每比特一个 ``rx`` 数据编码 + 可选 cx 环。

    参数数 = ``n_qubits``（一列一比特）。非线性/重上传特征映射请自建模板。
    """
    xs = [Parameter(f"x{i}") for i in range(int(n_qubits))]
    gates: list = [rx(xs[i], i) for i in range(n_qubits)]
    if entangle and n_qubits > 1:
        for q in range(n_qubits):
            gates.append(cx((q + 1) % n_qubits, [q]))
    return Circuit(*gates, n_qubits=n_qubits)


class QuantumKernel:
    """基于 BatchSV 的量子核：批量演化特征态 + 实/虚 gram。

    Args:
        feature_map: 含符号 ``Parameter`` 的特征映射模板（全部参数数据驱动）。
        backend: torch 系后端（``GPUBackend``/``NPUBackend``），决定 device/dtype。
    """

    def __init__(self, feature_map: Circuit, *, backend) -> None:
        if torch is None:  # pragma: no cover
            raise ImportError("QuantumKernel 需要 torch")
        params = list(feature_map.parameters)
        index_of = {id(p): i for i, p in enumerate(params)}
        specs = []  # (gate, param_index or None)
        for gate in feature_map.gates:
            raw = tuple(getattr(gate, "params", ()) or ())
            symbolic = [v for v in raw if isinstance(v, Parameter)]
            if symbolic:
                name = canonical_gate_name(instruction_name(gate))
                if name not in _BATCH_PARAM_GATES or len(raw) != 1:
                    raise ValueError(
                        f"QuantumKernel 参数门仅支持单参数的 {sorted(_BATCH_PARAM_GATES)}，得到 {name}")
                specs.append((gate, index_of[id(symbolic[0])]))
            else:
                specs.append((gate, None))
        self.n_qubits = int(feature_map.n_qubits)
        self.n_params = len(params)
        self._specs = specs
        self._backend = backend

    def _amplitudes(self, X: np.ndarray):
        """批量演化 N 个特征态，返回 (real, imag)，各 (N, 2^n) 后端张量。"""
        import dataclasses

        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != self.n_params:
            raise ValueError(f"输入形状 {arr.shape} 与参数数 {self.n_params} 不符（须 (N, {self.n_params})）")
        n = arr.shape[0]
        sv = BatchSV(self.n_qubits, n, self._backend)
        cols = torch.as_tensor(arr, dtype=sv.real_dtype, device=sv.device)
        for gate, idx in self._specs:
            if idx is None:
                sv.apply_gate(gate)
            else:
                sv.apply_gate(dataclasses.replace(gate, params=(cols[:, idx],)))
        return sv.real, sv.imag

    def matrix(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        """核矩阵 ``K[i,j] = |<Φ(Xᵢ)|Φ(Zⱼ)>|²``，返回 numpy ``(N, M)``。

        ``Z=None`` 时 ``Z=X``（对称，只演化一次）。全程实数 matmul，NPU 安全。
        """
        re_x, im_x = self._amplitudes(X)
        if Z is None:
            re_z, im_z = re_x, im_x
        else:
            re_z, im_z = self._amplitudes(Z)
        # <a|b> = Σ conj(a)·b：实部 R、虚部 I 均由实数 matmul 组成
        real = re_x @ re_z.transpose(0, 1) + im_x @ im_z.transpose(0, 1)
        imag = re_x @ im_z.transpose(0, 1) - im_x @ re_z.transpose(0, 1)
        gram = real * real + imag * imag
        return gram.detach().cpu().numpy()

    def __call__(self, x: np.ndarray, z: np.ndarray) -> float:
        """单对核值 ``K(x, z)``。"""
        K = self.matrix(np.asarray(x).reshape(1, -1), np.asarray(z).reshape(1, -1))
        return float(K[0, 0])
