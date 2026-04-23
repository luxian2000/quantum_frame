"""
quantum_sim/execution/sampler.py

采样器逻辑：基于后端概率向量进行 shots 次采样并转换为 bitstring 计数字典。
"""

from __future__ import annotations

from typing import Dict


class Sampler:
    """采样器，负责将概率分布转为离散测量计数。"""

    def __init__(self, backend):
        self.backend = backend

    def sample_counts(self, probs, n_qubits: int, shots: int = 1024) -> Dict[str, int]:
        """
        按概率采样并返回 bitstring 计数字典（只保留非零项）。

        参数:
            probs: 后端原生概率向量，shape (2^n,)
            n_qubits: 量子比特数
            shots: 采样次数
        """
        if shots <= 0:
            raise ValueError("shots 必须为正整数")

        counts_vec = self.backend.sample(probs, shots)
        counts_np = self.backend.to_numpy(counts_vec).astype(int).reshape(-1)
        return {
            f"|{idx:0{n_qubits}b}>": int(c)
            for idx, c in enumerate(counts_np)
            if c > 0
        }
