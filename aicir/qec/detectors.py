"""Detector / Observable / DetectorLayout —— 解码器唯一被允许知道的东西。

沿用 Stim 语义，使 M2 的互操作成为格式转换而非语义翻译：
- measurement record：一条 shot 内所有线路中测量的扁平有序列表，下标 i = 第 i 个执行的 measure。
- Detector：一组 record 下标，其奇偶在**无噪声线路中确定为 0**。
- Observable：一组 record 下标，其奇偶给出某逻辑算符取值。

解码器在 reset() 时拿到 DetectorLayout，此后只收 detection event 比特向量流。
它不持有线路、码、量子态或后端的任何引用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Detector:
    """一个 detector：records 的奇偶在无噪声下恒为 0。"""
    index: int
    records: tuple[int, ...]
    stabilizer: int
    round_index: int


@dataclass(frozen=True)
class Observable:
    """一个逻辑可观测量：records 的奇偶给出逻辑算符取值。"""
    index: int
    records: tuple[int, ...]


@dataclass(frozen=True)
class DetectorLayout:
    """解码器面向的布局描述。

    round0_stabilizers：轮 0 有 detector 的生成元下标。|0…0⟩ 一般不在码空间内，
    轮 0 的提取本身把态投影进码空间；只有在制备基下确定的生成元（|0⟩ 制备时即
    x 块全零的 Z 型）轮 0 读数才确定，其余是 50/50 随机的，不构成 detector。
    """
    n_detectors: int
    n_rounds: int
    n_stabilizers: int
    detectors: tuple[Detector, ...]
    observables: tuple[Observable, ...]
    coords: dict = field(default_factory=dict)
    round0_stabilizers: tuple[int, ...] = ()

    def detector_at(self, stabilizer: int, round_index: int) -> Detector:
        """按 (稳定子, 轮) 取 detector。"""
        for det in self.detectors:
            if det.stabilizer == stabilizer and det.round_index == round_index:
                return det
        raise KeyError(f"没有 (stabilizer={stabilizer}, round={round_index}) 对应的 detector")

    def round_slice(self, round_index: int) -> tuple[int, ...]:
        """该轮全部 detector 的全局下标，按稳定子序。"""
        return tuple(
            d.index for d in sorted(
                (d for d in self.detectors if d.round_index == round_index),
                key=lambda d: d.stabilizer,
            )
        )

    def detection_events(self, raw_syndromes, round_index: int, reference) -> np.ndarray:
        """由原始稳定子读数算出该轮的 detection event。

        raw_syndromes: (rounds, n_stabilizers) uint8 的原始读数
        reference:     (n_stabilizers,) uint8，轮 0 参考值（恒为 zeros）
        轮 0 与 reference 比较，且**只保留 round0_stabilizers 内的分量**（其余生成元
        轮 0 读数随机、不构成 detector，掩为 0）；其余轮与上一轮比较、不掩码。

        返回形状恒为 (n_stabilizers,)，使解码器协议与全部测试的形状保持统一。
        """
        raw = np.asarray(raw_syndromes, dtype=np.uint8)
        if round_index != 0:
            return (raw[round_index] ^ raw[round_index - 1]).astype(np.uint8)
        events = (raw[0] ^ np.asarray(reference, dtype=np.uint8)).astype(np.uint8)
        mask = np.zeros(self.n_stabilizers, dtype=np.uint8)
        for s in self.round0_stabilizers:
            mask[s] = 1
        return (events & mask).astype(np.uint8)
