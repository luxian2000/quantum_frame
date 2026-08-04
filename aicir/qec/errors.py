"""逐 shot 随机 Pauli 误差模型。

态始终保持**纯态矢量**——不走密度矩阵。现有噪声路径一挂 NoiseModel 就把态升为
密度矩阵，17 比特需 275 TB，对 QEC 完全不可行；而 QEC 基准本来就用随机 Pauli 采样。

测量误差作用在**经典记录**上（翻转读数比特），不作用于量子态。这与 Stim 的
「MR 前加 X_ERROR」等价，更省，且传播语义正确——解码器随后基于被污染的综合征
动作，这正是被测行为。
"""

from __future__ import annotations

from dataclasses import dataclass

_CHANNELS = {
    "bit_flip": ("X",),
    "phase_flip": ("Z",),
    "depolarizing": ("X", "Y", "Z"),
}


@dataclass(frozen=True)
class ErrorEvent:
    """一次错误注入。source="data" 作用于量子态，"measurement" 翻转经典读数位。"""
    round_index: int
    qubit: int
    pauli: str
    source: str


class PauliErrorModel:
    """每 data 比特每轮以 p_data 出错；每 ancilla 读数每轮以 p_measure 翻转。"""

    def __init__(self, p_data: float = 0.0, p_measure: float = 0.0,
                 channel: str = "depolarizing"):
        for label, p in (("p_data", p_data), ("p_measure", p_measure)):
            if not 0.0 <= float(p) <= 1.0:
                raise ValueError(f"{label} 必须是 [0,1] 区间内的概率，收到 {p}")
        channel = str(channel)
        if channel not in _CHANNELS:
            raise ValueError(f"未知 channel {channel!r}；可用：{sorted(_CHANNELS)}")
        self.p_data = float(p_data)
        self.p_measure = float(p_measure)
        self.channel = channel

    def sample_round(self, round_index: int, n_data: int, n_ancilla: int, rng) -> list[ErrorEvent]:
        """采样该轮的全部错误事件。"""
        paulis = _CHANNELS[self.channel]
        events: list[ErrorEvent] = []
        if self.p_data > 0.0:
            for q in range(int(n_data)):
                if rng.random() < self.p_data:
                    p = paulis[0] if len(paulis) == 1 else paulis[rng.integers(len(paulis))]
                    events.append(ErrorEvent(int(round_index), q, p, "data"))
        if self.p_measure > 0.0:
            for a in range(int(n_ancilla)):
                if rng.random() < self.p_measure:
                    events.append(ErrorEvent(int(round_index), a, "flip", "measurement"))
        return events

    @staticmethod
    def data_events(events) -> list[ErrorEvent]:
        return [e for e in events if e.source == "data"]

    @staticmethod
    def measurement_events(events) -> list[ErrorEvent]:
        return [e for e in events if e.source == "measurement"]

    def __repr__(self) -> str:
        return (f"PauliErrorModel(p_data={self.p_data}, p_measure={self.p_measure}, "
                f"channel={self.channel!r})")
