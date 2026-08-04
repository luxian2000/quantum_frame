"""逐 shot 与聚合的记录结构。

raw_syndromes 与 detection_events **并存**：M3 要画错误链，而「差分」表达不出错误链。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QECShotRecord:
    """一条 shot 的完整记录。"""
    shot: int
    seed: int
    injected_errors: list = field(default_factory=list)
    raw_syndromes: np.ndarray | None = None
    detection_events: np.ndarray | None = None
    decode_steps: list = field(default_factory=list)
    commit_latency: np.ndarray | None = None
    backlog: np.ndarray | None = None
    wall_clock: np.ndarray | None = None
    observable_raw: np.ndarray | None = None
    frame_flips: np.ndarray | None = None
    verdict: str = "corrected"


@dataclass
class QECResult:
    """一次 run(...) 的聚合结果。"""
    code_name: str
    decoder_name: str
    schedule_name: str
    rounds: int
    shots: int
    records: list = field(default_factory=list)
    failure_records: list = field(default_factory=list)
    logical_error_rate: float = 0.0
    logical_error_rate_stderr: float = 0.0
    verdict_counts: dict = field(default_factory=dict)
    max_backlog: float | None = None
    mean_commit_latency: float | None = None
    budget_violations: int | None = None

    def summary(self) -> str:
        lines = [
            f"码 {self.code_name} · 调度 {self.schedule_name} · 解码器 {self.decoder_name}",
            f"轮数 {self.rounds} · shots {self.shots}",
            f"逻辑错误率 {self.logical_error_rate:.6g} ± {self.logical_error_rate_stderr:.3g}",
            f"判定分布 {dict(sorted(self.verdict_counts.items()))}",
        ]
        if self.max_backlog is not None:
            lines.append(
                f"最大 backlog {self.max_backlog:.6g}s · 平均提交延迟 "
                f"{self.mean_commit_latency:.6g}s · 超预算轮数 {self.budget_violations}"
            )
        return "\n".join(lines)
