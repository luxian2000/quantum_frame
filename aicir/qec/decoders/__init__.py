"""在线解码协议。

因果性是**结构性**的，不是约定性的：update(t, …) 是唯一输入通道，按序每轮恰调用
一次；解码器不持有线路、码、量子态或未来轮次的引用。它无法偷看未来，因为轮 t+1
尚未被模拟。批式后处理平台只能靠自律保证这一点，这里它是架构性质。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


@dataclass
class DecodeStep:
    """一次 update/flush 的输出。"""
    frame_flips: np.ndarray | None = None
    corrections: list | None = None
    committed_through: int = -1
    cost: float = 0.0


class OnlineDecoder(Protocol):
    name: str
    window: int
    commit_lag: int

    def reset(self, layout) -> None: ...
    def update(self, round_index: int, events: np.ndarray) -> DecodeStep: ...
    def flush(self) -> DecodeStep: ...
    def cost_of(self, round_index: int, events: np.ndarray) -> float: ...


DECODERS: dict[str, Callable] = {}


def register_decoder(name: str, factory: Callable) -> None:
    DECODERS[str(name)] = factory


def resolve_decoder(name_or_obj, **kwargs):
    """名字 → 解码器实例；已经是实例则原样返回。"""
    if not isinstance(name_or_obj, str):
        return name_or_obj
    if name_or_obj not in DECODERS:
        raise KeyError(f"未知解码器 {name_or_obj!r}；可用：{sorted(DECODERS)}")
    return DECODERS[name_or_obj](**kwargs)


from .lookup import LookupDecoder  # noqa: E402  自注册

__all__ = ["DecodeStep", "OnlineDecoder", "DECODERS",
           "register_decoder", "resolve_decoder", "LookupDecoder"]
