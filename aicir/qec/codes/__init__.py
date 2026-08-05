"""内置稳定子码注册表。

沿用 aicir.chemistry.molecules 的惯例：每个码一个模块，模块内自注册进 CODES。
"""

from __future__ import annotations

from typing import Callable

CODES: dict[str, Callable] = {}


def register_code(name: str, builder: Callable) -> None:
    """把码构造器注册进 CODES。"""
    CODES[str(name)] = builder


def get_code(name: str, **kwargs):
    """按名取码；kwargs 透传给构造器（如 repetition 的 d/basis）。"""
    key = str(name)
    if key not in CODES:
        raise KeyError(f"未知码 {key!r}；可用：{sorted(CODES)}")
    return CODES[key](**kwargs)


from . import five_qubit, repetition, shor, steane, surface  # noqa: E402,F401  自注册

__all__ = ["CODES", "register_code", "get_code"]
