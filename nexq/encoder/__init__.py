"""nexq.encoder

编码器模块骨架（与 `circuit` 平级）。

该包用于放置量子态编码与态准备相关工具，例如角度编码（angle encoding）、幅值编码（amplitude encoding）、基态编码等。
当前仅创建包骨架，后续可在此添加具体编码器实现与工具函数。

示例::

    from nexq import encoder
    # TODO: from nexq.encoder import AngleEncoder, AmplitudeEncoder

"""

__all__ = []
# Encoder module for quantum data encoding

from .base import BaseEncoder
from .encoders import AmplitudeEncoder, AngleEncoder, BasisEncoder

__all__ = [
    "BaseEncoder",
    "AmplitudeEncoder",
    "AngleEncoder",
    "BasisEncoder",
]
