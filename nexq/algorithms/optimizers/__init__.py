"""nexq.algorithms.optimizers

优化器和参数更新策略的模块。可放置 Adam、SGD 变体、量子特定优化器（SPSA 等）。
"""

from . import qubo, sb

__all__ = ["qubo", "sb"]
