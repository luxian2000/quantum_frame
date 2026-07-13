"""Optimization tools.

This package currently exposes QUBO modeling utilities. ``aicir.optimization.sb``
（样本/子空间优化策略占位包）已从本包的公开导出中移除——它是已弃用（deprecated）
占位包，计划在后续版本删除；仍可通过 ``import aicir.optimization.sb`` 显式导入
（会触发 ``DeprecationWarning``），但不再是 ``aicir.optimization`` 的自动子属性。
"""

from . import qubo

__all__ = ["qubo"]
