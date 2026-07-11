"""跨模块结果/优化器协议（Phase 1：跨层契约统一）。

统一结果词汇表：``value``（目标值/能量）、``parameters``（最优参数）、
``history``（逐步记录，通常为 :class:`HistoryRecord` 或 dict 序列）、
``metadata``（附加信息字典）；``learning_rate`` 的常用别名为 ``lr``。

``circuit`` 不是协议成员：``VQEResult`` 等携带的 circuit/statevector 等字段
属于按约定提供的可选实现细节，不强制所有结果类型都暴露。

本模块只依赖 ``typing``/``dataclasses``，不引入任何重依赖（torch/scipy 等），
可在任意层安全导入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AlgorithmResult(Protocol):
    """跨算法结果协议：``value``/``parameters``/``history``/``metadata`` 四个只读成员。

    ``runtime_checkable`` 的 Protocol 在 ``isinstance`` 检查时只验证成员是否存在
    （``hasattr``），不检查类型或签名；数据类字段与等价 ``@property`` 别名都满足。
    """

    value: float
    parameters: Any
    history: Any
    metadata: Any


@runtime_checkable
class Optimizer(Protocol):
    """经典参数优化器协议：暴露 ``minimize(fn, init_params, ...) -> OptimizationResult``。

    ``runtime_checkable`` 只检查 ``minimize`` 方法是否存在，不检查具体签名。
    """

    def minimize(
        self,
        fn: Any,
        init_params: Any,
        *,
        gradient_fn: Any = None,
        callback: Any = None,
    ) -> Any:
        ...


@dataclass
class HistoryRecord:
    """优化器逐步历史记录，兼容旧版 dict 键访问（``record["fun"]``/``record.get(...)``）。

    查找顺序：具名字段（``step``/``fun``/``grad_norm``/``learning_rate``）优先，
    找不到再查 ``extras``（各优化器特有的额外字段，如 SPSA 的 ``perturbation``）。
    """

    step: int
    fun: float
    grad_norm: float | None = None
    learning_rate: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    _NAMED_FIELDS = ("step", "fun", "grad_norm", "learning_rate")

    def __getitem__(self, key: str) -> Any:
        if key in self._NAMED_FIELDS:
            return getattr(self, key)
        if key in self.extras:
            return self.extras[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        return key in self._NAMED_FIELDS or key in self.extras


__all__ = ["AlgorithmResult", "Optimizer", "HistoryRecord"]
