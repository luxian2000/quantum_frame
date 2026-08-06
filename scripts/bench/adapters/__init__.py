"""框架适配器注册表。

缺失的框架不是错误——``available_adapters()`` 只列出真正装上的，清单里会
如实记录哪些被跳过。基准因此在任何机器上都能跑，只是覆盖面不同。
"""

from __future__ import annotations

from .base import Adapter, reverse_qubit_order
from .aicir_adapter import AicirAdapter
from .cirq_adapter import CirqAdapter
from .qiskit_adapter import QiskitAdapter, QiskitAerAdapter
from .qulacs_adapter import QulacsAdapter
from .tensorcircuit_adapter import TensorCircuitAdapter

__all__ = [
    "Adapter",
    "reverse_qubit_order",
    "ADAPTERS",
    "available_adapters",
    "get_adapter",
    "adapter_versions",
]

#: 全部已知适配器（不代表已安装）
ADAPTERS: dict[str, type[Adapter]] = {
    cls.name: cls
    for cls in (
        AicirAdapter,
        QiskitAdapter,
        QiskitAerAdapter,
        CirqAdapter,
        QulacsAdapter,
        TensorCircuitAdapter,
    )
}


def available_adapters() -> tuple[str, ...]:
    """返回本机实际可用的适配器名。"""

    return tuple(name for name, cls in ADAPTERS.items() if cls.is_available())


def get_adapter(name: str, **kwargs) -> Adapter:
    """按名取适配器实例。

    未知名字抛 ``KeyError``；已知但未安装抛 ``RuntimeError``——两者要分开：
    前者是拼写错误，后者是环境缺依赖。
    """

    if name not in ADAPTERS:
        raise KeyError(f"未知适配器 {name!r}；已知：{tuple(ADAPTERS)}")
    cls = ADAPTERS[name]
    if not cls.is_available():
        raise RuntimeError(f"适配器 {name!r} 对应的框架未安装")
    return cls(**kwargs)


def adapter_versions() -> dict[str, str]:
    """返回可用框架的版本表，写入清单以保证数字可追溯。"""

    return {name: ADAPTERS[name].version() for name in available_adapters()}
