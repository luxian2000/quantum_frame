"""DiffMethod：fn-based 梯度方法的单点描述（NEXT.md §6 第一片）。

每个梯度方法在此注册一次，统一契约为 ``(fn, params, **kw) -> 梯度向量``。
镜像 ``aicir.gates.GateSpec`` 的 frozen dataclass + 模块级注册表习惯。
仅覆盖 fn-based 方法；电路型 ``ad`` 与预条件 ``qng`` 不在其中。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class DiffMethod:
    """单个梯度方法的元信息。

    - ``name``：规范方法名，如 ``"psr"``。
    - ``fn``：对应可调用，契约 ``(fn, params, **kw) -> 梯度向量``。
    - ``aliases``：等价写法，如 ``("multipsr",)``。
    - ``exact``：是否精确（psr/auto 精确；fd/spsa/spsr 近似）。
    - ``stochastic``：是否随机（spsa、spsr）。
    - ``requires_torch``：是否需要 Torch 系后端（auto）。
    - ``supports_shots``：是否支持有限 shots（auto=False）。
    - ``supports_noise``：是否支持噪声线路（auto=False）。
    """

    name: str
    fn: Callable[..., Any]
    aliases: tuple[str, ...] = ()
    exact: bool = False
    stochastic: bool = False
    requires_torch: bool = False
    supports_shots: bool = True
    supports_noise: bool = True

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("DiffMethod name cannot be empty")
        if not callable(self.fn):
            raise TypeError("DiffMethod fn must be callable")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "aliases", tuple(str(a) for a in self.aliases))
