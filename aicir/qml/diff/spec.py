"""DiffMethod：微分方法的单点描述（NEXT.md §6）。

注册表按 ``category`` 分三类，契约各异：

- ``fn_gradient``：``(fn, params, **kw) -> 梯度向量``（psr/fd/auto/spsa/spsr）；
  唯一参与 ``resolve_diff``/``select_diff`` 自动分发的类别。
- ``circuit_gradient``：``(circuit, observable, **kw) -> 梯度``（ad，伴随微分）。
- ``preconditioner``：``(fn, state_fn, params, **kw) -> 方向/度规``（qng 族）。

镜像 ``aicir.gates.GateSpec`` 的 frozen dataclass + 模块级注册表习惯。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

DIFF_CATEGORIES = ("fn_gradient", "circuit_gradient", "preconditioner")


@dataclass(frozen=True)
class DiffMethod:
    """单个微分方法的元信息。

    - ``name``：规范方法名，如 ``"psr"``。
    - ``fn``：对应可调用；契约由 ``category`` 决定（见模块 docstring）。
    - ``aliases``：等价写法，如 ``("multipsr",)``。
    - ``category``：``fn_gradient``/``circuit_gradient``/``preconditioner`` 之一。
    - ``exact``：是否精确（psr/auto/ad 精确；fd/spsa/spsr 近似）。
    - ``stochastic``：是否随机（spsa、spsr）。
    - ``requires_torch``：是否需要 Torch 系后端（auto）。
    - ``supports_shots``：是否支持有限 shots。
    - ``supports_noise``：是否支持噪声线路。

    其中 ``exact``/``stochastic``/``requires_torch``/``supports_*`` 只为
    ``fn_gradient`` 的 ``select_diff`` 自动优选服务；非梯度类别这些字段惰性。
    """

    name: str
    fn: Callable[..., Any]
    aliases: tuple[str, ...] = ()
    category: str = "fn_gradient"
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
        if self.category not in DIFF_CATEGORIES:
            raise ValueError(f"DiffMethod category must be one of {DIFF_CATEGORIES}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "aliases", tuple(str(a) for a in self.aliases))
