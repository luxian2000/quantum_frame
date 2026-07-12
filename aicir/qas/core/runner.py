"""Public QAS runner API for packaged aicir users."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import config as qas_config
from .problem import QASProblem, normalize_problem

QASMethod = str

# request 字段名 -> (QASProblem 归一化时使用的 kind, 是否为 ndarray 时才生效的显式 kind)。
# 顺序即向后兼容优先级；三者互斥，且与 problem= 互斥（见 _resolve_problem）。
_LEGACY_PROBLEM_FIELDS: tuple[tuple[str, str], ...] = (
    ("hamiltonian", "hamiltonian"),
    ("target_state", "state"),
    ("target_density_matrix", "density_matrix"),
)


@dataclass
class QASRunConfig:
    """Method-agnostic request object for running a QAS implementation.

    Users can pass this object to :func:`run`, or pass the same fields as
    keyword arguments directly to ``run(method, ...)``.

    ``problem``（3b 新增）是推荐的任务输入字段，接受
    :func:`~aicir.qas.core.problem.normalize_problem` 支持的任意形态
    （``Hamiltonian``/``State``/矩阵 ndarray/态向量 ndarray/Pauli 项列表/
    已构造好的 :class:`~aicir.qas.core.problem.QASProblem`）。旧字段
    ``hamiltonian``/``target_state``/``target_density_matrix`` 继续可用
    （按方法路由到同一个归一化问题），但不能与 ``problem=`` 同时传入，
    也不能互相同时传入——``run()`` 内部会据此报 ``ValueError``。

    ``initial_ansatz``（3b 新增）仅 ``mogvqe`` 使用：拓扑起点
    （``MOGVQEIndividual | Circuit | Sequence[MOGVQEBlock]``），不属于
    ``QASProblem``/``config`` 的语义范围。
    """

    method: QASMethod
    config: Any = None
    objective: Any = None
    dataset: Any = None
    hamiltonian: Any = None
    target_state: Any = None
    target_density_matrix: Any = None
    epsilon: float | None = None
    policy_library: Any = None
    problem: Any = None
    initial_ansatz: Any = None


def available_qas_methods() -> tuple[str, ...]:
    """Return canonical method names accepted by :func:`run`.

    派生自 ``SearchStrategy`` 注册表（``core.strategies``/``core.registry``），
    按字典序排列——每个 ``core.config._FACTORIES`` 方法都对应一个已注册策略
    （见 ``core.strategies``），二者集合始终一致。
    """

    from .registry import registered_strategies

    return registered_strategies()


def default_qas_config(method: QASMethod, **kwargs: Any) -> Any:
    """Return a config object for a QAS method.

    Prefer ``aicir.qas.core.config.<method>(...)`` in user-facing code. This helper
    remains as a method-name based compatibility wrapper.
    """

    return qas_config.create(method, **kwargs)


def _resolve_problem(run_config: QASRunConfig) -> QASProblem | None:
    """把 ``problem=``/旧版 ``hamiltonian=``/``target_state=``/``target_density_matrix=``
    归一化为单个 ``QASProblem``（或 ``None``，方法不需要任务输入时）。

    ``problem=`` 与旧字段互斥；旧字段之间也互斥——同时传入两个或以上会报
    ``ValueError``，避免用户以为两者都生效但实际只有一个被消费。
    """

    provided = [(name, value, kind) for name, kind in _LEGACY_PROBLEM_FIELDS if (value := getattr(run_config, name)) is not None]

    if run_config.problem is not None and provided:
        names = ", ".join(name for name, _, _ in provided)
        raise ValueError(f"不能同时传入 problem= 与旧版关键字参数 {names}；请只使用其一。")
    if len(provided) > 1:
        names = ", ".join(name for name, _, _ in provided)
        raise ValueError(f"只能传入以下旧版关键字参数中的一个：{names}。")

    if run_config.problem is not None:
        problem = run_config.problem
        return problem if isinstance(problem, QASProblem) else normalize_problem(problem)

    if provided:
        _name, value, kind = provided[0]
        explicit_kind = kind if isinstance(value, np.ndarray) else None
        return normalize_problem(value, kind=explicit_kind)

    return None


def run(request: QASRunConfig | QASMethod, **kwargs: Any) -> Any:
    """Run a QAS implementation with a common packaged-user interface.

    统一返回 :class:`~aicir.qas.core.results.QASResult`（Breaking：3b 之前部分
    方法直接返回底层结果对象/裸元组，见 CHANGELOG 迁移表）。

    Examples:
        ``run("supernet", config=config.supernet(...))``
        ``run("crlqas", problem=hamiltonian, config=config.crlqas(...))``
        ``run(QASRunConfig(method="pprdql", target_state=state))``
    """

    from .strategies import get_strategy

    run_config = _as_run_config(request, kwargs)
    method = qas_config.canonical_method(run_config.method)
    run_config.problem = _resolve_problem(run_config)

    strategy = get_strategy(method)
    if strategy is None:
        raise ValueError(f"Unsupported QAS method: {run_config.method!r}")
    return strategy.run(run_config)


def _as_run_config(request: QASRunConfig | QASMethod, kwargs: dict[str, Any]) -> QASRunConfig:
    if isinstance(request, QASRunConfig):
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Do not pass keyword overrides with QASRunConfig: {names}")
        return request
    return QASRunConfig(method=request, **kwargs)


# import 副作用：注册内置策略（全部 10 个 _FACTORIES 方法）。
from . import strategies as _strategies  # noqa: E402,F401

__all__ = [
    "QASRunConfig",
    "available_qas_methods",
    "default_qas_config",
    "run",
]
