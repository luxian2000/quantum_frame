"""Public QAS runner API for packaged aicir users."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from . import config as qas_config

QASMethod = str


@dataclass
class QASRunConfig:
    """Method-agnostic request object for running a QAS implementation.

    Users can pass this object to :func:`run`, or pass the same fields as
    keyword arguments directly to ``run(method, ...)``.
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


@dataclass(frozen=True)
class _Spec:
    """单个方法的分发规格：懒加载底层算法 + 需要的请求字段。

    - ``loader``：返回底层 train/搜索函数（懒导入，避免顶层 Torch 依赖）。
    - ``params``：透传给底层函数的 :class:`QASRunConfig` 字段名（同名映射）。
    - ``required``：调用前必须非 ``None`` 的字段名。
    """

    loader: Callable[[], Callable[..., Any]]
    params: tuple[str, ...]
    required: tuple[str, ...] = ()


def _load(module: str, name: str) -> Callable[[], Callable[..., Any]]:
    def loader() -> Callable[..., Any]:
        import importlib

        return getattr(importlib.import_module(f"..algorithms.{module}", __package__), name)

    return loader

def _load_vqe_loop() -> Callable[[], Callable[..., Any]]:
    def loader() -> Callable[..., Any]:
        import importlib

        return getattr(importlib.import_module("..vqe_loop", __package__), "run_vqe_qas_closed_loop")

    return loader


# Method name -> dispatch spec. `run` is table-driven.
_TABLE: dict[str, _Spec] = {
    "supernet": _Spec(_load("supernet", "train_supernet"), ("objective", "config", "dataset", "hamiltonian")),
    "supernet_classification": _Spec(_load("supernet", "classification_supernet"), ("config",)),
    "supernet_h2": _Spec(_load("supernet", "h2_vqe_supernet"), ("config",)),
    "pporb": _Spec(
        _load("pporb", "ppo_rb_qas"),
        ("target_density_matrix", "epsilon", "config"),
        ("target_density_matrix", "epsilon"),
    ),
    "pprdql": _Spec(_load("pprdql", "train_ppr_dql"), ("target_state", "config", "policy_library"), ("target_state",)),
    "crlqas": _Spec(_load("crlqas", "train_crlqas"), ("hamiltonian", "config"), ("hamiltonian",)),
    "qdrats": _Spec(_load("qdrats", "train_qdrats"), ("hamiltonian", "config"), ("hamiltonian",)),
    "vqe_loop": _Spec(_load_vqe_loop(), ("config",), ("config",)),
}


def available_qas_methods() -> tuple[str, ...]:
    """Return canonical method names accepted by :func:`run`."""

    return qas_config.method_names()


def default_qas_config(method: QASMethod, **kwargs: Any) -> Any:
    """Return a config object for a QAS method.

    Prefer ``aicir.qas.core.config.<method>(...)`` in user-facing code. This helper
    remains as a method-name based compatibility wrapper.
    """

    return qas_config.create(method, **kwargs)


def run(request: QASRunConfig | QASMethod, **kwargs: Any) -> Any:
    """Run a QAS implementation with a common packaged-user interface.

    Examples:
        ``run("supernet", config=config.supernet(...))``
        ``run(QASRunConfig(method="pprdql", target_state=state))``
    """

    run_config = _as_run_config(request, kwargs)
    method = qas_config.canonical_method(run_config.method)

    # 先查策略注册表（README §2.1）：命中则走 strategy.run，未命中回落到旧分发表。
    from .strategies import get_strategy

    strategy = get_strategy(method)
    if strategy is not None:
        return strategy.run(run_config)

    spec = _TABLE.get(method)
    if spec is None:
        raise ValueError(f"Unsupported QAS method: {run_config.method!r}")

    for name in spec.required:
        if getattr(run_config, name) is None:
            raise ValueError(f"{method} requires {name}.")

    fn = spec.loader()
    return fn(**{name: getattr(run_config, name) for name in spec.params})


def _as_run_config(request: QASRunConfig | QASMethod, kwargs: dict[str, Any]) -> QASRunConfig:
    if isinstance(request, QASRunConfig):
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Do not pass keyword overrides with QASRunConfig: {names}")
        return request
    return QASRunConfig(method=request, **kwargs)


# import 副作用：注册内置策略（当前仅 supernet）。轻量，不在 import 期触 torch。
from . import strategies as _strategies  # noqa: E402,F401

__all__ = [
    "QASRunConfig",
    "available_qas_methods",
    "default_qas_config",
    "run",
]
