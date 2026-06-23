"""Public QAS runner API for packaged aicir users."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import config as qas_config
from .registry import get_strategy
from ..algorithms import strategies as _strategies  # noqa: F401  注册内置策略（import 副作用）
from ..algorithms.CRLQAS import train_crlqas
from ..algorithms.PPO_RB import ppo_rb_qas
from ..algorithms.PPR_DQL import train_ppr_dql
from ..algorithms.qdrats import train_qdrats
from ..algorithms.supernet import classification_supernet, h2_vqe_supernet

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
        ``run(QASRunConfig(method="ppr_dql", target_state=state))``
    """

    run_config = _as_run_config(request, kwargs)
    method = qas_config.canonical_method(run_config.method)

    # 已迁移为 SearchStrategy 的方法走注册表；未注册的回落到下方旧分支。
    strategy = get_strategy(method)
    if strategy is not None:
        return strategy.run(run_config)

    if method == "supernet_classification":
        return classification_supernet(config=run_config.config)
    if method == "supernet_h2":
        return h2_vqe_supernet(config=run_config.config)
    if method == "ppo_rb":
        _require(run_config.target_density_matrix is not None, "ppo_rb requires target_density_matrix.")
        _require(run_config.epsilon is not None, "ppo_rb requires epsilon.")
        return ppo_rb_qas(
            target_density_matrix=run_config.target_density_matrix,
            epsilon=run_config.epsilon,
            config=run_config.config,
        )
    if method == "ppr_dql":
        _require(run_config.target_state is not None, "ppr_dql requires target_state.")
        return train_ppr_dql(
            target_state=run_config.target_state,
            config=run_config.config,
            policy_library=run_config.policy_library,
        )
    if method == "crlqas":
        _require(run_config.hamiltonian is not None, "crlqas requires hamiltonian.")
        return train_crlqas(hamiltonian=run_config.hamiltonian, config=run_config.config)
    if method == "qdrats":
        _require(run_config.hamiltonian is not None, "qdrats requires hamiltonian.")
        return train_qdrats(hamiltonian=run_config.hamiltonian, config=run_config.config)

    raise ValueError(f"Unsupported QAS method: {run_config.method!r}")


def _as_run_config(request: QASRunConfig | QASMethod, kwargs: dict[str, Any]) -> QASRunConfig:
    if isinstance(request, QASRunConfig):
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Do not pass keyword overrides with QASRunConfig: {names}")
        return request
    return QASRunConfig(method=request, **kwargs)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


__all__ = [
    "QASRunConfig",
    "available_qas_methods",
    "default_qas_config",
    "run",
]
