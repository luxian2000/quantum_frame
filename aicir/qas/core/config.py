"""Convenience config factories for public QAS methods."""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from typing import Any, Callable

QASMethod = str

# 字段别名：方法名 -> {旧字段名: 规范字段名}。仅收录含义完全相同的同义写法
# （见 aicir/qas/README.md 的“词汇对照表”）；语义不同的字段（如
# q_learning_rate/architecture_learning_rate/supernet_steps/search_epochs）不在此列。
# 在 ``_build`` 里于“未知字段报错”之前应用：命中别名会把旧字段名重写为规范字段名
# 并发出 DeprecationWarning，旧关键字参数因此继续可用。
_FIELD_ALIASES: dict[str, dict[str, str]] = {
    "pporb": {"episode_num": "max_episodes"},
    "pprdql": {"episode_num": "max_episodes"},
    "crlqas": {"q_hidden_dim": "hidden_dim"},
}

# 非规范名的等价写法 -> 规范方法名。规范名本身（``_FACTORIES`` 的键）无需登记。
_ALIASES = {
    "vqa": "supernet",
    "vqa_qas": "supernet",
    "vqa_classification": "supernet_classification",
    "classification": "supernet_classification",
    "vqa_h2": "supernet_h2",
    "h2": "supernet_h2",
    "h2_vqe": "supernet_h2",
    "ppo": "pporb",
    "ppo_rb": "pporb",
    "ppr": "pprdql",
    "ppr_dql": "pprdql",
    "crl": "crlqas",
    "qdarts": "qdrats",
    "quantumdarts": "qdrats",
    "quantum_darts": "qdrats",
    "differentiable_qas": "dqas",
    "differentiable_quantum_architecture_search": "dqas",
    "vqe_qas": "vqe_loop",
    "vqe_closed_loop": "vqe_loop",
}


def supernet(**kwargs: Any) -> Any:
    """Build a ``supernet`` config with optional field overrides."""

    from ..algorithms.supernet import SupernetConfig

    return _build(SupernetConfig, kwargs)


def supernet_classification(**kwargs: Any) -> Any:
    """Build a ``supernet`` config for the built-in classification task."""

    from ..algorithms.supernet import SupernetConfig

    values = {"task": "classification"}
    values.update(kwargs)
    return _build(SupernetConfig, values)


def supernet_h2(**kwargs: Any) -> Any:
    """Build a ``supernet`` config for the built-in H2 VQE task."""

    from ..algorithms.supernet import SupernetConfig

    values = {
        "n_qubits": 4,
        "layers": 3,
        "single_qubit_gates": ("ry", "rz"),
        "two_qubit_pairs": ((0, 1), (1, 2), (2, 3)),
        "supernet_steps": 500,
        "ranking_num": 500,
        "finetune_steps": 50,
        "task": "h2_vqe",
    }
    values.update(kwargs)
    return _build(SupernetConfig, values)


def pporb(**kwargs: Any) -> Any:
    """Build a ``PPO_RB`` config with optional field overrides."""

    from ..algorithms.pporb import PPORollbackConfig

    return _build(PPORollbackConfig, kwargs, method="pporb")


def pprdql(**kwargs: Any) -> Any:
    """Build a ``PPR_DQL`` config with optional field overrides."""

    from ..algorithms.pprdql import PPRDQLConfig

    return _build(PPRDQLConfig, kwargs, method="pprdql")


def crlqas(**kwargs: Any) -> Any:
    """Build a ``CRLQAS`` config with optional field overrides."""

    from ..algorithms.crlqas import CRLQASConfig

    values = dict(kwargs)
    if isinstance(values.get("adam_spsa"), dict):
        values["adam_spsa"] = adam_spsa(**values["adam_spsa"])
    return _build(CRLQASConfig, values, method="crlqas")


def qdrats(**kwargs: Any) -> Any:
    """Build a ``QDRATS`` config with optional field overrides."""

    from ..algorithms.qdrats import QDRATSConfig

    return _build(QDRATSConfig, kwargs)


def dqas(**kwargs: Any) -> Any:
    """Build a ``DQAS`` config with optional field overrides."""

    from ..algorithms.dqas import DQASConfig

    return _build(DQASConfig, kwargs)


def vqe_loop(**kwargs: Any) -> Any:
    """Build a ``vqe_loop`` config with optional field overrides."""

    from ..vqe_loop import P0BootstrapConfig

    values = dict(kwargs)
    if "output_dir" not in values:
        values["output_dir"] = Path("outputs") / "qas_vqe_loop"
    elif isinstance(values["output_dir"], str):
        values["output_dir"] = Path(values["output_dir"])
    if isinstance(values.get("protocol"), str):
        values["protocol"] = Path(values["protocol"])
    return _build(P0BootstrapConfig, values)


def adam_spsa(**kwargs: Any) -> Any:
    """Build the nested Adam-SPSA config used by ``CRLQAS``."""

    from ..algorithms.crlqas import AdamSPSAConfig

    return _build(AdamSPSAConfig, kwargs)


def create(method: QASMethod, **kwargs: Any) -> Any:
    """Build a config by method name.

    Method names are case-insensitive and accept aliases such as ``"VQA_QAS"``
    and ``"h2_vqe"``.
    """

    return _FACTORIES[canonical_method(method)](**kwargs)


def for_method(method: QASMethod, **kwargs: Any) -> Any:
    """Alias for :func:`create`."""

    return create(method, **kwargs)


def method_names() -> tuple[str, ...]:
    """Return public method names that have config factory functions."""

    return tuple(_FACTORIES)


def canonical_method(method: QASMethod) -> str:
    key = str(method).strip().lower().replace("-", "_")
    if key in _FACTORIES:
        return key
    canon = _ALIASES.get(key)
    if canon is not None:
        return canon
    methods = ", ".join(method_names())
    raise ValueError(f"Unsupported QAS method {method!r}. Available methods: {methods}.")


def _apply_field_aliases(method: str | None, kwargs: dict[str, Any]) -> dict[str, Any]:
    """把 ``method`` 对应的旧字段名重写为规范字段名，命中时发出 DeprecationWarning。"""

    aliases = _FIELD_ALIASES.get(method) if method else None
    if not aliases:
        return kwargs
    resolved = dict(kwargs)
    for alias, canonical in aliases.items():
        if alias not in resolved:
            continue
        if canonical in resolved:
            raise TypeError(
                f"不能同时传入别名字段 {alias!r} 与规范字段 {canonical!r}；请只使用其一。"
            )
        warnings.warn(
            f"{alias!r} 已废弃，请改用 {canonical!r}",
            DeprecationWarning,
            stacklevel=4,
        )
        resolved[canonical] = resolved.pop(alias)
    return resolved


def _build(config_type: Callable[..., Any], kwargs: dict[str, Any], method: str | None = None) -> Any:
    kwargs = _apply_field_aliases(method, kwargs)
    valid_fields = {f.name for f in dataclasses.fields(config_type)}
    unknown = sorted(set(kwargs) - valid_fields)
    if unknown:
        raise TypeError(
            f"{config_type.__name__} does not accept field(s) {unknown!r}; "
            f"valid fields are {sorted(valid_fields)!r}."
        )
    try:
        return config_type(**kwargs)
    except TypeError as exc:
        raise TypeError(f"{config_type.__name__} does not accept the provided config fields: {exc}") from exc


ppo_rb = pporb
ppr_dql = pprdql
ppo = pporb
ppr = pprdql
crl = crlqas

# 规范方法名 -> 配置工厂。方法集合的单一来源（``method_names`` 由此派生）。
_FACTORIES = {
    "supernet": supernet,
    "supernet_classification": supernet_classification,
    "supernet_h2": supernet_h2,
    "pporb": pporb,
    "pprdql": pprdql,
    "crlqas": crlqas,
    "qdrats": qdrats,
    "dqas": dqas,
    "vqe_loop": vqe_loop,
}

__all__ = [
    "adam_spsa",
    "canonical_method",
    "create",
    "crl",
    "crlqas",
    "dqas",
    "for_method",
    "method_names",
    "ppo",
    "ppo_rb",
    "pporb",
    "ppr",
    "ppr_dql",
    "pprdql",
    "qdrats",
    "supernet",
    "supernet_classification",
    "supernet_h2",
    "vqe_loop",
]
