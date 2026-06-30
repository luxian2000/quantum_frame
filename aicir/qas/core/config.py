"""Convenience config factories for public QAS methods."""

from __future__ import annotations

from typing import Any, Callable

QASMethod = str

# 非规范名的等价写法 → 规范方法名。规范名本身（``_FACTORIES`` 的键）无需登记。
_ALIASES = {
    "vqa": "supernet",
    "vqa_qas": "supernet",
    "vqa_classification": "supernet_classification",
    "classification": "supernet_classification",
    "vqa_h2": "supernet_h2",
    "h2": "supernet_h2",
    "h2_vqe": "supernet_h2",
    "ppo": "ppo_rb",
    "ppr": "ppr_dql",
    "crl": "crlqas",
    "qdarts": "qdrats",
    "quantumdarts": "qdrats",
    "quantum_darts": "qdrats",
    "differentiable_qas": "dqas",
    "differentiable_quantum_architecture_search": "dqas",
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


def ppo_rb(**kwargs: Any) -> Any:
    """Build a ``PPO_RB`` config with optional field overrides."""

    from ..algorithms.pporb import PPORollbackConfig

    return _build(PPORollbackConfig, kwargs)


def ppr_dql(**kwargs: Any) -> Any:
    """Build a ``PPR_DQL`` config with optional field overrides."""

    from ..algorithms.pprdql import PPRDQLConfig

    return _build(PPRDQLConfig, kwargs)


def crlqas(**kwargs: Any) -> Any:
    """Build a ``CRLQAS`` config with optional field overrides."""

    from ..algorithms.crlqas import CRLQASConfig

    values = dict(kwargs)
    if isinstance(values.get("adam_spsa"), dict):
        values["adam_spsa"] = adam_spsa(**values["adam_spsa"])
    return _build(CRLQASConfig, values)


def qdrats(**kwargs: Any) -> Any:
    """Build a ``QDRATS`` config with optional field overrides."""

    from ..algorithms.qdrats import QDRATSConfig

    return _build(QDRATSConfig, kwargs)


def dqas(**kwargs: Any) -> Any:
    """Build a ``DQAS`` config with optional field overrides."""

    from ..algorithms.dqas import DQASConfig

    return _build(DQASConfig, kwargs)


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


def _build(config_type: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    try:
        return config_type(**kwargs)
    except TypeError as exc:
        raise TypeError(f"{config_type.__name__} does not accept the provided config fields.") from exc


ppo = ppo_rb
ppr = ppr_dql
crl = crlqas

# 规范方法名 → 配置工厂。方法集合的单一来源（``method_names`` 由此派生）。
_FACTORIES = {
    "supernet": supernet,
    "supernet_classification": supernet_classification,
    "supernet_h2": supernet_h2,
    "ppo_rb": ppo_rb,
    "ppr_dql": ppr_dql,
    "crlqas": crlqas,
    "qdrats": qdrats,
    "dqas": dqas,
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
    "ppr",
    "ppr_dql",
    "qdrats",
    "supernet",
    "supernet_classification",
    "supernet_h2",
]
