"""Convenience config factories for public QAS methods."""

from __future__ import annotations

from typing import Any, Callable

QASMethod = str

_PUBLIC_METHODS = ("supernet", "supernet_classification", "supernet_h2", "ppo_rb", "ppr_dql", "crlqas", "vqe_loop")

_METHOD_ALIASES = {
    # supernet (formerly VQA_QAS): keep the old strings working for run().
    "supernet": "supernet",
    "vqa": "supernet",
    "vqa_qas": "supernet",
    "supernet_classification": "supernet_classification",
    "vqa_classification": "supernet_classification",
    "classification": "supernet_classification",
    "supernet_h2": "supernet_h2",
    "vqa_h2": "supernet_h2",
    "h2": "supernet_h2",
    "h2_vqe": "supernet_h2",
    "ppo": "ppo_rb",
    "ppo_rb": "ppo_rb",
    "ppr": "ppr_dql",
    "ppr_dql": "ppr_dql",
    "crl": "crlqas",
    "crlqas": "crlqas",
    "closed_loop": "vqe_loop",
    "vqe_qas": "vqe_loop",
    "vqe_qas_loop": "vqe_loop",
    "vqe_loop": "vqe_loop",
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

    from ..algorithms.PPO_RB import PPORollbackConfig

    return _build(PPORollbackConfig, kwargs)


def ppr_dql(**kwargs: Any) -> Any:
    """Build a ``PPR_DQL`` config with optional field overrides."""

    from ..algorithms.PPR_DQL import PPRDQLConfig

    return _build(PPRDQLConfig, kwargs)


def crlqas(**kwargs: Any) -> Any:
    """Build a ``CRLQAS`` config with optional field overrides."""

    from ..algorithms.CRLQAS import CRLQASConfig

    values = dict(kwargs)
    if isinstance(values.get("adam_spsa"), dict):
        values["adam_spsa"] = adam_spsa(**values["adam_spsa"])
    return _build(CRLQASConfig, values)


def vqe_loop(**kwargs: Any) -> Any:
    """Build a ``vqe_loop`` closed-loop config with optional field overrides."""

    from pathlib import Path

    from ..vqe_loop import ClosedLoopConfig

    values = dict(kwargs)
    if "output_dir" in values:
        values["output_dir"] = Path(values["output_dir"])
    if "protocol" in values:
        values["protocol"] = Path(values["protocol"])
    return _build(ClosedLoopConfig, values)


def adam_spsa(**kwargs: Any) -> Any:
    """Build the nested Adam-SPSA config used by ``CRLQAS``."""

    from ..algorithms.CRLQAS import AdamSPSAConfig

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

    return _PUBLIC_METHODS


def canonical_method(method: QASMethod) -> str:
    key = str(method).strip().lower().replace("-", "_")
    try:
        return _METHOD_ALIASES[key]
    except KeyError as exc:
        methods = ", ".join(method_names())
        raise ValueError(f"Unsupported QAS method {method!r}. Available methods: {methods}.") from exc


def _build(config_type: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    try:
        return config_type(**kwargs)
    except TypeError as exc:
        raise TypeError(f"{config_type.__name__} does not accept the provided config fields.") from exc


ppo = ppo_rb
ppr = ppr_dql
crl = crlqas

_FACTORIES = {
    "supernet": supernet,
    "supernet_classification": supernet_classification,
    "supernet_h2": supernet_h2,
    "ppo_rb": ppo_rb,
    "ppr_dql": ppr_dql,
    "crlqas": crlqas,
    "vqe_loop": vqe_loop,
}

__all__ = [
    "adam_spsa",
    "canonical_method",
    "create",
    "crl",
    "crlqas",
    "for_method",
    "method_names",
    "ppo",
    "ppo_rb",
    "ppr",
    "ppr_dql",
    "supernet",
    "supernet_classification",
    "supernet_h2",
    "vqe_loop",
]
