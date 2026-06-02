"""Convenience config factories for public QAS methods."""

from __future__ import annotations

from typing import Any, Callable

from .CRLQAS import AdamSPSAConfig, CRLQASConfig
from .PPO_RB import PPORollbackConfig
from .PPR_DQL import PPRDQLConfig
from .VQA_QAS import VQAQASConfig

QASMethod = str

_PUBLIC_METHODS = ("vqa_qas", "vqa_classification", "vqa_h2", "ppo_rb", "ppr_dql", "crlqas")

_METHOD_ALIASES = {
    "vqa": "vqa_qas",
    "vqa_qas": "vqa_qas",
    "vqa_classification": "vqa_classification",
    "classification": "vqa_classification",
    "vqa_h2": "vqa_h2",
    "h2": "vqa_h2",
    "h2_vqe": "vqa_h2",
    "ppo": "ppo_rb",
    "ppo_rb": "ppo_rb",
    "ppr": "ppr_dql",
    "ppr_dql": "ppr_dql",
    "crl": "crlqas",
    "crlqas": "crlqas",
}


def vqa_qas(**kwargs: Any) -> VQAQASConfig:
    """Build a ``VQA_QAS`` config with optional field overrides."""

    return _build(VQAQASConfig, kwargs)


def vqa_classification(**kwargs: Any) -> VQAQASConfig:
    """Build a ``VQA_QAS`` config for the built-in classification task."""

    values = {"task": "classification"}
    values.update(kwargs)
    return _build(VQAQASConfig, values)


def vqa_h2(**kwargs: Any) -> VQAQASConfig:
    """Build a ``VQA_QAS`` config for the built-in H2 VQE task."""

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
    return _build(VQAQASConfig, values)


def ppo_rb(**kwargs: Any) -> PPORollbackConfig:
    """Build a ``PPO_RB`` config with optional field overrides."""

    return _build(PPORollbackConfig, kwargs)


def ppr_dql(**kwargs: Any) -> PPRDQLConfig:
    """Build a ``PPR_DQL`` config with optional field overrides."""

    return _build(PPRDQLConfig, kwargs)


def crlqas(**kwargs: Any) -> CRLQASConfig:
    """Build a ``CRLQAS`` config with optional field overrides."""

    values = dict(kwargs)
    if isinstance(values.get("adam_spsa"), dict):
        values["adam_spsa"] = adam_spsa(**values["adam_spsa"])
    return _build(CRLQASConfig, values)


def adam_spsa(**kwargs: Any) -> AdamSPSAConfig:
    """Build the nested Adam-SPSA config used by ``CRLQAS``."""

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


vqa = vqa_qas
classification = vqa_classification
h2_vqe = vqa_h2
ppo = ppo_rb
ppr = ppr_dql
crl = crlqas

_FACTORIES = {
    "vqa_qas": vqa_qas,
    "vqa_classification": vqa_classification,
    "vqa_h2": vqa_h2,
    "ppo_rb": ppo_rb,
    "ppr_dql": ppr_dql,
    "crlqas": crlqas,
}

__all__ = [
    "adam_spsa",
    "canonical_method",
    "classification",
    "create",
    "crl",
    "crlqas",
    "for_method",
    "h2_vqe",
    "method_names",
    "ppo",
    "ppo_rb",
    "ppr",
    "ppr_dql",
    "vqa",
    "vqa_h2",
    "vqa_qas",
    "vqa_classification",
]
