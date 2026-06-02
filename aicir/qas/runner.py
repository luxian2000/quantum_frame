"""Public QAS runner API for packaged aicir users."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .CRLQAS import CRLQASConfig, train_crlqas
from .PPO_RB import PPORollbackConfig, ppo_rb_qas
from .PPR_DQL import PPRDQLConfig, train_ppr_dql
from .VQA_QAS import VQAQASConfig, classification_vqa_qas, h2_vqe_qas, train_vqa_qas

QASMethod = str

_METHOD_ALIASES = {
    "vqa": "vqa",
    "vqa_qas": "vqa",
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


@dataclass
class QASRunConfig:
    """Method-agnostic request object for running a QAS implementation.

    Users can pass this object to :func:`run_qas`, or pass the same fields as
    keyword arguments directly to ``run_qas(method, ...)``.
    """

    method: QASMethod
    config: Any = None
    objective_fn: Any = None
    dataset: Any = None
    hamiltonian: Any = None
    target_state: Any = None
    target_density_matrix: Any = None
    epsilon: float | None = None
    policy_library: Any = None


def available_qas_methods() -> tuple[str, ...]:
    """Return canonical method names accepted by :func:`run_qas`."""

    return ("vqa", "vqa_classification", "vqa_h2", "ppo_rb", "ppr_dql", "crlqas")


def default_qas_config(method: QASMethod) -> Any:
    """Return the default config object for a QAS method."""

    canonical = _canonical_method(method)
    if canonical == "vqa":
        return VQAQASConfig()
    if canonical == "vqa_classification":
        return VQAQASConfig(task="classification")
    if canonical == "vqa_h2":
        return VQAQASConfig(
            n_qubits=4,
            layers=3,
            single_qubit_gates=("ry", "rz"),
            two_qubit_pairs=((0, 1), (1, 2), (2, 3)),
            task="h2_vqe",
        )
    if canonical == "ppo_rb":
        return PPORollbackConfig()
    if canonical == "ppr_dql":
        return PPRDQLConfig()
    if canonical == "crlqas":
        return CRLQASConfig()
    raise ValueError(f"Unsupported QAS method: {method!r}")


def run_qas(request: QASRunConfig | QASMethod, **kwargs: Any) -> Any:
    """Run a QAS implementation with a common packaged-user interface.

    Examples:
        ``run_qas("vqa_classification", config=VQAQASConfig(...))``
        ``run_qas(QASRunConfig(method="ppr_dql", target_state=state))``
    """

    run_config = _as_run_config(request, kwargs)
    method = _canonical_method(run_config.method)

    if method == "vqa":
        return train_vqa_qas(
            objective_fn=run_config.objective_fn,
            config=run_config.config,
            dataset=run_config.dataset,
            hamiltonian=run_config.hamiltonian,
        )
    if method == "vqa_classification":
        return classification_vqa_qas(config=run_config.config)
    if method == "vqa_h2":
        return h2_vqe_qas(config=run_config.config)
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

    raise ValueError(f"Unsupported QAS method: {run_config.method!r}")


def _as_run_config(request: QASRunConfig | QASMethod, kwargs: dict[str, Any]) -> QASRunConfig:
    if isinstance(request, QASRunConfig):
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Do not pass keyword overrides with QASRunConfig: {names}")
        return request
    return QASRunConfig(method=request, **kwargs)


def _canonical_method(method: QASMethod) -> str:
    key = str(method).strip().lower().replace("-", "_")
    try:
        return _METHOD_ALIASES[key]
    except KeyError as exc:
        methods = ", ".join(available_qas_methods())
        raise ValueError(f"Unsupported QAS method {method!r}. Available methods: {methods}.") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


__all__ = [
    "QASRunConfig",
    "available_qas_methods",
    "default_qas_config",
    "run_qas",
]
