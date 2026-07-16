"""Quantum machine learning utilities."""

from .deriv import (
    auto,
    psr,
    psr4,
    spsr,
    spsa,
    mpsr,
    hessian,
    fd,
    ad,
    qfim,
    metric_tensor,
    qfim_diag,
    qfim_blocks,
    qng,
    bdqng,
    kqng,
    dqng,
    rotosolve,
)
from .diff import (
    DiffMethod,
    canonical_diff,
    get_diff,
    register_diff,
    registered_diffs,
    resolve_diff,
    select_diff,
    unregister_diff,
)
from .qfun import QFun, Expval, Probs, Sample, expval, probs, qfun, sample

try:  # torch 可选：无 torch 时不暴露 QLayer/BatchLayer/classifier
    from .qlayer import BatchLayer, QLayer
    from .classifier import build_classifier, classifier_template
    from .kernel import QuantumKernel, angle_feature_map
except ImportError:  # pragma: no cover - 取决于运行环境是否装 torch
    QLayer = None
    BatchLayer = None
    build_classifier = None
    classifier_template = None
    QuantumKernel = None
    angle_feature_map = None

__all__ = [
    "auto",
    "psr",
    "psr4",
    "spsr",
    "spsa",
    "mpsr",
    "hessian",
    "fd",
    "ad",
    "qfim",
    "metric_tensor",
    "qfim_diag",
    "qfim_blocks",
    "qng",
    "bdqng",
    "kqng",
    "dqng",
    "rotosolve",
    "DiffMethod",
    "canonical_diff",
    "get_diff",
    "register_diff",
    "registered_diffs",
    "resolve_diff",
    "select_diff",
    "unregister_diff",
    "QFun",
    "qfun",
    "Expval",
    "Probs",
    "Sample",
    "expval",
    "probs",
    "sample",
    "QLayer",
    "BatchLayer",
    "build_classifier",
    "classifier_template",
    "QuantumKernel",
    "angle_feature_map",
]
