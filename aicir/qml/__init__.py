"""Quantum machine learning utilities."""

from .deriv import auto, psr, spsr, spsa, mpsr, fd, ad, qng, bdqng, kqng, dqng, rotosolve
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

try:  # torch 可选：无 torch 时不暴露 QLayer
    from .qlayer import QLayer
except ImportError:  # pragma: no cover - 取决于运行环境是否装 torch
    QLayer = None

__all__ = [
    "auto",
    "psr",
    "spsr",
    "spsa",
    "mpsr",
    "fd",
    "ad",
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
]
