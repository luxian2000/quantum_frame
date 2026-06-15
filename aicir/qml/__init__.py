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
]
