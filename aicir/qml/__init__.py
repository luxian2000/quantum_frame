"""Quantum machine learning utilities."""

from .deriv import auto, psr, spsr, spsa, mpsr, fd, ad, qng, bdqng, kqng, dqng, rotosolve
from .diff import (
    DiffMethod,
    canonical_diff_name,
    get_diff_method,
    register_diff_method,
    registered_diff_methods,
    resolve_diff_method,
    select_diff_method,
    unregister_diff_method,
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
    "canonical_diff_name",
    "get_diff_method",
    "register_diff_method",
    "registered_diff_methods",
    "resolve_diff_method",
    "select_diff_method",
    "unregister_diff_method",
]
