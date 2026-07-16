"""DiffMethod 策略注册表（NEXT.md §6 第一片）。"""

from .registry import (
    canonical_diff,
    circuit_shift_rule,
    get_diff,
    register_diff,
    registered_diffs,
    resolve_diff,
    select_diff,
    unregister_diff,
)
from .spec import DiffMethod

__all__ = [
    "DiffMethod",
    "canonical_diff",
    "circuit_shift_rule",
    "get_diff",
    "register_diff",
    "registered_diffs",
    "resolve_diff",
    "select_diff",
    "unregister_diff",
]
