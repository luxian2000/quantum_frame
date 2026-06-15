"""DiffMethod 策略注册表（NEXT.md §6 第一片）。"""

from .registry import (
    canonical_diff_name,
    get_diff_method,
    register_diff_method,
    registered_diff_methods,
    resolve_diff_method,
    unregister_diff_method,
)
from .spec import DiffMethod

__all__ = [
    "DiffMethod",
    "canonical_diff_name",
    "get_diff_method",
    "register_diff_method",
    "registered_diff_methods",
    "resolve_diff_method",
    "unregister_diff_method",
]
