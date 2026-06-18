"""Reusable candidate-architecture libraries for QAS.

This package holds architecture catalogs used by library code and demos; it is
not a demo namespace.
"""

from .architectures import build_common_architectures, common_architecture_names

__all__ = ["build_common_architectures", "common_architecture_names"]
