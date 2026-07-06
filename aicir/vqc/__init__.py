"""Variational quantum computing algorithms."""

from __future__ import annotations

from .QAOA import BasicQAOA, QAOAResult, run_qaoa

__all__ = [
    "BasicQAOA",
    "QAOAResult",
    "run_qaoa",
]

try:
    from .SSVQE import BasicSSVQE, SSVQEResult, run_ssvqe
    from .VQD import BasicVQD, VQDResult, run_vqd
    from .VQE import BasicVQE, VQEResult, run_vqe
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.extend(
        [
            "BasicVQE",
            "VQEResult",
            "run_vqe",
            "BasicVQD",
            "VQDResult",
            "run_vqd",
            "BasicSSVQE",
            "SSVQEResult",
            "run_ssvqe",
        ]
    )
