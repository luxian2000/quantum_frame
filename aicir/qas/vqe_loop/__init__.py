"""Closed-loop VQE-QAS package.

The package keeps the online-search stack separate from legacy QAS algorithms:
`protocol` owns table/status semantics, `geometry` owns distances/TR/splits,
`selection_ops` owns Stage-2 selection operators, and `vqe_qas_loop` is the
one-call entry point.
"""

from .vqe_qas_loop import ClosedLoopConfig, ClosedLoopResult, run_vqe_qas_closed_loop, stamp_literal_hamiltonian_terms

__all__ = [
    "ClosedLoopConfig",
    "ClosedLoopResult",
    "run_vqe_qas_closed_loop",
    "stamp_literal_hamiltonian_terms",
]
