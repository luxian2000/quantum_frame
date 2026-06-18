"""Closed-loop VQE-QAS package.

The package keeps the online-search stack separate from legacy QAS algorithms:
`protocol` owns table/status semantics, `geometry` owns distances/TR/splits,
`selection_ops` owns Stage-2 selection operators, and `vqe_qas_loop` is the
one-call entry point.
"""

from .vqe_qas_loop import (
    ClosedLoopConfig,
    ClosedLoopResolvedDefaults,
    ClosedLoopResult,
    default_batch_quotas_for_qubits,
    default_initial_labels_for_qubits,
    default_max_rounds_for_qubits,
    resolve_closed_loop_defaults,
    run_vqe_qas_closed_loop,
    stamp_literal_hamiltonian_terms,
)

__all__ = [
    "ClosedLoopConfig",
    "ClosedLoopResolvedDefaults",
    "ClosedLoopResult",
    "default_batch_quotas_for_qubits",
    "default_initial_labels_for_qubits",
    "default_max_rounds_for_qubits",
    "resolve_closed_loop_defaults",
    "run_vqe_qas_closed_loop",
    "stamp_literal_hamiltonian_terms",
]
