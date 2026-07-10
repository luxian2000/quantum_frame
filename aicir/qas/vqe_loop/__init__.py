"""P0/P1 VQE-QAS package entry points.

`p0_bootstrap_fair` owns the supported P0 bootstrap plus fair-label path. Row-level schema, parsing, fair protocol defaults, and P1 quota policy live in `benchmark_table`; full P1 mutation/oracle/fallback planning lives in `p1_round.py` plus `demos/run_p1_round_demo.py`.
"""

from .benchmark_table import (
    QuotaDecision,
    decide_next_round_quotas,
    default_batch_quotas_for_qubits,
)
from .p0_bootstrap_fair import (
    ClosedLoopConfig,
    ClosedLoopResolvedDefaults,
    ClosedLoopResult,
    P0BootstrapConfig,
    P0BootstrapResult,
    effective_supernet_bootstrap_count,
    resolve_closed_loop_defaults,
    run_p0_bootstrap_fair,
    run_vqe_qas_closed_loop,
    stamp_literal_hamiltonian_terms,
)


__all__ = [
    "ClosedLoopConfig",
    "ClosedLoopResolvedDefaults",
    "ClosedLoopResult",
    "P0BootstrapConfig",
    "P0BootstrapResult",
    "QuotaDecision",
    "decide_next_round_quotas",
    "default_batch_quotas_for_qubits",
    "effective_supernet_bootstrap_count",
    "resolve_closed_loop_defaults",
    "run_p0_bootstrap_fair",
    "run_vqe_qas_closed_loop",
    "stamp_literal_hamiltonian_terms",
]
