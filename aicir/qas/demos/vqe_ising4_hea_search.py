"""Run the first 4-qubit Ising/TFIM VQE-QAS HEA reliability check.

This entry point checks whether B2 medium-budget VQE ranking tracks fair VQE
ranking before investing in the full Hamiltonian-aware beam-search framework.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aicir.qas.vqe_hea_demo import (
    resolve_qas_backend,
    run_ising4_b2_reliability_experiment,
    run_ising4_full_enumeration_baseline,
)


def main() -> None:
    backend = resolve_qas_backend()
    print(f"backend: {backend.name}")
    reliability = run_ising4_b2_reliability_experiment(backend=backend)
    print("\n".join(reliability.summary_lines()))
    print("\n" + "=" * 80 + "\n")
    enumeration = run_ising4_full_enumeration_baseline(candidate_limit=24, backend=backend)
    print("\n".join(enumeration.summary_lines(top_k=10)))


if __name__ == "__main__":
    main()
