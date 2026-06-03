"""Run the main 4-qubit Ising/TFIM VQE-QAS HEA demo.

The mainline uses trainability-prior Stage 1 ranking followed by fair final
VQE validation. Short-step VQE/SA runners are kept as diagnostics because the
4-qubit Ising sweep shows short-step fitness can be anti-correlated with fair
final VQE performance under small budgets.
"""

from __future__ import annotations

from aicir.qas.vqe_hea_demo import run_ising4_trainability_prior_demo


def main() -> None:
    report = run_ising4_trainability_prior_demo()
    print("\n".join(report.summary_lines()))


if __name__ == "__main__":
    main()
