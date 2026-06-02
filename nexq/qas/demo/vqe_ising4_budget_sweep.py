"""Run the 4-qubit Ising VQE-QAS SA budget sweep."""

from __future__ import annotations

from nexq.qas.vqe_hea_demo import run_ising4_budget_sweep


def main() -> None:
    report = run_ising4_budget_sweep()
    print("\n".join(report.summary_lines()))


if __name__ == "__main__":
    main()
