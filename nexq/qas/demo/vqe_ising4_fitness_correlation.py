"""Validate short-step VQE fitness against fair final VQE on 4-qubit Ising."""

from __future__ import annotations

from nexq.qas.vqe_hea_demo import run_ising4_fitness_correlation


def main() -> None:
    report = run_ising4_fitness_correlation()
    print("\n".join(report.summary_lines()))


if __name__ == "__main__":
    main()
