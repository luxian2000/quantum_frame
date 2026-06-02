"""Run trainability-prior Stage1 ranking on the 4-qubit Ising VQE demo."""

from __future__ import annotations

from nexq.qas.vqe_hea_demo import run_ising4_trainability_prior_demo


def main() -> None:
    report = run_ising4_trainability_prior_demo()
    print("\n".join(report.summary_lines()))


if __name__ == "__main__":
    main()
