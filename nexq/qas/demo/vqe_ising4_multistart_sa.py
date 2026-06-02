"""Run the 4-qubit Ising VQE-QAS diverse multi-start SA demo."""

from __future__ import annotations

from nexq.qas.vqe_hea_demo import run_ising4_multistart_sa


def main() -> None:
    report = run_ising4_multistart_sa()
    print("\n".join(report.summary_lines()))


if __name__ == "__main__":
    main()
