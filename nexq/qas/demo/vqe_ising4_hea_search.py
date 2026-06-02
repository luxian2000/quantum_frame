"""Run the 4-qubit Ising/TFIM VQE-QAS HEA demo."""

from __future__ import annotations

from nexq.qas.vqe_hea_demo import run_vqe_ising4_demo


def main() -> None:
    report = run_vqe_ising4_demo()
    print("\n".join(report.summary_lines()))


if __name__ == "__main__":
    main()
