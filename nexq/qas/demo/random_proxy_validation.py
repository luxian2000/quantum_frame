"""Demo: validate whether random-parameter objective is a useful search proxy."""

from __future__ import annotations

from nexq.qas import OptimizerConfig, maxcut_line, run_random_proxy_validation_experiment


def main() -> None:
    report = run_random_proxy_validation_experiment(
        maxcut_line(n_qubits=4),
        n_random_samples=80,
        optimizer_config=OptimizerConfig(max_evaluations=24, seed=2026),
        random_seed=2026,
    )
    print("\n".join(report.summary_lines()))


if __name__ == "__main__":
    main()
