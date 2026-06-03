"""Demo: task-level QAS validation against QAOA/HEA baselines."""

from __future__ import annotations

from aicir.qas import OptimizerConfig, SearchConfig, maxcut_line, run_validation_experiment


def main() -> None:
    problem = maxcut_line(n_qubits=4)
    report = run_validation_experiment(
        problem,
        search_config=SearchConfig(n_qubits=4, candidate_layers=1, n_samples=8),
        optimizer_config=OptimizerConfig(max_evaluations=16, seed=2026),
        qas_top_k=3,
    )
    print("\n".join(report.summary_lines()))


if __name__ == "__main__":
    main()
