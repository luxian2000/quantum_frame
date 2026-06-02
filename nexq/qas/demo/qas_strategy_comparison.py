"""Demo: compare QAS SuperCircuit search strategies on a small MaxCut task."""

from __future__ import annotations

from nexq.metrics.hardware import HardwareProfile
from nexq.qas import OptimizerConfig, SearchConfig, maxcut_line, run_search_strategy_comparison


def main() -> None:
    problem = maxcut_line(n_qubits=4)
    profile = HardwareProfile(coupling_map=[(0, 1), (1, 2), (2, 3)], max_depth=12)
    report = run_search_strategy_comparison(
        problem,
        search_config=SearchConfig(
            n_qubits=4,
            candidate_layers=2,
            n_samples=8,
            include_common_candidates=True,
            population_size=6,
            search_generations=2,
            beam_width=2,
            mutation_rate=0.5,
            top_k=4,
            active_metrics={
                "trainability": "gradient_variance",
                "hardware_efficiency": "topology_mapping_efficiency",
            },
        ),
        optimizer_config=OptimizerConfig(max_evaluations=8, seed=2026),
        qas_top_k=3,
        strategies=(
            "supercircuit_progressive",
            "supercircuit_evolution",
            "supercircuit_reflective",
            "task_feedback",
            "hybrid",
        ),
        feedback_generations=2,
        feedback_population_size=5,
        feedback_elite_count=2,
        hardware_profile=profile,
    )
    print("\n".join(report.summary_lines()))


if __name__ == "__main__":
    main()
