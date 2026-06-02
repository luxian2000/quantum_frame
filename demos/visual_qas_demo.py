"""Demo: visualize QAS candidates and metric scores.

Run from the repository root:
    python -m demos.visual_qas_demo
"""

from __future__ import annotations

import argparse
from pprint import pprint

import numpy as np

from aicir.channel.backends.numpy_backend import NumpyBackend
from aicir.qas import build_common_architectures, evaluate_architectures
from aicir.visual import (
    compare_architectures,
    plot_architecture_metrics,
    plot_qas_summary,
    plot_search_history,
    qas_scores_to_rows,
)

from ._visual_demo_utils import add_common_visual_args, configure_matplotlib, save_figure


def evaluate_demo_architectures():
    np.random.seed(7)
    backend = NumpyBackend()
    architectures = build_common_architectures(
        n_qubits=3,
        layers=1,
        backend=backend,
        names=["hea_linear", "qaoa_chain", "brickwork_cx", "ghz_ladder"],
    )
    return evaluate_architectures(architectures, backend=backend, n_samples=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize QAS search results and metrics.")
    add_common_visual_args(parser)
    args = parser.parse_args()

    plt = configure_matplotlib(args.show)
    scores = evaluate_demo_architectures()
    rows = qas_scores_to_rows(scores)

    print("=== QAS rows ===")
    for row in rows:
        pprint(row)

    best = scores[0]
    print("\n=== Best architecture ===")
    print(f"rank={best.rank} name={best.architecture.name} weighted_score={best.weighted_score:.6f}")

    fig, _ = plot_search_history(
        scores,
        metrics=["weighted_score", "expressibility", "trainability", "hardware_efficiency"],
        title="QAS ranking metrics",
    )
    print(f"Saved search history figure: {save_figure(fig, args.output_dir, 'visual_qas_history.png')}")

    fig, _ = compare_architectures(
        scores,
        metrics=["weighted_score", "n_gates", "two_qubit_gate_count"],
        title="QAS candidate comparison",
    )
    print(f"Saved comparison figure: {save_figure(fig, args.output_dir, 'visual_qas_compare.png')}")

    fig, _ = plot_architecture_metrics(
        best,
        metrics=["weighted_score", "expressibility", "trainability", "noise_robustness", "hardware_efficiency"],
        title=f"Best metrics: {best.architecture.name}",
    )
    print(f"Saved best metrics figure: {save_figure(fig, args.output_dir, 'visual_qas_best_metrics.png')}")

    fig, _ = plot_qas_summary(
        best,
        metrics=["weighted_score", "trainability", "hardware_efficiency"],
        title="Best QAS candidate summary",
    )
    print(f"Saved summary figure: {save_figure(fig, args.output_dir, 'visual_qas_summary.png')}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
