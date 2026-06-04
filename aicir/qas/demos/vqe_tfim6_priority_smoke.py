"""Run a budget-capped TFIM priority-seed smoke test.

This entry point is intentionally conservative: it checks whether
Hamiltonian-aware priority seeds are promising on 6q before launching the
larger all-space or multi-repeat experiments.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aicir.qas.vqe_hea_demo import (
    resolve_qas_backend,
    run_tfim_full_enumeration_baseline,
    run_tfim_priority_seed_validation,
    run_tfim_stage1_stage2_search,
)


def _parse_int_list(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or raw.strip() == "":
        return None
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _parse_layer_quota(raw: str | None) -> dict[int, int] | None:
    if raw is None or raw.strip() == "":
        return None
    result: dict[int, int] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        layer, quota = item.split(":", 1)
        result[int(layer.strip())] = int(quota.strip())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="6q TFIM Hamiltonian-aware priority-seed smoke test")
    parser.add_argument("--n-qubits", type=int, default=6)
    parser.add_argument("--mode", choices=("priority", "full", "pipeline"), default="priority")
    parser.add_argument("--priority-limit", type=int, default=12)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--only-name", default=None)
    parser.add_argument("--priority-layers", default="1,2,3")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--pool-count", type=int, default=4)
    parser.add_argument("--stage1-pool-size", type=int, default=16)
    parser.add_argument("--layer-quota", default="1:3,2:5,3:4")
    parser.add_argument("--beam-width", type=int, default=12)
    parser.add_argument("--stage2-rounds", type=int, default=2)
    parser.add_argument("--neighbors-per-parent", type=int, default=2)
    parser.add_argument("--b2-evals-per-param", type=int, default=20)
    parser.add_argument("--b2-max-evals", type=int, default=600)
    parser.add_argument("--fair-n-starts", type=int, default=1)
    parser.add_argument("--fair-evals-per-param", type=int, default=30)
    parser.add_argument("--fair-max-evals", type=int, default=3000)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--init-mode", choices=("zero", "random", "zero_then_random"), default="zero_then_random")
    parser.add_argument("--init-scale", type=float, default=3.141592653589793)
    parser.add_argument("--backend", choices=("numpy", "cpu", "npu", "torch"), default=None)
    args = parser.parse_args()
    priority_layers = _parse_int_list(args.priority_layers)
    layer_quota = _parse_layer_quota(args.layer_quota)

    backend = resolve_qas_backend(args.backend)
    print(f"backend: {backend.name}", flush=True)
    print(
        "config: "
        f"mode={args.mode}, n_qubits={args.n_qubits}, priority_limit={args.priority_limit}, "
        f"candidate_limit={args.candidate_limit}, only_name={args.only_name}, "
        f"priority_layers={priority_layers}, layer_quota={layer_quota}, repeats={args.repeats}, "
        f"fair_n_starts={args.fair_n_starts}, fair_evals_per_param={args.fair_evals_per_param}, "
        f"fair_max_evals={args.fair_max_evals}, init_mode={args.init_mode}, init_scale={args.init_scale}",
        flush=True,
    )
    if args.mode == "priority":
        report = run_tfim_priority_seed_validation(
            n_qubits=args.n_qubits,
            repeats=args.repeats,
            priority_limit=args.priority_limit,
            fair_n_starts=args.fair_n_starts,
            fair_evals_per_param=args.fair_evals_per_param,
            fair_max_evaluations=args.fair_max_evals,
            init_mode=args.init_mode,
            init_scale=args.init_scale,
            only_name=args.only_name,
            priority_layers=priority_layers,
            backend=backend,
        )
        print("\n".join(report.summary_lines()), flush=True)
    elif args.mode == "full":
        report = run_tfim_full_enumeration_baseline(
            n_qubits=args.n_qubits,
            candidate_limit=args.candidate_limit,
            fair_n_starts=args.fair_n_starts,
            fair_evals_per_param=args.fair_evals_per_param,
            fair_max_evaluations=args.fair_max_evals,
            init_mode=args.init_mode,
            init_scale=args.init_scale,
            priority_layers=priority_layers,
            layer_quota=layer_quota,
            backend=backend,
        )
        print("\n".join(report.summary_lines(top_k=args.top_k)), flush=True)
    else:
        report = run_tfim_stage1_stage2_search(
            n_qubits=args.n_qubits,
            pool_count=args.pool_count,
            candidate_limit=args.candidate_limit or 72,
            stage1_pool_size=args.stage1_pool_size,
            beam_width=args.beam_width,
            stage2_rounds=args.stage2_rounds,
            neighbors_per_parent=args.neighbors_per_parent,
            b2_evals_per_param=args.b2_evals_per_param,
            b2_max_evaluations=args.b2_max_evals,
            fair_repeats=args.repeats,
            fair_n_starts=args.fair_n_starts,
            fair_evals_per_param=args.fair_evals_per_param,
            fair_max_evaluations=args.fair_max_evals,
            init_mode=args.init_mode,
            init_scale=args.init_scale,
            backend=backend,
        )
        print("\n".join(report.summary_lines()), flush=True)


if __name__ == "__main__":
    main()
