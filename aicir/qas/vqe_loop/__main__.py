"""Command-line entry for the VQE-QAS closed loop.

Run with ``python -m aicir.qas.vqe_loop``.  The command is intentionally a thin
front door over ``run_vqe_qas_closed_loop``: all workflow logic stays in the
service modules so API and CLI runs share the same path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from aicir.chemistry.spec import GeneratedHamiltonian, load_hamiltonian_input
from .vqe_qas_loop import ClosedLoopConfig, run_vqe_qas_closed_loop


def _load_generated_hamiltonian(path: str | None) -> GeneratedHamiltonian | None:
    if not path:
        return None
    return load_hamiltonian_input(path)


def _infer_n_qubits(generated: GeneratedHamiltonian | None, explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    if generated is None:
        raise ValueError("--n-qubits is required when --hamiltonian is not provided")
    return int(generated.n_qubits)


def _auto_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "auto", "none"}:
        return None
    return int(text)


def _default_output_dir(hamiltonian_path: str | None, n_qubits: int | None = None) -> str:
    if hamiltonian_path:
        stem = Path(str(hamiltonian_path)).stem
        slug = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_").lower() or "hamiltonian"
        return str(Path("outputs") / f"qas_{slug}_loop")
    if n_qubits is not None:
        return str(Path("outputs") / f"qas_{int(n_qubits)}q_loop")
    return str(Path("outputs") / "qas_loop")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the VQE-QAS closed loop")
    parser.add_argument("hamiltonian_input", nargs="?", help="Optional positional Hamiltonian JSON path")
    parser.add_argument(
        "--hamiltonian",
        default=None,
        help="JSON Hamiltonian input: legacy [[coeff, pauli], ...], pauli_terms spec, or molecular spec",
    )
    parser.add_argument("--hamiltonian-id", default=None, help="Optional override for the auto-generated Hamiltonian id")
    parser.add_argument("--hamiltonian-class", default=None, help="Optional override for the Hamiltonian class")
    parser.add_argument("--n-qubits", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--rounds", default="auto")
    parser.add_argument("--initial-labels", default="auto")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--d-max", type=float, default=0.28125)
    parser.add_argument("--local", type=int, default=None)
    parser.add_argument("--boundary", type=int, default=None)
    parser.add_argument("--sparse", type=int, default=None)
    parser.add_argument("--control", type=int, default=None)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-improvement", type=float, default=1.0e-8)
    parser.add_argument("--ea-population", type=int, default=32)
    parser.add_argument("--ea-generations", type=int, default=8)
    parser.add_argument("--ea-seed-count", type=int, default=12)
    parser.add_argument("--ea-seed", type=int, default=101)
    parser.add_argument("--supernet-native-count", type=int, default=0)
    parser.add_argument("--supernet-native-layers", type=int, default=3)
    parser.add_argument("--supernet-native-supernet-num", type=int, default=2)
    parser.add_argument("--supernet-native-steps", type=int, default=20)
    parser.add_argument("--supernet-native-ranking-num", type=int, default=24)
    parser.add_argument("--supernet-native-finetune-steps", type=int, default=0)
    parser.add_argument("--supernet-native-seed", type=int, default=11)
    parser.add_argument("--supernet-native-device", default="cpu")
    parser.add_argument("--label-seed", type=int, default=5200)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--max-evals", type=int, default=100)
    parser.add_argument("--backend", default="npu")
    parser.add_argument("--dtype", default="complex64")
    parser.add_argument("--protocol", default="aicir/qas/vqe_loop/fair_label_protocol.json")
    parser.add_argument("--no-layerwise", action="store_true")
    parser.add_argument("--layerwise-count", default="auto")
    parser.add_argument("--layerwise-layers", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    hamiltonian_path = args.hamiltonian or args.hamiltonian_input
    generated = _load_generated_hamiltonian(hamiltonian_path)
    n_qubits = _infer_n_qubits(generated, args.n_qubits)
    config = ClosedLoopConfig(
        output_dir=Path(args.output_dir or _default_output_dir(hamiltonian_path, n_qubits)),
        n_qubits=n_qubits,
        hamiltonian_terms=generated.terms if generated is not None else None,
        hamiltonian_id=str(args.hamiltonian_id or (generated.hamiltonian_id if generated is not None else "literal_hamiltonian")),
        hamiltonian_class=str(args.hamiltonian_class or (generated.hamiltonian_class if generated is not None else "literal")),
        rounds=_auto_int(args.rounds),
        initial_labels=_auto_int(args.initial_labels),
        batch_size=_auto_int(args.batch_size),
        holdout_fraction=float(args.holdout_fraction),
        k_min=int(args.k_min),
        d_max=float(args.d_max),
        local=args.local,
        boundary=args.boundary,
        sparse=args.sparse,
        control=args.control,
        patience=int(args.patience),
        min_improvement=float(args.min_improvement),
        ea_population=int(args.ea_population),
        ea_generations=int(args.ea_generations),
        ea_seed_count=int(args.ea_seed_count),
        ea_seed=int(args.ea_seed),
        supernet_native_count=int(args.supernet_native_count),
        supernet_native_layers=int(args.supernet_native_layers),
        supernet_native_supernet_num=int(args.supernet_native_supernet_num),
        supernet_native_steps=int(args.supernet_native_steps),
        supernet_native_ranking_num=int(args.supernet_native_ranking_num),
        supernet_native_finetune_steps=int(args.supernet_native_finetune_steps),
        supernet_native_seed=int(args.supernet_native_seed),
        supernet_native_device=str(args.supernet_native_device),
        label_seed=int(args.label_seed),
        n_seeds=int(args.n_seeds),
        max_evals=int(args.max_evals),
        backend=str(args.backend),
        dtype=str(args.dtype),
        protocol=Path(args.protocol),
        include_layerwise=not bool(args.no_layerwise),
        layerwise_count=_auto_int(args.layerwise_count),
        layerwise_layers=int(args.layerwise_layers),
    )
    result = run_vqe_qas_closed_loop(config)
    print(json.dumps({key: str(value) for key, value in result.__dict__.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
