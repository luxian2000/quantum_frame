"""Command-line entry for the VQE-QAS closed loop.

Run with ``python -m aicir.qas.vqe_loop``.  The command is intentionally a thin
front door over ``run_vqe_qas_closed_loop``: all workflow logic stays in the
service modules so API and CLI runs share the same path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .vqe_qas_loop import ClosedLoopConfig, run_vqe_qas_closed_loop


def _load_hamiltonian_terms(path: str | None) -> tuple[tuple[float, str], ...] | None:
    if not path:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes, dict)):
        raise ValueError("Hamiltonian JSON must be a list of [coefficient, pauli] terms")
    terms: list[tuple[float, str]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid Hamiltonian term: {item!r}")
        coeff, pauli = item
        terms.append((float(coeff), str(pauli)))
    if not terms:
        raise ValueError("Hamiltonian JSON must contain at least one Pauli term")
    return tuple(terms)


def _infer_n_qubits(terms: tuple[tuple[float, str], ...] | None, explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    if not terms:
        raise ValueError("--n-qubits is required when --hamiltonian is not provided")
    widths = {len(pauli) for _coeff, pauli in terms}
    if len(widths) != 1:
        raise ValueError(f"Hamiltonian Pauli strings must have one width, found {sorted(widths)}")
    return int(next(iter(widths)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the VQE-QAS closed loop")
    parser.add_argument("--hamiltonian", default=None, help="JSON file containing [[coeff, pauli], ...] terms")
    parser.add_argument("--hamiltonian-id", default="literal_hamiltonian")
    parser.add_argument("--hamiltonian-class", default="literal")
    parser.add_argument("--n-qubits", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--initial-labels", type=int, default=24)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--d-max", type=float, default=0.28125)
    parser.add_argument("--local", type=int, default=4)
    parser.add_argument("--boundary", type=int, default=2)
    parser.add_argument("--sparse", type=int, default=2)
    parser.add_argument("--control", type=int, default=0)
    parser.add_argument("--ea-population", type=int, default=32)
    parser.add_argument("--ea-generations", type=int, default=8)
    parser.add_argument("--ea-seed-count", type=int, default=12)
    parser.add_argument("--ea-seed", type=int, default=101)
    parser.add_argument("--label-seed", type=int, default=5200)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--max-evals", type=int, default=100)
    parser.add_argument("--backend", default="numpy")
    parser.add_argument("--dtype", default="complex128")
    parser.add_argument("--protocol", default="aicir/qas/vqe_loop/fair_label_protocol.json")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    terms = _load_hamiltonian_terms(args.hamiltonian)
    config = ClosedLoopConfig(
        output_dir=Path(args.output_dir),
        n_qubits=_infer_n_qubits(terms, args.n_qubits),
        hamiltonian_terms=terms,
        hamiltonian_id=str(args.hamiltonian_id),
        hamiltonian_class=str(args.hamiltonian_class),
        rounds=int(args.rounds),
        initial_labels=int(args.initial_labels),
        holdout_fraction=float(args.holdout_fraction),
        k_min=int(args.k_min),
        d_max=float(args.d_max),
        local=int(args.local),
        boundary=int(args.boundary),
        sparse=int(args.sparse),
        control=int(args.control),
        ea_population=int(args.ea_population),
        ea_generations=int(args.ea_generations),
        ea_seed_count=int(args.ea_seed_count),
        ea_seed=int(args.ea_seed),
        label_seed=int(args.label_seed),
        n_seeds=int(args.n_seeds),
        max_evals=int(args.max_evals),
        backend=str(args.backend),
        dtype=str(args.dtype),
        protocol=Path(args.protocol),
    )
    result = run_vqe_qas_closed_loop(config)
    print(json.dumps({key: str(value) for key, value in result.__dict__.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
