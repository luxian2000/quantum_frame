"""Search a deeper H2O ground-state circuit with the supernet method on NPU.

This is the NPU counterpart of ``demos/H2O/H2O.py``. It runs the same supernet
QAS search but on the Ascend NPU backend (``device="npu"``).

Default hyperparameters were retuned after experiment 1 (see
``demos/H2O/result_npu.md``). There the ``L=8`` search reached only ~344.9 mHa
while the *fixed-ansatz* VQE baseline reached ~0.53 mHa, i.e. the circuit has
enough capacity but the search under-converged. The retuned defaults spend more
budget on optimisation: ``supernet_num`` 5 -> 8 (more supernets relieve
weight-sharing competition), ``supernet_steps`` 250 -> 500, ``ranking_num``
80 -> 150, and most importantly ``finetune_steps`` 250 -> 1000 (fine-tuning the
single selected circuit is the dominant lever for the final reported energy).
The default depth is also reduced to ``L=6`` to shrink the search space (a
smaller, easier-to-search ansatz), matching the CPU demo's depth. Expect a
longer wall-clock time than experiment 1's ~1320 s.

All information is written to a text report (default ``H2O_npu_result.txt``,
next to this file) and also echoed to stdout, so the remote NPU run leaves a
self-contained log even if the process is detached. The searched circuit is
additionally written to ``H2O_cir_npu.py`` / ``H2O_cir_npu.qasm`` and plotted
to ``H2O_cir_npu.png``, mirroring ``demos/H2O/H2O.py`` (the CPU demo's
``H2O_cir.*`` files are left untouched).

Run on the remote NPU platform (from the repository root)::

    python -m demos.H2O.H2O_npu                 # device=npu:0, layers=8
    python -m demos.H2O.H2O_npu --layers 10
    python -m demos.H2O.H2O_npu --device npu:1 --layers 12 --output run12.txt

A CPU dry run (no NPU needed) is possible with ``--device cpu`` to validate the
script before submitting it to the NPU queue.
"""

from __future__ import annotations

import argparse
import platform
import runpy
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Make ``demos`` / ``aicir`` importable when launched as a plain script
# (``python demos/H2O/H2O_npu.py``) and not only as ``python -m demos.H2O.H2O_npu``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch


def _enable_npu() -> object:
    """Import ``torch_npu`` to register the ``npu`` device. Return True or the error."""
    try:
        import torch_npu  # noqa: F401  registers torch.device("npu")

        return True
    except Exception as exc:  # pragma: no cover - depends on the remote platform
        return exc


def _device_report(device: str) -> list[str]:
    lines = [f"torch version            : {torch.__version__}"]
    npu_mod = getattr(torch, "npu", None)
    if npu_mod is not None:
        try:
            lines.append(f"torch.npu.is_available() : {npu_mod.is_available()}")
            lines.append(f"torch.npu.device_count() : {npu_mod.device_count()}")
        except Exception as exc:  # pragma: no cover
            lines.append(f"torch.npu query failed   : {exc!r}")
    else:
        lines.append("torch.npu                : not present")
    lines.append(f"requested device         : {device}")
    return lines


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deeper H2O supernet VQE on NPU.")
    parser.add_argument("--device", default="npu:0", help="torch device, e.g. npu:0 / npu / cpu")
    parser.add_argument("--layers", type=int, default=6, help="ansatz depth L")
    parser.add_argument("--supernet-num", type=int, default=8, help="number of supernets W (more relieves weight-sharing competition)")
    parser.add_argument("--supernet-steps", type=int, default=500, help="supernet optimisation steps")
    parser.add_argument("--ranking-num", type=int, default=150, help="number of ranked candidate ansatze")
    parser.add_argument("--finetune-steps", type=int, default=1000, help="fine-tuning steps (dominant lever for the final energy)")
    parser.add_argument("--seed", type=int, default=2, help="random seed")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("H2O_npu_result.txt")),
        help="text report path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).with_name(args.output)

    handle = output_path.open("w", encoding="utf-8")

    def log(line: str = "") -> None:
        print(line)
        handle.write(line + "\n")
        handle.flush()

    try:
        log("=" * 72)
        log("H2O supernet ground-state search on NPU (deeper ansatz)")
        log("=" * 72)
        log(f"timestamp                : {datetime.now().isoformat(timespec='seconds')}")
        log(f"platform                 : {platform.platform()}")

        # Enable the NPU device if requested.
        if "npu" in args.device:
            enabled = _enable_npu()
            if enabled is not True:
                log(f"[warning] import torch_npu failed: {enabled!r}")
                log("[warning] the run below will fail unless an 'npu' device is available.")
        for line in _device_report(args.device):
            log(line)

        # Import the shared H2O helpers (Hamiltonian, exact energy, IO) and the
        # encapsulated supernet entry point.
        from demos.H2O.H2O import (
            build_h2o_hamiltonian,
            exact_ground_energy,
            circuit_to_python_source,
            H2O_TWO_QUBIT_PAIRS,
        )
        from aicir.qas import supernet_qas
        from aicir.core.io.qasm import save_circuit_qasm3

        hamiltonian = build_h2o_hamiltonian()
        exact = exact_ground_energy(hamiltonian)  # exact value computed on CPU (numpy)

        log("")
        log("search configuration:")
        log(f"  qubits                 : {hamiltonian.n_qubits}")
        log(f"  layers (depth)         : {args.layers}")
        log("  single-qubit gates     : (i, h, rx, ry, rz)")
        log("  two-qubit gates        : (cx, rzz)")
        log(f"  two-qubit pairs        : {len(H2O_TWO_QUBIT_PAIRS)} (nearest + next-nearest)")
        log(f"  supernet_num (W)       : {args.supernet_num}")
        log(f"  supernet_steps         : {args.supernet_steps}")
        log(f"  ranking_num            : {args.ranking_num}")
        log(f"  finetune_steps         : {args.finetune_steps}")
        log(f"  seed                   : {args.seed}")
        log("")
        log(f"exact ground energy      : {exact:+.10f} Ha")
        log("running supernet search ... (this can take several minutes for deep L)")

        start = time.time()
        result = supernet_qas(
            hamiltonian,
            layers=args.layers,
            supernet_num=args.supernet_num,
            supernet_steps=args.supernet_steps,
            finetune_steps=args.finetune_steps,
            ranking_num=args.ranking_num,
            device=args.device,
            seed=args.seed,
        )
        elapsed = time.time() - start

        metrics = result.final_metrics
        qas_energy = float(metrics["fine_tuned_energy"])
        baseline_energy = float(metrics["baseline_vqe_energy"])
        abs_err = abs(qas_energy - exact)

        log("")
        log("-" * 72)
        log("results")
        log("-" * 72)
        log(f"exact ground energy      : {exact:+.10f} Ha")
        log(f"QAS fine-tuned energy    : {qas_energy:+.10f} Ha")
        log(f"fixed-ansatz VQE baseline: {baseline_energy:+.10f} Ha")
        log(f"|QAS - exact|            : {abs_err:.6e} Ha  ({abs_err * 1000:.3f} mHa)")
        log(f"chemical accuracy (<1.6 mHa): {abs_err < 1.6e-3}")
        log(f"variational bound (>= exact): {qas_energy >= exact - 1e-6}")
        log(
            "selected CNOT / two-qubit: "
            f"{metrics['selected_cnot_count']} / {metrics['selected_two_qubit_count']}"
        )
        log(f"wall-clock time          : {elapsed:.1f} s")
        log("")
        log("selected ansatz circuit (ASCII):")
        log(metrics["selected_circuit_ascii"])

        # Save the searched circuit and plot it, just like demos/H2O/H2O.py,
        # under NPU-specific names so the CPU demo artifacts (H2O_cir.*) stay.
        qasm_path = Path(__file__).with_name("H2O_cir_npu.qasm")
        save_circuit_qasm3(result.best_circuit, qasm_path)

        py_path = Path(__file__).with_name("H2O_cir_npu.py")
        source = circuit_to_python_source(
            result.best_circuit,
            func_name="build_h2o_npu_qas_circuit",
            figure_name="H2O_cir_npu.png",
            title=f"H2O supernet ground-state ansatz (NPU, L={args.layers})",
        )
        # Fix the auto-generated regenerate/plot hints for the NPU variant.
        source = (
            source.replace("by demos/H2O/H2O.py.", "by demos/H2O/H2O_npu.py.")
            .replace("python -m demos.H2O.H2O``.", "python -m demos.H2O.H2O_npu``.")
            .replace("python -m demos.H2O.H2O_cir``.", "python -m demos.H2O.H2O_cir_npu``.")
        )
        py_path.write_text(source, encoding="utf-8")

        log("")
        log(f"OpenQASM 3.0 saved to    : {qasm_path}")
        log(f"Python circuit saved to  : {py_path}")

        # Plot via the generated module's __main__ (runs on CPU; no NPU needed),
        # exactly as demos/H2O/H2O.py hands off to H2O_cir.py.
        figure_path = Path(__file__).with_name("H2O_cir_npu.png")
        try:
            runpy.run_path(str(py_path), run_name="__main__")
            log(f"circuit figure saved to  : {figure_path}")
        except Exception as plot_exc:  # pragma: no cover - matplotlib may be absent
            log(f"[warning] plotting failed ({plot_exc!r}); circuit .py/.qasm still saved.")

        log(f"text report saved to     : {output_path}")
    except Exception:  # pragma: no cover - surface remote failures into the log
        log("")
        log("[ERROR] the run failed with an exception:")
        log(traceback.format_exc())
        raise
    finally:
        handle.close()


if __name__ == "__main__":
    main()
