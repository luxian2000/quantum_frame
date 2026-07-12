"""Run DQAS for the H2O active-space ground state on NPU.

Run from the repository root:

    python -m demos.H2O.H2O_dqas
    python -m demos.H2O.H2O_dqas --device npu:1 --layers 4
    python -m demos.H2O.H2O_dqas --device cpu --search-epochs 1

This demo reuses the 6-qubit H2O active-space Hamiltonian from
``demos/H2O/H2O.py`` and searches a ground-state-preparing circuit with
``aicir.qas`` method name ``"dqas"`` over an **excitation gate pool**
(``gate_pool="excitation"``). DQAS samples each circuit placeholder from one
global categorical operation pool containing identity plus the spin-preserving
``single_excitation`` / ``double_excitation`` operators, while excitation angles
are shared through the DQAS parameter pool ``theta[position, operation, param]``.

The default device is ``npu:0``. A CPU dry run is available with ``--device cpu``
to validate the script on a machine without Ascend hardware. The discretised
circuit's energy is compared against the dense-matrix exact value, a text report
is written to ``H2O_dqas_npu_result.txt``, and the searched circuit is written to
``H2O_dqas_npu_cir.py``. Running that generated module transpile-optimises the
circuit and plots it to ``H2O_dqas_npu_cir.png``.

Requires ``torch`` (DQAS runs on a Torch/NPU backend).
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

import torch

# Make ``demos`` / ``aicir`` importable when launched as a plain script and not
# only as ``python -m demos.H2O.H2O_dqas``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aicir.qas import config, run
from demos.H2O.H2O import (
    H2O_DOUBLE_EXCITATIONS,
    H2O_HF_OCCUPIED_QUBITS,
    H2O_SINGLE_EXCITATIONS,
    build_h2o_hamiltonian,
    exact_ground_energy,
    save_circuit_python,
)


def _enable_npu() -> object:
    """Import ``torch_npu`` to register the NPU device, returning True or an error."""
    try:
        import torch_npu  # noqa: F401  registers torch.device("npu")

        return True
    except Exception as exc:  # pragma: no cover - platform dependent
        return exc


def _device_report(device: str) -> list[str]:
    lines = [f"torch version            : {torch.__version__}"]
    npu_mod = getattr(torch, "npu", None)
    if npu_mod is not None:
        try:
            lines.append(f"torch.npu.is_available() : {npu_mod.is_available()}")
            lines.append(f"torch.npu.device_count() : {npu_mod.device_count()}")
        except Exception as exc:  # pragma: no cover - platform dependent
            lines.append(f"torch.npu query failed   : {exc!r}")
    else:
        lines.append("torch.npu                : not present")
    lines.append(f"requested device         : {device}")
    return lines


def _validate_requested_device(device: str, npu_count: int | None = None) -> str | None:
    """Return a user-facing error if an explicit NPU index is outside range."""
    value = str(device).strip().lower()
    if not value.startswith("npu"):
        return None
    if ":" not in value:
        return None
    _, raw_index = value.split(":", 1)
    try:
        index = int(raw_index)
    except ValueError:
        return f"Invalid NPU device {device!r}; expected forms like 'npu', 'npu:0', or 'npu:1'."

    if npu_count is None:
        npu_mod = getattr(torch, "npu", None)
        if npu_mod is None:
            return None
        try:
            npu_count = int(npu_mod.device_count())
        except Exception:  # pragma: no cover - platform dependent
            return None

    if npu_count <= 0:
        return f"Invalid NPU device {device!r}; no NPU devices are visible to torch."
    if index < 0 or index >= npu_count:
        valid = "npu:0" if npu_count == 1 else f"npu:0..npu:{npu_count - 1}"
        return (
            f"Invalid NPU device {device!r}; torch reports {npu_count} visible "
            f"NPU device(s), so valid explicit devices are {valid}."
        )
    return None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H2O DQAS excitation search on NPU.")
    parser.add_argument("--device", default="npu:0", help="torch device, e.g. npu:0 / npu / cpu")
    parser.add_argument("--layers", type=int, default=2, help="DQAS placeholder count")
    parser.add_argument("--search-epochs", type=int, default=24, help="DQAS architecture search epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Monte Carlo architecture batch size")
    parser.add_argument("--theta-steps", type=int, default=1, help="theta updates per search epoch")
    parser.add_argument("--finetune-steps", type=int, default=80, help="fixed-circuit parameter fine-tuning steps")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("H2O_dqas_npu_result.txt")),
        help="text report path",
    )
    return parser.parse_args(argv)


def h2o_dqas_config(**overrides) -> object:
    """Build the DQAS excitation-pool search config for 6-qubit H2O on NPU.

    Defaults target an NPU run. Increase
    ``layers``, ``search_epochs`` and ``finetune_steps`` for a stronger search.
    """

    params: dict = dict(
        n_qubits=6,
        layers=2,
        gate_pool="excitation",
        single_excitations=H2O_SINGLE_EXCITATIONS,
        double_excitations=H2O_DOUBLE_EXCITATIONS,
        hf_occupied_qubits=H2O_HF_OCCUPIED_QUBITS,
        search_epochs=24,
        batch_size=8,
        theta_steps=1,
        finetune_steps=80,
        architecture_learning_rate=0.05,
        theta_learning_rate=0.05,
        finetune_learning_rate=0.03,
        seed=7,
        device="npu:0",
    )
    params.update(overrides)
    return config.dqas(**params)


def main(argv: list[str] | None = None) -> None:
    """Search an H2O ground-state circuit with DQAS, save it, and plot it."""

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
        log("H2O DQAS excitation search on NPU")
        log("=" * 72)
        log(f"timestamp                : {datetime.now().isoformat(timespec='seconds')}")
        log(f"platform                 : {platform.platform()}")

        if "npu" in args.device:
            enabled = _enable_npu()
            if enabled is not True:
                log(f"[warning] import torch_npu failed: {enabled!r}")
                log("[warning] the run below will fail unless an 'npu' device is available.")
        for line in _device_report(args.device):
            log(line)
        device_error = _validate_requested_device(args.device)
        if device_error is not None:
            log("")
            log(f"[error] {device_error}")
            log("[hint] In this run torch.npu.device_count() showed the valid range. Try --device npu:0 or --device npu:1.")
            raise SystemExit(2)

        hamiltonian = build_h2o_hamiltonian()
        exact = exact_ground_energy(hamiltonian)
        log("")
        log("H2O active-space Hamiltonian (PySCF/Qiskit Nature, STO-3G, JW)")
        log(f"  qubits: {hamiltonian.n_qubits}")
        log(f"  dense-matrix exact ground energy: {exact:+.10f} Ha")

        cfg = h2o_dqas_config(
            layers=args.layers,
            search_epochs=args.search_epochs,
            batch_size=args.batch_size,
            theta_steps=args.theta_steps,
            finetune_steps=args.finetune_steps,
            seed=args.seed,
            device=args.device,
        )
        log("")
        log("DQAS config")
        log(f"  device        : {cfg.device}")
        log(f"  layers        : {cfg.layers}")
        log(f"  gate_pool     : {cfg.gate_pool}")
        log(f"  singles       : {len(cfg.single_excitations)}")
        log(f"  doubles       : {len(cfg.double_excitations)}")
        log(f"  HF occupied   : {cfg.hf_occupied_qubits}")
        log(f"  search_epochs : {cfg.search_epochs}")
        log(f"  batch_size    : {cfg.batch_size}")
        log(f"  finetune_steps: {cfg.finetune_steps}")

        log("")
        log("Searching a ground-state circuit with DQAS...")
        start = time.perf_counter()
        # run() 统一返回 QASResult（3b breaking change）：能量走统一字段 value，
        # 方法专属的架构标签从 raw（原 DQASResult）取。
        result = run("dqas", hamiltonian=hamiltonian, config=cfg)
        elapsed = time.perf_counter() - start

        dqas_energy = float(result.value)
        log("")
        log(f"  exact ground energy : {exact:+.10f}")
        log(f"  DQAS minimum energy : {dqas_energy:+.10f}")
        log(f"  |DQAS - exact|      : {abs(dqas_energy - exact):.3e} Ha")
        log(f"  elapsed             : {elapsed:.1f} s")

        log("")
        log("  discretised architecture (one operation per DQAS placeholder):")
        for layer_index, label in enumerate(result.raw.architecture_labels):
            log(f"    placeholder {layer_index}: {label}")

        log("")
        log("  searched circuit:")
        result.circuit.show()

        circuit_py_path = Path(__file__).parent / "H2O_dqas_npu_cir.py"
        save_circuit_python(
            result.circuit,
            circuit_py_path,
            func_name="build_h2o_dqas_npu_circuit",
            figure_name="H2O_dqas_npu_cir.png",
            title="H2O DQAS ground-state ansatz (NPU)",
            function_docstring="Return the NPU DQAS-searched H2O ground-state circuit.",
            generated_by="demos/H2O/H2O_dqas.py",
            description="NPU DQAS searched excitation ansatz preparing the H2O active-space ground state.",
            regen_cmd="python -m demos.H2O.H2O_dqas",
            plot_cmd="python -m demos.H2O.H2O_dqas_npu_cir",
            optimize=True,
        )
        log(f"  Python circuit saved to: {circuit_py_path}")

        log("  Plotting via H2O_dqas_npu_cir.py ...")
        runpy.run_path(str(circuit_py_path), run_name="__main__")
    except Exception:
        log("")
        log("[error] H2O DQAS NPU run failed:")
        log(traceback.format_exc())
        raise
    finally:
        handle.close()


if __name__ == "__main__":
    main()
