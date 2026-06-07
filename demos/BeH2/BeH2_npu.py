"""Run BeH2 supernet VQE hyperparameter sweeps on NPU devices.

Run from repository root:

    python -m demos.BeH2.BeH2_npu
    python -m demos.BeH2.BeH2_npu --layers 4,6,8 --supernet-num 6,12
    python -m demos.BeH2.BeH2_npu --devices npu:0,npu:1,npu:2,npu:3

The parent process schedules one BeH2 search per ``(layers, supernet_num)``
configuration and runs several searches concurrently across the requested NPU
devices. Energies are summarized in ``output.txt`` and all successful best
circuits are written into one importable ``output.py`` module.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch


def _enable_npu() -> object:
    try:
        import torch_npu  # noqa: F401

        return True
    except Exception as exc:  # pragma: no cover
        return exc


def _device_report(device: str | None = None) -> list[str]:
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
    if device is not None:
        lines.append(f"requested device         : {device}")
    return lines


def _parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _parse_device_list(value: str) -> list[str]:
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one device")
    return values


def _path_next_to_this(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(__file__).with_name(path_value)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BeH2 supernet VQE hyperparameter sweep on NPU.")
    parser.add_argument(
        "--devices",
        type=_parse_device_list,
        default=None,
        help="comma-separated devices for the sweep, e.g. npu:0,npu:1,npu:2,npu:3",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="single-device compatibility option; use --devices for multi-NPU sweeps",
    )
    parser.add_argument(
        "--layers",
        type=_parse_int_list,
        default=[4, 6, 8],
        help="comma-separated ansatz depths, e.g. 4,6,8",
    )
    parser.add_argument(
        "--supernet-num",
        type=_parse_int_list,
        default=[6, 12],
        help="comma-separated supernet counts W, e.g. 6,8,12",
    )
    parser.add_argument("--supernet-steps", type=int, default=300)
    parser.add_argument("--ranking-num", type=int, default=120)
    parser.add_argument("--finetune-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        default="output.txt",
        help="combined text report path",
    )
    parser.add_argument(
        "--circuit-output",
        default="output.py",
        help="combined Python circuit module path",
    )

    # Hidden worker mode. The parent process starts one worker per experiment.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-id", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _jsonable(value.detach().cpu().item())
        return _jsonable(value.detach().cpu().tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def _run_single_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if "npu" in args.device:
        enabled = _enable_npu()
        if enabled is not True:
            raise RuntimeError(f"import torch_npu failed: {enabled!r}")

    from aicir.qas import supernet_qas
    from demos.BeH2.BeH2 import beh2_vqe_qas_kwargs, build_beh2_hamiltonian

    hamiltonian_start = time.time()
    hamiltonian = build_beh2_hamiltonian()
    hamiltonian_elapsed = time.time() - hamiltonian_start

    kwargs = beh2_vqe_qas_kwargs()
    kwargs.update(
        {
            "layers": args.layers[0],
            "supernet_num": args.supernet_num[0],
            "supernet_steps": args.supernet_steps,
            "ranking_num": args.ranking_num,
            "finetune_steps": args.finetune_steps,
            "seed": args.seed,
            "device": args.device,
        }
    )

    start = time.time()
    result = supernet_qas(hamiltonian, **kwargs)
    elapsed = time.time() - start
    metrics = result.final_metrics

    return {
        "status": "ok",
        "run_id": args.run_id,
        "device": args.device,
        "layers": args.layers[0],
        "supernet_num": args.supernet_num[0],
        "supernet_steps": args.supernet_steps,
        "ranking_num": args.ranking_num,
        "finetune_steps": args.finetune_steps,
        "seed": args.seed,
        "n_qubits": hamiltonian.n_qubits,
        "hamiltonian_terms": len(hamiltonian._terms),
        "hamiltonian_build_seconds": hamiltonian_elapsed,
        "wall_seconds": elapsed,
        "fine_tuned_energy": float(metrics["fine_tuned_energy"]),
        "baseline_vqe_energy": float(metrics["baseline_vqe_energy"]),
        "selected_cnot_count": int(metrics["selected_cnot_count"]),
        "selected_two_qubit_count": int(metrics["selected_two_qubit_count"]),
        "selected_circuit_ascii": str(metrics["selected_circuit_ascii"]),
        "circuit": {
            "n_qubits": int(result.best_circuit.n_qubits),
            "gates": _jsonable(result.best_circuit.gates),
        },
    }


def _worker_main(args: argparse.Namespace) -> int:
    if args.result_json is None:
        raise ValueError("--result-json is required in worker mode")
    result_path = Path(args.result_json)
    try:
        result = _run_single_experiment(args)
    except Exception:
        result = {
            "status": "error",
            "run_id": args.run_id,
            "device": args.device,
            "layers": args.layers[0],
            "supernet_num": args.supernet_num[0],
            "seed": args.seed,
            "traceback": traceback.format_exc(),
        }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["status"] == "ok" else 1


def _function_name(record: dict[str, Any]) -> str:
    return (
        "build_beh2_qas_"
        f"l{int(record['layers'])}_"
        f"w{int(record['supernet_num'])}_"
        f"seed{int(record['seed'])}"
    )


def _gate_call_from_record(gate: dict[str, Any]) -> tuple[str, str] | None:
    from demos.H2O.H2O import _gate_to_python_call

    return _gate_to_python_call(gate)


def _write_circuit_module(records: list[dict[str, Any]], output_path: Path) -> None:
    successful = [record for record in records if record.get("status") == "ok"]
    used_builders: set[str] = set()
    rendered: list[tuple[dict[str, Any], list[tuple[str, str]]]] = []
    for record in successful:
        calls: list[tuple[str, str]] = []
        for gate in record["circuit"]["gates"]:
            call = _gate_call_from_record(gate)
            if call is not None:
                calls.append(call)
                used_builders.add(call[0])
        rendered.append((record, calls))

    import_list = ", ".join(["Circuit", *sorted(used_builders)])
    lines = [
        '"""Auto-generated BeH2 NPU hyperparameter sweep circuits.',
        "",
        "Regenerate with: ``python -m demos.BeH2.BeH2_npu``.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        f"from aicir.core.circuit import {import_list}",
        "",
        "",
        "EXPERIMENTS = [",
    ]
    for record in successful:
        lines.append(
            "    "
            + repr(
                {
                    "function": _function_name(record),
                    "layers": int(record["layers"]),
                    "supernet_num": int(record["supernet_num"]),
                    "seed": int(record["seed"]),
                    "device": record["device"],
                    "fine_tuned_energy": float(record["fine_tuned_energy"]),
                    "baseline_vqe_energy": float(record["baseline_vqe_energy"]),
                    "wall_seconds": float(record["wall_seconds"]),
                }
            )
            + ","
        )
    lines.extend(["]", "", ""])

    for record, calls in rendered:
        function_name = _function_name(record)
        lines.append(f"def {function_name}():")
        lines.append(
            "    "
            + f'"""Return the best BeH2 circuit for layers={record["layers"]}, '
            + f'supernet_num={record["supernet_num"]}, seed={record["seed"]}."""'
        )
        lines.append("    gates = [")
        for name, args in calls:
            lines.append(f"        {name}({args}),")
        lines.append("    ]")
        lines.append(f"    return Circuit(*gates, n_qubits={int(record['circuit']['n_qubits'])})")
        lines.append("")
        lines.append("")

    if successful:
        best = min(successful, key=lambda record: float(record["fine_tuned_energy"]))
        lines.append(f"best_circuit = {_function_name(best)}()")
    else:
        lines.append("best_circuit = None")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(
    records: list[dict[str, Any]],
    output_path: Path,
    circuit_output_path: Path,
    started_at: str,
    elapsed: float,
    devices: list[str],
) -> None:
    successful = [record for record in records if record.get("status") == "ok"]
    failed = [record for record in records if record.get("status") != "ok"]
    ranked = sorted(successful, key=lambda record: float(record["fine_tuned_energy"]))

    lines = [
        "=" * 88,
        "BeH2 NPU hyperparameter sweep",
        "=" * 88,
        f"started at               : {started_at}",
        f"finished at              : {datetime.now().isoformat(timespec='seconds')}",
        f"platform                 : {platform.platform()}",
        f"devices                  : {', '.join(devices)}",
        f"total wall-clock time    : {elapsed:.1f} s",
        f"successful / failed      : {len(successful)} / {len(failed)}",
        f"circuit module           : {circuit_output_path}",
        "",
        "device report:",
    ]
    lines.extend(f"  {line}" for line in _device_report())
    lines.extend(
        [
            "",
            "results sorted by fine-tuned energy:",
            (
                "rank  status  device  layers  supernet_num  seed  "
                "fine_tuned_energy(Ha)  baseline_energy(Ha)  CNOT/2q  seconds  function"
            ),
            "-" * 140,
        ]
    )
    for rank, record in enumerate(ranked, start=1):
        lines.append(
            f"{rank:>4}  ok      {record['device']:<6}  "
            f"{int(record['layers']):>6}  {int(record['supernet_num']):>12}  "
            f"{int(record['seed']):>4}  {float(record['fine_tuned_energy']):>+21.10f}  "
            f"{float(record['baseline_vqe_energy']):>+19.10f}  "
            f"{int(record['selected_cnot_count'])}/{int(record['selected_two_qubit_count'])}  "
            f"{float(record['wall_seconds']):>7.1f}  {_function_name(record)}"
        )
    for record in failed:
        lines.append(
            f"   -  error   {record.get('device', ''):<6}  "
            f"{int(record.get('layers', 0)):>6}  {int(record.get('supernet_num', 0)):>12}  "
            f"{int(record.get('seed', 0)):>4}  {'nan':>21}  {'nan':>19}  -  -  -"
        )

    if ranked:
        best = ranked[0]
        lines.extend(
            [
                "",
                "best result:",
                f"  layers                 : {best['layers']}",
                f"  supernet_num           : {best['supernet_num']}",
                f"  device                 : {best['device']}",
                f"  fine-tuned energy      : {float(best['fine_tuned_energy']):+.10f} Ha",
                f"  circuit function       : {_function_name(best)}",
            ]
        )

    if failed:
        lines.extend(["", "failed runs:"])
        for record in failed:
            lines.extend(
                [
                    "-" * 88,
                    f"run_id={record.get('run_id')} device={record.get('device')} "
                    f"layers={record.get('layers')} supernet_num={record.get('supernet_num')}",
                    str(record.get("traceback", "")).rstrip(),
                ]
            )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _build_worker_command(
    args: argparse.Namespace,
    *,
    run_id: str,
    device: str,
    layers: int,
    supernet_num: int,
    result_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "demos.BeH2.BeH2_npu",
        "--worker",
        "--run-id",
        run_id,
        "--result-json",
        str(result_path),
        "--device",
        device,
        "--layers",
        str(layers),
        "--supernet-num",
        str(supernet_num),
        "--supernet-steps",
        str(args.supernet_steps),
        "--ranking-num",
        str(args.ranking_num),
        "--finetune-steps",
        str(args.finetune_steps),
        "--seed",
        str(args.seed),
    ]


def _run_sweep(args: argparse.Namespace) -> int:
    output_path = _path_next_to_this(args.output)
    circuit_output_path = _path_next_to_this(args.circuit_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    circuit_output_path.parent.mkdir(parents=True, exist_ok=True)

    devices = args.devices
    if devices is None:
        devices = [args.device] if args.device is not None else ["npu:0", "npu:1", "npu:2", "npu:3"]
    if any("npu" in device for device in devices):
        enabled = _enable_npu()
        if enabled is not True:
            print(f"[warning] import torch_npu failed in parent: {enabled!r}")

    jobs = [
        {
            "run_id": f"run_{index:03d}",
            "layers": layers,
            "supernet_num": supernet_num,
        }
        for index, (layers, supernet_num) in enumerate(
            (layer, width) for layer in args.layers for width in args.supernet_num
        )
    ]
    started_at = datetime.now().isoformat(timespec="seconds")
    start = time.time()

    print("=" * 88)
    print("BeH2 NPU hyperparameter sweep")
    print("=" * 88)
    print(f"experiments              : {len(jobs)}")
    print(f"devices                  : {', '.join(devices)}")
    print(f"layers                   : {', '.join(str(v) for v in args.layers)}")
    print(f"supernet_num             : {', '.join(str(v) for v in args.supernet_num)}")
    print(f"output                   : {output_path}")
    print(f"circuit output           : {circuit_output_path}")
    print("")

    records: list[dict[str, Any]] = []
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    with tempfile.TemporaryDirectory(prefix="beh2_npu_sweep_") as temp_dir:
        temp_root = Path(temp_dir)
        pending = list(jobs)
        running: dict[subprocess.Popen[str], dict[str, Any]] = {}
        idle_devices = list(devices)

        while pending or running:
            while pending and idle_devices:
                device = idle_devices.pop(0)
                job = pending.pop(0)
                result_path = temp_root / f"{job['run_id']}.json"
                command = _build_worker_command(
                    args,
                    run_id=job["run_id"],
                    device=device,
                    layers=job["layers"],
                    supernet_num=job["supernet_num"],
                    result_path=result_path,
                )
                process = subprocess.Popen(
                    command,
                    cwd=str(_REPO_ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                running[process] = {**job, "device": device, "result_path": result_path}
                print(
                    f"[start] {job['run_id']} device={device} "
                    f"layers={job['layers']} supernet_num={job['supernet_num']}"
                )

            finished = [process for process in running if process.poll() is not None]
            if not finished:
                time.sleep(5.0)
                continue

            for process in finished:
                job = running.pop(process)
                output, _ = process.communicate()
                result_path = job["result_path"]
                if result_path.exists():
                    record = json.loads(result_path.read_text(encoding="utf-8"))
                    record["worker_stdout"] = output
                else:
                    record = {
                        "status": "error",
                        "run_id": job["run_id"],
                        "device": job["device"],
                        "layers": job["layers"],
                        "supernet_num": job["supernet_num"],
                        "seed": args.seed,
                        "traceback": "worker exited without writing result JSON\n" + (output or ""),
                    }
                records.append(record)
                idle_devices.append(job["device"])
                if record.get("status") == "ok":
                    print(
                        f"[done]  {job['run_id']} device={job['device']} "
                        f"layers={job['layers']} supernet_num={job['supernet_num']} "
                        f"energy={float(record['fine_tuned_energy']):+.10f} Ha"
                    )
                else:
                    print(
                        f"[error] {job['run_id']} device={job['device']} "
                        f"layers={job['layers']} supernet_num={job['supernet_num']}"
                    )

    records.sort(key=lambda record: str(record.get("run_id", "")))
    elapsed = time.time() - start
    _write_circuit_module(records, circuit_output_path)
    _write_report(records, output_path, circuit_output_path, started_at, elapsed, devices)
    print("")
    print(f"text report saved to     : {output_path}")
    print(f"Python circuits saved to : {circuit_output_path}")
    return 0 if all(record.get("status") == "ok" for record in records) else 1


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.worker:
        raise SystemExit(_worker_main(args))
    raise SystemExit(_run_sweep(args))


if __name__ == "__main__":
    main()
