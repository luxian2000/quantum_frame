#!/usr/bin/env python3
"""跨框架基准运行器。

用法（仓库根目录）:
    python scripts/bench/run_bench.py --axis parity          # 先验有效性
    python scripts/bench/run_bench.py --axis micro --max-qubits 20
    python scripts/bench/run_bench.py --axis circuits --max-qubits 22
    python scripts/bench/run_bench.py --axis all --output-json out.json

设计要点见 ``scripts/bench/README.md``。三条铁律：
  1. **先跑 parity 再跑计时**——态对不上时任何计时表都是在比较不同的计算；
  2. 构建与执行分开计时；
  3. 中位数 + IQR，预热剔除。
"""

from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np

from scripts.bench.adapters import available_adapters, get_adapter
from scripts.bench.axes import (
    capability_matrix,
    measure_peak_memory,
    npu_vs_cpu_plan,
    run_vqe_axis,
    strong_scaling_plan,
)
from scripts.bench.core.manifest import build_manifest, write_manifest
from scripts.bench.core.spec import build_spec, gate_counts
from scripts.bench.core.timing import time_callable

#: parity 校验用的规格（小比特数，快）。
#:
#: **不要删掉 random / layered_ansatz。** ``ghz`` 与 ``qft`` 从 |0…0⟩ 得到的末态在
#: 比特反转下不变（GHZ 是 |00…0⟩+|11…1⟩，QFT|0⟩ 是均匀叠加），对比特序错误完全失明；
#: 只有带随机角度的那两族有区分力。``tests/bench`` 里有元测试钉住这一点。
PARITY_SPECS = (
    ("ghz", dict(n_qubits=4)),
    ("qft", dict(n_qubits=4)),
    ("random", dict(n_qubits=5, depth=3, seed=11)),
    ("layered_ansatz", dict(n_qubits=4, depth=2, seed=5)),
)

PARITY_TOLERANCE = 1e-5


def _instantiate(name: str, precision: str):
    """按目标精度实例化适配器；不支持该精度的框架返回 None。"""

    if name == "aicir":
        dtype = np.complex128 if precision == "double" else np.complex64
        return get_adapter(name, dtype=dtype)
    if name == "cirq":
        dtype = np.complex128 if precision == "double" else np.complex64
        return get_adapter(name, dtype=dtype)
    if name == "qiskit_aer":
        return get_adapter(name, precision=precision)
    if name == "tensorcircuit":
        return get_adapter(name, dtype="complex128" if precision == "double" else "complex64")
    if name == "qulacs":
        # Qulacs 无单精度构建。
        return get_adapter(name) if precision == "double" else None
    return get_adapter(name)


def run_parity(adapters: list[str], precision: str) -> tuple[list[dict], list[str]]:
    """校验所有可用框架对同一规格给出同一个态。返回 (记录, 失败条件)。"""

    records: list[dict] = []
    failed: list[str] = []

    for family, kwargs in PARITY_SPECS:
        spec = build_spec(family, **kwargs)
        reference = None
        for name in adapters:
            adapter = _instantiate(name, precision)
            if adapter is None:
                continue
            try:
                state = adapter.statevector(spec)
            except Exception as exc:
                records.append({"axis": "parity", "spec": spec.label(), "framework": name, "error": str(exc)})
                failed.append(f"parity_error:{name}:{spec.label()}")
                continue

            if reference is None:
                reference = state
                overlap = 1.0
            else:
                overlap = float(abs(np.vdot(reference, state)))

            ok = abs(overlap - 1.0) <= PARITY_TOLERANCE
            records.append(
                {
                    "axis": "parity",
                    "spec": spec.label(),
                    "framework": name,
                    "overlap": overlap,
                    "passed": ok,
                }
            )
            if not ok:
                failed.append(f"parity_mismatch:{name}:{spec.label()}")

    return records, failed


def run_timing(
    adapters: list[str],
    precision: str,
    family: str,
    qubit_range: range,
    *,
    depth: int | None,
    seed: int | None,
    repeats: int,
    warmup: int,
) -> list[dict]:
    """对一个线路族做逐比特数计时。"""

    records: list[dict] = []
    for n_qubits in qubit_range:
        spec = build_spec(family, n_qubits=n_qubits, depth=depth, seed=seed)
        for name in adapters:
            adapter = _instantiate(name, precision)
            if adapter is None:
                continue
            try:
                build_stats = time_callable(lambda: adapter.build(spec), repeats=repeats, warmup=warmup)
                circuit = adapter.build(spec)
                run_stats = time_callable(lambda: adapter.run(circuit, spec), repeats=repeats, warmup=warmup)
            except Exception as exc:
                records.append(
                    {
                        "axis": "timing",
                        "spec": spec.label(),
                        "framework": name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            records.append(
                {
                    "axis": "timing",
                    "spec": spec.label(),
                    "family": family,
                    "n_qubits": n_qubits,
                    "framework": name,
                    "precision": adapter.precision(),
                    "n_gates": spec.n_gates,
                    "n_two_qubit_gates": spec.n_two_qubit_gates,
                    "gate_counts": gate_counts(spec),
                    "build": build_stats.to_dict(),
                    "run": run_stats.to_dict(),
                }
            )
    return records


def _print_capability_matrix(matrix: dict) -> None:
    """轴 G：能力矩阵。论文里最强的一张表——它说明的是"谁跑得了"，不是"谁跑得快"。"""

    rows = sorted(matrix)
    caps = sorted({c for entry in matrix.values() for c in entry["capabilities"]})
    width = max(len(c) for c in caps) + 2

    print("\n=== 能力矩阵（轴 G）===")
    header = " " * width + " ".join(f"{name[:12]:>13}" for name in rows)
    print(header)
    print("-" * len(header))
    for cap in caps:
        cells = []
        for name in rows:
            value = matrix[name]["capabilities"].get(cap)
            cells.append(f"{'yes' if value else ('-' if value is False else '?'):>13}")
        print(f"{cap:<{width}}" + " ".join(cells))
    print()
    for name in rows:
        if not matrix[name]["available"]:
            print(f"  [note] {name} 未安装——表中为其声明能力，非本机实测")


def _print_axis_table(records: list[dict], axis: str, columns: list[tuple[str, str]], title: str) -> None:
    rows = [r for r in records if r.get("axis") == axis]
    if not rows:
        return
    print(f"\n=== {title} ===")
    header = " ".join(f"{label:>18}" for label, _ in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        for _, key in columns:
            value = row
            for part in key.split("."):
                value = value.get(part, "-") if isinstance(value, dict) else "-"
            cells.append(f"{value:>18.6f}" if isinstance(value, float) else f"{str(value):>18}")
        print(" ".join(cells))


def _print_timing_table(records: list[dict]) -> None:
    rows = [r for r in records if r.get("axis") == "timing" and "run" in r]
    if not rows:
        return
    frameworks = sorted({r["framework"] for r in rows})
    families = sorted({r["family"] for r in rows})

    for family in families:
        print(f"\n=== {family} — 执行时间中位数 (s) ===")
        header = f"{'n':>4} " + " ".join(f"{f:>16}" for f in frameworks)
        print(header)
        print("-" * len(header))
        qubits = sorted({r["n_qubits"] for r in rows if r["family"] == family})
        for n in qubits:
            cells = []
            for framework in frameworks:
                match = [r for r in rows if r["family"] == family and r["n_qubits"] == n and r["framework"] == framework]
                cells.append(f"{match[0]['run']['median']:>16.6f}" if match else f"{'-':>16}")
            print(f"{n:>4} " + " ".join(cells))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="aicir 跨框架基准")
    parser.add_argument(
        "--axis",
        default="parity",
        choices=["parity", "micro", "circuits", "vqe", "memory", "capability", "npu", "scaling", "all"],
    )
    parser.add_argument("--frameworks", default=None, help="逗号分隔；默认全部可用")
    parser.add_argument("--precision", default="double", choices=["double", "single"])
    parser.add_argument("--min-qubits", type=int, default=4)
    parser.add_argument("--max-qubits", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-json", default=None, help="清单写入路径")
    parser.add_argument("--skip-parity", action="store_true", help="跳过有效性校验（不建议）")
    args = parser.parse_args(argv)

    adapters = list(available_adapters())
    if args.frameworks:
        requested = [f.strip() for f in args.frameworks.split(",") if f.strip()]
        missing = [f for f in requested if f not in adapters]
        if missing:
            print(f"[warn] 请求的框架不可用，将跳过: {missing}", file=sys.stderr)
        adapters = [f for f in requested if f in adapters]

    if not adapters:
        print("[error] 没有可用的框架适配器", file=sys.stderr)
        return 2

    print(f"可用框架: {adapters}")
    print(f"精度: {args.precision}\n")

    records: list[dict] = []
    failed: list[str] = []

    if args.axis in ("parity", "all") or not args.skip_parity:
        parity_records, parity_failed = run_parity(adapters, args.precision)
        records.extend(parity_records)
        failed.extend(parity_failed)
        status = "FAIL" if parity_failed else "PASS"
        print(f"parity: {status} ({len(parity_records)} 项)")
        for item in parity_records:
            if not item.get("passed", True) or "error" in item:
                print(f"  ! {item}")
        # parity 不过就不该继续计时——比的会是不同的线路。
        if parity_failed and args.axis != "parity":
            print("\n[abort] parity 未通过，拒绝产出计时数据。", file=sys.stderr)
            args.axis = "parity"

    qubit_range = range(args.min_qubits, args.max_qubits + 1)

    if args.axis in ("micro", "all"):
        records.extend(
            run_timing(
                adapters, args.precision, "ghz", qubit_range,
                depth=None, seed=None, repeats=args.repeats, warmup=args.warmup,
            )
        )

    if args.axis in ("circuits", "all"):
        for family in ("qft", "random", "layered_ansatz"):
            records.extend(
                run_timing(
                    adapters, args.precision, family, qubit_range,
                    depth=args.depth, seed=args.seed,
                    repeats=args.repeats, warmup=args.warmup,
                )
            )

    if args.axis in ("vqe", "all"):
        for n_qubits in range(args.min_qubits, min(args.max_qubits, 14) + 1, 2):
            try:
                records.append(
                    run_vqe_axis("aicir", n_qubits=n_qubits, layers=2, repeats=args.repeats, warmup=args.warmup)
                )
            except NotImplementedError as exc:
                print(f"[skip] 轴 C n={n_qubits}: {exc}", file=sys.stderr)
                break

    if args.axis in ("memory", "all"):
        for name in adapters:
            for n_qubits in qubit_range:
                try:
                    records.append(measure_peak_memory(name, build_spec("ghz", n_qubits=n_qubits)))
                except Exception as exc:
                    records.append({"axis": "memory", "framework": name, "error": str(exc)})

    if args.axis in ("scaling", "all"):
        try:
            records.extend(strong_scaling_plan(n_qubits=args.max_qubits, world_sizes=(1, 2, 4, 8)))
            records.extend(strong_scaling_plan(n_qubits=args.max_qubits, world_sizes=(1, 2, 4, 8), mode="weak"))
        except ValueError as exc:
            failed.append(f"scaling_plan_invalid:{exc}")

    if args.axis in ("npu", "all"):
        plan = npu_vs_cpu_plan()
        records.append(plan)
        if not plan["npu_available"]:
            print(f"\n[skip] 轴 E：{plan['skipped_reason']}")

    _print_timing_table(records)
    _print_axis_table(
        records,
        "vqe",
        [("n_qubits", "n_qubits"), ("params", "n_parameters"), ("energy(s)", "energy_eval.median"),
         ("grad(s)", "gradient_eval.median")],
        "VQE 端到端（轴 C，aicir）",
    )
    _print_axis_table(
        records,
        "memory",
        [("framework", "framework"), ("n", "n_qubits"), ("peak_bytes", "peak_bytes"),
         ("overhead", "overhead_ratio")],
        "峰值内存（轴 D）",
    )

    if args.axis in ("capability", "all"):
        matrix = capability_matrix()
        _print_capability_matrix(matrix)
        records.append({"axis": "capability", "matrix": matrix})

    manifest = build_manifest(results=records, failed_conditions=failed)
    print(f"\nrelease_gate = {manifest['release_gate']}")
    if args.output_json:
        path = write_manifest(manifest, args.output_json)
        print(f"清单已写入 {path}")

    return 0 if manifest["release_gate"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[中断]", file=sys.stderr)
        raise SystemExit(130)
    except Exception:  # pragma: no cover
        traceback.print_exc()
        raise SystemExit(1)
