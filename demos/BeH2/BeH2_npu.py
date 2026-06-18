"""在 4 张 Ascend NPU 上用 supernet QAS 搜索 BeH2 的 VQE 线路（任务并行加速）。

方法选择：与 ``demos/BeH2/BeH2.py`` 一致，使用 ``supernet`` (``supernet_qas``)。
它是面向任意哈密顿量做 VQE ansatz 结构搜索的唯一 QAS 方法；CRLQAS / PPR-DQL /
PPO-RB 面向小规模目标态制备，不适合 16 量子比特分子哈密顿量。

4 张 NPU 的使用方式是**任务并行**：每张卡用不同随机种子并发跑一次独立 supernet
搜索。``supernet_qas`` 没有把单次搜索切到多卡的内置能力，因此这里用 4 卡在同样的
墙钟时间里并发探索 4 个种子，再取能量最优的线路——相对单卡单种子，这是“用 4 卡
加速找到好线路”的实际做法，而不是把单个态向量切成分布式张量。每个 rank 通过
``LOCAL_RANK`` 绑定一张 NPU，与 ``demos/demo_npu.py`` / ``aicir/qas/README.md``
的 NPU 任务并行约定一致。

运行结束后由 rank 0 汇总所有 rank 的结果，写出：

- ``output.txt``    : 各 rank 能量对比与最优结果的文本报告（反馈信息）
- ``BeH2_npu_cir.qasm`` / ``BeH2_npu_cir.py`` : 全局最优线路

运行方式（4 卡）：

    torchrun --nproc_per_node=4 demos/BeH2/BeH2_npu.py

无 NPU 的本地冒烟测试（单进程、允许 CPU 回退）：

    python demos/BeH2/BeH2_npu.py --allow-cpu-fallback
"""

from __future__ import annotations

import argparse
import platform
import time
from datetime import datetime
from pathlib import Path

from aicir.backends.npu_backend import (
    NPUBackend,
    is_npu_available,
    npu_runtime_context_from_env,
)
from aicir.core.circuit import Circuit
from aicir.core.io.qasm import save_circuit_qasm3
from aicir.measure import hamiltonian_pauli_terms
from aicir.qas import supernet_qas

from demos.BeH2.BeH2 import (
    BEH2_BASIS,
    BEH2_CHARGE,
    BEH2_QUBIT_MAPPER,
    BEH2_SPIN,
    beh2_vqe_qas_kwargs,
    build_beh2_hamiltonian,
    save_circuit_python,
)


def _log(message: str, rank: int) -> None:
    print(f"[rank {rank}] {message}", flush=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BeH2 16-qubit supernet VQE QAS on 4 Ascend NPU (task-parallel).",
    )
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--supernet-num", type=int, default=6)
    parser.add_argument("--supernet-steps", type=int, default=300)
    parser.add_argument("--ranking-num", type=int, default=120)
    parser.add_argument("--finetune-steps", type=int, default=500)
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="基础随机种子；每个 rank 实际使用 seed + rank，保证各卡搜索不同架构。",
    )
    parser.add_argument(
        "--output",
        default="output.txt",
        help="文本报告路径（默认写在脚本同目录）。",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="NPU 不可用时回退到 CPU（仅用于开发冒烟测试）。默认严格使用 NPU。",
    )
    parser.add_argument(
        "--no-init-process-group",
        action="store_true",
        help="即使存在 torchrun 环境变量也不初始化 torch.distributed。",
    )
    return parser.parse_args(argv)


def _write_report(
    records: list[dict],
    output_path: Path,
    best_rank: int,
    *,
    n_qubits: int,
    n_terms: int,
    world_size: int,
    started_at: str,
    elapsed: float,
) -> None:
    ranked = sorted(records, key=lambda r: r["fine_tuned_energy"])
    lines = [
        "=" * 88,
        "BeH2 supernet VQE QAS on Ascend NPU (task-parallel, one search per NPU)",
        "=" * 88,
        f"started at            : {started_at}",
        f"finished at           : {datetime.now().isoformat(timespec='seconds')}",
        f"platform              : {platform.platform()}",
        f"NPUs (world_size)     : {world_size}",
        f"qubits / terms        : {n_qubits} / {n_terms}",
        f"basis/charge/spin/map : {BEH2_BASIS}/{BEH2_CHARGE}/{BEH2_SPIN}/{BEH2_QUBIT_MAPPER}",
        f"total wall-clock time : {elapsed:.1f} s",
        "",
        "per-NPU results (sorted by fine-tuned energy):",
        "rank  seed  device     fine_tuned_energy(Ha)   baseline_energy(Ha)  CNOT/2q   seconds",
        "-" * 88,
    ]
    for r in ranked:
        mark = "  <- best" if r["rank"] == best_rank else ""
        lines.append(
            f"{r['rank']:>4}  {r['seed']:>4}  {r['device']:<9}  "
            f"{r['fine_tuned_energy']:>+21.10f}  {r['baseline_vqe_energy']:>+19.10f}  "
            f"{r['cnot']}/{r['two_qubit']}  {r['seconds']:>7.1f}{mark}"
        )
    best = ranked[0]
    lines.extend(
        [
            "",
            "best result:",
            f"  rank               : {best['rank']}",
            f"  seed               : {best['seed']}",
            f"  fine-tuned energy  : {best['fine_tuned_energy']:+.10f} Ha",
            f"  baseline VQE       : {best['baseline_vqe_energy']:+.10f} Ha",
            f"  CNOT / 2-qubit     : {best['cnot']} / {best['two_qubit']}",
            "  circuit files      : BeH2_npu_cir.qasm, BeH2_npu_cir.py",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    backend = NPUBackend.from_distributed_env(
        fallback_to_cpu=args.allow_cpu_fallback,
        init_process_group=not args.no_init_process_group,
    )
    ctx = npu_runtime_context_from_env()
    rank = backend.distributed_rank
    world_size = backend.distributed_world_size
    device = f"npu:{ctx.local_rank}" if is_npu_available() else "cpu"

    _log("=== BeH2 supernet VQE QAS on Ascend NPU ===", rank)
    _log(f"backend={backend.name} device={device}", rank)
    _log(f"world_size={world_size} distributed={backend.distributed_initialized}", rank)

    hamiltonian = build_beh2_hamiltonian()
    n_terms = len(hamiltonian_pauli_terms(hamiltonian))
    if rank == 0:
        print(
            f"BeH2 active-space Hamiltonian: basis={BEH2_BASIS} charge={BEH2_CHARGE} "
            f"spin={BEH2_SPIN} mapper={BEH2_QUBIT_MAPPER}"
        )
        print(f"Qubits: {hamiltonian.n_qubits}  Terms: {n_terms}")

    # 每个 rank 用不同 seed 做一次独立 supernet 搜索（任务并行）。
    seed = args.seed + rank
    kwargs = beh2_vqe_qas_kwargs()
    kwargs.update(
        {
            "layers": args.layers,
            "supernet_num": args.supernet_num,
            "supernet_steps": args.supernet_steps,
            "ranking_num": args.ranking_num,
            "finetune_steps": args.finetune_steps,
            "seed": seed,
            "device": device,
        }
    )

    _log(f"start supernet search (seed={seed}) ...", rank)
    started_at = datetime.now().isoformat(timespec="seconds")
    start = time.time()
    result = supernet_qas(hamiltonian, **kwargs)
    elapsed = time.time() - start

    metrics = result.final_metrics
    record = {
        "rank": rank,
        "seed": seed,
        "device": device,
        "fine_tuned_energy": float(metrics["fine_tuned_energy"]),
        "baseline_vqe_energy": float(metrics["baseline_vqe_energy"]),
        "cnot": int(metrics["selected_cnot_count"]),
        "two_qubit": int(metrics["selected_two_qubit_count"]),
        "seconds": elapsed,
        # 携带线路本身，便于 rank 0 集中落盘（gate dict 可被 pickle）。
        "n_qubits": int(result.best_circuit.n_qubits),
        "gates": list(result.best_circuit.gates),
    }
    _log(
        f"fine_tuned_energy={record['fine_tuned_energy']:+.10f} "
        f"baseline={record['baseline_vqe_energy']:+.10f} "
        f"cnot={record['cnot']} time={elapsed:.1f}s",
        rank,
    )

    # all-gather 各 rank 的完整结果（按 rank 排序），由 rank 0 统一汇总落盘。
    gathered = backend.gather_indexed_results([(rank, record)])
    records = [item[1] for item in gathered]
    best = min(records, key=lambda r: r["fine_tuned_energy"])
    best_rank = best["rank"]

    if rank == 0:
        print("\n=== gathered results across NPUs ===")
        for r in sorted(records, key=lambda r: r["rank"]):
            mark = "  <- best" if r["rank"] == best_rank else ""
            print(f"  rank {r['rank']} (seed {r['seed']}): energy={r['fine_tuned_energy']:+.10f}{mark}")
        print(f"global best energy: {best['fine_tuned_energy']:+.10f} (rank {best_rank})")

        out_dir = Path(__file__).parent
        report_path = (out_dir / args.output) if not Path(args.output).is_absolute() else Path(args.output)
        _write_report(
            records,
            report_path,
            best_rank,
            n_qubits=hamiltonian.n_qubits,
            n_terms=n_terms,
            world_size=world_size,
            started_at=started_at,
            elapsed=elapsed,
        )
        print(f"text report saved to: {report_path}")

        best_circuit = Circuit(*best["gates"], n_qubits=best["n_qubits"])
        qasm_path = out_dir / "BeH2_npu_cir.qasm"
        save_circuit_qasm3(best_circuit, qasm_path)
        print(f"OpenQASM 3.0 saved to: {qasm_path}")

        py_path = out_dir / "BeH2_npu_cir.py"
        save_circuit_python(best_circuit, py_path, func_name="build_beh2_npu_qas_circuit")
        print(f"Python circuit saved to: {py_path}")


if __name__ == "__main__":
    main()
