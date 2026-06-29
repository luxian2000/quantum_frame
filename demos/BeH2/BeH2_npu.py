"""在 4 张 Ascend NPU 上用 supernet QAS 搜索 BeH2 的 VQE 线路（内搜索分片）。

方法选择：与 ``demos/BeH2/BeH2.py`` 一致，使用 ``supernet`` (``supernet_qas``)。
它是面向任意哈密顿量做 VQE ansatz 结构搜索的唯一 QAS 方法；CRLQAS / PPR-DQL /
PPO-RB 面向小规模目标态制备，不适合 16 量子比特分子哈密顿量。

所有 rank 使用**相同随机种子**，``supernet_qas`` 在内部将训练/排名/微调阶段切分到
多 NPU 上并行执行（内搜索分片）。``--mode`` 控制分片策略，两种模式区别如下：

- ``safe``（默认）：与单卡数值完全等价（确定性可复现）。每步所有 rank 用相同
  种子采样同一架构，只把候选 supernet 的选择评估 切分到各 rank（``_sharded_select``）；
  仅 rank 0 计算梯度并执行优化器步，随后 ``broadcast_parameters(src=0)`` 把权重广播给
  全部 rank。优点：与单卡逐位等价、结果可复现；缺点：rank 1..N 在 broadcast 屏障空等
  rank 0，负载不均衡（即因此触发 HCCL 屏障超时，见 ``aicir/backends/README.md`` §5.5）。

- ``aggressive``：数据并行，约 world_size 倍吞吐。每步各 rank 采样 world_size 个
  不同架构，rank ``r`` 负责 ``arch[r]``，各自前向 + ``backward``；梯度经 ``all_reduce_mean``
  汇总，所有被选中 supernet 的优化器一起步进。优点：每步探索约 N 倍架构、吃满多卡；
  缺点：各 rank 动态步不同，数值不再与单卡等价，搜索轨迹随卡数 N 变化。

  | 维度 | safe | aggressive |
  | --- | --- | --- |
  | 数值 | 与单卡逐位等价（确定） | 数据并行，随 N 变化 |
  | 分片对象 | 仅候选选择评估 | 整架构前向 + 梯度 |
  | 吞吐 | ~1×（rank 0 瓶颈） | ~world_size× |
  | 适用 | 复现/正确性基线 | 提速、加大探索 |

16 比特 BeH2 Hamiltonian 含 1313 个 Pauli 项。NPU 版本默认使用参数移位梯度，
避免 autograd 为每个 Pauli 项保留大反向图；如需对比可传 ``--gradient ad``。

每个 rank 通过 ``LOCAL_RANK`` 绑定一张 NPU，与 ``demos/demo_npu.py`` /
``aicir/qas/README.md`` 的 NPU 约定一致。搜索完成后返回的 ``SupernetResult``
在所有 rank 上均为全局最优，由 rank 0 落盘：

- ``output.txt``         : 分片搜索结果的文本报告（模式、卡数、能量、线路指标）
- ``BeH2_npu_cir.qasm`` / ``BeH2_npu_cir.py`` : 全局最优线路

运行方式（4 卡）：

    torchrun --nproc_per_node=4 demos/BeH2/BeH2_npu.py [--mode safe|aggressive]

无 NPU 的本地冒烟测试（单进程、允许 CPU 回退，分片路径未激活）：

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
from aicir.measure import hamiltonian_pauli_terms
from aicir.qas import supernet_qas
from demos.chemistry_ansatz import save_qasm3_if_supported

from demos.BeH2.BeH2 import (
    BEH2_BASIS,
    BEH2_CHARGE,
    BEH2_QUBIT_MAPPER,
    BEH2_SPIN,
    beh2_vqe_qas_kwargs,
    build_beh2_hamiltonian,
    save_circuit_python,
)

import os, threading, time, tracemalloc
def _mem_watch():
    tracemalloc.start(5)
    p = os.getpid()
    while True:
        rss = int(open(f"/proc/{p}/status").read().split("VmRSS:")[1].split()[0]) // 1024
        cur, peak = tracemalloc.get_traced_memory()
        top = tracemalloc.take_snapshot().statistics("lineno")[:5]
        print(f"[memwatch pid={p}] RSS={rss}MB  pytracemalloc cur={cur//2**20}MB", flush=True)
        for s in top:
            print(f"   {s.traceback[0]}  {s.size//1024}KB x{s.count}", flush=True)
        time.sleep(30)
if os.environ.get("RANK", "0") == "0" and os.environ.get("MEM_WATCH"):
    threading.Thread(target=_mem_watch, daemon=True).start()


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
        help="随机种子；所有 rank 使用相同 seed，保证权重/候选集一致，由分片机制加速。",
    )
    parser.add_argument(
        "--mode",
        choices=("safe", "aggressive"),
        default="safe",
        help="训练分片模式：safe=与单卡数值等价；aggressive=数据并行(~world_size 倍，动态不同)。",
    )
    parser.add_argument(
        "--gradient",
        choices=("psr", "ad"),
        default="psr",
        help="梯度方式：psr=参数移位（默认）；ad=autograd（每步 1 次前向+1 次 backward，比 psr 快约 2P 倍；"
             "需 Ascend float32 autograd 可用，已通过 _NpuHamiltonianExpectationFn 消除 complex64 瓶颈）。",
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
    record: dict,
    output_path: Path,
    *,
    mode: str,
    n_qubits: int,
    n_terms: int,
    world_size: int,
    started_at: str,
    elapsed: float,
) -> None:
    lines = [
        "=" * 88,
        "BeH2 supernet VQE QAS on Ascend NPU (in-search sharding, single run)",
        "=" * 88,
        f"started at            : {started_at}",
        f"finished at           : {datetime.now().isoformat(timespec='seconds')}",
        f"platform              : {platform.platform()}",
        f"mode                  : {mode}",
        f"NPUs (world_size)     : {world_size}",
        f"seed                  : {record['seed']}",
        f"device                : {record['device']}",
        f"qubits / terms        : {n_qubits} / {n_terms}",
        f"basis/charge/spin/map : {BEH2_BASIS}/{BEH2_CHARGE}/{BEH2_SPIN}/{BEH2_QUBIT_MAPPER}",
        f"total wall-clock time : {elapsed:.1f} s",
        "",
        "global-best result (identical on all ranks):",
        "-" * 88,
        f"  fine-tuned energy  : {record['fine_tuned_energy']:+.10f} Ha",
        f"  baseline VQE       : {record['baseline_vqe_energy']:+.10f} Ha",
        f"  excitations         : {record['excitations']} "
        f"(single={record['two_qubit']}, double={record['four_qubit']})",
        "  circuit files      : BeH2_npu_cir.qasm, BeH2_npu_cir.py",
        "",
    ]
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

    # 所有 rank 使用相同 seed；supernet_qas 内部将训练/排名/微调切分到各 NPU（内搜索分片）。
    kwargs = beh2_vqe_qas_kwargs()
    kwargs.update(
        {
            "layers": args.layers,
            "supernet_num": args.supernet_num,
            "supernet_steps": args.supernet_steps,
            "ranking_num": args.ranking_num,
            "finetune_steps": args.finetune_steps,
            "seed": args.seed,          # 同一 seed，保证各 rank 权重/候选集一致
            "device": device,
            "mode": args.mode,
            "use_parameter_shift": args.gradient == "psr",
        }
    )

    _log(
        f"start sharded supernet search (seed={args.seed}, mode={args.mode}, gradient={args.gradient}) ...",
        rank,
    )
    started_at = datetime.now().isoformat(timespec="seconds")
    start = time.time()
    result = supernet_qas(hamiltonian, **kwargs)
    elapsed = time.time() - start

    # result 在所有 rank 上均为全局最优，无需 all-gather。
    metrics = result.final_metrics
    record = {
        "seed": args.seed,
        "device": device,
        "fine_tuned_energy": float(metrics["fine_tuned_energy"]),
        "baseline_vqe_energy": float(metrics["baseline_vqe_energy"]),
        "cnot": int(metrics["selected_cnot_count"]),
        "two_qubit": int(metrics["selected_two_qubit_count"]),
        "four_qubit": int(metrics["selected_four_qubit_count"]),
        "excitations": int(metrics["selected_excitation_count"]),
        "seconds": elapsed,
    }
    _log(
        f"fine_tuned_energy={record['fine_tuned_energy']:+.10f} "
        f"baseline={record['baseline_vqe_energy']:+.10f} "
        f"excitations={record['excitations']} time={elapsed:.1f}s",
        rank,
    )

    if rank == 0:
        print(f"\n=== sharded supernet search complete (mode={args.mode}, world_size={world_size}) ===")
        print(f"global best energy: {record['fine_tuned_energy']:+.10f} Ha")

        out_dir = Path(__file__).parent
        report_path = (out_dir / args.output) if not Path(args.output).is_absolute() else Path(args.output)
        _write_report(
            record,
            report_path,
            mode=args.mode,
            n_qubits=hamiltonian.n_qubits,
            n_terms=n_terms,
            world_size=world_size,
            started_at=started_at,
            elapsed=elapsed,
        )
        print(f"text report saved to: {report_path}")

        qasm_path = out_dir / "BeH2_npu_cir.qasm"
        saved_qasm, qasm_message = save_qasm3_if_supported(result.best_circuit, qasm_path)
        if saved_qasm:
            print(f"OpenQASM 3.0 saved to: {qasm_message}")
        else:
            print(f"OpenQASM 3.0 skipped: {qasm_message}")

        py_path = out_dir / "BeH2_npu_cir.py"
        save_circuit_python(result.best_circuit, py_path, func_name="build_beh2_npu_qas_circuit")
        print(f"Python circuit saved to: {py_path}")


if __name__ == "__main__":
    main()
