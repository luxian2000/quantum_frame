"""在 8 张 Ascend NPU 上用 supernet QAS 搜索 CH4 的 VQE 线路（内搜索分片）。

方法选择：与 ``demos/CH4/CH4.py`` 一致，使用 ``supernet`` (``supernet_qas``)。
它是面向任意哈密顿量做 VQE ansatz 结构搜索的唯一 QAS 方法；CRLQAS / PPR-DQL /
PPO-RB 面向小规模目标态制备，不适合 18 量子比特分子哈密顿量。

所有 rank 使用**相同随机种子**，``supernet_qas`` 在内部将训练/排名/微调阶段切分到
多 NPU 上并行执行（内搜索分片）。``--mode`` 控制分片策略，两种模式区别如下：

- ``safe``（默认）：**与单卡数值完全等价（确定性可复现）**。每步所有 rank 用相同
  种子采样**同一架构**，只把 *候选 supernet 的选择评估* 切分到各 rank（``_sharded_select``）；
  **仅 rank 0 计算梯度并执行优化器步**，随后 ``broadcast_parameters(src=0)`` 把权重广播给
  全部 rank。优点：与单卡逐位等价、结果可复现；缺点：rank 1..N 在 broadcast 屏障空等
  rank 0，负载不均衡（BeH2 上即因此触发 HCCL 屏障超时，见 ``aicir/backends/README.md`` §5.5）。

- ``aggressive``：**数据并行，约 world_size 倍吞吐**。每步各 rank 采样 world_size 个
  **不同架构**，rank ``r`` 负责 ``arch[r]``，各自前向 + ``backward``；梯度经 ``all_reduce_mean``
  汇总，所有被选中 supernet 的优化器一起步进。优点：每步探索约 N 倍架构、吃满多卡；
  缺点：各 rank 动态步不同，**数值不再与单卡等价**，搜索轨迹随卡数 N 变化。

  | 维度 | safe | aggressive |
  | --- | --- | --- |
  | 数值 | 与单卡逐位等价（确定） | 数据并行，随 N 变化 |
  | 分片对象 | 仅候选选择评估 | 整架构前向 + 梯度 |
  | 吞吐 | ~1×（rank 0 瓶颈） | ~world_size× |
  | 适用 | 复现/正确性基线 | 提速、加大探索 |

18 比特 CH4 Hamiltonian 含 6892 个 Pauli 项。NPU 版本默认使用参数移位梯度，
避免 autograd 为每个 Pauli 项保留大反向图；如需对比可传 ``--gradient ad``。

每个 rank 通过 ``LOCAL_RANK`` 绑定一张 NPU，与 ``demos/demo_npu.py`` /
``aicir/qas/README.md`` 的 NPU 约定一致。搜索完成后返回的 ``SupernetResult``
在所有 rank 上均为全局最优，由 rank 0 落盘：

- ``output.txt``         : 分片搜索结果的文本报告（模式、卡数、能量、线路指标）
- ``CH4_npu_cir.qasm`` / ``CH4_npu_cir.py`` : 全局最优线路

运行方式（8 卡）：

    torchrun --nproc_per_node=8 demos/CH4/CH4_npu.py [--mode safe|aggressive]

无 NPU 的本地冒烟测试（单进程、允许 CPU 回退，分片路径未激活）：

    python demos/CH4/CH4_npu.py --allow-cpu-fallback
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
from aicir.core.io.qasm import save_circuit_qasm3
from aicir.measure import hamiltonian_pauli_terms
from aicir.qas import supernet_qas

from demos.CH4.CH4 import (
    CH4_BASIS,
    CH4_CHARGE,
    CH4_QUBIT_MAPPER,
    CH4_SPIN,
    ch4_vqe_qas_kwargs,
    build_ch4_hamiltonian,
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
        description="CH4 18-qubit supernet VQE QAS on 8 Ascend NPU (task-parallel).",
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
        "CH4 supernet VQE QAS on Ascend NPU (in-search sharding, single run)",
        "=" * 88,
        f"started at            : {started_at}",
        f"finished at           : {datetime.now().isoformat(timespec='seconds')}",
        f"platform              : {platform.platform()}",
        f"mode                  : {mode}",
        f"NPUs (world_size)     : {world_size}",
        f"seed                  : {record['seed']}",
        f"device                : {record['device']}",
        f"qubits / terms        : {n_qubits} / {n_terms}",
        f"basis/charge/spin/map : {CH4_BASIS}/{CH4_CHARGE}/{CH4_SPIN}/{CH4_QUBIT_MAPPER}",
        f"total wall-clock time : {elapsed:.1f} s",
        "",
        "global-best result (identical on all ranks):",
        "-" * 88,
        f"  fine-tuned energy  : {record['fine_tuned_energy']:+.10f} Ha",
        f"  baseline VQE       : {record['baseline_vqe_energy']:+.10f} Ha",
        f"  CNOT / 2-qubit     : {record['cnot']} / {record['two_qubit']}",
        "  circuit files      : CH4_npu_cir.qasm, CH4_npu_cir.py",
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

    _log("=== CH4 supernet VQE QAS on Ascend NPU ===", rank)
    _log(f"backend={backend.name} device={device}", rank)
    _log(f"world_size={world_size} distributed={backend.distributed_initialized}", rank)

    hamiltonian = build_ch4_hamiltonian()
    n_terms = len(hamiltonian_pauli_terms(hamiltonian))
    if rank == 0:
        print(
            f"CH4 active-space Hamiltonian: basis={CH4_BASIS} charge={CH4_CHARGE} "
            f"spin={CH4_SPIN} mapper={CH4_QUBIT_MAPPER}"
        )
        print(f"Qubits: {hamiltonian.n_qubits}  Terms: {n_terms}")

    # 所有 rank 使用相同 seed；supernet_qas 内部将训练/排名/微调切分到各 NPU（内搜索分片）。
    kwargs = ch4_vqe_qas_kwargs()
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
        "seconds": elapsed,
    }
    _log(
        f"fine_tuned_energy={record['fine_tuned_energy']:+.10f} "
        f"baseline={record['baseline_vqe_energy']:+.10f} "
        f"cnot={record['cnot']} time={elapsed:.1f}s",
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

        qasm_path = out_dir / "CH4_npu_cir.qasm"
        save_circuit_qasm3(result.best_circuit, qasm_path)
        print(f"OpenQASM 3.0 saved to: {qasm_path}")

        py_path = out_dir / "CH4_npu_cir.py"
        save_circuit_python(result.best_circuit, py_path, func_name="build_ch4_npu_qas_circuit")
        print(f"Python circuit saved to: {py_path}")


if __name__ == "__main__":
    main()
