# tests/test_supernet_sharding_dist.py
# CPU-gloo distributed reproducibility test for the NPU sharding primitives.
#
# How this works:
# - `shard_context` activates when backend is NPUBackend AND dist is initialized
#   AND world_size > 1.
# - `NPUBackend(device="npu:0")` on a CPU-only machine transparently falls back
#   to CPU but remains an NPUBackend instance, so the sharding predicate fires.
# - Two spawned gloo workers (world_size=2) therefore exercise the sharded code
#   paths on plain CPU, validating correctness without Ascend hardware.

import os
import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist
import torch.multiprocessing as mp

from aicir.operators import Hamiltonian
from aicir.qas.algorithms.supernet import supernet_qas


HAM_TERMS = [("ZZ", -1.0), ("XI", 0.2), ("IX", 0.2)]
COMMON = dict(
    layers=1,
    supernet_num=2,
    supernet_steps=4,
    finetune_steps=3,
    ranking_num=4,
    seed=3,   # seed=5 触发库边界：最优架构恰好全是零参数门（i/h），导致空参数列表崩溃
    device="npu:0",
)

# 用不同端口隔离两个测试，避免 TIME_WAIT 冲突
_PORT_SAFE = "29555"
_PORT_AGGRESSIVE = "29556"


def _single_card_energy() -> float:
    """单卡（无 process group）运行 safe 模式，返回 fine_tuned_energy。"""
    ham = Hamiltonian(n_qubits=2, terms=HAM_TERMS)
    res = supernet_qas(ham, mode="safe", **COMMON)
    return float(res.final_metrics["fine_tuned_energy"])


def _worker(rank: int, world_size: int, mode: str, port: str, return_dict) -> None:
    """每个 spawn 子进程的入口。rank 由 mp.spawn 自动注入为第一个参数。"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        ham = Hamiltonian(n_qubits=2, terms=HAM_TERMS)
        res = supernet_qas(ham, mode=mode, **COMMON)
        if rank == 0:
            return_dict["energy"] = float(res.final_metrics["fine_tuned_energy"])
    finally:
        dist.destroy_process_group()


def _run(mode: str, port: str) -> float:
    """在 2 个 gloo worker 里运行 supernet_qas，返回 rank-0 的能量。"""
    mgr = mp.Manager()
    return_dict = mgr.dict()
    mp.spawn(_worker, args=(2, mode, port, return_dict), nprocs=2, join=True)
    return return_dict["energy"]


def test_safe_mode_matches_single_card():
    """safe 模式：分布式结果必须在 abs=1e-4 以内复现单卡结果。"""
    single = _single_card_energy()
    sharded = _run("safe", _PORT_SAFE)
    assert sharded == pytest.approx(single, abs=1e-4), (
        f"safe 模式分布式能量 {sharded} 与单卡能量 {single} 差值 "
        f"{abs(sharded - single):.2e} 超出 1e-4"
    )


def test_aggressive_mode_runs_and_is_finite():
    """aggressive 模式：分布式运行正常完成且返回有限能量。"""
    energy = _run("aggressive", _PORT_AGGRESSIVE)
    assert energy == energy, "aggressive 模式返回了 NaN"
    assert energy < 1.0, f"aggressive 模式能量 {energy} 超出平凡上界 1.0"
