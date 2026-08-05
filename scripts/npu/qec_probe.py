#!/usr/bin/env python3
"""真机 NPU 上的 aicir.qec 在线纠错平台探针。

`tests/qec` 全程用 NumpyBackend 跑，不触 NPU 设备；本探针把同一套 QEC 流程
放到 `NPUBackend` 上实跑，补设备缺口。

`aicir.qec` 自身只依赖 numpy（包内不 import torch），但它把 `backend` 一路透传给
`State` / `run_trajectory`，因此**后端相关的风险全部落在 measure 热路径上**，而不在
qec 的代数层：
  - 线路内 `measure(creg=...)` 的逐比特 Z 投影与 `reset`（每轮 syndrome 提取都走）
  - 轮间态串联（上一轮 `.pre` 作为下一轮 `init_state`）
  - active 模式下把修正 Pauli 作为真实门施加
QEC 路径**不含任何梯度**，故 CLAUDE.md 记录的 complex64 反向/梯度累加缺失与此无关；
真正要在真机上确认的是上述投影/reset/门施加在 complex64 下能否跑通。

用法：
    scripts/npu/qec.sh                        # 严格 NPU
    scripts/npu/qec.sh --allow-cpu-fallback   # 仅本地开发用，不作为真机证据
    scripts/npu/qec.sh --include-surface      # 追加 surface d=3（17 比特，较慢）
    scripts/npu/qec.sh --rounds 4 --shots 8   # 调整轮数/采样数
"""
from __future__ import annotations

import argparse
import traceback

import numpy as np

from aicir import NPUBackend
from aicir.backends.npu_backend import is_npu_available
from aicir.qec.code import StabilizerCode
from aicir.qec.codes import get_code
from aicir.qec.decoders.lookup import LookupDecoder
from aicir.qec.errors import PauliErrorModel
from aicir.qec.runner import TimingModel, run
from aicir.qec.schedules import BareAncillaSchedule, verify_schedule


def _backend(allow_cpu_fallback: bool) -> NPUBackend:
    if not allow_cpu_fallback and not is_npu_available():
        raise SystemExit("NPU is unavailable. Use --allow-cpu-fallback only for local script validation.")
    backend = NPUBackend.from_distributed_env(fallback_to_cpu=allow_cpu_fallback)
    if not allow_cpu_fallback and getattr(backend._device, "type", None) != "npu":
        raise AssertionError(f"strict NPU probe resolved device={backend._device!r}, expected npu")
    print(f"[backend] {backend.name}")
    print(f"[runtime] {backend.runtime_context}")
    return backend


# ---------- case 1：最小设备路径 ----------

def case_round_circuit_on_device(backend, cfg):
    """单轮 syndrome 提取在设备上跑通，且经典寄存器读得回 m 位。

    这是最小的设备路径：H → 受控 P → H → measure(creg) → reset，
    覆盖 complex64 门施加 + 逐比特 Z 投影 + reset 三件事。
    """
    from aicir.core.state import State
    from aicir.measure.trajectory import run_trajectory

    code = get_code("repetition", d=3, basis="Z")
    sched = BareAncillaSchedule()
    n_total = code.n + code.m
    rng = np.random.default_rng(0)

    state = run_trajectory(sched.build_encode(code, "0"), State.zero_state(n_total, backend),
                           backend, tm=False, measure_qubits=None, snap_ops=set(), rng=rng).pre
    rc = sched.build_round(code, 0)
    res = run_trajectory(rc.circuit, state, backend, tm=False, measure_qubits=None,
                         snap_ops=set(), rng=rng)
    bits = list(res.classical.get(rc.creg_name, []))
    if len(bits) < code.m:
        raise AssertionError(f"经典寄存器只读到 {len(bits)} 位，应有 {code.m} 位")
    if any(b not in (0, 1) for b in bits):
        raise AssertionError(f"综合征位取值非 0/1: {bits}")


# ---------- case 2：detector 确定性（结构性检验） ----------

def case_detector_determinism(backend, cfg):
    """无噪声下每个 detector 恒为 0 —— 提取调度最有力的结构性检验。

    在真机上跑通它，等于确认设备侧的投影/reset 语义与 numpy 一致：
    只要 ancilla 复用或投影塌缩在 NPU 上有任何偏差，这里立刻炸。
    """
    for name, kwargs in cfg["codes"]:
        code = get_code(name, **kwargs)
        verify_schedule(code, BareAncillaSchedule(), cfg["rounds"],
                        backend=backend, shots=cfg["verify_shots"])


# ---------- case 3：无噪声端到端 ----------

def case_noiseless_run_is_clean(backend, cfg):
    """无噪声运行的逻辑错误率必须恰好为 0。"""
    for name, kwargs in cfg["codes"]:
        code = get_code(name, **kwargs)
        dec = _decoder_for(code, name, kwargs)
        result = run(code, errors=PauliErrorModel(), decoder=dec,
                     rounds=cfg["rounds"], shots=cfg["shots"], seed=0, backend=backend)
        if result.logical_error_rate != 0.0:
            raise AssertionError(
                f"{code.name} 无噪声逻辑错误率 {result.logical_error_rate} != 0；"
                f"判定分布 {result.verdict_counts}"
            )


# ---------- case 4：含噪端到端 ----------

def case_noisy_run_produces_records(backend, cfg):
    """含噪运行跑通，且记录结构形状正确、判定合法。"""
    code = get_code("steane")
    result = run(code, errors=PauliErrorModel(p_data=0.05, p_measure=0.02,
                                              channel="depolarizing"),
                 decoder=LookupDecoder(code), rounds=cfg["rounds"], shots=cfg["shots"],
                 seed=3, backend=backend)
    if not result.records:
        raise AssertionError("未保留任何 shot 记录")
    rec = result.records[0]
    if rec.raw_syndromes.shape != (cfg["rounds"], code.m):
        raise AssertionError(f"raw_syndromes 形状 {rec.raw_syndromes.shape}")
    if rec.detection_events.shape != (cfg["rounds"], code.m):
        raise AssertionError(f"detection_events 形状 {rec.detection_events.shape}")
    if sum(result.verdict_counts.values()) != cfg["shots"]:
        raise AssertionError("判定计数未覆盖全部 shot")
    if any(e.round_index < 1 for e in rec.injected_errors):
        raise AssertionError("轮 0 是制备轮，不应注入错误")


# ---------- case 5：frame / active 一致（最细的一处物理） ----------

def case_frame_active_agree(backend, cfg):
    """两种修正模式必须交出逐字节相同的 detection event 流与相同判定。

    active 模式会把修正作为真实门施加，因而额外覆盖了设备侧的门施加路径；
    两模式一致则说明 `_applied_syndrome_delta` 的相邻差扣除与末端读出回退
    在 NPU 上与 numpy 行为一致。轮数取 >=4，否则该处 bug 不会暴露。
    """
    rounds = max(4, cfg["rounds"])
    code = get_code("steane")
    common = dict(errors=PauliErrorModel(p_data=0.08, channel="depolarizing"),
                  rounds=rounds, shots=cfg["shots"], seed=17, backend=backend)
    a = run(code, decoder=LookupDecoder(code), correction_mode="frame", **common)
    b = run(code, decoder=LookupDecoder(code), correction_mode="active", **common)
    if len(a.records) != len(b.records):
        raise AssertionError("两模式保留的记录数不一致")
    for ra, rb in zip(a.records, b.records):
        if not np.array_equal(ra.detection_events, rb.detection_events):
            raise AssertionError(f"shot {ra.shot} 的 detection event 流不一致")
    if a.verdict_counts != b.verdict_counts:
        raise AssertionError(f"判定分布不一致：frame={a.verdict_counts} active={b.verdict_counts}")


# ---------- case 6：k=2 多逻辑比特 ----------

def case_multi_logical_code(backend, cfg):
    """[[4,2,2]] 检测码（k=2）端到端，覆盖 k>1 才走到的分支。"""
    code = StabilizerCode.from_paulis(
        ["XXXX", "ZZZZ"],
        logical_x=["XXII", "XIXI"], logical_z=["ZIZI", "ZZII"],
        name="detection_422", coords={q: (0, q) for q in range(4)},
    )
    code.validate()

    class _Passive:
        """只做记账、不产出修正的最简在线解码器（frame 模式）。"""
        name, window, commit_lag = "passive", 1, 0

        def reset(self, layout):
            self._c = -1

        def update(self, round_index, events):
            from aicir.qec.decoders import DecodeStep
            self._c = int(round_index)
            return DecodeStep(committed_through=self._c, cost=1.0)

        def flush(self):
            from aicir.qec.decoders import DecodeStep
            return DecodeStep(committed_through=self._c, cost=0.0)

        def cost_of(self, round_index, events):
            return 1.0

    result = run(code, errors=PauliErrorModel(p_data=0.05), decoder=_Passive(),
                 rounds=cfg["rounds"], shots=cfg["shots"], seed=41, backend=backend)
    if sum(result.verdict_counts.values()) != cfg["shots"]:
        raise AssertionError("k=2 码的判定计数未覆盖全部 shot")
    if result.records[0].detection_events.shape != (cfg["rounds"], code.m):
        raise AssertionError("k=2 码的 detection_events 形状异常")


# ---------- case 7：实时模型 ----------

def case_timing_model(backend, cfg):
    """实时预算模型在设备路径下照常产出 backlog / 提交延迟。"""
    code = get_code("steane")
    timing = TimingModel(round_duration=1e-6, cost_to_seconds=lambda c: c * 4e-6)
    result = run(code, errors=PauliErrorModel(p_data=0.05), decoder=LookupDecoder(code),
                 rounds=cfg["rounds"], shots=cfg["shots"], seed=13, timing=timing,
                 backend=backend)
    if result.max_backlog is None or result.budget_violations is None:
        raise AssertionError("给了 TimingModel 却没填 timing 聚合量")
    rec = result.records[0]
    if rec.backlog.shape != (cfg["rounds"],) or rec.commit_latency.shape != (cfg["rounds"],):
        raise AssertionError("逐轮 backlog / commit_latency 形状异常")
    # 建模解码时长 4e-6s 远超 1e-6s 的轮时长 → 每轮必超预算、backlog 单调增
    if result.budget_violations != cfg["rounds"] * len(result.records):
        raise AssertionError(f"超预算轮数 {result.budget_violations} 与预期不符")
    if not np.all(np.diff(rec.backlog) > 0):
        raise AssertionError("解码持续慢于轮时长时 backlog 应单调增长")


def _decoder_for(code, name, kwargs):
    """重复码的 distance() 是 1（对未受保护基无防护），必须显式给 t 与 error_basis。"""
    if name == "repetition":
        basis = "X" if kwargs.get("basis", "Z") == "Z" else "Z"
        return LookupDecoder(code, t=(int(kwargs.get("d", 3)) - 1) // 2, error_basis=basis)
    return LookupDecoder(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--allow-cpu-fallback", action="store_true",
                        help="允许回落 CPU；仅本地开发用，不可作为真机证据")
    parser.add_argument("--rounds", type=int, default=3,
                        help="每次运行的轮数（轮 0 是制备轮，纠错检验需 >=2）")
    parser.add_argument("--shots", type=int, default=4, help="每个 case 的采样数")
    parser.add_argument("--verify-shots", type=int, default=2,
                        help="verify_schedule 的 detector 确定性采样数")
    parser.add_argument("--include-surface", action="store_true",
                        help="追加 surface d=3（9+8=17 比特，明显更慢）")
    args = parser.parse_args()

    if args.rounds < 2:
        raise SystemExit("--rounds 必须 >=2：轮 0 是投影式制备轮，不注入错误")

    backend = _backend(args.allow_cpu_fallback)

    codes = [("repetition", {"d": 3, "basis": "Z"}), ("steane", {})]
    if args.include_surface:
        codes.append(("surface", {"d": 3}))

    cfg = {
        "codes": codes,
        "rounds": args.rounds,
        "shots": args.shots,
        "verify_shots": args.verify_shots,
    }
    print(f"[config] rounds={args.rounds} shots={args.shots} "
          f"codes={[n for n, _ in codes]}")

    cases = [
        ("round_circuit_on_device", case_round_circuit_on_device),
        ("detector_determinism", case_detector_determinism),
        ("noiseless_run_is_clean", case_noiseless_run_is_clean),
        ("noisy_run_produces_records", case_noisy_run_produces_records),
        ("frame_active_agree", case_frame_active_agree),
        ("multi_logical_code", case_multi_logical_code),
        ("timing_model", case_timing_model),
    ]

    passed, failures = 0, []
    for name, fn in cases:
        try:
            fn(backend, cfg)
            print(f"[PASS] {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            first = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
            print(f"[FAIL] {name}: {first}")
            failures.append((name, traceback.format_exc()))
    print(f"qec_probe: {passed}/{len(cases)} cases passed")
    if failures:
        print("\n===== 失败详情 =====")
        for name, tb in failures:
            print(f"\n--- {name} ---\n{tb}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
