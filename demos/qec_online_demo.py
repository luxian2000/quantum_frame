"""aicir.qec 在线实时解码最小演示。

运行：PYTHONPATH=. python demos/qec_online_demo.py
"""

from aicir.qec import (LookupDecoder, PauliErrorModel, TimingModel, get_code,
                       run, verify_schedule)
from aicir.qec.schedules import BareAncillaSchedule


def main() -> None:
    code = get_code("surface", d=3)
    print(f"码：{code}  距离 = {code.distance()}")

    # 提取调度的结构性检验：无噪声下每个 detector 必须恒为 0
    verify_schedule(code, BareAncillaSchedule(), rounds=3)
    print("detector 确定性检验通过")

    timing = TimingModel(round_duration=1e-6, cost_to_seconds=lambda c: c * 2e-7)
    result = run(
        code,
        errors=PauliErrorModel(p_data=0.01, p_measure=0.01, channel="depolarizing"),
        decoder=LookupDecoder(code),
        rounds=5, shots=200, seed=0, timing=timing,
    )
    print(result.summary())

    if result.failure_records:
        rec = result.failure_records[0]
        print(f"\n首个失败 shot #{rec.shot}（判定 {rec.verdict}）注入的错误：")
        for e in rec.injected_errors[:8]:
            print(f"  轮 {e.round_index}  比特 {e.qubit}  {e.pauli}  ({e.source})")


if __name__ == "__main__":
    main()
