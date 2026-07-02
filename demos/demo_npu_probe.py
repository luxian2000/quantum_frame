"""NPU 硬件能力探测脚本：打印能力表并构建 Target。

用法：
    python demos/demo_npu_probe.py                       # 严格 NPU
    python demos/demo_npu_probe.py --allow-cpu-fallback  # 允许 CPU 回退
    python demos/demo_npu_probe.py --refresh             # 忽略缓存重探
"""

from __future__ import annotations

import argparse

from aicir.backends.npu_probe import probe_npu, target_from_npu


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="探测 Ascend NPU 硬件能力")
    parser.add_argument("--allow-cpu-fallback", action="store_true", help="无 NPU 时在 CPU 上探测")
    parser.add_argument("--refresh", action="store_true", help="忽略磁盘缓存，强制重探")
    args = parser.parse_args(argv)

    caps = probe_npu(allow_cpu_fallback=args.allow_cpu_fallback, refresh=args.refresh)

    print("== NpuCapabilities ==")
    for key, value in caps.to_dict().items():
        print(f"  {key}: {value}")

    if caps.max_qubits is not None:
        target = target_from_npu(caps)
        print("== Target ==")
        print(f"  {target}")
    else:
        print("== Target ==")
        print("  跳过：max_qubits 为 None（无法派生 n_qubits）")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
