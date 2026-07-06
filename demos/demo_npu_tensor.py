"""NPU 张量网络引擎远程验证脚本。

用法：
    python demos/demo_npu_tensor.py                    # 严格要求 NPU
    python demos/demo_npu_tensor.py --allow-cpu-fallback  # 无卡开发

核对项：
    1) NPU 上 TN 全态矢量 vs NPU 态矢量引擎一致；
    2) NPU 上 single_amplitude / partial_amplitude 正确；
    3) NPU 上 tn_expectation 可微（反传得到 .grad）；
并打印后端 name / 设备 / 是否复数（触发 real/imag 分解路径）。
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from aicir import Circuit, Hamiltonian, State, cnot, ry, rzz
from aicir.backends.npu_backend import NPUBackend, is_npu_available
from aicir.simulator import partial_amplitude, single_amplitude, tn_expectation, tn_statevector


def _demo_circuit(theta=0.4):
    return Circuit(ry(theta, 0), cnot(1, [0]), ry(0.9, 1), rzz(0.3, 0, 1), n_qubits=2)


def run_checks(allow_cpu_fallback: bool = False) -> bool:
    bk = NPUBackend(fallback_to_cpu=allow_cpu_fallback)
    print(f"[backend] {bk.name}")
    print(f"[device ] {bk._device}  npu_available={is_npu_available()}")

    c = _demo_circuit()
    ref = State.zero_state(c.n_qubits, bk).evolve(c.unitary(backend=bk)).to_numpy()

    # 1) 全态矢量一致
    tn = tn_statevector(c, backend=bk).to_numpy()
    ok_sv = np.allclose(tn, ref, atol=1e-3)
    print(f"[check ] tn_statevector vs statevector: {'OK' if ok_sv else 'FAIL'}")

    # 2) 单 / 部分振幅
    ok_single = np.isclose(single_amplitude(c, "11", backend=bk), ref[3], atol=1e-3)
    part = partial_amplitude(c, open_qubits=[1], backend=bk)
    ok_partial = np.allclose(part, [ref[0], ref[1]], atol=1e-3)
    print(f"[check ] single_amplitude: {'OK' if ok_single else 'FAIL'}")
    print(f"[check ] partial_amplitude: {'OK' if ok_partial else 'FAIL'}")

    # 3) 可微
    ok_grad = True
    try:
        import torch

        theta = torch.tensor(0.4, dtype=torch.float32, requires_grad=True)
        energy = tn_expectation(_demo_circuit(theta), Hamiltonian([("ZI", 1.0)]), backend=bk)
        energy.backward()
        grad = float(theta.grad)
        ok_grad = np.isclose(grad, -np.sin(0.4), atol=1e-2)
        print(f"[check ] tn_expectation grad={grad:.5f} vs -sin(0.4)={-np.sin(0.4):.5f}: "
              f"{'OK' if ok_grad else 'FAIL'}")
    except ImportError:
        print("[check ] torch 缺失，跳过可微检查")

    # 4) 切片收缩 parity（cotengra 装了才跑）
    ok_slice = True
    try:
        import cotengra  # noqa: F401
        sliced = tn_statevector(c, backend=bk, optimize="cotengra", memory_limit=4)
        ok_slice = np.allclose(sliced.to_numpy(), ref, atol=1e-3)
        print(f"[check ] sliced contraction: {'OK' if ok_slice else 'MISMATCH'}")
    except ImportError:
        print("[check ] sliced contraction: SKIP (cotengra 未安装)")

    passed = ok_sv and ok_single and ok_partial and ok_grad and ok_slice
    print(f"[result] {'ALL PASSED' if passed else 'FAILED'}")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    args = parser.parse_args()
    if not args.allow_cpu_fallback and not is_npu_available():
        print("NPU 不可用；如需在 CPU 上开发请加 --allow-cpu-fallback", file=sys.stderr)
        sys.exit(2)
    sys.exit(0 if run_checks(allow_cpu_fallback=args.allow_cpu_fallback) else 1)


if __name__ == "__main__":
    main()
