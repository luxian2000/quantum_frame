"""真实 Ascend NPU 上 MPS 引擎验收：NPU-vs-CPU 平价 + 参数移位梯度。

    python demos/demo_npu_mps.py                     # 严格要求 NPU
    python demos/demo_npu_mps.py --allow-cpu-fallback # 无 NPU 时回退 CPU（开发调试）

对齐 demos/demo_npu_tensor.py 风格：打印后端与各项核对，退出码 0/1 表示整体通过，
无 NPU 且未加 --allow-cpu-fallback 时退出码 2。run_checks() 可在 CPU-only 下作导入冒烟。
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _build_circuit():
    from aicir.core import Circuit
    from aicir import rx, rz, cnot

    rng = np.random.default_rng(0)
    c = Circuit(n_qubits=4)
    for q in range(4):
        c.append(rx(float(rng.uniform(0, np.pi)), q))
        c.append(rz(float(rng.uniform(0, np.pi)), q))
    for q in range(3):
        c.append(cnot(q + 1, [q]))
    return c


def run_checks(allow_cpu_fallback: bool = False) -> bool:
    from aicir.backends import NumpyBackend
    from aicir.backends.npu_backend import NPUBackend, is_npu_available
    from aicir.core.operators import Hamiltonian
    from aicir.primitives import MPSEstimator
    from aicir.simulator import mps_statevector, mps_expectation

    if is_npu_available():
        try:
            import torch

            torch.ones(1).npu()  # 预热，规避 ACL 初始化毒化
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] NPU 预热失败：{exc}")
    bk = NPUBackend(fallback_to_cpu=allow_cpu_fallback)
    print(f"[info] backend: {bk.name}")

    c = _build_circuit()
    cpu = NumpyBackend()
    passed = True

    sv_cpu = np.asarray(mps_statevector(c, backend=cpu).to_statevector().to_numpy()).reshape(-1)
    sv_npu = np.asarray(bk.to_numpy(mps_statevector(c, backend=bk).to_statevector().to_numpy())).reshape(-1)
    ok_sv = np.allclose(sv_npu, sv_cpu, atol=1e-4)
    print(f"[check] statevector NPU-vs-CPU: {'OK' if ok_sv else 'FAIL'}")
    passed = passed and ok_sv

    H = Hamiltonian([("ZZII", -1.0), ("IXXI", 0.5), ("IIZZ", 0.3)])
    e_cpu = float(np.real(complex(mps_expectation(c, H, backend=cpu))))
    e_npu = float(np.real(complex(bk.to_numpy(mps_expectation(c, H, backend=bk)))))
    ok_e = abs(e_npu - e_cpu) < 1e-3
    print(f"[check] expectation NPU-vs-CPU: {'OK' if ok_e else 'FAIL'} (npu={e_npu:.6f} cpu={e_cpu:.6f})")
    passed = passed and ok_e

    try:
        from aicir.core import Circuit
        from aicir import rx, cnot
        from aicir import Parameter

        theta = Parameter("theta")
        cc = Circuit(n_qubits=2)
        cc.append(rx(theta, 0))
        cc.append(cnot(1, [0]))
        grad = MPSEstimator(backend=bk).gradient(
            cc, Hamiltonian([("ZI", 1.0)]), parameter_values=[0.7]
        )
        ok_g = abs(float(grad.gradient[0]) - (-np.sin(0.7))) < 1e-3
        print(f"[check] parameter-shift gradient: {'OK' if ok_g else 'FAIL'}")
        passed = passed and ok_g
    except Exception as exc:  # noqa: BLE001
        print(f"[check] parameter-shift gradient: FAIL ({exc})")
        passed = False

    print(f"[result] {'ALL PASSED' if passed else 'FAILED'}")
    return bool(passed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    args = parser.parse_args()
    from aicir.backends.npu_backend import is_npu_available

    if not args.allow_cpu_fallback and not is_npu_available():
        print("NPU 不可用；如需在 CPU 上开发请加 --allow-cpu-fallback", file=sys.stderr)
        sys.exit(2)
    sys.exit(0 if run_checks(allow_cpu_fallback=args.allow_cpu_fallback) else 1)


if __name__ == "__main__":
    main()
