"""Ascend NPU 上 MPS 可行性探针：只测未知项（SVD / QR / 复数分解 / 反传），不改任何源码。

MPS 引擎需要什么、哪些已知能跑、哪些是未知：

- 已知能跑（NPUBackend 已用 real/imag 分解实现，autograd 安全）：matmul / tensordot /
  transpose / reshape / conj / add / take —— 单/双比特门就地作用、transfer 收缩全靠这些。
- 未知（本探针要回答）：
  1. NPU 上 `torch.linalg.svd` 前向能否跑？real float32 / complex64 各如何？
  2. SVD 反传（autograd）能否跑？梯度是否有限？—— 这是 MPS 截断可微的关键。
  3. `torch.linalg.qr` 前向/反传能否跑？—— QR 可替代「正交中心搬移」（不截断那步），
     若 SVD 不行但 QR 行，仍可做「NPU 上 QR 演化 + CPU 截断」的折中。
  4. 复数 SVD 的 real-embedding 变通（2m×2n 实块矩阵）前向/反传能否跑、数值是否正确？
  5. MPS 截断真实算子：complex theta 矩阵 (Dl*2, 2*Dr) 约化 SVD → 截断 top-k → 反传。

初始化顺序很重要：`torch_npu` 在模块顶层紧随 `torch` 导入，且第一次 NPU 触碰用一个
平凡 `.npu()` 预热（与已验证可跑的最小用例一致），避免某些 CANN 版本上 `get_device_name`
先行触发 `AclSetCompileopt(ACL_PRECISION_MODE)` 失败而毒化整个进程的 NPU 初始化。

用法（在 Ascend 机器、仓库根目录）：

    PYTHONPATH=. python demos/probe_npu_svd.py                 # 严格要求 NPU
    PYTHONPATH=. python demos/probe_npu_svd.py --allow-cpu     # 无 NPU 时也能干跑一遍看逻辑

把最后 `SUMMARY` 整段贴回即可。每项都 try/except 隔离，一项失败不影响其余。
"""

from __future__ import annotations

import argparse
import platform
import sys

import torch

try:  # 顶层紧随 torch 导入（与已验证可跑的最小用例一致）
    import torch_npu  # noqa: F401

    _TORCH_NPU_VER = getattr(torch_npu, "__version__", "unknown")
    _TORCH_NPU_ERR = None
except Exception as e:  # noqa: BLE001
    _TORCH_NPU_VER = None
    _TORCH_NPU_ERR = f"{type(e).__name__}: {e}"

RESULTS = []  # (name, ok, detail)


def record(name, ok, detail=""):
    RESULTS.append((name, bool(ok), str(detail)))
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}: {detail}")


def short_exc(e):
    return f"{type(e).__name__}: {e}".replace("\n", " ")[:300]


def probe(name, fn):
    try:
        detail = fn()
        record(name, True, detail if detail is not None else "ok")
    except Exception as e:  # noqa: BLE001 —— 探针要吞掉一切，逐项报告
        record(name, False, short_exc(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-cpu", action="store_true", help="无 NPU 时允许在 CPU 上干跑")
    args = ap.parse_args()

    import numpy as np

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_npu": _TORCH_NPU_VER or f"IMPORT_FAIL {_TORCH_NPU_ERR}",
    }

    npu_ok = hasattr(torch, "npu") and getattr(torch.npu, "is_available", lambda: False)()
    env["npu_available"] = npu_ok

    if not npu_ok and not args.allow_cpu:
        print("NPU 不可用；如需在 CPU 上干跑请加 --allow-cpu", file=sys.stderr)
        for k, v in env.items():
            print(f"  {k}: {v}")
        sys.exit(2)

    dev = torch.device("npu:0") if npu_ok else torch.device("cpu")

    # ---- 预热：第一次 NPU 触碰用平凡 add（已验证可跑的路径），先把设备初始化好 ----
    if npu_ok:
        try:
            w = torch.ones(2, 2).npu()
            _ = (w + w).cpu()
            env["warmup"] = "ok"
        except Exception as e:  # noqa: BLE001
            env["warmup"] = f"FAIL {short_exc(e)}"
        # 设备信息放到预热之后再取（避免先行触发坏初始化）
        try:
            env["npu_count"] = torch.npu.device_count()
            env["npu_name"] = torch.npu.get_device_name(0)
        except Exception as e:  # noqa: BLE001
            env["npu_info_err"] = short_exc(e)

    cdtype = torch.complex64
    print(f"\n=== 探针设备: {dev} ===\n")

    # ---- NPUBackend 冒烟：确认 MPS 依赖的 real/imag 复数原语在此机可跑 ----
    def backend_smoke():
        from aicir.backends.npu_backend import NPUBackend

        bk = NPUBackend(fallback_to_cpu=not npu_ok)
        a = bk.cast(np.array([[1 + 1j, 2], [0, 1j]], dtype=np.complex64))
        b = bk.cast(np.array([[1, 0], [1j, 1]], dtype=np.complex64))
        mm = bk.matmul(a, b)
        td = bk.tensordot(bk.reshape(a, (2, 2)), bk.reshape(b, (2, 2)), ([1], [0]))
        _ = bk.conj(a)
        _ = bk.add(a, b)
        return f"matmul&tensordot&conj&add ok, mm.dtype={mm.dtype}, td.dtype={td.dtype}"

    probe("00_npu_backend_complex_primitives", backend_smoke)

    def cpu_singular_ref(x):
        return torch.linalg.svdvals(x.detach().cpu().to(torch.complex128 if x.is_complex() else torch.float64))

    def svd_forward(dtype, shape):
        def fn():
            x = torch.randn(*shape, dtype=dtype, device=dev)
            u, s, vh = torch.linalg.svd(x, full_matrices=False)
            ref = cpu_singular_ref(x)
            diff = float((s.detach().cpu().to(ref.dtype) - ref).abs().max())
            recon = u @ torch.diag(s.to(u.dtype)) @ vh
            rerr = float((recon.detach().cpu() - x.detach().cpu()).abs().max())
            return f"shape={tuple(shape)} sv_maxdiff_vs_cpu={diff:.2e} recon_maxerr={rerr:.2e} S.dtype={s.dtype}"

        return fn

    def svd_backward(dtype, shape):
        def fn():
            x = torch.randn(*shape, dtype=dtype, device=dev, requires_grad=True)
            u, s, vh = torch.linalg.svd(x, full_matrices=False)
            s.sum().backward()  # 奇异值实数，可反传
            g = x.grad
            gok = g is not None and bool(torch.isfinite(g.abs() if g.is_complex() else g).all())
            gnorm = float(g.abs().sum()) if g is not None else -1.0
            return f"shape={tuple(shape)} grad_finite={gok} grad_absnorm={gnorm:.3e}"

        return fn

    def qr_forward(dtype, shape):
        def fn():
            x = torch.randn(*shape, dtype=dtype, device=dev)
            q, r = torch.linalg.qr(x, mode="reduced")
            rerr = float(((q @ r).detach().cpu() - x.detach().cpu()).abs().max())
            return f"shape={tuple(shape)} recon_maxerr={rerr:.2e} Q.dtype={q.dtype}"

        return fn

    def qr_backward(dtype, shape):
        def fn():
            x = torch.randn(*shape, dtype=dtype, device=dev, requires_grad=True)
            q, r = torch.linalg.qr(x, mode="reduced")
            r.abs().sum().backward()
            g = x.grad
            gok = g is not None and bool(torch.isfinite(g.abs()).all())
            return f"shape={tuple(shape)} grad_finite={gok}"

        return fn

    shapes = [(4, 4), (16, 16), (32, 8), (8, 32)]
    for sh in shapes:
        probe(f"01_svd_fwd_real_{sh[0]}x{sh[1]}", svd_forward(torch.float32, sh))
    for sh in shapes:
        probe(f"02_svd_bwd_real_{sh[0]}x{sh[1]}", svd_backward(torch.float32, sh))
    for sh in shapes:
        probe(f"03_svd_fwd_cplx_{sh[0]}x{sh[1]}", svd_forward(cdtype, sh))
    for sh in shapes:
        probe(f"04_svd_bwd_cplx_{sh[0]}x{sh[1]}", svd_backward(cdtype, sh))
    probe("05_qr_fwd_real_16x8", qr_forward(torch.float32, (16, 8)))
    probe("06_qr_bwd_real_16x8", qr_backward(torch.float32, (16, 8)))
    probe("07_qr_fwd_cplx_16x8", qr_forward(cdtype, (16, 8)))
    probe("08_qr_bwd_cplx_16x8", qr_backward(cdtype, (16, 8)))

    def real_embed_svd():
        m, n = 12, 8
        a = torch.randn(m, n, dtype=cdtype, device=dev, requires_grad=True)
        ar, ai = torch.real(a), torch.imag(a)
        big = torch.cat([torch.cat([ar, -ai], dim=1), torch.cat([ai, ar], dim=1)], dim=0)  # (2m,2n) 实
        u, s, vh = torch.linalg.svd(big, full_matrices=False)
        ref = cpu_singular_ref(a)
        s_sorted = torch.sort(s.detach().cpu(), descending=True).values
        s_dedup = s_sorted[0 : 2 * n : 2][:n].to(ref.dtype)
        diff = float((s_dedup - ref[:n]).abs().max())
        s.sum().backward()
        gok = a.grad is not None and bool(torch.isfinite(a.grad.abs()).all())
        return f"sv_maxdiff_vs_cplx={diff:.2e} grad_finite={gok}"

    probe("09_real_embed_complex_svd_fwd_bwd", real_embed_svd)

    def mps_truncation_op():
        Dl, Dr, k = 4, 4, 2
        theta = torch.randn(Dl * 2, 2 * Dr, dtype=cdtype, device=dev, requires_grad=True)
        u, s, vh = torch.linalg.svd(theta, full_matrices=False)
        u_k, s_k, vh_k = u[:, :k], s[:k], vh[:k, :]
        vh_scaled = vh_k * s_k.to(vh_k.dtype).reshape(k, 1)
        a_left = u_k.reshape(Dl, 2, k)
        a_right = vh_scaled.reshape(k, 2, Dr)
        (a_left.abs().sum() + a_right.abs().sum()).backward()
        gok = theta.grad is not None and bool(torch.isfinite(theta.grad.abs()).all())
        return f"k={k} slice+scale+reshape ok; trunc_grad_finite={gok}"

    probe("10_mps_truncation_op_fwd_bwd", mps_truncation_op)

    print("\n" + "=" * 60)
    print("SUMMARY (把这一整段贴回)")
    print("=" * 60)
    print("ENV:")
    for k, v in env.items():
        print(f"  {k}: {v}")
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\nPROBES: {npass}/{len(RESULTS)} passed")
    for name, ok, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print("=" * 60)


if __name__ == "__main__":
    main()
