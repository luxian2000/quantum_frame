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

用法（在 Ascend 机器、仓库根目录）：

    PYTHONPATH=. python demos/probe_npu_svd.py                 # 严格要求 NPU
    PYTHONPATH=. python demos/probe_npu_svd.py --allow-cpu     # 无 NPU 时也能干跑一遍看逻辑

把最后 `SUMMARY` 整段贴回即可。每项都 try/except 隔离，一项失败不影响其余。
"""

from __future__ import annotations

import argparse
import platform
import sys
import traceback

RESULTS = []  # (name, ok, detail)


def record(name, ok, detail=""):
    RESULTS.append((name, bool(ok), str(detail)))
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}: {detail}")


def short_exc(e):
    s = f"{type(e).__name__}: {e}"
    return s.replace("\n", " ")[:300]


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
    import torch

    # ---- 环境信息 ----
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    try:
        import torch_npu  # noqa: F401

        env["torch_npu"] = getattr(torch_npu, "__version__", "unknown")
    except Exception as e:  # noqa: BLE001
        env["torch_npu"] = f"IMPORT_FAIL {short_exc(e)}"

    npu_ok = hasattr(torch, "npu") and getattr(torch.npu, "is_available", lambda: False)()
    env["npu_available"] = npu_ok
    if npu_ok:
        try:
            env["npu_count"] = torch.npu.device_count()
            env["npu_name"] = torch.npu.get_device_name(0)
        except Exception as e:  # noqa: BLE001
            env["npu_info_err"] = short_exc(e)

    if not npu_ok and not args.allow_cpu:
        print("NPU 不可用；如需在 CPU 上干跑请加 --allow-cpu", file=sys.stderr)
        for k, v in env.items():
            print(f"  {k}: {v}")
        sys.exit(2)

    dev = torch.device("npu:0") if npu_ok else torch.device("cpu")
    cdtype = torch.complex64
    print(f"\n=== 探针设备: {dev} ===\n")

    # ---- NPUBackend 冒烟：确认 MPS 依赖的 real/imag 复数原语在此机可跑 ----
    def backend_smoke():
        from aicir.backends.npu_backend import NPUBackend

        bk = NPUBackend(fallback_to_cpu=not npu_ok)
        a = bk.cast(np.array([[1 + 1j, 2], [0, 1j]], dtype=np.complex64))
        b = bk.cast(np.array([[1, 0], [1j, 1]], dtype=np.complex64))
        mm = bk.matmul(a, b)  # 复数 matmul（real/imag 分解）
        td = bk.tensordot(bk.reshape(a, (2, 2)), bk.reshape(b, (2, 2)), ([1], [0]))
        _ = bk.conj(a)
        _ = bk.add(a, b)
        return f"backend={bk.name.split('(')[0]}; matmul&tensordot&conj&add ok, mm.dtype={mm.dtype}, td.dtype={td.dtype}"

    probe("00_npu_backend_complex_primitives", backend_smoke)

    # ---- 通用工具：前向+CPU 数值核对，再反传 ----
    def cpu_singular_ref(x):
        return torch.linalg.svdvals(x.detach().cpu().to(torch.complex128 if x.is_complex() else torch.float64))

    def svd_forward(dtype_name, dtype, shape):
        def fn():
            if dtype.is_complex:
                x = torch.randn(*shape, dtype=dtype, device=dev)
            else:
                x = torch.randn(*shape, dtype=dtype, device=dev)
            u, s, vh = torch.linalg.svd(x, full_matrices=False)
            # 数值核对：奇异值 vs CPU 参考
            ref = cpu_singular_ref(x)
            diff = float((s.detach().cpu().to(ref.dtype) - ref).abs().max())
            # 重建核对
            recon = (u @ torch.diag(s.to(u.dtype)) @ vh)
            rerr = float((recon.detach().cpu() - x.detach().cpu()).abs().max())
            return f"shape={tuple(shape)} sv_maxdiff_vs_cpu={diff:.2e} recon_maxerr={rerr:.2e} S.dtype={s.dtype}"

        return fn

    def svd_backward(dtype_name, dtype, shape):
        def fn():
            x = torch.randn(*shape, dtype=dtype, device=dev, requires_grad=True)
            u, s, vh = torch.linalg.svd(x, full_matrices=False)
            loss = s.sum()  # 奇异值是实数，sum 可反传
            loss.backward()
            g = x.grad
            gok = g is not None and bool(torch.isfinite(g.float() if not g.is_complex() else g.abs()).all())
            gnorm = float(g.abs().sum()) if g is not None else -1.0
            return f"shape={tuple(shape)} grad_finite={gok} grad_absnorm={gnorm:.3e}"

        return fn

    def qr_forward(dtype, shape):
        def fn():
            x = torch.randn(*shape, dtype=dtype, device=dev)
            q, r = torch.linalg.qr(x, mode="reduced")
            recon = q @ r
            rerr = float((recon.detach().cpu() - x.detach().cpu()).abs().max())
            return f"shape={tuple(shape)} recon_maxerr={rerr:.2e} Q.dtype={q.dtype}"

        return fn

    def qr_backward(dtype, shape):
        def fn():
            x = torch.randn(*shape, dtype=dtype, device=dev, requires_grad=True)
            q, r = torch.linalg.qr(x, mode="reduced")
            loss = r.abs().sum()
            loss.backward()
            g = x.grad
            gok = g is not None and bool(torch.isfinite(g.abs()).all())
            return f"shape={tuple(shape)} grad_finite={gok}"

        return fn

    # 覆盖几种 MPS 常见形状：小方阵、tall、theta 形 (Dl*2, 2*Dr)
    shapes = [(4, 4), (16, 16), (32, 8), (8, 32)]

    # ---- 1. real SVD 前向 ----
    for sh in shapes:
        probe(f"01_svd_fwd_real_{sh[0]}x{sh[1]}", svd_forward("real", torch.float32, sh))
    # ---- 2. real SVD 反传 ----
    for sh in shapes:
        probe(f"02_svd_bwd_real_{sh[0]}x{sh[1]}", svd_backward("real", torch.float32, sh))
    # ---- 3. complex64 SVD 前向 ----
    for sh in shapes:
        probe(f"03_svd_fwd_cplx_{sh[0]}x{sh[1]}", svd_forward("cplx", cdtype, sh))
    # ---- 4. complex64 SVD 反传 ----
    for sh in shapes:
        probe(f"04_svd_bwd_cplx_{sh[0]}x{sh[1]}", svd_backward("cplx", cdtype, sh))
    # ---- 5. real QR 前向/反传 ----
    probe("05_qr_fwd_real_16x8", qr_forward(torch.float32, (16, 8)))
    probe("06_qr_bwd_real_16x8", qr_backward(torch.float32, (16, 8)))
    # ---- 7. complex QR 前向/反传 ----
    probe("07_qr_fwd_cplx_16x8", qr_forward(cdtype, (16, 8)))
    probe("08_qr_bwd_cplx_16x8", qr_backward(cdtype, (16, 8)))

    # ---- 9. real-embedding 复数 SVD 变通（若 complex SVD 不行的备选路径）----
    def real_embed_svd():
        m, n = 12, 8
        a = torch.randn(m, n, dtype=cdtype, device=dev, requires_grad=True)
        ar, ai = torch.real(a), torch.imag(a)
        top = torch.cat([ar, -ai], dim=1)
        bot = torch.cat([ai, ar], dim=1)
        big = torch.cat([top, bot], dim=0)  # (2m, 2n) 实矩阵
        u, s, vh = torch.linalg.svd(big, full_matrices=False)
        # big 的奇异值是 |a| 奇异值各出现两次；去重后应与复数 SVD 一致
        ref = cpu_singular_ref(a)  # 复数 a 的奇异值（CPU complex128）
        s_sorted = torch.sort(s.detach().cpu(), descending=True).values
        # 取每隔一个（成对），比对前 n 个
        s_dedup = s_sorted[0 : 2 * n : 2][:n].to(ref.dtype)
        diff = float((s_dedup - ref[:n]).abs().max())
        loss = s.sum()
        loss.backward()
        gok = a.grad is not None and bool(torch.isfinite(a.grad.abs()).all())
        return f"real-embed sv_maxdiff_vs_cplx={diff:.2e} grad_finite={gok}"

    probe("09_real_embed_complex_svd_fwd_bwd", real_embed_svd)

    # ---- 10. MPS 截断真实算子：complex theta (Dl*2, 2*Dr) 约化 SVD → 截断 top-k → 反传 ----
    def mps_truncation_op():
        Dl, Dr, k = 4, 4, 2
        theta = torch.randn(Dl * 2, 2 * Dr, dtype=cdtype, device=dev, requires_grad=True)
        u, s, vh = torch.linalg.svd(theta, full_matrices=False)
        u_k = u[:, :k]
        s_k = s[:k]
        vh_k = vh[:k, :]
        # 吸收奇异值进右张量（MPS 里的写法）
        vh_scaled = vh_k * s_k.to(vh_k.dtype).reshape(k, 1)
        a_left = u_k.reshape(Dl, 2, k)
        a_right = vh_scaled.reshape(k, 2, Dr)
        # 用一个标量损失反传（模拟期望值对参数的梯度）
        loss = a_left.abs().sum() + a_right.abs().sum()
        loss.backward()
        gok = theta.grad is not None and bool(torch.isfinite(theta.grad.abs()).all())
        # 截断重建误差（前向数值合理性）
        recon = u_k @ torch.diag(s_k.to(u.dtype)) @ vh_k
        return f"k={k} left/right built; slice+scale+reshape ok; trunc_grad_finite={gok}"

    probe("10_mps_truncation_op_fwd_bwd", mps_truncation_op)

    # ---- SUMMARY ----
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
