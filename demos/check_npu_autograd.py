"""最小复现：Ascend NPU 上 **纯实数（float32）** autograd 是否可用。

不依赖 aicir，只用 torch + torch_npu，单进程运行：

    python demos/check_npu_autograd.py

**有意完全不用 complex64**（Ascend 连 complex64 前向都缺算子，aicir 的 NPU 路径全程实/虚部
拆成实数张量）。本测确认的是：若把 autodiff 也完全用实数表示（real/imag 分解、无 complex64
节点），NPU 的 autograd 反向是否能跑通——尤其是反向里的**梯度累加 add**（complex64 版本正是
这里报 `aclnnAdd ... DT_COMPLEX64`）。

逐项 try/except，打印 OK 或确切错误串。
"""

import torch

try:
    import torch_npu  # noqa: F401  导入即注册 npu 设备
except Exception as exc:  # noqa: BLE001
    print("torch_npu 不可用：", repr(exc))
    raise SystemExit(1)

DEV = "npu:0"
DT = torch.float32
print("torch:", torch.__version__, " torch_npu:", getattr(torch_npu, "__version__", "?"))
print("npu available:", torch.npu.is_available())


def check(label, fn):
    try:
        out = fn()
        print(f"[OK]   {label}: {out}")
        return True
    except Exception as exc:  # noqa: BLE001  探测即为捕获不支持的算子
        print(f"[FAIL] {label}: {exc!r}")
        return False


# --- 1) 纯实数前向 ---
def _add():
    a = torch.ones(2, dtype=DT, device=DEV)
    return (a + a).sum().item()


def _matmul():
    m = torch.eye(2, dtype=DT, device=DEV)
    v = torch.ones(2, dtype=DT, device=DEV)
    return torch.matmul(m, v).sum().item()


check("forward float32 add (a+a)", _add)
check("forward float32 matmul", _matmul)


# --- 2) 纯实数 autograd 反向 ---
def _backward_simple():
    x = torch.tensor([1.0, 2.0], requires_grad=True, dtype=DT, device=DEV)
    loss = (x * x).sum()
    loss.backward()
    return x.grad


def _backward_accumulate():
    # x 用两次 → 反向把两路梯度相加（float32 add）。complex64 版本正是此处报 aclnnAdd 失败；
    # 本测看纯实数下该累加是否可用。
    x = torch.tensor([1.0, 2.0], requires_grad=True, dtype=DT, device=DEV)
    loss = (x * x).sum() + (x * x).sum()
    loss.backward()
    return x.grad


check("autograd backward (single use)", _backward_simple)
check("autograd backward (grad accumulation = float32 add)", _backward_accumulate)


# --- 3) real/imag 分解的「复数」期望值，全程实数 + autograd ---
def _backward_real_imag_expectation():
    # 把复数态 |psi> 用 (re, im) 两个实数张量表示，复数 matmul 用 4 个实数 matmul，
    # 期望值 <psi|Z|psi> 取实部 → 实标量 loss → backward。全程无 complex64 节点。
    theta = torch.tensor(0.3, requires_grad=True, dtype=DT, device=DEV)
    # ry(theta)|0> = [cos(t/2), sin(t/2)]（实数态）
    re = torch.stack([torch.cos(theta / 2), torch.sin(theta / 2)])
    im = torch.zeros(2, dtype=DT, device=DEV)
    # 算符 Z = diag(1, -1)，实矩阵
    z_re = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=DT, device=DEV)
    # (Z @ psi) 实/虚部
    hz_re = torch.matmul(z_re, re)
    hz_im = torch.matmul(z_re, im)
    # <psi|Z|psi> = conj(psi) . (Z psi)，实部 = re·hz_re + im·hz_im
    val = (re * hz_re + im * hz_im).sum()
    val.backward()
    return float(val.detach().cpu()), theta.grad  # 期望 <Z>=cos(0.3)，grad=-sin(0.3)


check("autograd real/imag expectation <Z> + grad", _backward_real_imag_expectation)

print("\n结论：")
print("- 若 1/2 全 [OK] 但 complex64 版（之前那版）在反向 [FAIL]，则瓶颈确是 complex64 算子，非 autograd 本身。")
print("- 若第 3 项 [OK]（real/imag 全实数表示的期望值可反向 + 梯度正确），则一个 **complex64-free 的 autodiff**")
print("  在 NPU 上原则可行——这才是 NPU 上让 autodiff 可用的方向，而非直接用 complex64。")
