import numpy as np
from aicir.backends import NumpyBackend


def test_numpy_tensordot_matches_numpy():
    bk = NumpyBackend()
    a = np.random.rand(2, 3, 4) + 1j * np.random.rand(2, 3, 4)
    b = np.random.rand(4, 3, 5) + 1j * np.random.rand(4, 3, 5)
    out = bk.tensordot(bk.cast(a), bk.cast(b), ([2, 1], [0, 1]))
    assert np.allclose(bk.to_numpy(out), np.tensordot(a, b, axes=([2, 1], [0, 1])), atol=1e-5)


def test_numpy_transpose_reshape_conj():
    bk = NumpyBackend()
    a = (np.arange(24) + 1j * np.arange(24)).reshape(2, 3, 4)
    ca = bk.cast(a)
    assert np.allclose(bk.to_numpy(bk.transpose(ca, (2, 0, 1))), np.transpose(a, (2, 0, 1)), atol=1e-5)
    assert np.allclose(bk.to_numpy(bk.reshape(ca, (6, 4))), a.reshape(6, 4), atol=1e-5)
    assert np.allclose(bk.to_numpy(bk.conj(ca)), np.conj(a), atol=1e-5)


def test_tensordot_via_matmul_matches_numpy():
    from aicir.backends._contract import tensordot_via_matmul
    bk = NumpyBackend()
    a = np.random.rand(2, 3, 4) + 1j * np.random.rand(2, 3, 4)
    b = np.random.rand(4, 3, 5) + 1j * np.random.rand(4, 3, 5)
    out = tensordot_via_matmul(bk, bk.cast(a), bk.cast(b), ([2, 1], [0, 1]))
    assert np.allclose(bk.to_numpy(out), np.tensordot(a, b, axes=([2, 1], [0, 1])), atol=1e-5)


def test_tensordot_via_matmul_outer_product():
    from aicir.backends._contract import tensordot_via_matmul
    bk = NumpyBackend()
    a = np.array([1.0, 2.0], dtype=np.complex64)
    b = np.array([3.0, 4.0, 5.0], dtype=np.complex64)
    out = tensordot_via_matmul(bk, bk.cast(a), bk.cast(b), ([], []))
    assert np.allclose(bk.to_numpy(out), np.tensordot(a, b, axes=([], [])), atol=1e-5)


def test_gpu_tensordot_matches_numpy():
    import pytest
    pytest.importorskip("torch")
    from aicir.backends import GPUBackend
    bk = GPUBackend(device="cpu")
    a = np.random.rand(2, 3, 4) + 1j * np.random.rand(2, 3, 4)
    b = np.random.rand(4, 3, 5) + 1j * np.random.rand(4, 3, 5)
    out = bk.tensordot(bk.cast(a), bk.cast(b), ([2, 1], [0, 1]))
    assert np.allclose(bk.to_numpy(out), np.tensordot(a, b, axes=([2, 1], [0, 1])), atol=1e-4)


def test_npu_transpose_reshape_conj_fn_forward_and_backward():
    """``_NpuTransposeFn``/``_NpuReshapeFn``/``_NpuConjFn`` 的前向/反向正确性。

    这三个 Function 是为规避 Ascend 上 complex64 梯度累加（aclnnAdd 缺失）而设计的
    自动微分安全包装：把复数张量作为 Function 的单条输入边，避免其在追踪图中被
    ``real(a)``/``imag(a)`` 两次读取。该 fan-out 触发条件（aclnnAdd）只在真实 NPU
    设备上出现，本测试在 CPU 上直接调用 ``.apply()`` 验证数值与标准 torch
    permute/reshape/conj + autograd 一致；NPU 设备门控（``_is_npu_complex``）与
    真实硬件下的 aclnnAdd 规避需在真实 NPU 上另行验证（见 demos/demo_npu_tensor.py）。
    """
    import pytest
    torch = pytest.importorskip("torch")
    from aicir.backends.npu_backend import _NpuConjFn, _NpuReshapeFn, _NpuTransposeFn

    a = torch.randn(2, 3, 4, dtype=torch.complex64, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)

    out = _NpuTransposeFn.apply(a, [2, 0, 1])
    ref = a_ref.permute(2, 0, 1)
    assert torch.allclose(out, ref, atol=1e-5)
    out.real.sum().backward()
    ref.real.sum().backward()
    assert torch.allclose(a.grad, a_ref.grad, atol=1e-5)

    a2 = torch.randn(2, 3, 4, dtype=torch.complex64, requires_grad=True)
    a2_ref = a2.detach().clone().requires_grad_(True)
    out2 = _NpuReshapeFn.apply(a2, (6, 4))
    ref2 = a2_ref.reshape(6, 4)
    assert torch.allclose(out2, ref2, atol=1e-5)
    out2.real.sum().backward()
    ref2.real.sum().backward()
    assert torch.allclose(a2.grad, a2_ref.grad, atol=1e-5)

    a3 = torch.randn(5, dtype=torch.complex64, requires_grad=True)
    a3_ref = a3.detach().clone().requires_grad_(True)
    out3 = _NpuConjFn.apply(a3)
    ref3 = torch.conj(a3_ref)
    assert torch.allclose(out3, ref3, atol=1e-5)
    out3.real.sum().backward()
    ref3.real.sum().backward()
    assert torch.allclose(a3.grad, a3_ref.grad, atol=1e-5)
