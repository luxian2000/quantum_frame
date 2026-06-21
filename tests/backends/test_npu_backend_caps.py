"""NPUBackend.caps() 构造 + sizing guard（消费 npu_probe 能力 sheet）。"""

import pytest

pytest.importorskip("torch")

from aicir import NPUBackend
from aicir.backends.npu_probe import NpuCapabilities


def _caps(*, max_qubits, needs_real_imag=True, device="cpu"):
    return NpuCapabilities(
        device=device,
        available=False,
        torch_version="x",
        torch_npu_version=None,
        complex_dtype="complex64",
        supports_complex_matmul=not needs_real_imag,
        supports_complex_conj=not needs_real_imag,
        supports_complex_add=not needs_real_imag,
        needs_real_imag_decomp=needs_real_imag,
        max_ndim=64,
        max_elements=None,
        max_qubits=max_qubits,
        max_qubits_sharded=max_qubits,
        total_memory=None,
        world_size=1,
        probe_errors=(),
    )


def test_plain_backend_has_safe_defaults():
    b = NPUBackend()  # 无 NPU 时回退 cpu
    assert b._max_qubits is None
    assert b._needs_real_imag is True


def test_caps_sets_fields_from_sheet():
    b = NPUBackend.caps(_caps(max_qubits=10, needs_real_imag=True))
    assert b._max_qubits == 10
    assert b._needs_real_imag is True


def test_ensure_capacity_raises_when_exceeded():
    b = NPUBackend.caps(_caps(max_qubits=4))
    with pytest.raises(ValueError):
        b.ensure_capacity(5)
    b.ensure_capacity(4)  # 等于上限，放行
    b.ensure_capacity(1)


def test_ensure_capacity_noop_without_max_qubits():
    b = NPUBackend()  # max_qubits None → 无数据不守卫
    b.ensure_capacity(99)  # 不抛


def test_zeros_state_guards_allocation():
    b = NPUBackend.caps(_caps(max_qubits=3))
    with pytest.raises(ValueError):
        b.zeros_state(4)
    state = b.zeros_state(3)  # 上限内，正常分配
    assert state is not None
