import pytest

pytest.importorskip("torch")

from aicir.backends.npu_probe import NpuCapabilities, _collect_capabilities


def _sample_caps(**over):
    base = dict(
        device="npu:0",
        available=True,
        torch_version="2.1.0",
        torch_npu_version="2.1.0",
        complex_dtype="complex64",
        supports_complex_matmul=True,
        supports_complex_conj=False,
        supports_complex_add=False,
        needs_real_imag_decomp=True,
        max_ndim=8,
        max_elements=1024,
        max_qubits=10,
        max_qubits_sharded=12,
        total_memory=8192,
        world_size=4,
        probe_errors=("conj failed",),
    )
    base.update(over)
    return NpuCapabilities(**base)


def test_to_dict_from_dict_round_trip():
    caps = _sample_caps()
    restored = NpuCapabilities.from_dict(caps.to_dict())
    assert restored == caps


def test_cache_key_uses_device_and_versions():
    caps = _sample_caps()
    assert caps.cache_key() == "npu:0|2.1.0|2.1.0"


def test_collect_capabilities_cpu_fallback_does_not_crash():
    caps = _collect_capabilities(allow_cpu_fallback=True)
    assert caps.device == "cpu"
    # CPU 支持复数算子 → 无需 real/imag 分解
    assert caps.supports_complex_matmul is True
    assert caps.supports_complex_conj is True
    assert caps.supports_complex_add is True
    assert caps.needs_real_imag_decomp is False
    # CPU 路径不查询设备内存 → 尺寸上限为 None
    assert caps.total_memory is None
    assert caps.max_qubits is None
    assert caps.max_qubits_sharded is None
    assert isinstance(caps.max_ndim, int) and caps.max_ndim >= 1


def test_collect_capabilities_requires_npu_without_fallback(monkeypatch):
    import aicir.backends.npu_probe as mod

    monkeypatch.setattr(mod, "is_npu_available", lambda: False)
    with pytest.raises(RuntimeError):
        _collect_capabilities(allow_cpu_fallback=False)


def test_collect_capabilities_records_probe_failures(monkeypatch):
    import aicir.backends.npu_probe as mod

    monkeypatch.setattr(mod, "_probe_total_memory", lambda device: (None, "total_memory: boom"))
    caps = mod._collect_capabilities(allow_cpu_fallback=True)
    assert "total_memory: boom" in caps.probe_errors
