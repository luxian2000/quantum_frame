import pytest

pytest.importorskip("torch")

from aicir.backends.npu_probe import NpuCapabilities


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
