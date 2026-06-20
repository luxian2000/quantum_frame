"""Ascend NPU 运行时硬件能力探测（设计见 docs/superpowers/specs/2026-06-20-npu-capability-probe-design.md）。

把 NPU 实际支持的 dtype/算子、张量维度与尺寸上限、设备内存探测成 ``NpuCapabilities``，
缓存到磁盘供后续脚本复用，并可映射为 ``aicir.devices.Target`` 执行标志。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class NpuCapabilities:
    """一次 NPU 能力探测的结构化结果。静态字段入磁盘缓存；实时内存用 :func:`free_memory`。"""

    device: str
    available: bool
    torch_version: str
    torch_npu_version: str | None
    complex_dtype: str
    supports_complex_matmul: bool
    supports_complex_conj: bool
    supports_complex_add: bool
    needs_real_imag_decomp: bool
    max_ndim: int | None
    max_elements: int | None
    max_qubits: int | None
    max_qubits_sharded: int | None
    total_memory: int | None
    world_size: int
    probe_errors: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """转可 JSON 序列化字典（``probe_errors`` 元组转列表）。"""
        data = asdict(self)
        data["probe_errors"] = list(self.probe_errors)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "NpuCapabilities":
        """从 :meth:`to_dict` 的字典重建（列表还原为元组）。"""
        kwargs = dict(data)
        kwargs["probe_errors"] = tuple(kwargs.get("probe_errors", ()))
        return cls(**kwargs)

    def cache_key(self) -> str:
        """缓存失效键：设备 + torch / torch_npu 版本。"""
        return f"{self.device}|{self.torch_version}|{self.torch_npu_version}"
