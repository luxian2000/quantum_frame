"""Ascend NPU 运行时硬件能力探测（设计见 docs/superpowers/specs/2026-06-20-npu-capability-probe-design.md）。

把 NPU 实际支持的 dtype/算子、张量维度与尺寸上限、设备内存探测成 ``NpuCapabilities``，
缓存到磁盘供后续脚本复用，并可映射为 ``aicir.devices.Target`` 执行标志。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from .npu_backend import is_npu_available, npu_runtime_context_from_env

# 模块常量
BYTES_COMPLEX64 = 8
MEMORY_SAFETY = 0.9
MAX_PROBE_NDIM = 64


def _torch_npu_version() -> str | None:
    """获取 torch_npu 版本字符串，失败返回 None。"""
    try:
        import torch_npu  # type: ignore

        return str(getattr(torch_npu, "__version__", "unknown"))
    except Exception:
        return None


def _resolve_probe_device(backend, allow_cpu_fallback: bool) -> str:
    """选定探测设备：优先 backend 设备 / NPU；无 NPU 时按 allow_cpu_fallback 决定回退或报错。"""
    if is_npu_available():
        dev = getattr(backend, "_device", None)
        return str(dev) if dev is not None else "npu:0"
    if allow_cpu_fallback:
        return "cpu"
    raise RuntimeError("NPU 不可用；如需在 CPU 上探测请传 allow_cpu_fallback=True")


def _probe_op_support(device: str):
    """在设备上跑微 complex64 算子，返回 (matmul, conj, add, errors)。任何失败记入 errors。"""
    errors: list[str] = []

    def _ok(label: str, fn) -> bool:
        try:
            fn()
            return True
        except Exception as exc:  # noqa: BLE001  探测即为捕获不支持的算子
            errors.append(f"{label}: {exc!r}")
            return False

    try:
        a = torch.ones((2, 2), dtype=torch.complex64, device=device)
    except Exception as exc:  # noqa: BLE001  连分配都失败 → 全部不支持
        return False, False, False, (f"alloc complex64: {exc!r}",)

    matmul = _ok("matmul", lambda: torch.matmul(a, a))
    conj = _ok("conj", lambda: torch.conj(a))
    add = _ok("add", lambda: a + a)
    return matmul, conj, add, tuple(errors)


def _probe_max_ndim(device: str) -> int | None:
    """用递增轴数的微张量试到抛错；返回成功的最大维数，全失败返回 None。"""
    last_ok: int | None = None
    for ndim in range(1, MAX_PROBE_NDIM + 1):
        try:
            torch.empty((1,) * ndim, dtype=torch.complex64, device=device)
            last_ok = ndim
        except Exception:  # noqa: BLE001  到达维度上限即停
            break
    return last_ok


def _probe_total_memory(device: str) -> int | None:
    """查询设备总内存（字节）；CPU 或不可用时返回 None（不做 allocate-until-OOM）。"""
    if not device.startswith("npu"):
        return None
    try:
        free, total = torch.npu.mem_get_info()  # type: ignore[attr-defined]
        return int(total)
    except Exception:  # noqa: BLE001
        return None


def _collect_capabilities(backend=None, *, allow_cpu_fallback: bool = False) -> NpuCapabilities:
    """实际探测（不读写缓存），组装并返回 NpuCapabilities。"""
    device = _resolve_probe_device(backend, allow_cpu_fallback)
    ctx = npu_runtime_context_from_env()

    matmul, conj, add, op_errors = _probe_op_support(device)
    needs_decomp = not (matmul and conj and add)
    max_ndim = _probe_max_ndim(device)
    total_memory = _probe_total_memory(device)

    if total_memory is not None:
        max_elements = int(total_memory * MEMORY_SAFETY) // BYTES_COMPLEX64
        max_qubits = int(math.floor(math.log2(max_elements))) if max_elements >= 1 else None
    else:
        max_elements = None
        max_qubits = None

    if max_qubits is not None:
        max_qubits_sharded = max_qubits + int(math.floor(math.log2(ctx.world_size)))
    else:
        max_qubits_sharded = None

    return NpuCapabilities(
        device=device,
        available=is_npu_available(),
        torch_version=str(torch.__version__),
        torch_npu_version=_torch_npu_version(),
        complex_dtype="complex64",
        supports_complex_matmul=matmul,
        supports_complex_conj=conj,
        supports_complex_add=add,
        needs_real_imag_decomp=needs_decomp,
        max_ndim=max_ndim,
        max_elements=max_elements,
        max_qubits=max_qubits,
        max_qubits_sharded=max_qubits_sharded,
        total_memory=total_memory,
        world_size=ctx.world_size,
        probe_errors=op_errors,
    )


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
