# NPU 硬件能力探测（npu_probe）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Ascend NPU 提供运行时硬件能力探测器，把支持的 dtype/算子、张量维度与尺寸上限、设备内存探测成结构化参数，缓存到磁盘供后续脚本复用，并映射为 `Target` 执行标志。

**Architecture:** 三个单一职责单元——`probe_npu`（运行时微算子探测，含磁盘缓存）→ `NpuCapabilities`（冻结数据类，纯数据）→ `target_from_npu`（纯映射到 `Target`）。一个 `demos/demo_npu_probe.py` 脚本串联三者。`Target` 不改动。

**Tech Stack:** Python、PyTorch（可选，测试 `importorskip`）、`torch_npu`（仅 NPU）、JSON 磁盘缓存。

## Global Constraints

- 仅 NPU 范围；不做通用跨后端探测、不做吞吐量基准、不实现状态向量分片执行。
- 任何探测失败必须优雅降级为 `None`/`False` + 一条 `probe_errors`，绝不让调用方崩溃。
- 不做 allocate-until-OOM；尺寸上限由内存查询派生。
- `torch` 为可选依赖：核心模块顶层 `import torch` 允许（模块本就属 NPU 路径），但测试用 `pytest.importorskip("torch")`。
- 注释/文档用中文，匹配仓库风格。
- 运行测试：仓库根目录 `PYTHONPATH=. pytest`。
- 复数 dtype 固定 `complex64`（8 字节/元素）；内存安全系数 `0.9`。

---

### Task 1: `NpuCapabilities` 数据模型 + JSON 序列化

**Files:**
- Create: `aicir/backends/npu_probe.py`
- Test: `tests/backends/test_npu_probe.py`

**Interfaces:**
- Consumes: 无。
- Produces:
  - `NpuCapabilities` 冻结数据类，字段见下。
  - `NpuCapabilities.to_dict(self) -> dict`（仅静态字段）。
  - `NpuCapabilities.from_dict(cls, data: dict) -> NpuCapabilities`。
  - `NpuCapabilities.cache_key(self) -> str`（`f"{device}|{torch_version}|{torch_npu_version}"`）。

- [ ] **Step 1: 写失败测试**

```python
# tests/backends/test_npu_probe.py
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/backends/test_npu_probe.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'aicir.backends.npu_probe'`

- [ ] **Step 3: 写最小实现**

```python
# aicir/backends/npu_probe.py
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. pytest tests/backends/test_npu_probe.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add aicir/backends/npu_probe.py tests/backends/test_npu_probe.py
git commit -m "feat(npu_probe): NpuCapabilities 数据模型 + JSON 序列化

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: 探测内核 `_collect_capabilities`（无缓存）

**Files:**
- Modify: `aicir/backends/npu_probe.py`
- Test: `tests/backends/test_npu_probe.py`

**Interfaces:**
- Consumes: `NpuCapabilities`（Task 1）；`aicir.backends.npu_backend.is_npu_available`、`npu_runtime_context_from_env`。
- Produces:
  - `BYTES_COMPLEX64 = 8`、`MEMORY_SAFETY = 0.9`、`MAX_PROBE_NDIM = 64` 模块常量。
  - `_resolve_probe_device(backend, allow_cpu_fallback) -> str`。
  - `_probe_op_support(device) -> tuple[bool, bool, bool, tuple[str, ...]]`（matmul, conj, add, errors）。
  - `_probe_max_ndim(device) -> int | None`。
  - `_probe_total_memory(device) -> int | None`。
  - `_collect_capabilities(backend=None, *, allow_cpu_fallback=False) -> NpuCapabilities`。

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/backends/test_npu_probe.py
from aicir.backends.npu_probe import _collect_capabilities


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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/backends/test_npu_probe.py -k collect -v`
Expected: FAIL，`ImportError: cannot import name '_collect_capabilities'`

- [ ] **Step 3: 写最小实现**

```python
# aicir/backends/npu_probe.py 顶部 import 区追加：
import math

import torch

from .npu_backend import is_npu_available, npu_runtime_context_from_env

# 模块常量
BYTES_COMPLEX64 = 8
MEMORY_SAFETY = 0.9
MAX_PROBE_NDIM = 64


def _torch_npu_version() -> str | None:
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. pytest tests/backends/test_npu_probe.py -v`
Expected: PASS（全部）

- [ ] **Step 5: 提交**

```bash
git add aicir/backends/npu_probe.py tests/backends/test_npu_probe.py
git commit -m "feat(npu_probe): 探测内核 _collect_capabilities（dtype/维度/内存）

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: 磁盘缓存 + 公共 `probe_npu`

**Files:**
- Modify: `aicir/backends/npu_probe.py`
- Test: `tests/backends/test_npu_probe.py`

**Interfaces:**
- Consumes: `_collect_capabilities`、`NpuCapabilities`（Task 1/2）。
- Produces:
  - `cache_path() -> pathlib.Path`（`$AICIR_CACHE_DIR/npu_caps.json`，默认 `~/.cache/aicir/`）。
  - `probe_npu(backend=None, *, allow_cpu_fallback=False, refresh=False) -> NpuCapabilities`。

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/backends/test_npu_probe.py
import json

from aicir.backends.npu_probe import cache_path, probe_npu


def test_probe_npu_writes_then_loads_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("AICIR_CACHE_DIR", str(tmp_path))
    first = probe_npu(allow_cpu_fallback=True)
    assert cache_path().exists()

    # 篡改缓存内容；refresh=False 且键匹配 → 应读到被篡改值，证明走了缓存
    data = json.loads(cache_path().read_text())
    data["max_ndim"] = 999
    cache_path().write_text(json.dumps(data))

    cached = probe_npu(allow_cpu_fallback=True)
    assert cached.max_ndim == 999
    assert cached.cache_key() == first.cache_key()


def test_probe_npu_refresh_ignores_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("AICIR_CACHE_DIR", str(tmp_path))
    probe_npu(allow_cpu_fallback=True)
    data = json.loads(cache_path().read_text())
    data["max_ndim"] = 999
    cache_path().write_text(json.dumps(data))

    fresh = probe_npu(allow_cpu_fallback=True, refresh=True)
    assert fresh.max_ndim != 999  # 重新探测，覆盖篡改值


def test_probe_npu_stale_key_reprobes(tmp_path, monkeypatch):
    monkeypatch.setenv("AICIR_CACHE_DIR", str(tmp_path))
    probe_npu(allow_cpu_fallback=True)
    data = json.loads(cache_path().read_text())
    data["torch_version"] = "0.0.0-stale"
    data["max_ndim"] = 999
    cache_path().write_text(json.dumps(data))

    result = probe_npu(allow_cpu_fallback=True)
    assert result.max_ndim != 999  # 键不匹配 → 忽略缓存重探
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/backends/test_npu_probe.py -k probe_npu -v`
Expected: FAIL，`ImportError: cannot import name 'cache_path'`

- [ ] **Step 3: 写最小实现**

```python
# aicir/backends/npu_probe.py 顶部 import 区追加：
import json
import os
import pathlib


def cache_path() -> pathlib.Path:
    """能力缓存文件路径；可经 AICIR_CACHE_DIR 覆盖，默认 ~/.cache/aicir/。"""
    base = os.environ.get("AICIR_CACHE_DIR")
    root = pathlib.Path(base) if base else pathlib.Path.home() / ".cache" / "aicir"
    return root / "npu_caps.json"


def _load_cached(key: str) -> NpuCapabilities | None:
    """读缓存；文件缺失/损坏/键不匹配返回 None。"""
    path = cache_path()
    if not path.exists():
        return None
    try:
        caps = NpuCapabilities.from_dict(json.loads(path.read_text()))
    except Exception:  # noqa: BLE001  损坏缓存视为未命中
        return None
    return caps if caps.cache_key() == key else None


def _save_cached(caps: NpuCapabilities) -> None:
    """写缓存（静态字段）。"""
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(caps.to_dict(), ensure_ascii=False, indent=2))


def probe_npu(
    backend=None, *, allow_cpu_fallback: bool = False, refresh: bool = False
) -> NpuCapabilities:
    """探测 NPU 能力。``refresh=False`` 且缓存键匹配时读缓存，否则探测并写回。

    缓存仅持久化静态字段；实时空闲内存请另行查询（本探测不缓存空闲内存）。
    """
    probe_key = NpuCapabilities(
        device=_resolve_probe_device(backend, allow_cpu_fallback),
        available=is_npu_available(),
        torch_version=str(torch.__version__),
        torch_npu_version=_torch_npu_version(),
        complex_dtype="complex64",
        supports_complex_matmul=False,
        supports_complex_conj=False,
        supports_complex_add=False,
        needs_real_imag_decomp=True,
        max_ndim=None,
        max_elements=None,
        max_qubits=None,
        max_qubits_sharded=None,
        total_memory=None,
        world_size=1,
        probe_errors=(),
    ).cache_key()

    if not refresh:
        cached = _load_cached(probe_key)
        if cached is not None:
            return cached

    caps = _collect_capabilities(backend, allow_cpu_fallback=allow_cpu_fallback)
    _save_cached(caps)
    return caps
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. pytest tests/backends/test_npu_probe.py -v`
Expected: PASS（全部）

- [ ] **Step 5: 提交**

```bash
git add aicir/backends/npu_probe.py tests/backends/test_npu_probe.py
git commit -m "feat(npu_probe): 磁盘缓存 + 公共 probe_npu（版本键失效/refresh）

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `target_from_npu` 映射

**Files:**
- Modify: `aicir/backends/npu_probe.py`
- Test: `tests/backends/test_npu_probe.py`

**Interfaces:**
- Consumes: `NpuCapabilities`（Task 1）；`aicir.devices.Target`。
- Produces: `target_from_npu(caps, n_qubits=None) -> Target`。

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/backends/test_npu_probe.py
from aicir.backends.npu_probe import target_from_npu
from aicir.devices import Target


def test_target_from_npu_maps_flags_and_uses_explicit_n_qubits():
    caps = _sample_caps(max_qubits=10)
    target = target_from_npu(caps, n_qubits=5)
    assert isinstance(target, Target)
    assert target.n_qubits == 5
    assert target.supports_statevector is True
    assert target.supports_autodiff is True


def test_target_from_npu_defaults_n_qubits_to_max_qubits():
    caps = _sample_caps(max_qubits=7)
    target = target_from_npu(caps)
    assert target.n_qubits == 7


def test_target_from_npu_requires_some_n_qubits():
    caps = _sample_caps(max_qubits=None)
    with pytest.raises(ValueError):
        target_from_npu(caps)  # 无显式 n_qubits 且 max_qubits 为 None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/backends/test_npu_probe.py -k target_from_npu -v`
Expected: FAIL，`ImportError: cannot import name 'target_from_npu'`

- [ ] **Step 3: 写最小实现**

```python
# aicir/backends/npu_probe.py 顶部 import 区追加：
from ..devices import Target


def target_from_npu(caps: NpuCapabilities, n_qubits: int | None = None) -> Target:
    """把 NpuCapabilities 映射为 Target 执行标志。

    ``n_qubits`` 缺省时取 ``caps.max_qubits``；两者皆 None 抛 ValueError。
    丰富的 dtype/内存细节留在 caps，Target 只消费它能建模的子集。
    """
    resolved = n_qubits if n_qubits is not None else caps.max_qubits
    if resolved is None:
        raise ValueError("target_from_npu 需要 n_qubits（caps.max_qubits 为 None 时必须显式传入）")
    return Target(
        n_qubits=int(resolved),
        supports_statevector=True,
        supports_density_matrix=False,
        supports_autodiff=True,
    )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. pytest tests/backends/test_npu_probe.py -v`
Expected: PASS（全部）

- [ ] **Step 5: 提交**

```bash
git add aicir/backends/npu_probe.py tests/backends/test_npu_probe.py
git commit -m "feat(npu_probe): target_from_npu 映射到 Target 执行标志

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: 探测脚本 `demos/demo_npu_probe.py`

**Files:**
- Create: `demos/demo_npu_probe.py`
- Test: `tests/backends/test_npu_probe.py`

**Interfaces:**
- Consumes: `probe_npu`、`target_from_npu`（Task 3/4）。
- Produces: `demos/demo_npu_probe.py` 含 `main(argv=None) -> int`。

- [ ] **Step 1: 写失败测试**

```python
# 追加到 tests/backends/test_npu_probe.py
def test_demo_main_cpu_fallback_returns_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AICIR_CACHE_DIR", str(tmp_path))
    from demos.demo_npu_probe import main

    rc = main(["--allow-cpu-fallback"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "NpuCapabilities" in out or "device" in out
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/backends/test_npu_probe.py -k demo_main -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'demos.demo_npu_probe'`

- [ ] **Step 3: 写最小实现**

```python
# demos/demo_npu_probe.py
"""NPU 硬件能力探测脚本：打印能力表并构建 Target。

用法：
    python demos/demo_npu_probe.py                       # 严格 NPU
    python demos/demo_npu_probe.py --allow-cpu-fallback  # 允许 CPU 回退
    python demos/demo_npu_probe.py --refresh             # 忽略缓存重探
"""

from __future__ import annotations

import argparse

from aicir.backends.npu_probe import probe_npu, target_from_npu


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="探测 Ascend NPU 硬件能力")
    parser.add_argument("--allow-cpu-fallback", action="store_true", help="无 NPU 时在 CPU 上探测")
    parser.add_argument("--refresh", action="store_true", help="忽略磁盘缓存，强制重探")
    args = parser.parse_args(argv)

    caps = probe_npu(allow_cpu_fallback=args.allow_cpu_fallback, refresh=args.refresh)

    print("== NpuCapabilities ==")
    for key, value in caps.to_dict().items():
        print(f"  {key}: {value}")

    if caps.max_qubits is not None:
        target = target_from_npu(caps)
        print("== Target ==")
        print(f"  {target}")
    else:
        print("== Target ==")
        print("  跳过：max_qubits 为 None（无法派生 n_qubits）")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. pytest tests/backends/test_npu_probe.py -v`
Expected: PASS（全部）

- [ ] **Step 5: 提交**

```bash
git add demos/demo_npu_probe.py tests/backends/test_npu_probe.py
git commit -m "feat(npu_probe): demo_npu_probe 探测脚本（打印能力表 + Target）

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: 文档收尾

**Files:**
- Modify: `aicir/backends/README.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: 全部前序任务的公共 API。
- Produces: 无代码接口。

- [ ] **Step 1: 在 `aicir/backends/README.md` §5 NPUBackend 下新增 §5.x 探测小节**

在 §5.3（QAS supernet 集成）之前插入：

```markdown
### 5.x  硬件能力探测（npu_probe）

`aicir.backends.npu_probe.probe_npu(allow_cpu_fallback=False, refresh=False)` 运行时探测
NPU 实际支持的 dtype/算子、张量维度与尺寸上限、设备内存，返回 `NpuCapabilities`，并把静态
字段缓存到 `$AICIR_CACHE_DIR/npu_caps.json`（默认 `~/.cache/aicir/`）。后续脚本默认读缓存，
`refresh=True` 强制重探。`target_from_npu(caps, n_qubits=None)` 映射为 `Target` 执行标志。

```bash
python demos/demo_npu_probe.py --allow-cpu-fallback
```

注：空闲内存随分配变化，不缓存；探测失败的算子降级为 `False`/`None` 并记入 `probe_errors`。
单条电路的态向量/门矩阵仍完整驻留单设备（见 §5.2），本探测只产出未来分片所需的尺寸输入。
```

- [ ] **Step 2: 在 `CHANGELOG.md` 当日 `### Added` 追加条目**

```markdown
- **`aicir.backends.npu_probe`：NPU 运行时硬件能力探测。** `probe_npu(...)` 探测 dtype/算子
  支持、张量维度与尺寸上限、设备内存 → `NpuCapabilities`，静态字段磁盘缓存（版本键失效、
  `refresh=` 重探），`target_from_npu(caps, n_qubits=None)` 映射为 `Target`；脚本
  `demos/demo_npu_probe.py`。配套 `tests/backends/test_npu_probe.py`。
```

- [ ] **Step 3: 跑全套确认无回归**

Run: `PYTHONPATH=. pytest tests/backends/ -q`
Expected: PASS（含新 `test_npu_probe.py`）

- [ ] **Step 4: 提交**

```bash
git add aicir/backends/README.md CHANGELOG.md
git commit -m "docs(npu_probe): README §5 探测小节 + CHANGELOG

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- NpuCapabilities 数据模型 → Task 1 ✓
- dtype/算子 + 维度 + 内存探测 + 失败降级 → Task 2 ✓
- 磁盘缓存（版本键、refresh、静态/实时分离）→ Task 3 ✓
- Target 映射（Target 不改）→ Task 4 ✓
- 探测脚本（--allow-cpu-fallback / --refresh）→ Task 5 ✓
- 文档（README §5 + CHANGELOG）→ Task 6 ✓
- 与 SV 分片关系（max_qubits_sharded 派生字段）→ Task 2 字段 + README 注 ✓

**Placeholder scan:** 无 TBD/TODO；每个代码步骤含完整代码。

**Type consistency:** `NpuCapabilities` 字段在 Task 1 定义，Task 2/3/4 使用一致；`probe_npu`/`_collect_capabilities`/`target_from_npu`/`cache_path` 签名跨任务一致；`cache_key()` 格式（`device|torch_version|torch_npu_version`）Task 1 定义、Task 3 使用一致。

**已知取舍:** Task 3 的 `probe_npu` 为构建缓存键复制了一遍身份字段（device/版本）以避免缓存未命中时也做完整探测；实现时若 `_resolve_probe_device` 等调用开销可忽略，可保留此写法。
```
