"""证据清单。

形状对齐 ``docs/evidence/distributed-autograd/<sha>/manifest.json``：钉住 commit、
run_id、内容哈希与门禁结论。基准数字若不能追溯到具体 commit 与框架版本，就没有
论文价值——半年后没人说得清那张表是在哪个版本上跑出来的。

同时记录**线程环境**：不报告 ``OMP_NUM_THREADS``/BLAS 线程数，跨机器的 CPU 计时
根本无法解释。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone

__all__ = ["build_manifest", "write_manifest", "current_commit"]


def current_commit() -> str:
    """当前 HEAD 的完整 SHA；不在 git 仓库时返回 ``"unknown"``。"""

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"


def _worktree_is_dirty() -> bool:
    """工作区是否有未提交改动。脏工作区跑出来的数字不可复现，必须记录。"""

    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return False


def _thread_environment() -> dict:
    """采集影响 CPU 计时的线程设置。"""

    keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    env = {key: os.environ.get(key) for key in keys}
    env["cpu_count"] = os.cpu_count()
    return env


def _numpy_blas() -> str:
    """numpy 链接的 BLAS 实现名——Accelerate/OpenBLAS/MKL 的性能差异极大。"""

    try:
        import numpy as np

        config = np.__config__.get_info("blas_opt") if hasattr(np.__config__, "get_info") else {}
        if config:
            return str(config.get("libraries", ["unknown"])[0])
        blas = getattr(np.__config__, "_built_with_meson", None)
        if blas is not None:
            cfg = np.__config__.show(mode="dicts") if hasattr(np.__config__, "show") else {}
            return str(cfg.get("Build Dependencies", {}).get("blas", {}).get("name", "unknown"))
    except Exception:  # pragma: no cover - 仅为记录，失败不影响基准
        pass
    return "unknown"


def build_manifest(*, results, failed_conditions, extra: dict | None = None) -> dict:
    """构造清单字典。

    参数:
        results:            每条基准记录（可 JSON 序列化）。
        failed_conditions:  失败的不变量名列表；非空即门禁 FAIL。
        extra:              附加字段（如 parity 报告）。
    """

    from ..adapters import adapter_versions

    payload_results = list(results)
    raw = json.dumps(payload_results, sort_keys=True, ensure_ascii=False).encode("utf-8")

    manifest = {
        "commit": current_commit(),
        "worktree_dirty": _worktree_is_dirty(),
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_gate": "FAIL" if failed_conditions else "PASS",
        "failed_conditions": list(failed_conditions),
        "frameworks": adapter_versions(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "threads": _thread_environment(),
            "numpy_blas": _numpy_blas(),
        },
        "n_results": len(payload_results),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "results": payload_results,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(manifest: dict, path) -> str:
    """把清单写成 JSON，返回写入路径。"""

    import pathlib

    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(target)
