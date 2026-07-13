"""``import aicir``/``aicir.vqc`` 在无 torch 环境下必须可用（numpy 路径端到端可跑）。

模拟方式必须是"真"的模块不存在，而非 ``sys.modules["torch"] = None`` 这类伪造
（后者会被 ``import torch`` 直接命中 ``sys.modules`` 缓存并返回 ``None``，让很多本该
触发 ``ModuleNotFoundError`` 的代码路径悄悄跳过，产生假绿）。这里改用一个 meta path
finder，对 ``torch``/``torch_npu`` 的任意子模块导入都主动抛出 ``ModuleNotFoundError``，
并且整段验证跑在独立子进程里——子进程从未 import 过 torch/aicir，不存在缓存模块可以
"复用"，比在当前进程里操纵 ``sys.modules`` 更接近真实的"未安装 torch"环境。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_BLOCK_TORCH_AND_RUN = r"""
import sys
import importlib.abc


class _TorchBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch.") or name == "torch_npu" or name.startswith("torch_npu."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


sys.meta_path.insert(0, _TorchBlocker())

import aicir  # noqa: E402

assert "torch" not in sys.modules, "import aicir must not pull in torch"

# 顶层命名空间不能因为缺少 torch 而被静默削减：纯 numpy 路径的核心导出必须仍在。
for _name in ("Circuit", "Parameter", "pauli_x", "cnot", "rx", "Hamiltonian", "Measure",
              "circuit_to_qasm", "circuit_to_json", "NumpyBackend"):
    assert hasattr(aicir, _name), f"aicir.{_name} must exist without torch"

# torch 专属导出则应当缺席（而不是牵连上面的核心导出一起消失）。
assert not hasattr(aicir, "BatchSV"), "BatchSV is torch-only and must be absent without torch"

from aicir.vqc import BasicVQE  # noqa: E402
from aicir.backends import NumpyBackend  # noqa: E402
from aicir.core.operators import Hamiltonian  # noqa: E402

hamiltonian = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])
solver = BasicVQE(hamiltonian, depth=1, seed=0, backend=NumpyBackend())
result = solver.run(max_iters=3, optimizer=None)

import math  # noqa: E402

assert math.isfinite(result.energy), f"energy must be finite, got {result.energy!r}"
assert len(result.energy_history) == 3

assert "torch" not in sys.modules, "running a numpy-backend VQE must not pull in torch"

print("TORCH_ABSENCE_OK")
"""


def test_aicir_and_basic_vqe_import_and_run_without_torch() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)

    proc = subprocess.run(
        [sys.executable, "-c", _BLOCK_TORCH_AND_RUN],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode == 0, (
        f"subprocess failed (stdout={proc.stdout!r}, stderr={proc.stderr!r})"
    )
    assert "TORCH_ABSENCE_OK" in proc.stdout
