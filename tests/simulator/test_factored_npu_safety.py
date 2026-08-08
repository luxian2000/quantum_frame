"""因子化引擎的 NPU-safe 边界（无需真机的那部分）。

**这份测试的范围是有意收窄的，先说清楚为什么。**

昇腾有三类会被本引擎命中的内核缺口：复数高级索引（``aclnnIndex``）、
复数加/乘（``aclnnAdd``/``aclnnMul``）、以及位运算
（``aten::bitwise_right_shift``——它**不报错**，而是静默回落 CPU，每次调用一趟
device→host→device）。

但 ``NPUBackend`` 里那些 real/imag 分解是**按设备门控**的：
``_is_npu_complex`` 要求 ``device.type == "npu"``，因此在 CPU 上
（``fallback_to_cpu=True``）``kron``/``matmul`` 等会走原生复数分支。
**这意味着复数算子的 NPU 安全性无法在 CPU 上验证**——CPU 上看到的复数 ``mul``
是回退路径的产物，不是真机上会执行的东西。

于是职责这样切分：

- **本引擎自己负责**、且可在 CPU 上钉死的：不发出任何位运算、不直接索引复数
  张量。位运算尤其重要，因为它是唯一"不崩、只是悄悄变慢"的缺口。
- **委托给 ``NPUBackend`` 的**：``kron`` 与局部门应用内部的复数算术。其 NPU
  安全性是那些方法自身的契约，由它们自己的测试与 ``scripts/npu/factored.sh``
  的真机探针验证，不在本文件范围内。

每条断言都配了反证测试——一个永远不会触发的拦截器什么也没证明，本仓库已经
出过一次这种断言（见 CHANGELOG 里 device_residency 探针的记录）。
"""

import pytest

torch = pytest.importorskip("torch")

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

import aicir as A  # noqa: E402
from aicir import Circuit, Hamiltonian, NPUBackend  # noqa: E402
from aicir.simulator.factored import factored_expectation, factored_statevector  # noqa: E402


class _RecordEngineHazards(TorchDispatchMode):
    """记录本引擎**自身**不该发出的算子。

    位运算：昇腾无内核，静默回落 CPU。
    复数索引：``aclnnIndex`` 不支持复数张量。
    """

    BITWISE = ("bitwise_", "__lshift__", "__rshift__")
    COMPLEX_INDEX = ("index_select", "index.Tensor", "gather")

    def __init__(self):
        super().__init__()
        self.bitwise = []
        self.complex_index = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        if any(tag in name for tag in self.BITWISE):
            self.bitwise.append(name)
        if any(tag in name for tag in self.COMPLEX_INDEX):
            for value in list(args) + list(kwargs.values()):
                if isinstance(value, torch.Tensor) and torch.is_complex(value):
                    self.complex_index.append(name)
                    break
        return func(*args, **kwargs)


def _circuit(n=6):
    gates = [A.hadamard(0), A.cnot(1, [0]), A.ry(0.3, 3), A.cnot(4, [3]), A.rz(0.2, 5)]
    return Circuit(*gates, n_qubits=n)


class TestEngineIssuesNoBitwiseOps:
    """位运算是唯一"不崩、只悄悄变慢"的缺口，因此最需要自动化守卫。"""

    def test_evolution_issues_no_bitwise_op(self):
        backend = NPUBackend(fallback_to_cpu=True)
        mode = _RecordEngineHazards()
        with mode:
            factored_statevector(_circuit(), backend)
        assert not mode.bitwise, f"引擎发出位运算（昇腾会回落 CPU）: {sorted(set(mode.bitwise))}"

    def test_materialisation_issues_no_bitwise_op(self):
        backend = NPUBackend(fallback_to_cpu=True)
        state = factored_statevector(_circuit(), backend)
        mode = _RecordEngineHazards()
        with mode:
            state.to_statevector()
        assert not mode.bitwise, f"稠密化发出位运算: {sorted(set(mode.bitwise))}"

    def test_expectation_issues_no_bitwise_op(self):
        backend = NPUBackend(fallback_to_cpu=True)
        state = factored_statevector(_circuit(), backend)
        ham = Hamiltonian(n_qubits=6, terms=[("ZZ", [0, 1], 1.0), ("Y", [5], 0.5)])
        mode = _RecordEngineHazards()
        with mode:
            factored_expectation(state, ham)
        assert not mode.bitwise, f"期望值发出位运算: {sorted(set(mode.bitwise))}"


class TestEngineDoesNotIndexComplexTensors:
    def test_materialisation_does_not_index_complex_tensors(self):
        """比特序置换必须走 mps._permute_basis 的实/虚部 index_select。"""
        backend = NPUBackend(fallback_to_cpu=True)
        state = factored_statevector(_circuit(), backend)
        mode = _RecordEngineHazards()
        with mode:
            state.to_statevector()
        assert not mode.complex_index, (
            f"对复数张量做了索引（昇腾 aclnnIndex 不支持）: {sorted(set(mode.complex_index))}"
        )


class TestDetectorHasTeeth:
    """反证：拦截器必须能抓到真实违规。"""

    def test_detects_bitwise(self):
        mode = _RecordEngineHazards()
        with mode:
            idx = torch.arange(8, dtype=torch.int64)
            torch.bitwise_xor(idx, torch.bitwise_right_shift(idx, 1))
        assert mode.bitwise, "位运算拦截器没有检测能力"

    def test_detects_complex_indexing(self):
        mode = _RecordEngineHazards()
        with mode:
            values = torch.ones(8, dtype=torch.complex64)
            values.index_select(0, torch.arange(4))
        assert mode.complex_index, "复数索引拦截器没有检测能力"
