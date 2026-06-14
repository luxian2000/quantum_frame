"""Tests for aicir 的批量态矢量模拟路径 (BatchSV)。

该路径为上层(例如 csinet 的量子补偿块)提供:
- 一次性模拟一批态矢量 (batch),
- 支持逐样本 (per-sample) 的旋转门角度,
- NPU 安全 (全程实部/虚部实张量, 不出现 complex64 加/乘),
- 自动求导可用。

正确性以 aicir 单态路径 (apply_gate_to_state) 逐样本循环为基准。
"""

import math
import unittest

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from aicir import BatchSV, hadamard, ry, crz, pauli_x
from aicir import NPUBackend
from aicir.core.gates import apply_gate_to_state


class _BanComplexAddMul(TorchDispatchMode):
    """在 CPU 上复现 Ascend NPU 缺失的 complex64 加/乘内核 (与 test_npu_backend 一致)。"""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        if "add" in name or "mul" in name or "sub" in name:
            for value in list(args) + list(kwargs.values()):
                if isinstance(value, torch.Tensor) and torch.is_complex(value):
                    raise RuntimeError(f"unsupported complex op on NPU: {name}")
        return func(*args, **kwargs)


def _z_expectations_reference(backend, n_qubits, per_sample_gates):
    """逐样本走 aicir 单态路径, 返回 (batch, n_qubits) 的 <Z_q>。"""
    batch = len(per_sample_gates)
    out = np.zeros((batch, n_qubits), dtype=np.float64)
    # aicir 约定: qubit 0 为最高位, qubit q 的比特权重为 2^(n-1-q)。
    idx = np.arange(1 << n_qubits)
    zsign = np.stack(
        [1.0 - 2.0 * ((idx >> (n_qubits - 1 - q)) & 1) for q in range(n_qubits)],
        axis=0,
    )  # (n, D)
    for b, gates in enumerate(per_sample_gates):
        state = backend.zeros_state(n_qubits)
        for gate in gates:
            state = apply_gate_to_state(gate, state, n_qubits, backend)
        probs = backend.to_numpy(backend.measure_probs(state)).reshape(-1)
        out[b] = zsign @ probs
    return out


class TestBatchSV(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self.backend = NPUBackend(fallback_to_cpu=True)

    def test_all_zero_state_z_is_plus_one(self):
        bsv = BatchSV(n_qubits=3, batch_size=4, backend=self.backend)
        z = bsv.z_expectations()
        self.assertEqual(tuple(z.shape), (4, 3))
        self.assertTrue(torch.allclose(z, torch.ones_like(z), atol=1e-6))

    def test_pauli_x_flips_z(self):
        bsv = BatchSV(n_qubits=3, batch_size=2, backend=self.backend)
        bsv.apply_gate(pauli_x(1))
        z = bsv.z_expectations().detach().cpu().numpy()
        expected = np.array([[1.0, -1.0, 1.0]] * 2)
        self.assertTrue(np.allclose(z, expected, atol=1e-6))

    def test_hadamard_all_zero_z(self):
        bsv = BatchSV(n_qubits=4, batch_size=3, backend=self.backend)
        for q in range(4):
            bsv.apply_gate(hadamard(q))
        z = bsv.z_expectations().detach().cpu().numpy()
        self.assertTrue(np.allclose(z, 0.0, atol=1e-6))

    def test_matches_single_state_reference_per_sample_angles(self):
        n, batch = 4, 5
        # 逐样本不同的编码角度。
        enc = torch.randn(batch, n)
        crz_theta = torch.randn(n)
        ry_theta = torch.randn(n)

        bsv = BatchSV(n_qubits=n, batch_size=batch, backend=self.backend)
        for q in range(n):
            bsv.apply_gate(hadamard(q))
            bsv.apply_gate(ry(enc[:, q], q))
        for q in range(n):
            target = (q + 1) % n
            bsv.apply_gate(crz(crz_theta[q], target, [q]))
        for q in range(n):
            bsv.apply_gate(ry(ry_theta[q], q))
        z_batched = bsv.z_expectations().detach().cpu().numpy()

        per_sample_gates = []
        for b in range(batch):
            gates = []
            for q in range(n):
                gates.append(hadamard(q))
                gates.append(ry(float(enc[b, q]), q))
            for q in range(n):
                target = (q + 1) % n
                gates.append(crz(float(crz_theta[q]), target, [q]))
            for q in range(n):
                gates.append(ry(float(ry_theta[q]), q))
            per_sample_gates.append(gates)
        z_ref = _z_expectations_reference(self.backend, n, per_sample_gates)

        self.assertTrue(np.allclose(z_batched, z_ref, atol=1e-5),
                        msg=f"max diff={np.abs(z_batched - z_ref).max()}")

    def test_autograd_matches_finite_difference(self):
        n, batch = 3, 2
        theta = torch.tensor(0.7, dtype=torch.float32, requires_grad=True)

        def loss_fn(t):
            bsv = BatchSV(n_qubits=n, batch_size=batch, backend=self.backend)
            for q in range(n):
                bsv.apply_gate(hadamard(q))
            bsv.apply_gate(crz(t, 1, [0]))
            bsv.apply_gate(ry(t, 2))
            return bsv.z_expectations().sum()

        loss = loss_fn(theta)
        loss.backward()
        grad = float(theta.grad)

        eps = 1e-3
        with torch.no_grad():
            fp = float(loss_fn(torch.tensor(0.7 + eps)))
            fm = float(loss_fn(torch.tensor(0.7 - eps)))
        fd = (fp - fm) / (2 * eps)
        self.assertAlmostEqual(grad, fd, places=3)

    def test_backward_has_no_complex_ops(self):
        # NPU 守卫: 反向不得出现 complex64 加/乘 (整条路径应是实张量)。
        n, batch = 3, 2
        enc = torch.randn(batch, n, requires_grad=True)
        theta = torch.zeros(n, requires_grad=True)

        bsv = BatchSV(n_qubits=n, batch_size=batch, backend=self.backend)
        for q in range(n):
            bsv.apply_gate(hadamard(q))
            bsv.apply_gate(ry(enc[:, q], q))
        for q in range(n):
            bsv.apply_gate(crz(theta[q], (q + 1) % n, [q]))
        loss = bsv.z_expectations().pow(2).sum()
        with _BanComplexAddMul():
            loss.backward()  # 不得抛错
        self.assertIsNotNone(enc.grad)
        self.assertIsNotNone(theta.grad)


if __name__ == "__main__":
    unittest.main()
