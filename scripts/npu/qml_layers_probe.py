#!/usr/bin/env python3
"""真机 NPU qml 层栈探针（成熟化 #4 的设备覆盖）。

tests/qml 用 GPUBackend（cpu-torch）跑，不触 NPU 设备；本探针在 NPUBackend 上
实跑上层 qml 栈，补设备缺口：
  - QFun expval 梯度（select_diff auto + PSR，设备侧期望）
  - QFun probs Jacobian（投影算符期望的参数移位）
  - BatchLayer 前向 + 反向（BatchSV 整批演化，实/虚分离 autograd）
  - build_classifier 一步训练（端到端量子分类器）

每 case 独立兜底。用法：
    python scripts/npu/qml_layers_probe.py [--allow-cpu-fallback]
"""
from __future__ import annotations

import argparse
import traceback

import numpy as np

from aicir import Circuit, Hamiltonian, NPUBackend, PauliString, cx, hadamard, rx, ry, rzz
from aicir.backends.npu_backend import is_npu_available
from aicir.core.circuit import Parameter
from aicir.qml import qfun, expval, probs, BatchLayer, build_classifier


def _backend(allow_cpu_fallback: bool) -> NPUBackend:
    if not allow_cpu_fallback and not is_npu_available():
        raise SystemExit("NPU is unavailable. Use --allow-cpu-fallback only for local script validation.")
    backend = NPUBackend.from_distributed_env(fallback_to_cpu=allow_cpu_fallback)
    if not allow_cpu_fallback and getattr(backend._device, "type", None) != "npu":
        raise AssertionError(f"strict NPU probe resolved device={backend._device!r}, expected npu")
    print(f"[backend] {backend.name}")
    print(f"[runtime] {backend.runtime_context}")
    return backend


def case_qfun_expval_grad(backend):
    obs = Hamiltonian([PauliString("Z", n_qubits=2, qubits=[0])])

    @qfun(device=backend, differential="auto", observable=obs)
    def f(t):
        return Circuit(ry(t[0], 0), cx(1, [0]), ry(t[1], 1), n_qubits=2)

    x = np.array([0.4, -0.7])
    g = np.asarray(f.grad(x), dtype=float)
    if g.shape != (2,) or not np.all(np.isfinite(g)):
        raise AssertionError(f"expval 梯度形状/有限性异常: {g}")


def case_qfun_probs_grad(backend):
    @qfun(device=backend)
    def f(t):
        return probs(Circuit(ry(t[0], 0), rx(t[1], 1), cx(1, [0]), n_qubits=2), wires=[0])

    x = np.array([0.5, 0.3])
    jac = np.asarray(f.grad(x), dtype=float)
    if jac.shape != (2, 2):
        raise AssertionError(f"probs Jacobian 形状 {jac.shape} != (2,2)")
    if not np.allclose(jac.sum(axis=0), 0.0, atol=1e-4):
        raise AssertionError("probs 梯度逐参数求和应为 0（概率守恒）")


def case_batchlayer_forward_backward(backend):
    import torch

    x0, x1 = Parameter("x0"), Parameter("x1")
    w0, w1, w2 = (Parameter(f"w{i}") for i in range(3))
    template = Circuit(rx(x0, 0), rx(x1, 1), ry(w0, 0), ry(w1, 1),
                       rzz(w2, 0, 1), cx(1, [0]), n_qubits=2)
    layer = BatchLayer(template, n_inputs=2, backend=backend)
    xb = torch.as_tensor(np.random.default_rng(0).uniform(-1, 1, size=(8, 2)))
    out = layer(xb)
    if tuple(out.shape) != (8, 2):
        raise AssertionError(f"BatchLayer 输出形状 {tuple(out.shape)} != (8,2)")
    out.sum().backward()
    if layer.weights.grad is None or not bool(torch.all(torch.isfinite(layer.weights.grad))):
        raise AssertionError("BatchLayer 权重梯度缺失/非有限")


def case_classifier_train_step(backend):
    import torch

    model = build_classifier(n_features=2, n_classes=2, backend=backend,
                             n_qubits=2, layers=1, seed=0)
    x = torch.as_tensor(np.random.default_rng(1).uniform(-1, 1, size=(16, 2)), dtype=torch.float32)
    y = torch.as_tensor((x[:, 0] * x[:, 1] > 0).long())
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    lossfn = torch.nn.CrossEntropyLoss()
    first = None
    for _ in range(5):
        opt.zero_grad()
        loss = lossfn(model(x), y)
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.detach())
    if not np.isfinite(float(loss.detach())):
        raise AssertionError("classifier 训练损失非有限")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    args = parser.parse_args()
    backend = _backend(args.allow_cpu_fallback)

    cases = [
        ("qfun_expval_grad", lambda: case_qfun_expval_grad(backend)),
        ("qfun_probs_grad", lambda: case_qfun_probs_grad(backend)),
        ("batchlayer_forward_backward", lambda: case_batchlayer_forward_backward(backend)),
        ("classifier_train_step", lambda: case_classifier_train_step(backend)),
    ]
    passed, failures = 0, []
    for name, fn in cases:
        try:
            fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            first = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
            print(f"[FAIL] {name}: {first}")
            failures.append((name, traceback.format_exc()))
    print(f"qml_layers_probe: {passed}/{len(cases)} cases passed")
    if failures:
        print("\n===== 失败详情 =====")
        for name, tb in failures:
            print(f"\n--- {name} ---\n{tb}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
