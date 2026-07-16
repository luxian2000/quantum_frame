"""端到端量子分类器（成熟化 #6）：encoder → HEA → BatchLayer → torch 训练。

自包含、确定性：numpy 生成 XOR 四象限（非线性可分，需纠缠层），
断言训练收敛到高准确率。sklearn 非必需。
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aicir.backends.gpu_backend import GPUBackend
from aicir.qml import build_classifier


def _xor_data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(np.int64)  # 四象限 XOR
    return x, y


def _train(model, x, y, *, epochs=80, lr=0.1):
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.CrossEntropyLoss()
    first = None
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossfn(model(xt), yt)
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.detach())
    return first, float(loss.detach())


def _accuracy(model, x, y):
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32))
        pred = logits.argmax(dim=1).cpu().numpy()
    return float((pred == y).mean())


def test_classifier_learns_xor():
    torch.manual_seed(0)
    model = build_classifier(n_features=2, n_classes=2, backend=GPUBackend(),
                             n_qubits=2, layers=2, seed=0)
    x, y = _xor_data(80, seed=1)
    first, last = _train(model, x, y)
    assert last < first  # 损失下降
    assert _accuracy(model, x, y) > 0.85  # 训练集拟合非线性 XOR

    # 泛化到独立测试集
    xt, yt = _xor_data(60, seed=2)
    assert _accuracy(model, xt, yt) > 0.80


def test_classifier_is_nn_module_and_composable():
    # 是标准 nn.Module，可嵌入 Sequential，参数含量子权重
    model = build_classifier(n_features=2, n_classes=2, backend=GPUBackend(),
                             n_qubits=2, layers=1, seed=0)
    assert isinstance(model, torch.nn.Module)
    assert sum(p.numel() for p in model.parameters()) > 0
    out = model(torch.randn(5, 2))
    assert out.shape == (5, 2)
    out.sum().backward()
    assert any(p.grad is not None for p in model.parameters())
