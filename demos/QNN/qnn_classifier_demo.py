"""Demo: 端到端量子分类器（qml 成熟化 #6）。

用 ``aicir.qml.build_classifier`` 训练一个混合量子-经典分类器：
角度编码 → 硬件高效纠缠层（BatchLayer 批量演化）→ 逐比特 <Z_q> → 线性头。
量子层走 BatchSV（实/虚分离、NPU 安全、原生 autograd），故可直接用 torch
优化器训练、可放上 NPU/GPU 大 batch 跑。

默认用 numpy 合成的 XOR 四象限数据（非线性可分，需纠缠层，无需 sklearn）；
装了 sklearn 时可 ``--dataset moons`` 换成 make_moons。

仓库根目录运行：
    python -m demos.QNN.qnn_classifier_demo
    python -m demos.QNN.qnn_classifier_demo --dataset moons --epochs 120
    python -m demos.QNN.qnn_classifier_demo --device npu     # 真机 NPU
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from aicir.qml import build_classifier


def _xor_data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(np.int64)
    return x, y


def _moons_data(n, seed):
    from sklearn.datasets import make_moons  # 可选依赖

    x, y = make_moons(n_samples=n, noise=0.15, random_state=seed)
    x = (x - x.mean(0)) / x.std(0)  # 标准化到旋转角度友好范围
    return x.astype(np.float64), y.astype(np.int64)


def _make_backend(device: str):
    if device == "npu":
        from aicir import NPUBackend

        return NPUBackend.from_distributed_env(fallback_to_cpu=False)
    from aicir.backends.gpu_backend import GPUBackend

    return GPUBackend()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["xor", "moons"], default="xor")
    parser.add_argument("--n-train", type=int, default=120)
    parser.add_argument("--n-test", type=int, default=80)
    parser.add_argument("--n-qubits", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--device", choices=["cpu", "npu"], default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    load = _moons_data if args.dataset == "moons" else _xor_data
    x_tr, y_tr = load(args.n_train, args.seed)
    x_te, y_te = load(args.n_test, args.seed + 1)

    torch.manual_seed(args.seed)
    model = build_classifier(
        n_features=x_tr.shape[1], n_classes=2, backend=_make_backend(args.device),
        n_qubits=args.n_qubits, layers=args.layers, seed=args.seed,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossfn = torch.nn.CrossEntropyLoss()
    xt = torch.as_tensor(x_tr, dtype=torch.float32)
    yt = torch.as_tensor(y_tr, dtype=torch.long)

    for epoch in range(args.epochs):
        opt.zero_grad()
        loss = lossfn(model(xt), yt)
        loss.backward()
        opt.step()
        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  loss {float(loss.detach()):.4f}")

    def acc(x, y):
        with torch.no_grad():
            pred = model(torch.as_tensor(x, dtype=torch.float32)).argmax(1).cpu().numpy()
        return float((pred == y).mean())

    print(f"\ndataset={args.dataset} device={args.device} "
          f"n_qubits={args.n_qubits} layers={args.layers}")
    print(f"train accuracy: {acc(x_tr, y_tr):.3f}")
    print(f"test  accuracy: {acc(x_te, y_te):.3f}")


if __name__ == "__main__":
    main()
