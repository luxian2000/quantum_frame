from __future__ import annotations

"""一个使用 aicir 实现的 6 比特 MNIST QCNN 示例。

这个 demo 读取本地 MNIST idx 数据集，把 28x28 图像平均池化到 7x7，
再用 amplitude encoding 装载到 6 个量子比特上。QCNN 本体采用两层局部
卷积 + 池化结构，量子线路输出 64 维概率特征，再接一个 10 类 softmax
线性头，用交叉熵训练 MNIST 分类器。

从仓库根目录运行：

    python -m demos.mnist.qcnn_demo
    python -m demos.mnist.qcnn_demo --steps 40 --lr 0.05 --show
"""

import struct
from pathlib import Path

import argparse
from dataclasses import dataclass
import pickle

import numpy as np

from aicir import Circuit, NumpyBackend, State, cnot, cry, ry, rz
from aicir.encoder.amplitude import AmplitudeEncoder
from aicir.optimizer import Adam
from aicir.visual import plot

from .._visual_demo_utils import add_common_visual_args, configure_matplotlib, save_figure


DATA_ROOT = Path("/Volumes/Right/DataSpace/MNIST")
OUTPUT_DIR = Path(__file__).resolve().parent
N_QUBITS = 6
N_FEATURES = 49
N_PARAMS = 24
DEFAULT_STEPS = 80
DEFAULT_LR = 0.03
DEFAULT_TRAIN_PER_CLASS = 20
DEFAULT_TEST_PER_CLASS = 8
DEFAULT_SEED = 7
DEFAULT_SPSA_EPS = 0.03
DEFAULT_SPSA_SAMPLES = 8
DEFAULT_DIGITS = tuple(range(10))
N_CLASSES = 10
FEATURE_DIM = 1 << N_QUBITS
N_HEAD_PARAMS = FEATURE_DIM * N_CLASSES + N_CLASSES
N_TOTAL_PARAMS = N_PARAMS + N_HEAD_PARAMS
DEFAULT_L2 = 1e-4


@dataclass(frozen=True)
class DemoData:
    train_images: np.ndarray
    train_states: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_states: np.ndarray
    test_labels: np.ndarray
    digits: tuple[int, ...]


def _load_idx_images(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"{path.name} 不是 IDX image 文件")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    expected = count * rows * cols
    if data.size != expected:
        raise ValueError(f"{path.name} 数据长度不匹配：期望 {expected}，实际 {data.size}")
    return data.reshape(count, rows, cols)


def _load_idx_labels(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"{path.name} 不是 IDX label 文件")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    if data.size != count:
        raise ValueError(f"{path.name} 标签数量不匹配：期望 {count}，实际 {data.size}")
    return data.reshape(count)


def _parse_digits(raw: str | None) -> tuple[int, ...]:
    if raw is None or not raw.strip():
        return DEFAULT_DIGITS
    digits = tuple(dict.fromkeys(int(part) for part in raw.replace(";", ",").split(",") if part.strip()))
    if not digits:
        raise ValueError("--digits 不能为空")
    for digit in digits:
        if digit < 0 or digit > 9:
            raise ValueError("--digits 只能包含 0 到 9")
    return digits


def _downsample_to_7x7(image: np.ndarray) -> np.ndarray:
    if image.shape != (28, 28):
        raise ValueError(f"MNIST image shape must be 28x28, got {image.shape}")
    pooled = image.astype(np.float32).reshape(7, 4, 7, 4).mean(axis=(1, 3))
    return pooled / 255.0


def _select_subset(
    images: np.ndarray,
    labels: np.ndarray,
    digits: tuple[int, ...],
    per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    selected_images: list[np.ndarray] = []
    selected_labels: list[int] = []
    for digit in digits:
        indices = np.flatnonzero(labels == digit)
        if indices.size < per_class:
            raise ValueError(f"digit {digit} 只有 {indices.size} 个样本，不足 {per_class}")
        choice = rng.choice(indices, size=per_class, replace=False)
        selected_images.extend(images[choice])
        selected_labels.extend([digit] * per_class)
    order = rng.permutation(len(selected_labels))
    return np.asarray(selected_images, dtype=np.uint8)[order], np.asarray(selected_labels, dtype=np.int64)[order]


def _precompute_encoded_states(images: np.ndarray, backend: NumpyBackend) -> np.ndarray:
    encoder = AmplitudeEncoder(n_qubits=N_QUBITS)
    states: list[np.ndarray] = []
    for image in images:
        features = _downsample_to_7x7(image).reshape(-1)
        _, state = encoder.encode(features, cir="dict", backend=backend)
        states.append(state.to_numpy().reshape(-1).astype(np.complex64))
    return np.asarray(states, dtype=np.complex64)


def prepare_mnist_dataset(
    data_root: Path,
    digits: tuple[int, ...],
    train_per_class: int,
    test_per_class: int,
    rng: np.random.Generator,
    backend: NumpyBackend,
) -> DemoData:
    train_images = _load_idx_images(data_root / "raw" / "train-images-idx3-ubyte")
    train_labels = _load_idx_labels(data_root / "raw" / "train-labels-idx1-ubyte")
    test_images = _load_idx_images(data_root / "raw" / "t10k-images-idx3-ubyte")
    test_labels = _load_idx_labels(data_root / "raw" / "t10k-labels-idx1-ubyte")

    train_images, train_labels = _select_subset(train_images, train_labels, digits, train_per_class, rng)
    test_images, test_labels = _select_subset(test_images, test_labels, digits, test_per_class, rng)

    train_states = _precompute_encoded_states(train_images, backend)
    test_states = _precompute_encoded_states(test_images, backend)

    return DemoData(
        train_images=train_images,
        train_states=train_states,
        train_labels=train_labels,
        test_images=test_images,
        test_states=test_states,
        test_labels=test_labels,
        digits=digits,
    )


def build_qcnn_circuit(params: np.ndarray) -> Circuit:
    """构造一个 6-qubit 的三层 QCNN 主体电路。"""

    theta = np.asarray(params, dtype=float).reshape(-1)
    if theta.shape[0] != N_PARAMS:
        raise ValueError(f"params must have length {N_PARAMS}")

    gates: list = []

    def conv_block(left: int, right: int, offset: int) -> None:
        gates.extend(
            [
                ry(float(theta[offset + 0]), left),
                rz(float(theta[offset + 1]), right),
                cnot(right, [left]),
                ry(float(theta[offset + 2]), left),
                rz(float(theta[offset + 3]), right),
            ]
        )

    def pool_block(left: int, right: int, offset: int) -> None:
        gates.extend(
            [
                cry(float(theta[offset + 0]), left, [right]),
                rz(float(theta[offset + 1]), left),
            ]
        )

    for left, right in ((0, 1), (2, 3), (4, 5)):
        conv_block(left, right, 0)
        pool_block(left, right, 4)

    for left, right in ((0, 2), (2, 4)):
        conv_block(left, right, 6)
        pool_block(left, right, 10)

    conv_block(0, 4, 12)
    pool_block(0, 4, 16)

    for qubit, offset in enumerate(range(18, 24)):
        gates.append(ry(float(theta[offset]), qubit))

    return Circuit(*gates, n_qubits=N_QUBITS)


def _quantum_feature_vector(amplitudes: np.ndarray) -> np.ndarray:
    return np.abs(amplitudes.reshape(-1)) ** 2


def _split_theta(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.shape[0] != N_TOTAL_PARAMS:
        raise ValueError(f"params must have length {N_TOTAL_PARAMS}")
    q_params = theta[:N_PARAMS]
    head = theta[N_PARAMS:]
    head_w = head[: FEATURE_DIM * N_CLASSES].reshape(FEATURE_DIM, N_CLASSES)
    head_b = head[FEATURE_DIM * N_CLASSES :]
    return q_params, head_w, head_b


def _head_l2(params: np.ndarray) -> float:
    _, head_w, _ = _split_theta(params)
    return float(np.sum(head_w * head_w))


def predict_feature_matrix(states: np.ndarray, q_params: np.ndarray, backend: NumpyBackend) -> np.ndarray:
    unitary = build_qcnn_circuit(q_params).unitary()
    outputs: list[np.ndarray] = []
    for encoded_state in states:
        state = State.from_array(encoded_state, n_qubits=N_QUBITS, backend=backend).evolve(unitary)
        outputs.append(_quantum_feature_vector(state.to_numpy()))
    return np.asarray(outputs, dtype=float)


def predict_logits(states: np.ndarray, params: np.ndarray, backend: NumpyBackend) -> np.ndarray:
    q_params, head_w, head_b = _split_theta(params)
    quantum_features = predict_feature_matrix(states, q_params, backend)
    return quantum_features @ head_w + head_b


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    log_probs = shifted - log_sum_exp
    return float(-np.mean(log_probs[np.arange(labels.size), labels]))


def dataset_loss(
    states: np.ndarray,
    labels: np.ndarray,
    params: np.ndarray,
    backend: NumpyBackend,
    *,
    l2: float = 0.0,
) -> float:
    logits = predict_logits(states, params, backend)
    return cross_entropy_loss(logits, labels) + float(l2) * _head_l2(params)


def predict_digits(states: np.ndarray, params: np.ndarray, backend: NumpyBackend, digits: tuple[int, ...]) -> np.ndarray:
    logits = predict_logits(states, params, backend)
    class_indices = np.argmax(logits, axis=1)
    return np.asarray(class_indices, dtype=int)


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(predictions == labels))


def summarize_dataset(train_labels: np.ndarray, test_labels: np.ndarray, digits: tuple[int, ...]) -> None:
    print(f"  digits       : {', '.join(str(d) for d in digits)}")
    print(f"  train samples: {len(train_labels)}")
    print(f"  test  samples: {len(test_labels)}")
    train_counts = {digit: int(np.sum(train_labels == digit)) for digit in digits}
    test_counts = {digit: int(np.sum(test_labels == digit)) for digit in digits}
    print(f"  train dist   : {train_counts}")
    print(f"  test  dist   : {test_counts}")


def save_model(output_dir: Path, params: np.ndarray, metadata: dict[str, object]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "qcnn_mnist_model.pkl"
    payload: dict[str, np.ndarray] = {"params": np.asarray(params, dtype=np.float32)}
    for key, value in metadata.items():
        payload[key] = np.asarray(value)
    with model_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return model_path


def plot_test_samples(
    images: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    output_dir: Path,
    *,
    filename: str = "qcnn_mnist_test_predictions.png",
    seed: int = DEFAULT_SEED,
) -> Path:
    count = min(10, len(images))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(images), size=count, replace=False)

    plt = configure_matplotlib(False)
    import matplotlib.pyplot as mpl_plt

    fig, axes = mpl_plt.subplots(2, 5, figsize=(12.0, 5.2))
    axes = np.asarray(axes).reshape(-1)
    for axis, index in zip(axes, indices, strict=False):
        axis.imshow(images[index], cmap="gray_r")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"T:{int(true_labels[index])}  P:{int(predicted_labels[index])}", fontsize=10)
    for axis in axes[count:]:
        axis.axis("off")
    fig.suptitle("MNIST QCNN test predictions", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = save_figure(fig, output_dir, filename)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="6-qubit MNIST QCNN demo built with aicir.")
    add_common_visual_args(parser)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="MNIST root directory containing raw/.")
    parser.add_argument("--digits", type=str, default=",".join(str(d) for d in DEFAULT_DIGITS), help="Comma-separated digits to use, e.g. 0,1,2,3.")
    parser.add_argument("--train-per-class", type=int, default=DEFAULT_TRAIN_PER_CLASS, help="Training samples per digit.")
    parser.add_argument("--test-per-class", type=int, default=DEFAULT_TEST_PER_CLASS, help="Test samples per digit.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Optimizer steps.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate.")
    parser.add_argument("--spsa-eps", type=float, default=DEFAULT_SPSA_EPS, help="SPSA perturbation magnitude.")
    parser.add_argument("--spsa-samples", type=int, default=DEFAULT_SPSA_SAMPLES, help="Number of SPSA perturbation directions.")
    parser.add_argument("--l2", type=float, default=DEFAULT_L2, help="L2 regularization on softmax head weights.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    args = parser.parse_args()

    backend = NumpyBackend()
    rng = np.random.default_rng(args.seed)
    digits = _parse_digits(args.digits)
    data = prepare_mnist_dataset(
        args.data_root,
        digits,
        args.train_per_class,
        args.test_per_class,
        rng,
        backend,
    )
    init_params = rng.normal(0.0, 0.05, size=N_TOTAL_PARAMS)
    output_dir = OUTPUT_DIR

    print("=== MNIST QCNN demo ===")
    print(f"  data root   : {args.data_root}")
    print(f"  qubits      : {N_QUBITS}")
    print(f"  qcnn params : {N_PARAMS}")
    print(f"  head params : {N_HEAD_PARAMS}")
    print(f"  features    : {FEATURE_DIM} (full quantum probability vector)")
    print(f"  steps / lr  : {args.steps} / {args.lr}")
    print(f"  l2 penalty  : {args.l2}")
    summarize_dataset(data.train_labels, data.test_labels, data.digits)

    def objective(theta: np.ndarray) -> float:
        return dataset_loss(data.train_states, data.train_labels, theta, backend, l2=args.l2)

    optimizer = Adam(
        max_iters=args.steps,
        learning_rate=args.lr,
        gradient_method="spsa",
        gradient_kwargs={"eps": args.spsa_eps, "n_samples": args.spsa_samples, "rng": np.random.default_rng(args.seed + 1)},
    )
    result = optimizer.minimize(objective, init_params)
    best_params = np.asarray(result.best_x, dtype=float)

    train_logits = predict_logits(data.train_states, best_params, backend)
    test_logits = predict_logits(data.test_states, best_params, backend)
    test_probs = _softmax(test_logits)
    train_loss = dataset_loss(data.train_states, data.train_labels, best_params, backend, l2=args.l2)
    test_loss = dataset_loss(data.test_states, data.test_labels, best_params, backend, l2=args.l2)
    train_pred = predict_digits(data.train_states, best_params, backend, data.digits)
    test_pred = predict_digits(data.test_states, best_params, backend, data.digits)
    qcnn_params, _, _ = _split_theta(best_params)

    print()
    print("=== Training result ===")
    print(f"  final loss : {result.fun:.6f}")
    print(f"  train loss : {train_loss:.6f}")
    print(f"  test  loss : {test_loss:.6f}")
    print(f"  train acc  : {accuracy(train_pred, data.train_labels):.3f}")
    print(f"  test  acc  : {accuracy(test_pred, data.test_labels):.3f}")
    print(f"  nfev       : {result.nfev}")

    model_path = save_model(
        output_dir,
        best_params,
        {
            "digits": np.asarray(data.digits, dtype=np.int64),
            "train_labels": data.train_labels,
            "test_labels": data.test_labels,
            "train_loss": np.asarray([train_loss], dtype=np.float32),
            "test_loss": np.asarray([test_loss], dtype=np.float32),
            "train_acc": np.asarray([accuracy(train_pred, data.train_labels)], dtype=np.float32),
            "test_acc": np.asarray([accuracy(test_pred, data.test_labels)], dtype=np.float32),
            "steps": np.asarray([args.steps], dtype=np.int64),
            "learning_rate": np.asarray([args.lr], dtype=np.float32),
            "l2": np.asarray([args.l2], dtype=np.float32),
        },
    )
    print(f"Saved model      : {model_path}")

    print()
    print("=== Sample predictions ===")
    for index in range(min(5, len(data.test_labels))):
        label = int(data.test_labels[index])
        pred = int(test_pred[index])
        class_probs = ", ".join(f"{digit}:{value:.2f}" for digit, value in zip(range(N_CLASSES), test_probs[index], strict=False))
        print(f"  sample {index:02d}: y={label}  pred={pred}  probs=[{class_probs}]")

    fig = plot(
        build_qcnn_circuit(qcnn_params),
        path=output_dir,
        name="qcnn_mnist_circuit",
        title="6-qubit MNIST QCNN feature extractor",
    )
    if fig is not None:
        print(f"\nSaved circuit figure: {output_dir / 'qcnn_mnist_circuit.png'}")

    test_grid_path = plot_test_samples(
        data.test_images,
        data.test_labels,
        test_pred,
        output_dir,
        seed=args.seed + 11,
    )
    print(f"Saved test grid     : {test_grid_path}")

    plt = configure_matplotlib(args.show)
    import matplotlib.pyplot as mpl_plt

    history = [entry["fun"] for entry in result.history]
    if history:
        history.append(result.fun)
        fig, ax = mpl_plt.subplots(figsize=(6.2, 3.6))
        ax.plot(history, color="#1565C0", linewidth=2)
        ax.set_title("QCNN training loss")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cross-entropy loss")
        ax.grid(alpha=0.3)
        save_figure(fig, output_dir, "qcnn_mnist_loss.png")
        print(f"Saved loss curve: {output_dir / 'qcnn_mnist_loss.png'}")
        if args.show:
            plt.show()
        else:
            plt.close(fig)
    elif args.show:
        plt.show()


if __name__ == "__main__":
    main()