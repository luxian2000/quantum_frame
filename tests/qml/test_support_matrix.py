"""qml 梯度支持矩阵的可执行契约（成熟化 #4）。

把「方法 × 后端族 × shots × 噪声 × 线路生成元谱 → select_diff 选择」这一散落
在 DiffMethod capability 标志 + README §1 指南里的约定，固化为一张被测表；每个
「支持」单元都有对应断言，防止能力标志与实际分发漂移。
"""

import numpy as np
import pytest

from aicir import Circuit, Hamiltonian, PauliString, cx, hadamard, rx, ry, single_excitation
from aicir.backends.numpy_backend import NumpyBackend
from aicir.qml import qfun, expval, resolve_diff, select_diff
from aicir.qml.diff import registered_diffs


def _torch_backend():
    torch = pytest.importorskip("torch")
    from aicir.backends.gpu_backend import GPUBackend
    return GPUBackend()


_ROT = Circuit(rx(0.1, 0), ry(0.2, 1), n_qubits=2)
_EXC = Circuit(hadamard(1), single_excitation(0.3, 0, 1), n_qubits=2)
_MIXED = Circuit(rx(0.1, 0), single_excitation(0.3, 0, 1), n_qubits=2)


# (backend_kind, shots, noisy, circuit, expected) — numpy 后端族契约
_NUMPY_MATRIX = [
    ("numpy", None, False, None, "psr"),
    ("numpy", None, False, _ROT, "psr"),
    ("numpy", None, False, _EXC, "psr4"),      # 激发门谱 {-1,0,1} → psr4
    ("numpy", None, False, _MIXED, "fd"),      # 混合谱 → fd（psr/psr4 均非逐参数正确）
    ("numpy", 128, False, None, "psr"),        # psr 支持 shots
    ("numpy", None, True, None, "psr"),        # psr 支持 noise
]


@pytest.mark.parametrize("kind,shots,noisy,circuit,expected", _NUMPY_MATRIX)
def test_select_diff_matrix_numpy(kind, shots, noisy, circuit, expected):
    got = select_diff(backend=NumpyBackend(), shots=shots, noisy=noisy, circuit=circuit)
    assert got == expected


# torch 后端族契约：auto（伴随 AD）优先，但 shots/noise 使其失格降级
_TORCH_MATRIX = [
    (None, False, None, "auto"),      # torch 精确 AD
    (128, False, None, "psr"),        # auto 不支持 shots → psr
    (None, True, None, "psr"),        # auto 不支持 noise → psr
    (None, False, "exc", "auto"),     # 激发线路但 chosen=auto（AD 对任意门精确），不升级
    (128, False, "exc", "psr4"),      # auto 失格→psr，激发线路升级 psr4
]


@pytest.mark.parametrize("shots,noisy,circ_key,expected", _TORCH_MATRIX)
def test_select_diff_matrix_torch(shots, noisy, circ_key, expected):
    backend = _torch_backend()
    circuit = _EXC if circ_key == "exc" else None
    got = select_diff(backend=backend, shots=shots, noisy=noisy, circuit=circuit)
    assert got == expected


def test_all_fn_gradient_methods_resolvable():
    # 每个 fn_gradient 方法名都能经注册表解析为可调用
    names = registered_diffs(category="fn_gradient")
    assert names  # 非空
    for name in names:
        assert callable(resolve_diff(name))


@pytest.mark.parametrize("method", ["psr", "psr4", "fd", "spsa", "spsr"])
def test_declared_methods_compute_finite_gradient(method):
    # 每个显式方法在合适线路上算出有限梯度（psr4 用激发线路，其余用旋转线路）
    obs = Hamiltonian([PauliString("X", n_qubits=2, qubits=[0])])
    use_exc = method == "psr4"

    kwargs = {}
    if method in ("spsa", "spsr"):
        kwargs["rng"] = np.random.default_rng(0)

    @qfun(device="numpy", differential=method, observable=obs)
    def f(t):
        if use_exc:
            return Circuit(hadamard(1), single_excitation(t[0], 0, 1), n_qubits=2)
        return Circuit(rx(t[0], 0), ry(t[1], 1), cx(1, [0]), n_qubits=2)

    x = np.array([0.5]) if use_exc else np.array([0.5, -0.3])
    # spsa/spsr 经 differential 名解析，无需 kwargs（用默认）；此处只验有限性
    g = np.asarray(f.grad(x), dtype=float)
    assert g.shape == x.shape
    assert np.all(np.isfinite(g))


def test_torch_only_method_filtered_on_numpy():
    # auto 需要 torch：numpy 后端绝不选它
    assert select_diff(backend=NumpyBackend()) != "auto"


def test_qfun_auto_on_torch_backend_uses_psr_family():
    # QFun.grad 为 float 接口，无法驱动 torch-AD auto；device="gpu"+auto 应退回
    # 参数移位家族并算出有限梯度，而非崩溃。
    torch = pytest.importorskip("torch")
    obs = Hamiltonian([PauliString("Z", n_qubits=2, qubits=[0])])

    @qfun(device="gpu", differential="auto", observable=obs)
    def f(t):
        return Circuit(ry(t[0], 0), cx(1, [0]), ry(t[1], 1), n_qubits=2)

    g = np.asarray(f.grad(np.array([0.4, -0.7])), dtype=float)
    assert g.shape == (2,)
    assert np.all(np.isfinite(g))


def test_qfun_backend_instance_accepted_as_device():
    # device 可直接传后端实例（与 BatchLayer/build_classifier 一致）
    obs = Hamiltonian([PauliString("Z", n_qubits=1, qubits=[0])])

    @qfun(device=NumpyBackend(), differential="psr", observable=obs)
    def f(t):
        return Circuit(ry(t[0], 0), n_qubits=1)

    val = float(f(np.array([0.5])))
    assert np.isfinite(val)
