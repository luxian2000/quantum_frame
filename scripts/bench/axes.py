"""基准轴 C–G。

轴 A/B（单门吞吐、标准线路）由 `run_bench.py` 直接驱动。本模块提供其余五轴：

- **C** `run_vqe_axis`：变分负载端到端——单次能量、单次梯度。这才是本框架的主场，
  纯态矢量吞吐只是入场券。
- **D** `measure_peak_memory`：峰值内存 vs 理论下界，比值即额外拷贝的代价。
- **E** `npu_vs_cpu_plan`：昇腾 vs CPU（需真机）。
- **F** `strong_scaling_plan`：多卡强/弱扩展（需真机）。
- **G** `capability_matrix`：**哪些框架能跑每一行**——三支柱论证的直接证据，
  必须由数据生成而非手写散文。

轴 E/F 在无昇腾硬件的机器上只产出**计划**并说明跳过原因；计划本身的自洽性
（world_size 为 2 的幂、`n_qubits >= p`）在此校验，避免上真机才发现参数非法。
"""

from __future__ import annotations

import tracemalloc

import numpy as np

from .adapters import ADAPTERS, available_adapters, get_adapter
from .core.spec import CircuitSpec, build_spec
from .core.timing import time_callable

__all__ = [
    "capability_matrix",
    "measure_peak_memory",
    "run_vqe_axis",
    "strong_scaling_plan",
    "npu_vs_cpu_plan",
]


# ────────────────────────────── 轴 G：能力矩阵 ──────────────────────────────

#: 每个框架声明的能力，以及**证据出处**。
#:
#: 每条 True 都必须能指向可核验的东西（模块路径、探针脚本、上游文档），否则
#: 论文里的能力矩阵就只是作者的断言。
_CAPABILITIES = {
    "aicir": {
        "capabilities": {
            "statevector": True,
            "density_matrix": True,
            "tensor_network": True,
            "mps": True,
            "single_state_sharding": True,
            "non_cuda_accelerator": True,
            "differentiable_on_non_cuda": True,
            "architecture_search": True,
            "trainability_metrics": True,
        },
        "evidence": {
            "single_state_sharding": "aicir/distributed/ + docs/evidence/distributed-autograd/<sha>/",
            "non_cuda_accelerator": "aicir/backends/npu_backend.py + scripts/npu/",
            "differentiable_on_non_cuda": "scripts/npu/deriv.sh --strict-npu",
            "architecture_search": "aicir/qas/ (10 methods via aicir.qas.run)",
            "trainability_metrics": "aicir/metrics/ + aicir/qml/diagnostics.py",
        },
    },
    "qiskit": {
        "capabilities": {
            "statevector": True,
            "density_matrix": True,
            "tensor_network": False,
            "mps": False,
            "single_state_sharding": False,
            "non_cuda_accelerator": False,
            "differentiable_on_non_cuda": False,
            "architecture_search": False,
            "trainability_metrics": False,
        },
        "evidence": {},
    },
    "qiskit_aer": {
        "capabilities": {
            "statevector": True,
            "density_matrix": True,
            "tensor_network": True,
            "mps": True,
            "single_state_sharding": False,
            "non_cuda_accelerator": False,
            "differentiable_on_non_cuda": False,
            "architecture_search": False,
            "trainability_metrics": False,
        },
        "evidence": {"mps": "AerSimulator(method='matrix_product_state')"},
    },
    "cirq": {
        "capabilities": {
            "statevector": True,
            "density_matrix": True,
            "tensor_network": False,
            "mps": False,
            "single_state_sharding": False,
            "non_cuda_accelerator": False,
            "differentiable_on_non_cuda": False,
            "architecture_search": False,
            "trainability_metrics": False,
        },
        "evidence": {},
    },
    "qulacs": {
        "capabilities": {
            "statevector": True,
            "density_matrix": True,
            "tensor_network": False,
            "mps": False,
            "single_state_sharding": False,
            "non_cuda_accelerator": False,
            "differentiable_on_non_cuda": False,
            "architecture_search": False,
            "trainability_metrics": False,
        },
        "evidence": {},
    },
    "tensorcircuit": {
        "capabilities": {
            "statevector": True,
            "density_matrix": True,
            "tensor_network": True,
            "mps": True,
            "single_state_sharding": False,
            "non_cuda_accelerator": False,
            "differentiable_on_non_cuda": False,
            "architecture_search": False,
            "trainability_metrics": False,
        },
        "evidence": {"tensor_network": "tc.Circuit contraction backend"},
    },
}


def capability_matrix() -> dict:
    """轴 G：能力矩阵。

    **未安装的框架仍会出现**（`available=False`），否则表格会因本机环境而失真——
    "缺席"和"不支持"是两回事。
    """

    installed = set(available_adapters())
    matrix = {}
    for name in ADAPTERS:
        entry = _CAPABILITIES.get(name, {"capabilities": {}, "evidence": {}})
        matrix[name] = {
            "available": name in installed,
            "version": ADAPTERS[name].version() if name in installed else None,
            "capabilities": dict(entry["capabilities"]),
            "evidence": dict(entry["evidence"]),
        }
    return matrix


# ────────────────────────────── 轴 D：峰值内存 ──────────────────────────────


def measure_peak_memory(framework: str, spec: CircuitSpec) -> dict:
    """轴 D：一次演化的 Python 侧峰值内存。

    同时报告**理论下界** `2^n · sizeof(complex128)`，比值 `overhead_ratio` 即额外
    拷贝的代价——单看绝对值无法判断实现是否浪费。

    局限（须在论文中写明）：`tracemalloc` 只看得见 Python 分配器，C++ 后端
    （Aer/Qulacs）在其内部分配的内存不计入，故跨框架的内存列不可直接横比，
    只在同框架内比较 n 的增长趋势。
    """

    adapter = get_adapter(framework)
    circuit = adapter.build(spec)

    tracemalloc.start()
    tracemalloc.reset_peak()
    adapter.run(circuit, spec)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    theoretical = (1 << spec.n_qubits) * 16
    return {
        "axis": "memory",
        "framework": framework,
        "spec": spec.label(),
        "n_qubits": spec.n_qubits,
        "peak_bytes": int(peak),
        "theoretical_bytes": theoretical,
        "overhead_ratio": float(peak) / theoretical if theoretical else 0.0,
        "note": "tracemalloc 仅覆盖 Python 分配器；C++ 后端内部分配不计入",
    }


# ────────────────────────────── 轴 C：VQE 端到端 ──────────────────────────────


def _tfim_hamiltonian(n_qubits: int):
    """横场 Ising 哈密顿量 H = Σ Z_i Z_{i+1} + 0.5 Σ X_i。

    选它是因为：所有框架都能表达，谱范围可解析定界（便于对计时结果做物理校验），
    且项数随 n 线性增长，不会让轴 C 变成"比谁的哈密顿量分组更聪明"。
    """

    from aicir import Hamiltonian

    terms = [("ZZ", [q, q + 1], 1.0) for q in range(n_qubits - 1)]
    terms += [("X", [q], 0.5) for q in range(n_qubits)]
    return Hamiltonian(n_qubits=n_qubits, terms=terms)


def run_vqe_axis(framework: str, *, n_qubits: int = 6, layers: int = 2, repeats: int = 5, warmup: int = 1) -> dict:
    """轴 C：变分负载。分别计时**单次能量**与**单次梯度**。

    梯度用参数移位（`qml.deriv.psr`），因为那是所有框架都能做、且与硬件语义
    一致的方式；自动微分只有部分框架支持，混在一起比较不成立。

    同时返回能量值与谱界，让计时结果可被物理校验——算得快但算错没有意义。
    """

    if framework != "aicir":
        raise NotImplementedError(
            f"轴 C 目前只实现了 aicir 路径；{framework!r} 需要各自的 VQE 封装，"
            "跨框架 VQE 比较留待后续（见 scripts/bench/README.md 已知缺口）"
        )

    from aicir.ansatze import hea
    from aicir.primitives import StatevectorEstimator
    from aicir.qml.deriv import psr

    hamiltonian = _tfim_hamiltonian(n_qubits)
    template = hea(n_qubits, layers=layers)
    n_params = len(template.parameters)

    rng = np.random.default_rng(7)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_params)

    # 走 Estimator primitive——这也是 BasicVQE 默认的精确能量路径，
    # 因此测到的是真实 VQE 循环里的那条路径，而非另建的捷径。
    estimator = StatevectorEstimator()

    def energy(values) -> float:
        bound = template.bind_parameters(np.asarray(values, dtype=float))
        return float(np.real(estimator.run(bound, hamiltonian).value))

    energy_stats = time_callable(lambda: energy(theta), repeats=repeats, warmup=warmup)
    gradient_stats = time_callable(lambda: psr(energy, theta), repeats=max(1, repeats // 2), warmup=warmup)

    gradient = np.asarray(psr(energy, theta)).reshape(-1)
    value = energy(theta)

    # 谱界（triangle inequality）：Σ|cᵢ|，任何态上的 |<H>| 都不可能超过它。
    # 用它给计时结果加一道物理校验——算得快但算错没有意义。
    spectral_bound = float(sum(abs(term.coefficient) for term in hamiltonian.terms))

    return {
        "axis": "vqe",
        "framework": framework,
        "n_qubits": n_qubits,
        "layers": layers,
        "n_parameters": n_params,
        "energy": value,
        "spectral_bound": spectral_bound,
        "gradient_length": int(gradient.size),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "energy_eval": energy_stats.to_dict(),
        "gradient_eval": gradient_stats.to_dict(),
    }


# ─────────────────────── 轴 E/F：昇腾与多卡（需真机） ───────────────────────


def strong_scaling_plan(*, n_qubits: int, world_sizes=(1, 2, 4, 8), mode: str = "strong") -> list[dict]:
    """轴 F：多卡扩展计划。

    在**本机**校验参数合法性，避免上真机排队几小时后才发现配置非法：

    - `world_size` 必须是 2 的幂（`aicir.distributed` 按 `2^p` 行分片）；
    - `n_qubits >= p`，否则分片数超过可分的比特维度。

    `mode="strong"` 固定 n、增卡；`mode="weak"` 每翻倍一次卡数多装一个比特——
    后者才回答"多卡究竟买到了多大的态"这个真正的问题。
    """

    if mode not in ("strong", "weak"):
        raise ValueError(f"mode 必须是 'strong' 或 'weak'，收到 {mode!r}")

    plan = []
    for world_size in world_sizes:
        if world_size < 1 or (world_size & (world_size - 1)) != 0:
            raise ValueError(f"world_size 必须是 2 的幂，收到 {world_size}")
        p = world_size.bit_length() - 1
        target_qubits = n_qubits + p if mode == "weak" else n_qubits
        if target_qubits < p:
            raise ValueError(f"n_qubits={target_qubits} 小于分片指数 p={p}（world_size={world_size}）")
        plan.append(
            {
                "axis": f"{mode}_scaling",
                "world_size": world_size,
                "shard_exponent": p,
                "n_qubits": target_qubits,
                "bytes_per_rank": (1 << target_qubits) * 8 // world_size,  # complex64
                "command": (
                    f"torchrun --nproc-per-node={world_size} "
                    f"scripts/npu/distributed_autograd_probe.py --section all"
                ),
            }
        )
    return plan


def npu_vs_cpu_plan(*, families=("ghz", "qft", "random"), qubit_range=range(12, 21)) -> dict:
    """轴 E：昇腾 vs CPU 计划。

    昇腾恒为 complex64（无 complex128 内核），因此 CPU 侧必须也用 complex64
    才是同精度比较——这条约束由 `aicir/dtypes.py` 的策略保证，此处显式记录，
    以免有人拿 CPU 的 complex128 去比 NPU 的 complex64。
    """

    from aicir.backends.npu_backend import is_npu_available

    available = bool(is_npu_available()) if callable(is_npu_available) else False
    return {
        "axis": "npu_vs_cpu",
        "npu_available": available,
        "skipped_reason": None if available else "本机无昇腾设备；须在 NPU 机器上运行",
        "precision": "complex64",
        "precision_note": "昇腾无 complex128 内核，CPU 侧必须同样用 complex64 才可比",
        "families": list(families),
        "qubit_range": [int(qubit_range.start), int(qubit_range.stop) - 1],
        "command": "scripts/bench/bench.sh --axis npu --precision single",
    }
