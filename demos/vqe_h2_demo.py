"""用 VQE 求解氢分子 H2 基态能量的完整示例。

从仓库根目录运行：

    python -m demos.vqe_h2_demo
    python -m demos.vqe_h2_demo --estimator pauli --optimizer spsa --shots 4096

本示例只使用 aicir 的公开 API，完整流程为：

1. 从 ``aicir.chemistry`` 读取预置的 2-qubit H2 Hamiltonian。
2. 构造带符号 ``Parameter`` 的 hardware-efficient ansatz。
3. 使用 ``BasicVQE`` 和经典优化器训练 ansatz 参数。
4. 将 VQE 能量与 Hamiltonian 矩阵精确对角化结果对比。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from aicir import NumpyBackend, PauliEstimator
from aicir.chemistry import get_molecule, molecule_hamiltonian, molecule_matrix
from aicir.measure import hamiltonian_pauli_terms
from aicir.optimizer import COBYLA, SPSA, Adam
from aicir.vqc import BasicVQE
from aicir.ansatze import hea


@dataclass(frozen=True)
class DemoConfig:
    """命令行参数解析后的只读配置。

    这个配置对象集中保存 demo 的关键超参数，避免在多个函数之间传递
    argparse 的原始 Namespace。
    """

    estimator: str
    optimizer: str
    layers: int
    maxiter: int
    shots: int
    seed: int


def build_h2_ansatz(n_qubits: int, layers: int, backend: NumpyBackend):
    """构造 H2 VQE 使用的 hardware-efficient ansatz。

    这里选择最小而有效的结构：每层使用单比特 ``Ry`` 旋转，量子比特之间
    用线性 ``CX`` 纠缠，并追加末尾旋转层。返回值是带符号参数的
    ``Circuit``，后续由 ``BasicVQE`` 负责绑定数值参数。
    """

    return hea(
        n_qubits,
        layers=layers,
        rotation_gates=("ry",),
        entangler="cx",
        topology="linear",
        final_rotation_layer=True,
        backend=backend,
    )


def build_optimizer(name: str, maxiter: int, seed: int):
    """根据命令行名称创建经典参数优化器。

    - ``cobyla``：默认选择，适合小规模无梯度 VQE，结果稳定。
    - ``spsa``：适合 noisy / shot-based 目标函数。
    - ``adam``：配合 parameter-shift 梯度使用。
    """

    method = name.strip().lower()
    if method == "cobyla":
        return COBYLA(options={"maxiter": maxiter, "rhobeg": 0.3, "tol": 1e-8})
    if method == "spsa":
        return SPSA(
            max_iters=maxiter,
            learning_rate=0.08,
            perturbation=0.08,
            rng=seed,
        )
    if method == "adam":
        return Adam(
            max_iters=maxiter,
            learning_rate=0.08,
            gradient_method="psr",
        )
    raise ValueError(f"Unknown optimizer {name!r}")


def build_energy_estimator(config: DemoConfig, backend: NumpyBackend):
    """创建 VQE 的能量估计器。

    ``exact`` 使用 full-matrix observable 精确期望值，适合本地模拟和基准
    对比；``pauli`` 使用 ``PauliEstimator``，按 Pauli 项拆分 Hamiltonian，
    做测量基变换和有限 shots 统计，更接近硬件 VQE 的测量流程。
    """

    if config.estimator == "exact":
        return "exact"
    if config.estimator == "pauli":
        return PauliEstimator(
            backend,
            shots=config.shots,
            grouping="qwc",
            shot_allocation="coefficient",
        )
    raise ValueError(f"Unknown estimator {config.estimator!r}")


def exact_ground_energy(backend: NumpyBackend) -> float:
    """通过精确矩阵对角化计算 H2 Hamiltonian 的基态能量。

    这个值不是 VQE 训练的一部分，只作为 demo 结尾的参考答案，用来判断
    VQE 优化后的能量是否接近真实基态。
    """

    matrix = molecule_matrix("h2", backend=backend)
    return float(np.linalg.eigvalsh(matrix).min())


def print_hamiltonian_summary() -> None:
    """打印 H2 Hamiltonian preset 的元数据和 Pauli 项。

    VQE 优化的目标函数来自这些 Pauli 项；在 shot-based estimator 模式下，
    这些项会被分组、做基变换并通过采样 counts 估计期望值。
    """

    preset = get_molecule("h2")
    terms = hamiltonian_pauli_terms(preset.to_hamiltonian())

    print("Hamiltonian preset")
    print(f"  name    : {preset.name}")
    print(f"  formula : {preset.formula}")
    print(f"  basis   : {preset.basis}")
    print(f"  mapping : {preset.mapping}")
    print("  terms   :")
    for term in terms:
        print(f"    {term.coefficient:+.8f} * {term.pauli}")


def run(config: DemoConfig):
    """执行一次完整 H2 VQE 训练并返回结果。

    该函数把问题定义、ansatz、优化器和能量估计器串起来：
    ``Hamiltonian -> ansatz -> BasicVQE -> optimizer``。返回 VQE 结果、
    精确基态能量、误差以及 ansatz，供 ``main()`` 统一打印。
    """

    backend = NumpyBackend()
    hamiltonian = molecule_hamiltonian("h2")
    ansatz = build_h2_ansatz(hamiltonian.n_qubits, config.layers, backend)
    optimizer = build_optimizer(config.optimizer, config.maxiter, config.seed)
    energy_estimator = build_energy_estimator(config, backend)

    rng = np.random.default_rng(config.seed)
    init_params = rng.uniform(-0.05, 0.05, size=len(ansatz.parameters))

    solver = BasicVQE(
        hamiltonian,
        ansatz=ansatz,
        backend=backend,
        optimizer=optimizer,
        energy_estimator=energy_estimator,
    )
    result = solver.run(init_params=init_params)

    exact_energy = exact_ground_energy(backend)
    error = result.energy - exact_energy
    return result, exact_energy, error, ansatz


def parse_args() -> DemoConfig:
    """解析命令行参数并转换为 ``DemoConfig``。

    这里同时做简单合法性检查，例如层数不能为负、最大迭代数和 shots
    必须为正数。这样后续训练函数可以假设配置已经有效。
    """

    parser = argparse.ArgumentParser(description="VQE demo for H2 ground-state energy.")
    parser.add_argument(
        "--estimator",
        choices=("exact", "pauli"),
        default="exact",
        help="Energy estimator used during VQE optimization.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("cobyla", "spsa", "adam"),
        default="cobyla",
        help="Classical optimizer.",
    )
    parser.add_argument("--layers", type=int, default=1, help="HEA layers.")
    parser.add_argument("--maxiter", type=int, default=200, help="Optimizer iterations.")
    parser.add_argument("--shots", type=int, default=4096, help="Shots for --estimator pauli.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for initial parameters.")
    args = parser.parse_args()

    if args.layers < 0:
        raise ValueError("--layers must be non-negative")
    if args.maxiter <= 0:
        raise ValueError("--maxiter must be positive")
    if args.shots <= 0:
        raise ValueError("--shots must be positive")

    return DemoConfig(
        estimator=args.estimator,
        optimizer=args.optimizer,
        layers=args.layers,
        maxiter=args.maxiter,
        shots=args.shots,
        seed=args.seed,
    )


def main() -> None:
    """命令行入口函数。

    负责打印问题信息、运行 VQE、展示最终能量和误差。如果使用
    ``PauliEstimator``，还会额外打印分组数量、shots 和标准误。
    """

    config = parse_args()

    print("=" * 72)
    print("VQE demo: H2 ground-state energy")
    print("=" * 72)
    print_hamiltonian_summary()
    print()
    print("VQE configuration")
    print(f"  ansatz    : HEA Ry + CX, layers={config.layers}")
    print(f"  estimator : {config.estimator}")
    print(f"  optimizer : {config.optimizer}, maxiter={config.maxiter}")
    if config.estimator == "pauli":
        print(f"  shots     : {config.shots}")
    print()

    result, exact_energy, error, ansatz = run(config)

    print("Result")
    print(f"  parameters       : {np.array2string(result.parameters.reshape(-1), precision=6)}")
    print(f"  VQE energy       : {result.energy:.10f} Ha")
    print(f"  exact ground     : {exact_energy:.10f} Ha")
    print(f"  VQE - exact      : {error:+.3e} Ha")
    print(f"  ansatz gates     : {len(ansatz.gates)}")
    nfev = getattr(result.optimizer_result, "nfev", None)
    if nfev is not None:
        print(f"  energy evals     : {nfev}")
    else:
        print(f"  history entries  : {len(result.energy_history)}")
    if result.estimator_result is not None:
        print(f"  estimator groups : {result.estimator_result.metadata['n_groups']}")
        print(f"  estimator shots  : {result.estimator_result.shots}")
        print(f"  estimator stderr : {result.estimator_result.std_error:.3e} Ha")


if __name__ == "__main__":
    main()
