"""QAS 架构评估器（ArchitectureEvaluator）演示。

`aicir.qas.evaluator` 提供「架构级」打分：对每个候选线路，在四个正交目标
组上各用一个激活指标打分，再按权重合成一个 weighted_score 用于排名。

四个目标组（默认激活指标）：
  - expressibility   表达能力       (kl_haar)
  - trainability     可训练性       (structure_proxy)
  - noise_robustness 噪声鲁棒性     (ion_trap_error_budget_proxy)
  - hardware_efficiency 硬件效率    (native_depth_twoq_efficiency)

本演示**不依赖** aicir.qas.architecture_candidates，而是用 aicir 的
Circuit 与门工厂（ry/rz/cnot/cz/crx/rzz/hadamard ...）手写候选线路，
再用 ArchitectureSpec 包装后交给评估器。

运行（在仓库根目录）：
    PYTHONPATH=. python demos/qas_evaluator_demo.py
"""

from __future__ import annotations

from typing import List

import numpy as np

from aicir import Circuit, cnot, crx, cz, hadamard, rx, ry, rz, rzz
from aicir.qas import (
    ArchitectureEvaluator,
    ArchitectureSpec,
    RewardWeights,
    evaluate_architectures,
    metric_catalog,
)
from aicir.vqc.ansatz import hea, hea_parameter_count, hea_ti, hea_ti_parameter_count


# 用一个简单计数器为每个参数门给出确定性的占位角度，便于复现。
class _Angle:
    def __init__(self) -> None:
        self._i = 0

    def next(self) -> float:
        self._i += 1
        return 0.071 * self._i


def hea_linear(n_qubits: int, layers: int) -> ArchitectureSpec:
    """硬件高效 ansatz：每层 RY/RZ 旋转 + 线性链 CNOT 纠缠。"""
    a = _Angle()
    circuit = Circuit(n_qubits=n_qubits)
    for _ in range(layers):
        for q in range(n_qubits):
            circuit.append(ry(a.next(), q))
            circuit.append(rz(a.next(), q))
        for q in range(n_qubits - 1):
            circuit.append(cnot(q + 1, [q]))  # 工厂签名: cnot(target, control_qubits)
    for q in range(n_qubits):
        circuit.append(ry(a.next(), q))
    return ArchitectureSpec(name="hea_linear", circuit=circuit, tags=["HEA"])


def real_amplitudes(n_qubits: int, layers: int) -> ArchitectureSpec:
    """RealAmplitudes 风格：仅 RY 旋转 + 线性链 CNOT。"""
    a = _Angle()
    circuit = Circuit(n_qubits=n_qubits)
    for _ in range(layers):
        for q in range(n_qubits):
            circuit.append(ry(a.next(), q))
        for q in range(n_qubits - 1):
            circuit.append(cnot(q + 1, [q]))
    for q in range(n_qubits):
        circuit.append(ry(a.next(), q))
    return ArchitectureSpec(name="real_amplitudes", circuit=circuit, tags=["RealAmplitudes"])


def qaoa_chain(n_qubits: int, layers: int) -> ArchitectureSpec:
    """QAOA 风格：初始 H + 每层链式 RZZ 代价项 + RX 混合项。"""
    a = _Angle()
    circuit = Circuit(n_qubits=n_qubits)
    for q in range(n_qubits):
        circuit.append(hadamard(q))
    for _ in range(layers):
        for q in range(n_qubits - 1):
            circuit.append(rzz(a.next(), q, q + 1))  # 工厂签名: rzz(theta, qubit_1, qubit_2)
        for q in range(n_qubits):
            circuit.append(rx(a.next(), q))
    return ArchitectureSpec(name="qaoa_chain", circuit=circuit, tags=["QAOA"])


def strongly_entangling(n_qubits: int, layers: int) -> ArchitectureSpec:
    """强纠缠层：每层 RX/RY/RZ + 环形受控 RX（crx）。"""
    a = _Angle()
    circuit = Circuit(n_qubits=n_qubits)
    for _ in range(layers):
        for q in range(n_qubits):
            circuit.append(rx(a.next(), q))
            circuit.append(ry(a.next(), q))
            circuit.append(rz(a.next(), q))
        for q in range(n_qubits):
            circuit.append(crx(a.next(), (q + 1) % n_qubits, [q]))  # crx(theta, target, control)
    return ArchitectureSpec(name="strongly_entangling", circuit=circuit, tags=["SEL"])


def two_local_cz(n_qubits: int, layers: int) -> ArchitectureSpec:
    """TwoLocal 风格：RY 旋转 + 全连接 CZ 纠缠（双比特门更密集）。"""
    a = _Angle()
    circuit = Circuit(n_qubits=n_qubits)
    for _ in range(layers):
        for q in range(n_qubits):
            circuit.append(ry(a.next(), q))
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                circuit.append(cz(j, [i]))
    for q in range(n_qubits):
        circuit.append(ry(a.next(), q))
    return ArchitectureSpec(name="two_local_cz", circuit=circuit, tags=["TwoLocal"])


def vqc_hea(n_qubits: int, layers: int) -> ArchitectureSpec:
    """直接复用 aicir.vqc.ansatz 的 hea 模板（RY/RZ + 线性 CX）。

    模板默认生成符号 Parameter；这里给出确定性数值绑定，
    便于基于采样的指标稳定复现。
    """
    n_params = hea_parameter_count(n_qubits, layers, rotation_gates=("ry", "rz"), entangler="cx")
    values = (0.071 * (1 + np.arange(n_params))).tolist()
    circuit = hea(n_qubits, layers, rotation_gates=("ry", "rz"), entangler="cx", parameters=values)
    return ArchitectureSpec(name="vqc_hea", circuit=circuit, tags=["HEA", "vqc.ansatz"])


def vqc_hea_ti(n_qubits: int, layers: int) -> ArchitectureSpec:
    """直接复用 aicir.vqc.ansatz 的 hea_ti 模板（离子阱：Rx Ry Rx + 全局 TFIM 演化）。"""
    n_params = hea_ti_parameter_count(n_qubits, layers, variant="general")
    values = (0.071 * (1 + np.arange(n_params))).tolist()
    circuit = hea_ti(n_qubits, layers, variant="general", parameters=values)
    return ArchitectureSpec(name="vqc_hea_ti", circuit=circuit, tags=["HEA-TI", "trapped_ion", "vqc.ansatz"])


def build_candidates(n_qubits: int = 4, layers: int = 2) -> List[ArchitectureSpec]:
    return [
        hea_linear(n_qubits, layers),
        real_amplitudes(n_qubits, layers),
        qaoa_chain(n_qubits, layers),
        strongly_entangling(n_qubits, layers),
        two_local_cz(n_qubits, layers),
        vqc_hea(n_qubits, layers),
        vqc_hea_ti(n_qubits, layers),
    ]


def show_catalog() -> None:
    """打印指标目录，标出每个目标组当前激活的指标。"""
    print("=" * 70)
    print("指标目录（[*] 为当前激活，status 区分已实现 / 待办）")
    print("=" * 70)
    for group_name, metrics in metric_catalog().items():
        print(f"\n[{group_name}]")
        for metric in metrics:
            mark = "*" if metric.active else " "
            print(f"  [{mark}] {metric.name:<28} ({metric.status})  {metric.purpose}")


def show_score_detail(score) -> None:
    """逐组打印一个候选的得分与规模信息。"""
    arch = score.architecture
    print(f"\n候选：{arch.name}")
    print(
        f"  规模：n_qubits={arch.n_qubits}  n_gates={arch.n_gates}  "
        f"参数={arch.parameter_count}  双比特门={arch.two_qubit_gate_count}"
    )
    for group_name, group in score.groups().items():
        print(f"  {group_name:<20} 激活={group.active_metric:<28} score={group.score:.4f}")
    print(f"  -> weighted_score = {score.weighted_score:.4f}  (rank={score.rank})")


def main() -> None:
    # 1) 看一眼指标目录
    show_catalog()

    # 2) 用 aicir 手写一批候选架构（4 比特、2 层）
    architectures = build_candidates(n_qubits=4, layers=2)
    print("\n" + "=" * 70)
    print(f"已用 aicir 手写 {len(architectures)} 个候选架构：")
    print("  " + ", ".join(arch.name for arch in architectures))

    # 3) 单个评估：直接使用 ArchitectureEvaluator
    #    n_samples 较小以加快演示（表达能力指标基于采样）。
    evaluator = ArchitectureEvaluator(n_samples=64)
    single = evaluator.evaluate(architectures[0])
    print("\n" + "=" * 70)
    print("单个候选评估（默认等权重）：")
    show_score_detail(single)

    # 4) 批量评估并排名：evaluate_many 按 weighted_score 排序并填 rank
    print("\n" + "=" * 70)
    print("批量评估并排名（默认等权重）：")
    ranked = evaluator.evaluate_many(architectures)
    print(f"\n{'rank':>4}  {'name':<22}{'expr':>7}{'train':>7}{'noise':>7}{'hw':>7}{'total':>8}")
    print("-" * 62)
    for score in ranked:
        row = score.to_row()
        print(
            f"{row['rank']:>4}  {row['name']:<22}"
            f"{row['expressibility']:>7.3f}{row['trainability']:>7.3f}"
            f"{row['noise_robustness']:>7.3f}{row['hardware_efficiency']:>7.3f}"
            f"{row['weighted_score']:>8.3f}"
        )
    print(f"\n最佳架构：{ranked[0].architecture.name}")

    # 5) 自定义权重：偏向可训练性与硬件效率，看排名如何变化
    #    （便捷函数 evaluate_architectures 等价于 Evaluator(...).evaluate_many）
    weights = RewardWeights(
        expressibility=0.1,
        trainability=0.4,
        noise_robustness=0.1,
        hardware_efficiency=0.4,
    )
    print("\n" + "=" * 70)
    print("自定义权重（偏向 trainability + hardware_efficiency）后的排名：")
    reweighted = evaluate_architectures(architectures, weights=weights, n_samples=64)
    for score in reweighted:
        print(f"  rank {score.rank}: {score.architecture.name:<22} total={score.weighted_score:.4f}")


if __name__ == "__main__":
    main()
