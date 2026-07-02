"""Demo: score and rank a library of candidate QAS architectures.

Run from the repository root:
    C:/ProgramData/anaconda3/python.exe aicir/qas/demos/architecture_scoring_demo.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aicir.qas import ArchitectureScore, RewardWeights, build_common_architectures, evaluate_architectures
from aicir.backends.numpy_backend import NumpyBackend


RESULT_PATH = Path(__file__).with_name("architecture_scoring_results.txt")


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _render_table(scores: Iterable[ArchitectureScore]) -> str:
    headers = [
        "rank",
        "architecture",
        "expr",
        "train",
        "noise",
        "hw",
        "weighted",
        "gates",
        "2q",
        "params",
        "err_budget",
    ]
    rows: List[List[str]] = []
    for score in scores:
        row = score.to_row()
        noise_raw = score.noise_robustness.raw_values
        rows.append(
            [
                str(row["rank"]),
                str(row["name"]),
                _fmt(row["expressibility"]),
                _fmt(row["trainability"]),
                _fmt(row["noise_robustness"]),
                _fmt(row["hardware_efficiency"]),
                _fmt(row["weighted_score"]),
                str(row["n_gates"]),
                str(row["two_qubit_gate_count"]),
                str(row["n_parameters"]),
                _fmt(float(noise_raw.get("total_error_budget", 0.0)), digits=6),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for column_index, cell in enumerate(row):
            widths[column_index] = max(widths[column_index], len(cell))

    def line(cells: List[str]) -> str:
        return "  ".join(cell.ljust(widths[column_index]) for column_index, cell in enumerate(cells))

    lines = [line(headers), line(["-" * width for width in widths])]
    for row in rows:
        lines.append(line(row))
    return "\n".join(lines)


def _render_top_details(scores: List[ArchitectureScore], top_k: int = 3) -> str:
    lines = ["=== Top Candidate Details ==="]
    for score in scores[:top_k]:
        architecture = score.architecture
        noise_raw = score.noise_robustness.raw_values
        lines.extend(
            [
                "",
                f"#{score.rank} {architecture.name}",
                f"description: {architecture.description}",
                f"tags: {', '.join(architecture.tags)}",
            ]
        )
        lines.append(
            "scores: "
            f"expr={_fmt(score.expressibility.score)}, "
            f"train={_fmt(score.trainability.score)}, "
            f"noise={_fmt(score.noise_robustness.score)}, "
            f"hw={_fmt(score.hardware_efficiency.score)}, "
            f"weighted={_fmt(score.weighted_score)}"
        )
        lines.append(
            "ion_trap_error_budget: "
            f"total={_fmt(float(noise_raw['total_error_budget']), 6)}, "
            f"gate={_fmt(float(noise_raw['gate_error_budget']), 6)}, "
            f"idle={_fmt(float(noise_raw['idle_error_budget']), 6)}, "
            f"crosstalk={_fmt(float(noise_raw['crosstalk_error_budget']), 6)}"
        )
    return "\n".join(lines)


def _render_field_definitions() -> str:
    return "\n".join(
        [
            "=== 字段说明 ===",
            "candidates: 本次参与评分的候选架构数量，也就是 ArchitectureSpec 的个数。",
            "n_qubits: 每个候选架构使用的量子比特数。",
            "layers: 传入 build_common_architectures() 的重复结构层数。",
            "expressibility_samples: 表达能力 active 指标使用的 Monte Carlo 采样次数。",
            "weights: 顶层四组目标权重，weighted = sum(weight_i * score_i)。",
            "rank: 按 weighted 从高到低排序后的名次。",
            "architecture: 候选架构名称，对应 ArchitectureSpec.name。",
            "expr: 表达能力组评分；当前 active 指标是 kl_haar，越高越好。",
            "train: 可训练性组评分；当前 active 指标是 structure_proxy，越高越好。",
            "noise: 噪声鲁棒性组评分；当前 active 指标是 ion_trap_error_budget_proxy，越高越好。",
            "hw: 硬件效率组评分；当前 active 指标是 native_depth_twoq_efficiency，越高越好。",
            "weighted: 四组评分按 weights 加权后的总分，越高排名越靠前。",
            "gates: 候选线路中的总门数。",
            "2q: 候选线路中的双比特门或受控门数量。",
            "params: 候选线路中的标量可训练参数数量。",
            "err_budget: 离子阱总 error budget，计算 noise = exp(-err_budget) 之前的误差预算；越低越好。",
            "description: 候选架构库中给出的自然语言架构定义。",
            "tags: 架构族、拓扑、entangler 等标签。",
            "ion_trap_error_budget.total: 与表格中的 err_budget 相同。",
            "ion_trap_error_budget.gate: 单比特门和双比特门退极化错误贡献。",
            "ion_trap_error_budget.idle: 串行执行时非 active qubit 的 idle dephasing 贡献。",
            "ion_trap_error_budget.crosstalk: 默认离子阱配置中的串扰贡献。",
            "",
            "分数范围: expr/train/noise/hw/weighted 均裁剪到 [0, 1]；越高越好。",
            "噪声关系: noise = exp(-err_budget)，所以 err_budget 越大，噪声鲁棒评分越低。",
        ]
    )


def _render_report(scores: List[ArchitectureScore], n_qubits: int, layers: int, n_samples: int, weights: RewardWeights) -> str:
    lines = [
        "=== Noise-Adaptive QAS Candidate Architecture Scoring ===",
        f"candidates: {len(scores)}",
        f"n_qubits: {n_qubits}",
        f"layers: {layers}",
        f"expressibility_samples: {n_samples}",
        f"weights: {weights.to_dict()}",
        "",
        _render_field_definitions(),
        "",
        "=== Ranking Table ===",
        _render_table(scores),
        "",
        _render_top_details(scores),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    np.random.seed(11)

    backend = NumpyBackend()
    n_qubits = 3
    layers = 2
    n_samples = 12
    weights = RewardWeights(
        expressibility=0.25,
        trainability=0.20,
        noise_robustness=0.35,
        hardware_efficiency=0.20,
    )

    architectures = build_common_architectures(n_qubits=n_qubits, layers=layers, backend=backend)
    scores = evaluate_architectures(
        architectures,
        backend=backend,
        weights=weights,
        n_samples=n_samples,
    )

    report = _render_report(scores, n_qubits=n_qubits, layers=layers, n_samples=n_samples, weights=weights)
    RESULT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"results_file: {RESULT_PATH}")


if __name__ == "__main__":
    main()
