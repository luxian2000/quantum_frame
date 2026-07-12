"""QAS ``SearchStrategy`` 具体适配器实现（QAS README §2.3）。

本文件只做“翻译”：把 ``QASRunConfig`` 请求（含已由 ``runner.run()`` 归一化的
``QASProblem``）转换成底层算法函数的调用参数，再把底层返回值包装成
``QASResult``。底层函数（``train_supernet``/``classification_supernet``/
``h2_vqe_supernet``/``train_crlqas``/``ppo_rb_qas``/``train_ppr_dql``/
``train_qdrats``/``train_dqas``/``mogvqe``/``run_vqe_qas_closed_loop``）的
签名与返回类型保持不变——它们是兼容性接缝，不在本文件内改动。

注册（``strategies.py``）与实现（本文件）分开：新增/调整某个方法的翻译逻辑
只改本文件；`SearchStrategy`/`StrategySpec` 等框架部分见 ``registry.py``。
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ...core.operators import Hamiltonian
from .problem import QASProblem, terms_coeff_first
from .registry import SearchStrategy
from .results import QASResult

__all__ = [
    "CRLQASStrategy",
    "DQASStrategy",
    "MOGVQEStrategy",
    "PPORBStrategy",
    "PPRDQLStrategy",
    "QDRATSStrategy",
    "SupernetClassificationStrategy",
    "SupernetH2Strategy",
    "SupernetStrategy",
    "VqeLoopStrategy",
]


# ──────────────────────────────────────────────────────────────────────────────
# request.problem 取值辅助：底层函数各自要的是 Hamiltonian/矩阵、State 或密度矩阵。
# ──────────────────────────────────────────────────────────────────────────────


def _hamiltonian_input_from_problem(problem: QASProblem | None, *, method: str) -> Any:
    """从 ``QASProblem`` 取出可直接喂给
    ``train_supernet``/``train_crlqas``/``train_qdrats``/``train_dqas``/``mogvqe``
    的哈密顿量入参（这些底层函数本就接受 ``Hamiltonian`` 或稠密矩阵）。

    ``problem is None`` 时返回 ``None``（调用方决定是否必需）。
    """

    if problem is None:
        return None
    if problem.kind != "hamiltonian":
        raise ValueError(f"{method} requires a hamiltonian-kind problem; got problem.kind={problem.kind!r}.")
    return problem.hamiltonian if problem.hamiltonian is not None else problem.matrix


def _require_hamiltonian_input(problem: QASProblem | None, *, method: str) -> Any:
    if problem is None:
        raise ValueError(f"{method} requires hamiltonian.")
    return _hamiltonian_input_from_problem(problem, method=method)


def _require_state(problem: QASProblem | None, *, method: str) -> Any:
    if problem is None or problem.kind != "state":
        raise ValueError(f"{method} requires target_state.")
    return problem.state


def _require_density_matrix(problem: QASProblem | None, *, method: str) -> np.ndarray:
    if problem is None or problem.kind != "density_matrix":
        raise ValueError(f"{method} requires target_density_matrix.")
    return problem.matrix


def _pauli_terms_coeff_first(hamiltonian: Hamiltonian) -> list[tuple[float, str]]:
    """把 ``Hamiltonian`` 转成 ``[(coeff, label), ...]``（vqe_loop 的 ``PauliTerm`` 顺序）。

    每个 ``PauliString.qubit_labels`` 已补全到 ``n_qubits`` 长度（未参与的比特为
    ``"I"``），直接拼接即为 ``problems/hamiltonians.py`` 期望的定长标签字符串；
    再经 :func:`~aicir.qas.core.problem.terms_coeff_first` 把
    ``(label, coeff)`` 顺序转换为 vqe_loop 需要的 ``(coeff, label)`` 顺序。
    """

    label_first: list[tuple[str, float]] = []
    for term in hamiltonian.terms:
        coeff = complex(term.coefficient)
        if abs(coeff.imag) > 1e-9:
            raise ValueError("vqe_loop hamiltonian_terms 要求系数为实数（Hermitian 哈密顿量）")
        label_first.append(("".join(term.qubit_labels), coeff.real))
    return terms_coeff_first(label_first)


# ──────────────────────────────────────────────────────────────────────────────
# supernet 系列：train_supernet / classification_supernet / h2_vqe_supernet
# ──────────────────────────────────────────────────────────────────────────────


def _wrap_supernet_result(raw: Any, *, method: str) -> QASResult:
    """``SupernetResult`` 没有独立的连续参数向量字段：微调后的角度值已经作为
    torch 标量直接写入 ``best_circuit`` 的门参数里（``build_circuit`` 用
    ``shared_parameters`` 的当前值构造门），因此 ``parameters`` 留空，完整信息
    见 ``raw``/``metadata``。``history`` 取 ``supernet_log`` 与 ``finetune_log``
    顺序拼接（两段训练日志），排序记录与最终指标放进 ``metadata``。
    """

    history = list(raw.supernet_log) + list(raw.finetune_log)
    return QASResult(
        method=method,
        value=float(raw.best_score),
        circuit=raw.best_circuit,
        parameters=None,
        history=history,
        metadata={
            "method": method,
            "config": raw.config,
            "best_architecture": raw.best_architecture,
            "best_supernet_id": raw.best_supernet_id,
            "ranking_records": raw.ranking_records,
            "final_metrics": raw.final_metrics,
        },
        raw=raw,
    )


class SupernetStrategy(SearchStrategy):
    """``run("supernet", ...)`` 的适配器：分发到 ``train_supernet``。"""

    _CANONICAL = "supernet"
    _PARAMS = ("objective", "config", "dataset")

    def run(self, request: Any) -> Any:
        # 懒导入：supernet 依赖 torch，避免在无 torch 环境 import 本模块即失败。
        from ..algorithms.supernet import train_supernet

        problem = getattr(request, "problem", None)
        hamiltonian = _hamiltonian_input_from_problem(problem, method=self._CANONICAL)
        kwargs = {name: getattr(request, name, None) for name in self._PARAMS}
        raw = train_supernet(hamiltonian=hamiltonian, **kwargs)
        return _wrap_supernet_result(raw, method=self._CANONICAL)


class SupernetClassificationStrategy(SearchStrategy):
    """``run("supernet_classification", ...)`` 的适配器：分发到 ``classification_supernet``。"""

    _CANONICAL = "supernet_classification"

    def run(self, request: Any) -> Any:
        from ..algorithms.supernet import classification_supernet

        raw = classification_supernet(config=getattr(request, "config", None))
        return _wrap_supernet_result(raw, method=self._CANONICAL)


class SupernetH2Strategy(SearchStrategy):
    """``run("supernet_h2", ...)`` 的适配器：分发到 ``h2_vqe_supernet``。

    ``h2_vqe_supernet`` 固定使用内置 4 量子比特 H2 哈密顿量，不消费
    ``request.problem``（与旧 ``_TABLE`` 分发行为一致）。
    """

    _CANONICAL = "supernet_h2"

    def run(self, request: Any) -> Any:
        from ..algorithms.supernet import h2_vqe_supernet

        raw = h2_vqe_supernet(config=getattr(request, "config", None))
        return _wrap_supernet_result(raw, method=self._CANONICAL)


# ──────────────────────────────────────────────────────────────────────────────
# dqas / qdrats：结构同构（circuit/parameters/minimum_energy/search_log/finetune_log）
# ──────────────────────────────────────────────────────────────────────────────


class DQASStrategy(SearchStrategy):
    """``run("dqas", ...)`` 的适配器：分发到 ``train_dqas``。"""

    _CANONICAL = "dqas"

    def run(self, request: Any) -> Any:
        from ..algorithms.dqas import train_dqas

        problem = getattr(request, "problem", None)
        hamiltonian = _require_hamiltonian_input(problem, method=self._CANONICAL)
        raw = train_dqas(hamiltonian=hamiltonian, config=getattr(request, "config", None))
        return QASResult(
            method=self._CANONICAL,
            value=float(raw.minimum_energy),
            circuit=raw.circuit,
            parameters=raw.parameters,
            history=list(raw.search_log),
            metadata={
                "method": self._CANONICAL,
                "config": raw.config,
                "finetune_log": raw.finetune_log,
                "architecture_indices": raw.architecture_indices,
                "architecture_labels": raw.architecture_labels,
                "architecture_probabilities": raw.architecture_probabilities,
            },
            raw=raw,
        )


class QDRATSStrategy(SearchStrategy):
    """``run("qdrats", ...)`` 的适配器：分发到 ``train_qdrats``。"""

    _CANONICAL = "qdrats"

    def run(self, request: Any) -> Any:
        from ..algorithms.qdrats import train_qdrats

        problem = getattr(request, "problem", None)
        hamiltonian = _require_hamiltonian_input(problem, method=self._CANONICAL)
        raw = train_qdrats(hamiltonian=hamiltonian, config=getattr(request, "config", None))
        return QASResult(
            method=self._CANONICAL,
            value=float(raw.minimum_energy),
            circuit=raw.circuit,
            parameters=raw.parameters,
            history=list(raw.search_log),
            metadata={
                "method": self._CANONICAL,
                "config": raw.config,
                "finetune_log": raw.finetune_log,
                "architecture_indices": raw.architecture_indices,
                "architecture_labels": raw.architecture_labels,
                "architecture_probabilities": raw.architecture_probabilities,
            },
            raw=raw,
        )


# ──────────────────────────────────────────────────────────────────────────────
# crlqas
# ──────────────────────────────────────────────────────────────────────────────


class CRLQASStrategy(SearchStrategy):
    """``run("crlqas", ...)`` 的适配器：分发到 ``train_crlqas``。"""

    _CANONICAL = "crlqas"

    def run(self, request: Any) -> Any:
        from ..algorithms.crlqas import train_crlqas

        problem = getattr(request, "problem", None)
        hamiltonian = _require_hamiltonian_input(problem, method=self._CANONICAL)
        config = getattr(request, "config", None)
        raw = train_crlqas(hamiltonian=hamiltonian, config=config)
        return QASResult(
            method=self._CANONICAL,
            value=float(raw.minimum_energy),
            circuit=raw.circuit,
            parameters=raw.parameters,
            history=list(raw.episode_best_energies),
            metadata={
                "method": self._CANONICAL,
                "config": config,
                "curriculum_threshold": raw.curriculum_threshold,
                "q_network_state_dict": raw.q_network_state_dict,
            },
            raw=raw,
        )


# ──────────────────────────────────────────────────────────────────────────────
# pporb：底层只返回裸 (theta, circuit) 元组，没有可用的保真度/能量字段。
# ──────────────────────────────────────────────────────────────────────────────


class PPORBStrategy(SearchStrategy):
    """``run("pporb", ...)`` 的适配器：分发到 ``ppo_rb_qas``。

    ``ppo_rb_qas`` 只返回 ``(theta, circuit)``，训练过程中记录的最佳保真度
    （``best_fidelity``）是函数内部局部变量，不在返回值里；重新计算保真度需要
    重跑环境仿真（不是廉价操作），3b 不做该重算，因此 ``value=None``——完整信息
    仍在 ``raw``/``parameters`` 里（``parameters=theta``）。
    """

    _CANONICAL = "pporb"

    def run(self, request: Any) -> Any:
        from ..algorithms.pporb import ppo_rb_qas

        problem = getattr(request, "problem", None)
        target_density_matrix = _require_density_matrix(problem, method=self._CANONICAL)
        epsilon = getattr(request, "epsilon", None)
        if epsilon is None:
            raise ValueError(f"{self._CANONICAL} requires epsilon.")
        config = getattr(request, "config", None)
        theta, circuit = ppo_rb_qas(
            target_density_matrix=target_density_matrix,
            epsilon=epsilon,
            config=config,
        )
        return QASResult(
            method=self._CANONICAL,
            value=None,
            circuit=circuit,
            parameters=theta,
            history=[],
            metadata={"method": self._CANONICAL, "config": config},
            raw=(theta, circuit),
        )


# ──────────────────────────────────────────────────────────────────────────────
# pprdql
# ──────────────────────────────────────────────────────────────────────────────


class PPRDQLStrategy(SearchStrategy):
    """``run("pprdql", ...)`` 的适配器：分发到 ``train_ppr_dql``。"""

    _CANONICAL = "pprdql"

    def run(self, request: Any) -> Any:
        from ..algorithms.pprdql import train_ppr_dql

        problem = getattr(request, "problem", None)
        target_state = _require_state(problem, method=self._CANONICAL)
        config = getattr(request, "config", None)
        policy_library = getattr(request, "policy_library", None)
        raw = train_ppr_dql(target_state=target_state, config=config, policy_library=policy_library)
        return QASResult(
            method=self._CANONICAL,
            value=float(raw.best_fidelity),
            circuit=raw.circuit,
            parameters=raw.policy,
            history=list(raw.episode_rewards),
            metadata={
                "method": self._CANONICAL,
                "config": config,
                "selected_policy_indices": raw.selected_policy_indices,
            },
            raw=raw,
        )


# ──────────────────────────────────────────────────────────────────────────────
# vqe_loop：闭环产出的是 benchmark-table CSV 路径，没有内存态的 circuit/energy。
# ──────────────────────────────────────────────────────────────────────────────


class VqeLoopStrategy(SearchStrategy):
    """``run("vqe_loop", ...)`` 的适配器：分发到 ``run_vqe_qas_closed_loop``。

    ``ClosedLoopResult``（``P0BootstrapResult``）只携带输出文件路径
    （``candidates``/``initial_queue``/``final_benchmark_table`` 等），没有内存
    态的最优 circuit/energy——最优能量需要解析 ``final_benchmark_table`` CSV，
    不是廉价操作，3b 不做该解析，因此 ``value``/``circuit``/``parameters``
    均为 ``None``；完整路径信息见 ``raw``/``metadata``。

    ``request.problem`` 若给定，须是可分解为 Pauli 项的 ``Hamiltonian``（经
    :func:`_pauli_terms_coeff_first` 转换为 ``config.hamiltonian_terms``）；
    稠密矩阵输入不受支持（无法廉价分解为 Pauli 项）。``problem`` 为 ``None``
    时 ``config`` 原样透传，与旧 ``_TABLE`` 分发行为一致。
    """

    _CANONICAL = "vqe_loop"

    def run(self, request: Any) -> Any:
        from ..vqe_loop import run_vqe_qas_closed_loop

        config = getattr(request, "config", None)
        if config is None:
            raise ValueError(f"{self._CANONICAL} requires config.")
        problem = getattr(request, "problem", None)
        if problem is not None:
            hamiltonian_input = _hamiltonian_input_from_problem(problem, method=self._CANONICAL)
            if not isinstance(hamiltonian_input, Hamiltonian):
                raise ValueError(
                    f"{self._CANONICAL} problem 需要可分解为 Pauli 项的 Hamiltonian（不支持稠密矩阵输入）。"
                )
            config = dataclasses.replace(config, hamiltonian_terms=_pauli_terms_coeff_first(hamiltonian_input))
        raw = run_vqe_qas_closed_loop(config=config)
        return QASResult(
            method=self._CANONICAL,
            value=None,
            circuit=None,
            parameters=None,
            history=[],
            metadata={
                "method": self._CANONICAL,
                "config": config,
                "output_dir": str(raw.output_dir),
                "final_benchmark_table": str(raw.final_benchmark_table),
            },
            raw=raw,
        )


# ──────────────────────────────────────────────────────────────────────────────
# mogvqe：唯一不需要 torch 的方法；额外需要 request.initial_ansatz（拓扑起点）。
# ──────────────────────────────────────────────────────────────────────────────


class MOGVQEStrategy(SearchStrategy):
    """``run("mogvqe", ...)`` 的适配器：分发到 ``mogvqe()``。

    ``mogvqe(initial_ansatz, *, hamiltonian=None, energy_evaluator=None, config=None,
    backend=None)`` 的第一个位置参数是拓扑起点，不属于 ``QASProblem``/``config``
    的语义范围，因此走 ``QASRunConfig.initial_ansatz``（3b 新增字段）。
    ``energy_evaluator``/``backend`` 注入未经 ``run()`` 打通——需要自定义能量函数
    或非默认后端时请直接调用 ``mogvqe(...)``。
    """

    _CANONICAL = "mogvqe"

    def run(self, request: Any) -> Any:
        from ..algorithms.mogvqe import mogvqe as run_mogvqe

        initial_ansatz = getattr(request, "initial_ansatz", None)
        if initial_ansatz is None:
            raise ValueError(f"{self._CANONICAL} requires initial_ansatz.")
        problem = getattr(request, "problem", None)
        hamiltonian = _hamiltonian_input_from_problem(problem, method=self._CANONICAL)
        if hamiltonian is None:
            raise ValueError(f"{self._CANONICAL} requires hamiltonian (problem= or legacy hamiltonian=).")
        config = getattr(request, "config", None)
        raw = run_mogvqe(initial_ansatz, hamiltonian=hamiltonian, config=config)
        return QASResult(
            method=self._CANONICAL,
            value=float(raw.best_energy),
            circuit=raw.best_circuit,
            parameters=raw.best_parameters,
            history=list(raw.history),
            metadata={
                "method": self._CANONICAL,
                "config": raw.config,
                "best_individual": raw.best_individual,
                "pareto_front_size": len(raw.pareto_front),
            },
            raw=raw,
        )
