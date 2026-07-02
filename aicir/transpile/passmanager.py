"""Pass manager for ordered circuit transformations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..core.circuit import Circuit
from ..ir import circuit_gate_dicts, has_circuit_instructions
from .base import TransformationPass


def _pass_from_name(name: str) -> TransformationPass:
    from .passes import (
        CancelInversePass,
        CanonicalizePass,
        CommuteSingleQubitPass,
        DecomposePass,
        LayoutPass,
        MergeRotationsPass,
        ValidatePass,
    )

    key = str(name).strip().lower()
    mapping = {
        "validate": ValidatePass,
        "canonicalize": CanonicalizePass,
        "cancel_inverse": CancelInversePass,
        "cancel": CancelInversePass,
        "merge_rotations": MergeRotationsPass,
        "merge_rotation": MergeRotationsPass,
        "commute_single_qubit": CommuteSingleQubitPass,
        "commute": CommuteSingleQubitPass,
        "decompose": DecomposePass,
        "layout": LayoutPass,
    }
    try:
        return mapping[key]()
    except KeyError as exc:
        raise ValueError(f"Unknown transpile pass: {name}") from exc


def _coerce_pass(item: str | TransformationPass) -> TransformationPass:
    if isinstance(item, str):
        return _pass_from_name(item)
    if isinstance(item, TransformationPass):
        return item
    if hasattr(item, "run"):
        return item
    raise TypeError("passes must be pass names or TransformationPass objects")


class PassManager:
    """Run circuit transformation passes in sequence."""

    def __init__(
        self,
        passes: Iterable[str | TransformationPass],
        *,
        fixed_point: bool = False,
        max_rounds: int = 64,
    ) -> None:
        self.passes = tuple(_coerce_pass(item) for item in passes)
        self.fixed_point = bool(fixed_point)
        self.max_rounds = int(max_rounds)
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")

    def run(self, circuit: Circuit) -> Circuit:
        if not hasattr(circuit, "n_qubits") or not has_circuit_instructions(circuit):
            raise TypeError("PassManager.run expects a Circuit or CircuitIR-like object")
        if not isinstance(circuit, Circuit):
            circuit = Circuit(*circuit_gate_dicts(circuit), n_qubits=int(circuit.n_qubits))

        current = circuit
        rounds = self.max_rounds if self.fixed_point else 1
        for _ in range(rounds):
            before = circuit_gate_dicts(current)
            for item in self.passes:
                current = item.run(current)
            if not self.fixed_point or circuit_gate_dicts(current) == before:
                return current
        return current

    def run_with_result(self, circuit: Circuit):
        """运行流水线并返回 :class:`TranspileResult`（NEXT.md 第 9 节）。

        记录编译前后深度、pass 名序列与布局映射（取自含 ``last_layout`` 的 pass，
        如 ``LayoutPass``；无则为 ``None``）。``circuit`` 返回值同 :meth:`run`。
        """
        from ..metrics._utils import depth_proxy
        from .result import TranspileResult

        if not hasattr(circuit, "n_qubits") or not has_circuit_instructions(circuit):
            raise TypeError("PassManager.run_with_result expects a Circuit or CircuitIR-like object")
        if not isinstance(circuit, Circuit):
            circuit = Circuit(*circuit_gate_dicts(circuit), n_qubits=int(circuit.n_qubits))

        depth_before = int(depth_proxy(circuit))
        result_circuit = self.run(circuit)
        # 按 pass 顺序组合各 last_layout：LayoutPass(logical->physical) 与
        # RoutingPass(physical->physical 置换) 链式组合为 logical->final wire，
        # 即 composed[q] = later[earlier[q]]。
        layout = None
        for item in self.passes:
            mapping = getattr(item, "last_layout", None)
            if mapping is None:
                continue
            if layout is None:
                layout = dict(mapping)
            else:
                layout = {q: mapping.get(p, p) for q, p in layout.items()}
        return TranspileResult(
            circuit=result_circuit,
            layout=layout,
            passes=tuple(getattr(item, "name", type(item).__name__) for item in self.passes),
            depth_before=depth_before,
            depth_after=int(depth_proxy(result_circuit)),
        )


def _optimize_pipeline(*, max_rounds: int = 64, max_reorder_hops: int = 8) -> PassManager:
    """构造默认的本地线路优化流水线（cancel→merge→commute，跑到不动点）。"""

    from .passes import CancelInversePass, CommuteSingleQubitPass, MergeRotationsPass

    return PassManager(
        [
            CancelInversePass(),
            MergeRotationsPass(),
            CommuteSingleQubitPass(max_reorder_hops=max_reorder_hops),
        ],
        fixed_point=True,
        max_rounds=max_rounds,
    )


def optimize(circuit: Circuit, *, max_rounds: int = 64, max_reorder_hops: int = 8) -> Circuit:
    """对线路应用默认本地优化流水线，返回优化后的新线路。

    线路结构优化的统一入口；等价于
    ``_optimize_pipeline(...).run(circuit)``。需要自定义 pass 顺序时
    直接用 :class:`PassManager`。
    """

    return _optimize_pipeline(max_rounds=max_rounds, max_reorder_hops=max_reorder_hops).run(circuit)
