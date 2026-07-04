"""Basic validation and canonicalization passes."""

from __future__ import annotations

from dataclasses import replace

from ...core.circuit import Circuit
from ...gates import canonical_gate_name
from ...ir import (
    circuit_gate_dicts,
    circuit_instructions,
    instruction_controls,
    instruction_name,
    Operation,
)
from ..base import TransformationPass
from ._local_rewrite import circuit_from_gates


class ValidatePass(TransformationPass):
    """实质校验：qubit/控制位越界、目标与控制冲突、重复比特。

    已注册门的目标比特数/参数个数/控制位要求在构造 typed IR 时由
    GateSpec 自动校验（``Operation._validate_against_spec``），本 pass
    负责需要线路上下文（``n_qubits``）的结构检查。
    """

    def run(self, circuit: Circuit) -> Circuit:
        n_qubits = int(circuit.n_qubits)
        for index, instruction in enumerate(circuit_instructions(circuit)):
            name = instruction_name(instruction)
            qubits = instruction.qubits
            controls = instruction_controls(instruction)

            for qubit in (*qubits, *controls):
                if qubit < 0 or qubit >= n_qubits:
                    raise ValueError(
                        f"gate #{index} '{name}': qubit {qubit} out of range "
                        f"for {n_qubits}-qubit circuit"
                    )
            if len(set(qubits)) != len(qubits):
                raise ValueError(f"gate #{index} '{name}': duplicate target qubits {qubits}")
            if len(set(controls)) != len(controls):
                raise ValueError(f"gate #{index} '{name}': duplicate control qubits {controls}")
            overlap = set(qubits) & set(controls)
            if overlap:
                raise ValueError(
                    f"gate #{index} '{name}': target and control qubits overlap: {sorted(overlap)}"
                )
        return circuit_from_gates(circuit, circuit_gate_dicts(circuit))


class CanonicalizePass(TransformationPass):
    """实质规范化：把别名门名（``X``/``cnot``/``ccnot`` 等）重写为 GateSpec 规范名。

    未注册的门名原样保留；其余字段不变。同时经过 ``Circuit`` 构造器
    round-trip 完成字典格式归一。
    """

    def run(self, circuit: Circuit) -> Circuit:
        instructions = []
        for instruction in circuit_instructions(circuit):
            canonical = canonical_gate_name(instruction_name(instruction))
            if isinstance(instruction, Operation) and canonical != instruction.name:
                instruction = replace(instruction, name=canonical)
            instructions.append(instruction)
        return Circuit(*instructions, n_qubits=circuit.n_qubits, backend=getattr(circuit, "backend", None))
