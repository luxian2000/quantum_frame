"""Shared noise-related circuit metrics."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ...core.circuit import Circuit
from ...ir import circuit_instructions, instruction_controls, instruction_name, instruction_qubits
from .ion_trap import ONEQ_GATE_TYPES, TWOQ_GATE_TYPES, load_default_ion_trap_noise_config


def ion_trap_error_budget_proxy(circuit: Circuit) -> Tuple[float, Dict[str, Any]]:
    """Estimate an ion-trap error-budget score for a circuit.

    Returns a score in ``[0, 1]`` together with raw budget details. The score
    is intentionally hardware/noise-layer logic; QAS can use it as one scoring
    signal, but the calculation is reusable by other algorithms.
    """
    config = load_default_ion_trap_noise_config()
    resolved = config.resolved_parameters()

    oneq_gate_count = 0
    twoq_gate_count = 0
    measure_count = 0
    reset_count = 0
    for gate in circuit_instructions(circuit):
        gate_type = instruction_name(gate)
        if gate_type == "measure":
            measure_count += 1
        elif gate_type == "reset":
            reset_count += 1
        elif gate_type in TWOQ_GATE_TYPES or instruction_controls(gate) or gate_type in {"swap", "rzz", "rxx"}:
            twoq_gate_count += 1
        elif gate_type in ONEQ_GATE_TYPES or instruction_qubits(gate):
            oneq_gate_count += 1

    n_qubits = int(circuit.n_qubits)
    oneq_p = float(resolved.get("oneq_depol", 0.0) or 0.0) if resolved.get("enable_oneq_gate_noise", True) else 0.0
    twoq_p = float(resolved.get("twoq_depol", 0.0) or 0.0) if resolved.get("enable_twoq_gate_noise", True) else 0.0
    crosstalk_p = float(resolved.get("cross_talk", 0.0) or 0.0) if resolved.get("enable_crosstalk_noise", True) else 0.0
    measure_p = float(resolved.get("meas_bitflip", 0.0) or 0.0) if resolved.get("enable_measurement_noise", True) else 0.0
    reset_p = float(resolved.get("reset_bitflip", 0.0) or 0.0) if resolved.get("enable_initialization_noise", True) else 0.0
    if resolved.get("enable_idle_dephasing_noise", True):
        idle_oneq_p = config.idle_dephasing_probability(gate_family="oneq")
        idle_twoq_p = config.idle_dephasing_probability(gate_family="twoq")
    else:
        idle_oneq_p = 0.0
        idle_twoq_p = 0.0

    gate_error_budget = oneq_gate_count * oneq_p + twoq_gate_count * twoq_p
    idle_error_budget = (
        oneq_gate_count * max(n_qubits - 1, 0) * idle_oneq_p
        + twoq_gate_count * max(n_qubits - 2, 0) * idle_twoq_p
    )
    crosstalk_error_budget = (oneq_gate_count + twoq_gate_count) * n_qubits * crosstalk_p
    readout_reset_error_budget = measure_count * measure_p + reset_count * reset_p
    total_error_budget = gate_error_budget + idle_error_budget + crosstalk_error_budget + readout_reset_error_budget
    score = float(np.exp(-max(0.0, total_error_budget)))

    return score, {
        "oneq_gate_count": oneq_gate_count,
        "twoq_gate_count": twoq_gate_count,
        "measure_count": measure_count,
        "reset_count": reset_count,
        "oneq_depol": oneq_p,
        "twoq_depol": twoq_p,
        "cross_talk": crosstalk_p,
        "idle_oneq": idle_oneq_p,
        "idle_twoq": idle_twoq_p,
        "gate_error_budget": gate_error_budget,
        "idle_error_budget": idle_error_budget,
        "crosstalk_error_budget": crosstalk_error_budget,
        "readout_reset_error_budget": readout_reset_error_budget,
        "total_error_budget": total_error_budget,
    }


__all__ = ["ion_trap_error_budget_proxy"]
