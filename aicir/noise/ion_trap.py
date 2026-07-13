"""Unified ion-trap noise configuration loader.

This module reads a flat Markdown parameter table by default and turns it into
a runtime NoiseModel for the current QAS stack.

The Markdown file is the single source of truth. It keeps the numbers human-
readable and makes it easy to swap in a different hardware profile by editing
one file only.
"""

from __future__ import annotations

import ast
import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import (
    BitFlipChannel,
    DepolarizingChannel,
    NoiseModel,
    PhaseFlipChannel,
)

DEFAULT_MD_NAME = "README.md"

TRUE_VALUES = {"true", "1", "yes", "on"}
FALSE_VALUES = {"false", "0", "no", "off"}

ONEQ_GATE_TYPES = {
    "pauli_x",
    "pauli_y",
    "pauli_z",
    "hadamard",
    "rx",
    "ry",
    "rz",
    "s_gate",
    "t_gate",
    "x",
    "y",
    "z",
    "h",
    "s",
    "t",
    "u2",
    "u3",
}

TWOQ_GATE_TYPES = {
    "cx",
    "cnot",
    "cy",
    "cz",
    "crx",
    "cry",
    "crz",
    "swap",
    "toffoli",
    "ccnot",
    "rzz",
    "rxx",
    "zz",
    "zz_opt",
    "zz_sp_opt",
    "cz_opt",
    "cz_sp_opt",
    "cz_from_cx",
    "zz_from_cx",
}


def _safe_eval_numeric_expr(value: Any) -> Any:
    """Parse a numeric string safely.

    The flat parameter table uses literal floating-point values such as
    `2.8e-06` and `0.0005`, so we only need lightweight conversion logic.
    """
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return value

    try:
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        return float(text)
    except ValueError:
        return value


def _coerce_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    text = value.strip()
    lowered = text.lower()
    if lowered in TRUE_VALUES:
        return True
    if lowered in FALSE_VALUES:
        return False
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
        if isinstance(parsed, list):
            return parsed
    return _safe_eval_numeric_expr(text)


def _as_bool(value: Any, default: bool = True) -> bool:
    value = _coerce_scalar(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _float_or_none(value: Any) -> Optional[float]:
    value = _safe_eval_numeric_expr(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _apply_2025_document_formula(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Derive runtime noise parameters according to the 2025 ion-trap report."""
    resolved = deepcopy(parameters)

    delta_f1 = _float_or_none(resolved.get("oneq_avg_infidelity_deltaF1"))
    if delta_f1 is not None:
        # Report formula: p_s = 3/2 * deltaF^(1)
        resolved["oneq_depol"] = 1.5 * delta_f1

    delta_f2 = _float_or_none(resolved.get("twoq_avg_infidelity_deltaF2"))
    if delta_f2 is None:
        fidelity_f2 = _float_or_none(resolved.get("twoq_avg_fidelity_F2"))
        if fidelity_f2 is not None:
            delta_f2 = 1.0 - fidelity_f2
            resolved["twoq_avg_infidelity_deltaF2"] = delta_f2
    if delta_f2 is not None:
        # Report formula: p_MS = 5/4 * deltaF
        resolved["twoq_depol"] = 1.25 * delta_f2

    cross_talk = _float_or_none(resolved.get("cross_talk"))
    if cross_talk is not None:
        resolved["cross_talk"] = cross_talk

    resolved["formula_profile"] = "ion_trap_doc_2025"
    return resolved


def _duration_to_seconds(value: Any) -> Optional[float]:
    value = _safe_eval_numeric_expr(value)
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        return None

    text = value.strip().lower()
    if text.endswith("us"):
        return float(text[:-2]) * 1e-6
    if text.endswith("ms"):
        return float(text[:-2]) * 1e-3
    if text.endswith("s"):
        return float(text[:-1])
    return None


@dataclass
class IonTrapNoiseConfig:
    """Unified ion-trap hardware noise configuration."""

    parameters: Dict[str, Any] = field(default_factory=dict)
    sources: Dict[str, Optional[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameters": deepcopy(self.parameters),
            "sources": deepcopy(self.sources),
        }

    def summary(self) -> str:
        resolved = self.resolved_parameters()
        lines = [
            "=== Ion Trap Unified Noise Config ===",
            f"md: {self.sources.get('md')}",
            "--- resolved parameters ---",
        ]

        for key in sorted(resolved):
            lines.append(f"{key}: {resolved[key]}")

        return "\n".join(lines)

    def resolved_parameters(self) -> Dict[str, Any]:
        """Return a flattened parameter dictionary ready for runtime use."""
        resolved = deepcopy(self.parameters)
        for key in list(resolved):
            resolved[key] = _coerce_scalar(resolved[key])
            if isinstance(resolved[key], str):
                try:
                    resolved[key] = float(resolved[key])
                except ValueError:
                    pass

        profile = str(resolved.get("formula_profile", "")).strip().lower()
        if profile == "ion_trap_doc_2025":
            resolved = _apply_2025_document_formula(resolved)

        return resolved

    def build_noise_model(
        self,
        qubits: Optional[Sequence[int]] = None,
        *,
        include_readout: Optional[bool] = None,
        include_idle: Optional[bool] = None,
    ) -> NoiseModel:
        """Create a runtime NoiseModel using the resolved parameters.

        The current simulator applies noise after gates, so readout/reset/idle
        channels are represented as optional gate-triggered rules for future
        compatibility. They only activate when the circuit contains gate types
        named `measure`, `reset`, or `idle`.
        """
        resolved = self.resolved_parameters()
        noise_model = NoiseModel()

        if qubits is not None:
            all_qubits = list(qubits)
        else:
            data_qubits = resolved.get("data_qubits", []) or []
            ancillas = resolved.get("ancillas", []) or []
            all_qubits = list(data_qubits) + list(ancillas)
            if not all_qubits:
                inferred_n_qubits = int(resolved.get("n_qubits", 0) or 0)
                all_qubits = list(range(inferred_n_qubits))

        oneq_p = float(resolved.get("oneq_depol", 0.0) or 0.0)
        twoq_p = float(resolved.get("twoq_depol", 0.0) or 0.0)
        readout_p = float(resolved.get("meas_bitflip", 0.0) or 0.0)
        reset_p = float(resolved.get("reset_bitflip", readout_p) or readout_p)
        cross_talk_p = float(resolved.get("cross_talk", 0.0) or 0.0)

        enable_initialization = _as_bool(resolved.get("enable_initialization_noise"), True)
        enable_measurement = _as_bool(resolved.get("enable_measurement_noise"), True)
        enable_oneq = _as_bool(resolved.get("enable_oneq_gate_noise"), True)
        enable_twoq = _as_bool(resolved.get("enable_twoq_gate_noise"), True)
        enable_crosstalk = _as_bool(resolved.get("enable_crosstalk_noise"), True)
        enable_idle = _as_bool(resolved.get("enable_idle_dephasing_noise"), True)
        if include_readout is not None:
            enable_initialization = bool(include_readout)
            enable_measurement = bool(include_readout)
        if include_idle is not None:
            enable_idle = bool(include_idle)

        if enable_oneq and oneq_p > 0:
            for qubit in all_qubits:
                noise_model.add_channel(
                    DepolarizingChannel(target_qubit=int(qubit), p=oneq_p),
                    after_gates=sorted(ONEQ_GATE_TYPES),
                )

        if enable_twoq and twoq_p > 0:
            effective_twoq_p = twoq_p / 2.0
            for qubit in all_qubits:
                noise_model.add_channel(
                    DepolarizingChannel(target_qubit=int(qubit), p=effective_twoq_p),
                    after_gates=sorted(TWOQ_GATE_TYPES),
                )

        if enable_crosstalk and cross_talk_p > 0:
            # Effective crosstalk is modeled as a small additional depolarizing
            # channel on every qubit after addressed operations.
            for qubit in all_qubits:
                noise_model.add_channel(
                    DepolarizingChannel(target_qubit=int(qubit), p=cross_talk_p),
                    after_gates=sorted(ONEQ_GATE_TYPES | TWOQ_GATE_TYPES),
                )

        if enable_measurement and readout_p > 0:
            for qubit in all_qubits:
                noise_model.add_channel(
                    BitFlipChannel(target_qubit=int(qubit), p=readout_p),
                    after_gates=["measure"],
                )
            if enable_initialization and reset_p > 0:
                for qubit in all_qubits:
                    noise_model.add_channel(
                        BitFlipChannel(target_qubit=int(qubit), p=reset_p),
                        after_gates=["reset"],
                    )

        if enable_idle:
            idle_oneq_p = self.idle_dephasing_probability(gate_family="oneq")
            idle_twoq_p = self.idle_dephasing_probability(gate_family="twoq")
            if idle_oneq_p > 0:
                for qubit in all_qubits:
                    noise_model.add_channel(
                        PhaseFlipChannel(target_qubit=int(qubit), p=idle_oneq_p),
                        after_gates=sorted(ONEQ_GATE_TYPES),
                        exclude_gate_qubits=True,
                    )
            if idle_twoq_p > 0:
                for qubit in all_qubits:
                    noise_model.add_channel(
                        PhaseFlipChannel(target_qubit=int(qubit), p=idle_twoq_p),
                        after_gates=sorted(TWOQ_GATE_TYPES),
                        exclude_gate_qubits=True,
                    )

        return noise_model

    def idle_dephasing_probability(self, *, gate_family: str = "twoq") -> float:
        """Derive idle dephasing probability using the 2025 document formula.

        Document formula:
            p_idle = 1/2 * (1 - exp(-t / T2))

        Time selection strategy:
        - oneq: t = oneq_gate_time
        - twoq: t = twoq_gate_time
        """
        resolved = self.resolved_parameters()
        t2 = float(resolved.get("T2", 0.0) or 0.0)
        if t2 <= 0:
            return 0.0

        strategy = str(resolved.get("idle_time_strategy", "switch_by_gate_arity")).strip().lower()
        if strategy != "switch_by_gate_arity":
            raise ValueError(f"不支持的 idle_time_strategy: {strategy}")

        if gate_family == "oneq":
            duration = float(resolved.get("oneq_gate_time", 0.0) or 0.0)
        elif gate_family == "twoq":
            duration = float(resolved.get("twoq_gate_time", 0.0) or 0.0)
        else:
            raise ValueError(f"不支持的 gate_family: {gate_family}")

        if duration <= 0:
            return 0.0
        return float(0.5 * (1.0 - math.exp(-duration / t2)))


def _parse_parameter_lines(lines: Iterable[str], md_path: Path) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            line = line[1:].strip()
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.split("#", 1)[0].strip()
        if not key:
            continue

        parameters[key] = _coerce_scalar(value)

    if not parameters:
        raise ValueError(f"Markdown 参数文件中没有解析到任何参数: {md_path}")

    return parameters


def _extract_fenced_parameter_lines(lines: Sequence[str]) -> List[str]:
    blocks: List[List[str]] = []
    current: Optional[List[str]] = None
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            if current is None:
                current = []
            else:
                blocks.append(current)
                current = None
            continue
        if current is not None:
            current.append(raw_line)

    for block in blocks:
        if any(line.strip().startswith("formula_profile:") for line in block):
            return block
    return []


def _load_markdown_parameters(md_path: Path) -> Dict[str, Any]:
    if not md_path.exists():
        raise FileNotFoundError(f"找不到 Markdown 参数文件: {md_path}")

    lines = md_path.read_text(encoding="utf-8").splitlines()
    fenced_lines = _extract_fenced_parameter_lines(lines)
    if fenced_lines:
        return _parse_parameter_lines(fenced_lines, md_path)
    return _parse_parameter_lines(lines, md_path)


def load_ion_trap_noise_config(
    md_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> IonTrapNoiseConfig:
    """Load the default ion-trap parameters from a Markdown file."""
    default_md = Path(__file__).with_name(DEFAULT_MD_NAME)
    md_path = Path(md_path) if md_path is not None else default_md

    parameters = _load_markdown_parameters(md_path)
    config = IonTrapNoiseConfig(
        parameters=parameters,
        sources={"md": str(md_path)},
    )

    if overrides:
        config = IonTrapNoiseConfig(**_deep_update(config.to_dict(), overrides))

    return config


def load_default_ion_trap_noise_config() -> IonTrapNoiseConfig:
    """Convenience wrapper for the repository default ion-trap parameters."""
    return load_ion_trap_noise_config()


__all__ = [
    "IonTrapNoiseConfig",
    "load_ion_trap_noise_config",
    "load_default_ion_trap_noise_config",
    "DEFAULT_MD_NAME",
]
