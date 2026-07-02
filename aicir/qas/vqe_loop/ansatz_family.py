"""Ansatz-family capability declarations for VQE-QAS planning.

This module keeps family-specific support explicit without forcing every
family into the supernet-native path.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping, Sequence


_DEFAULT_CAPABILITIES: dict[str, Any] = {
    "candidate_generation": False,
    "mutation": False,
    "oracle": False,
    "native_supernet_screening": False,
    "evaluators": {
        "E2": True,
        "E5": False,
        "VQE_TASK_PROXY": False,
        "GNN_PROXY": False,
        "fair_label": True,
    },
}

_FAMILY_CAPABILITIES: dict[str, dict[str, Any]] = {
    "supernet_native": {
        "candidate_generation": True,
        "mutation": True,
        "oracle": True,
        "native_supernet_screening": True,
        "evaluators": {
            "E2": True,
            "E5": True,
            "VQE_TASK_PROXY": True,
            "GNN_PROXY": True,
            "fair_label": True,
        },
    },
    "operator_sequence": {
        "candidate_generation": True,
        "mutation": True,
        "oracle": True,
        "native_supernet_screening": False,
        "evaluators": {
            "E2": True,
            "E5": False,
            "VQE_TASK_PROXY": False,
            "GNN_PROXY": False,
            "fair_label": True,
        },
    },
    "chemistry_excitation": {
        "candidate_generation": True,
        "mutation": True,
        "oracle": True,
        "native_supernet_screening": False,
        "evaluators": {
            "E2": True,
            "E5": False,
            "VQE_TASK_PROXY": False,
            "GNN_PROXY": False,
            "fair_label": True,
        },
    },
}


def ansatz_family_from_row(row: Mapping[str, Any]) -> str:
    family = str(row.get("family", "") or "").strip().lower()
    if family:
        return family
    raw = row.get("ansatz_gene")
    if raw is None or str(raw).strip() == "":
        return "unknown"
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError, json.JSONDecodeError):
        return "unknown"
    if isinstance(payload, Mapping):
        return str(payload.get("kind", "unknown") or "unknown").strip().lower()
    return "unknown"


def ansatz_family_capabilities(family: str) -> dict[str, Any]:
    normalized = str(family or "unknown").strip().lower()
    capabilities = deepcopy(_DEFAULT_CAPABILITIES)
    override = _FAMILY_CAPABILITIES.get(normalized)
    if override is None:
        return capabilities
    for key, value in override.items():
        if key == "evaluators":
            capabilities["evaluators"].update(value)
        else:
            capabilities[key] = value
    return capabilities


def summarize_ansatz_families(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for row in rows:
        family = ansatz_family_from_row(row)
        if family not in summary:
            summary[family] = {"count": 0, "capabilities": ansatz_family_capabilities(family)}
        summary[family]["count"] += 1
    return summary


__all__ = [
    "ansatz_family_capabilities",
    "ansatz_family_from_row",
    "summarize_ansatz_families",
]
