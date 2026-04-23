"""
quantum_sim/circuit/io/json_io.py

Circuit 的 JSON 序列化与反序列化。

与当前仓库的 `Circuit` 对象兼容：
- circuit.n_qubits
- circuit.gates (list[dict])
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from ..model import Circuit

_FORMAT_VERSION = "1.0"


def circuit_to_json_dict(circuit: Circuit) -> Dict[str, Any]:
    """将 Circuit 转换为可 JSON 序列化的 Python 字典。"""
    if not hasattr(circuit, "n_qubits") or not hasattr(circuit, "gates"):
        raise TypeError("circuit 需要具备 n_qubits 和 gates 属性")

    return {
        "format": "quantum_sim.circuit",
        "version": _FORMAT_VERSION,
        "n_qubits": int(circuit.n_qubits),
        "gates": deepcopy(list(circuit.gates)),
    }


def circuit_to_json(circuit: Circuit, indent: int = 2) -> str:
    """将 Circuit 序列化为 JSON 字符串。"""
    payload = circuit_to_json_dict(circuit)
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def circuit_from_json_dict(data: Dict[str, Any]) -> Circuit:
    """从 JSON 字典重建 Circuit。"""
    if not isinstance(data, dict):
        raise TypeError("data 必须是 dict")

    if "n_qubits" not in data or "gates" not in data:
        raise ValueError("JSON 缺少必要字段：n_qubits 或 gates")

    n_qubits = int(data["n_qubits"])
    gates = data["gates"]
    if not isinstance(gates, list):
        raise ValueError("gates 必须是 list")

    return Circuit(*gates, n_qubits=n_qubits)


def circuit_from_json(json_text: str) -> Circuit:
    """从 JSON 字符串反序列化 Circuit。"""
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"非法 JSON: {exc}") from exc
    return circuit_from_json_dict(data)


def save_circuit_json(circuit: Circuit, file_path: str | Path, indent: int = 2) -> None:
    """将 Circuit 保存为 JSON 文件。"""
    path = Path(file_path)
    path.write_text(circuit_to_json(circuit, indent=indent), encoding="utf-8")


def load_circuit_json(file_path: str | Path) -> Circuit:
    """从 JSON 文件加载 Circuit。"""
    path = Path(file_path)
    return circuit_from_json(path.read_text(encoding="utf-8"))
