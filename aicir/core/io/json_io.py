"""
aicir/core/io/json_io.py

Circuit 的 JSON 序列化与反序列化。

与当前仓库的 `Circuit` 对象兼容：
- circuit.n_qubits
- typed IR operations / circuit.gates
- legacy dict snapshots from circuit.to_gate_dicts()
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ...ir import circuit_gate_dicts, has_circuit_instructions
from ..circuit import Circuit, Parameter

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for JSON IO.
    torch = None

_FORMAT_VERSION = "1.0"


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, Parameter):
        return {"__aicir_type__": "Parameter", "name": value.name}
    if isinstance(value, np.ndarray):
        return {
            "__aicir_type__": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": _jsonable_value(value.tolist()),
        }
    if isinstance(value, np.generic):
        return _jsonable_value(value.item())
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim == 0:
            return _jsonable_value(tensor.item())
        return _jsonable_value(tensor.numpy())
    if isinstance(value, complex):
        return {"__aicir_type__": "complex", "real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {key: _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    return deepcopy(value)


def _restore_json_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore_json_value(item) for item in value]
    if not isinstance(value, dict):
        return value

    marker = value.get("__aicir_type__")
    if marker == "Parameter":
        return Parameter(value["name"])
    if marker == "complex":
        return complex(value["real"], value["imag"])
    if marker == "ndarray":
        data = _restore_json_value(value["data"])
        return np.asarray(data, dtype=np.dtype(value["dtype"])).reshape(value["shape"])

    return {key: _restore_json_value(item) for key, item in value.items()}


def circuit_to_json_dict(circuit: Circuit) -> Dict[str, Any]:
    """将 Circuit 转换为可 JSON 序列化的 Python 字典。"""
    if not hasattr(circuit, "n_qubits") or not has_circuit_instructions(circuit):
        raise TypeError("circuit 需要具备 n_qubits 和 typed IR operations 或 gates 序列")

    return {
        "format": "aicir.circuit",
        "version": _FORMAT_VERSION,
        "n_qubits": int(circuit.n_qubits),
        "gates": _jsonable_value(circuit_gate_dicts(circuit)),
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
    gates = _restore_json_value(data["gates"])
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
