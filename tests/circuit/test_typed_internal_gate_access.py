import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AICIR_ROOT = PROJECT_ROOT / "aicir"

LEGACY_GATE_DICT_BOUNDARIES = {
    # Typed IR conversion and legacy read-only compatibility layer.
    "aicir/ir/accessors.py",
    "aicir/ir/circuit_ir.py",
    "aicir/ir/control_flow.py",
    "aicir/ir/measurement.py",
    "aicir/ir/operation.py",
    # Legacy transpile APIs that explicitly operate on gate-dict lists.
    "aicir/transpile/passes/_local_rewrite.py",
    "aicir/transpile/rewrite.py",
    # QAS 显式门序列基因：gates 字段是 JSON DTO（json.dumps 往返、to_jsonable 导出），
    # 属于允许的 dict 互操作边界，而非运行时线路读取。
    "aicir/qas/library/ansatz.py",
}

LEGACY_GATE_DICT_PREFIXES = (
    "aicir/core/io/",
    "aicir/qas/demos/",
)

GATE_FIELD_KEYS = {
    "basis",
    "classical_bit",
    "classical_bits",
    "clbits",
    "control_qubits",
    "control_states",
    "id",
    "n_qubits",
    "parameter",
    "qubit_1",
    "qubit_2",
    "qubit_3",
    "qubit_4",
    "qubits",
    "target_qubit",
    "targets",
    "type",
}


def _is_legacy_boundary(path: Path) -> bool:
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    return rel in LEGACY_GATE_DICT_BOUNDARIES or rel.startswith(LEGACY_GATE_DICT_PREFIXES)


def _name_looks_like_gate(value: ast.AST) -> bool:
    if isinstance(value, ast.Name):
        name = value.id.lower()
        return name in {"g", "gate", "gate_dict"} or "gate" in name
    if isinstance(value, ast.Subscript):
        return _name_looks_like_gate(value.value)
    return False


def _constant_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _legacy_gate_field_uses(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and _name_looks_like_gate(node.value):
            key = _constant_string(node.slice)
            if key in GATE_FIELD_KEYS:
                findings.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} uses gate[{key!r}]")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _name_looks_like_gate(node.func.value)
            and node.args
        ):
            key = _constant_string(node.args[0])
            if key in GATE_FIELD_KEYS:
                findings.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} uses gate.get({key!r})")
    return findings


def test_runtime_code_does_not_read_circuit_gates_as_legacy_dicts():
    findings: list[str] = []
    for path in sorted(AICIR_ROOT.rglob("*.py")):
        if _is_legacy_boundary(path):
            continue
        findings.extend(_legacy_gate_field_uses(path))

    assert findings == []
