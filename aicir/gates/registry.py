"""GateSpec 注册表：按规范名或别名查询门元信息。

设计约定：

- 查询未注册的门名返回 ``None``，调用方应保持宽松（自定义门不受限）。
- 注册表对 ``aicir`` 内置门集在导入时填充一次；新增内置门时在
  ``_STANDARD_GATES`` 中补一条记录即可，矩阵/QASM/绘图等模块后续
  逐步改为从这里读取元信息。
"""

from __future__ import annotations

from .decompositions import decompose_cy, decompose_cz, decompose_swap
from .spec import GateSpec

_REGISTRY: dict[str, GateSpec] = {}
_LOOKUP: dict[str, GateSpec] = {}


def register_gate(spec: GateSpec, *, overwrite: bool = False) -> GateSpec:
    """注册一个门规格；``overwrite=False`` 时与现有名称/别名冲突会报错。"""

    if not isinstance(spec, GateSpec):
        raise TypeError("register_gate expects a GateSpec")

    names = (spec.name, *spec.aliases)
    if not overwrite:
        for name in names:
            existing = _LOOKUP.get(name)
            if existing is not None and existing.name != spec.name:
                raise ValueError(f"gate name or alias already registered: {name!r}")
        if spec.name in _REGISTRY:
            raise ValueError(f"gate already registered: {spec.name!r}")
    else:
        unregister_gate(spec.name)

    _REGISTRY[spec.name] = spec
    for name in names:
        _LOOKUP[name] = spec
    return spec


def unregister_gate(name: str) -> None:
    """移除一个已注册门（含其全部别名）；未注册时静默返回。"""

    spec = _REGISTRY.pop(str(name), None)
    if spec is None:
        return
    for key in (spec.name, *spec.aliases):
        if _LOOKUP.get(key) is spec:
            del _LOOKUP[key]


def get_gate_spec(name: str) -> GateSpec | None:
    """按规范名或别名查询门规格；未注册返回 ``None``。"""

    return _LOOKUP.get(str(name))


def registered_gate_names() -> tuple[str, ...]:
    """返回全部已注册门的规范名。"""

    return tuple(_REGISTRY)


def canonical_gate_name(name: str) -> str:
    """把门名（含别名）解析为规范名；未注册的名称原样返回。"""

    spec = _LOOKUP.get(str(name))
    return spec.name if spec is not None else str(name)


def gate_generator(name: str) -> str | None:
    """返回门的 Pauli 生成元标签（``rx``->``"X"``、``rzz``->``"ZZ"`` 等）。

    非（标准 Pauli）参数化旋转门、或未注册门返回 ``None``。供 QML 自省门类
    能否使用解析参数移位，替代散落的硬编码门集。
    """

    spec = _LOOKUP.get(str(name))
    return spec.generator if spec is not None else None


def gate_shift_rule(name: str) -> str | None:
    """返回该门的参数移位规则类别（``two_term``/``four_term``/``None``）。"""
    spec = get_gate_spec(name)
    return spec.shift_rule if spec is not None else None


def parametric_pauli_gates() -> frozenset[str]:
    """返回所有带 Pauli 生成元的门规范名集合（即解析参数移位适用门）。"""

    return frozenset(spec.name for spec in _REGISTRY.values() if spec.generator is not None)


def gate_decomposition(name: str):
    """返回门的分解规则可调用；无规则或未注册门返回 ``None``。

    规则签名见 :mod:`aicir.gates.decompositions`：
    ``(qubits, controls, control_states, params) -> list[dict] | None``。
    """

    spec = _LOOKUP.get(str(name))
    return spec.decomposition if spec is not None else None


# ---------------------------------------------------------------------------
# aicir 内置门集（与 gate_to_matrix / QASM 导出表保持一致）
# ---------------------------------------------------------------------------

_STANDARD_GATES = (
    GateSpec("pauli_x", 1, 0, aliases=("X",), qasm_name="x", symbol="X"),
    GateSpec("pauli_y", 1, 0, aliases=("Y",), qasm_name="y", symbol="Y"),
    GateSpec("pauli_z", 1, 0, aliases=("Z",), qasm_name="z", symbol="Z"),
    GateSpec("hadamard", 1, 0, aliases=("H",), qasm_name="h", symbol="H"),
    GateSpec("s_gate", 1, 0, aliases=("S",), qasm_name="s", symbol="S"),
    GateSpec("t_gate", 1, 0, aliases=("T",), qasm_name="t", symbol="T"),
    # identity 允许整寄存器形式 {"type": "identity", "n_qubits": n}（QAS 动作空间），
    # 因此目标比特数可变。
    GateSpec("identity", None, 0, aliases=("I",), qasm_name="id", symbol="I"),
    GateSpec("rx", 1, 1, qasm_name="rx", symbol="Rx", generator="X"),
    GateSpec("ry", 1, 1, qasm_name="ry", symbol="Ry", generator="Y"),
    GateSpec("rz", 1, 1, qasm_name="rz", symbol="Rz", generator="Z"),
    GateSpec("u2", 1, 2, qasm_name="u2", symbol="U2"),
    GateSpec("u3", 1, 3, qasm_name="u3", symbol="U3"),
    # num_qubits=None：cx/cnot 支持单目标或多目标（多目标等价于多个单目标 CX）。
    GateSpec("cx", None, 0, aliases=("cnot",), controlled=True, num_controls=1, qasm_name="cx", symbol="X"),
    GateSpec("cy", 1, 0, controlled=True, num_controls=1, qasm_name="cy", symbol="Y", decomposition=decompose_cy),
    GateSpec("cz", 1, 0, controlled=True, num_controls=1, qasm_name="cz", symbol="Z", decomposition=decompose_cz),
    GateSpec("crx", 1, 1, controlled=True, num_controls=1, qasm_name="crx", symbol="Rx", generator="X"),
    GateSpec("cry", 1, 1, controlled=True, num_controls=1, qasm_name="cry", symbol="Ry", generator="Y"),
    GateSpec("crz", 1, 1, controlled=True, num_controls=1, qasm_name="crz", symbol="Rz", generator="Z"),
    GateSpec("toffoli", 1, 0, aliases=("ccnot",), controlled=True, num_controls=2, qasm_name="ccx", symbol="X"),
    # 这些门在绘图中使用专门形状，不携带通用 symbol。
    GateSpec("swap", 2, 0, qasm_name="swap", decomposition=decompose_swap),
    GateSpec("rzz", 2, 1, qasm_name="rzz", generator="ZZ"),
    GateSpec("rxx", 2, 1, qasm_name="rxx", generator="XX"),
    GateSpec("single_excitation", 2, 1, aliases=("givens",), qasm_name=None, shift_rule="four_term"),
    GateSpec("double_excitation", 4, 1, qasm_name=None, shift_rule="four_term"),
    # unitary 的矩阵经 "parameter" 携带，但绘图占位场景允许缺省，故参数个数可变。
    GateSpec("unitary", None, None, symbol="U"),
    GateSpec("measure", None, 0, aliases=("measurement",)),
    GateSpec("reset", None, 0, symbol="|0>"),
)

for _spec in _STANDARD_GATES:
    register_gate(_spec)
del _spec
