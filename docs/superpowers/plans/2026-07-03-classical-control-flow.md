# 经典控制流（if/else/while）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 aicir 加基于测量结果的经典控制流：`ClassicalRegister`/`Bit`、`measure→creg`、`if_(含else)`/`while_`，运行在测量轨迹执行路径上。

**Architecture:** 控制流是每-shot 运行时概念，落在 `run_trajectory` 的递归逐操作执行器（非 `unitary()`）。控制流作为携带嵌套 body（gate-dict 元组 + n_qubits）的 IR 指令 `ControlFlow`；条件 `Condition` 存 `(register_name, index)` 而非活对象，故求值与序列化都不依赖活寄存器实例。

**Tech Stack:** Python，numpy（唯一硬依赖）。纯态 shot 采样，`NumpyBackend` 即可，不强制 torch。

## Global Constraints

- 仓库根目录 `PYTHONPATH=.` 运行；测试 `PYTHONPATH=. pytest`。
- `numpy` 唯一硬依赖。注释/docstring/README 中文，跟随周边风格。
- 经典数据模型用 `ClassicalRegister`/`Bit`；条件仅 `==` / `!=`（位或整个寄存器整数值，`creg[0]` 为 LSB）。
- `while_` 的 `max_iterations` keyword-only 必填；达上限仍满足条件抛 `RuntimeError`（不静默截断）。
- `if_` 含可选 `else_body`。序列化仅 JSON。QASM3 控制流导出为非目标。
- `Condition` 内部存 `(register_name, index|None, op, value)`；`index=None` 表示整个寄存器。
- measure 有经典目标时：per-qubit Z 基投影，比特 `|0>→0 / |1>→1`；basis≠Z 抛 `ValueError`；cbits 跨多寄存器抛 `ValueError`；位数不符抛 `ValueError`。无经典目标时行为**完全不变**（向后兼容）。
- `unitary()` 与张量网络引擎遇控制流指令抛 `ValueError`。
- 每任务结束提交；提交信息末尾 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。

---

### Task 1: 经典数据模型 `aicir/core/classical.py`

**Files:**
- Create: `aicir/core/classical.py`
- Modify: `aicir/core/__init__.py`（导出 `ClassicalRegister`）、`aicir/__init__.py`（顶层再导出 `ClassicalRegister`）
- Test: `tests/core/test_classical.py`（新建）

**Interfaces:**
- Produces:
  - `ClassicalRegister(size:int, name:str)`；`reg[i]->Bit`；`len(reg)`；`reg == v`/`reg != v -> Condition`；`.name`,`.size`。
  - `Bit`（`.register_name`,`.index`）；`bit == v`/`bit != v -> Condition`（v∈{0,1}）。
  - `Condition(register_name:str, index:int|None, op:str, value:int)`；`.evaluate(store:Mapping[str,list[int]])->bool`；`.to_dict()`/`Condition.from_dict(d)`。

- [ ] **Step 1: 写失败测试**

`tests/core/test_classical.py`：

```python
import pytest
from aicir.core.classical import Bit, ClassicalRegister, Condition


def test_register_indexing_and_len():
    reg = ClassicalRegister(3, "c")
    assert reg.name == "c" and len(reg) == 3
    b = reg[2]
    assert isinstance(b, Bit) and b.register_name == "c" and b.index == 2
    with pytest.raises(IndexError):
        reg[3]


def test_bit_condition_sugar():
    reg = ClassicalRegister(2, "c")
    cond = reg[0] == 1
    assert isinstance(cond, Condition)
    assert (cond.register_name, cond.index, cond.op, cond.value) == ("c", 0, "==", 1)
    ne = reg[1] != 0
    assert (ne.index, ne.op, ne.value) == (1, "!=", 0)
    with pytest.raises(ValueError):
        reg[0] == 2  # 位只能 0/1


def test_register_int_condition():
    reg = ClassicalRegister(2, "c")
    cond = reg == 3
    assert (cond.register_name, cond.index, cond.op, cond.value) == ("c", None, "==", 3)


def test_evaluate_lsb_convention():
    reg = ClassicalRegister(2, "c")
    store = {"c": [1, 0]}  # bit0=1,bit1=0 -> int 1 (LSB=bit0)
    assert (reg[0] == 1).evaluate(store) is True
    assert (reg[1] == 0).evaluate(store) is True
    assert (reg == 1).evaluate(store) is True
    assert (reg == 2).evaluate(store) is False
    assert (reg != 2).evaluate(store) is True


def test_evaluate_missing_register_defaults_zero():
    assert (ClassicalRegister(1, "x")[0] == 0).evaluate({}) is True


def test_condition_roundtrip():
    c = ClassicalRegister(2, "c") == 3
    assert Condition.from_dict(c.to_dict()).to_dict() == c.to_dict()


def test_hashable():
    reg = ClassicalRegister(2, "c")
    {reg, reg[0]}  # 不应抛错
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/core/test_classical.py -v`
Expected: FAIL（`ModuleNotFoundError: aicir.core.classical`）

- [ ] **Step 3: 实现 `aicir/core/classical.py`**

```python
"""经典寄存器与条件模型：支撑测量反馈的 if/while 控制流。

Condition 内部只存 (register_name, index, op, value)，不引用活寄存器对象，
因此求值与 JSON 序列化都与具体 ClassicalRegister 实例解耦。
"""

from __future__ import annotations

from collections.abc import Mapping

_VALID_OPS = ("==", "!=")


class Condition:
    """经典条件：位或整个寄存器整数值与常量的 == / != 比较。"""

    __slots__ = ("register_name", "index", "op", "value")

    def __init__(self, register_name: str, index: int | None, op: str, value: int):
        if op not in _VALID_OPS:
            raise ValueError(f"op 必须是 {_VALID_OPS} 之一，收到 {op!r}")
        self.register_name = str(register_name)
        self.index = None if index is None else int(index)
        self.op = op
        self.value = int(value)

    def evaluate(self, store: Mapping[str, list]) -> bool:
        bits = store.get(self.register_name)
        if bits is None:
            actual = 0  # 从未写入的寄存器默认全 0
        elif self.index is None:
            actual = sum(int(b) << i for i, b in enumerate(bits))  # LSB=bit0
        else:
            actual = int(bits[self.index])
        return actual == self.value if self.op == "==" else actual != self.value

    def to_dict(self) -> dict:
        return {
            "target": {"register": self.register_name, "index": self.index},
            "op": self.op,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "Condition":
        t = d["target"]
        return cls(t["register"], t.get("index"), d["op"], int(d["value"]))

    def __repr__(self) -> str:
        tgt = self.register_name if self.index is None else f"{self.register_name}[{self.index}]"
        return f"Condition({tgt} {self.op} {self.value})"


class Bit:
    """经典寄存器中的单个位引用（register_name, index）。"""

    __slots__ = ("register_name", "index")

    def __init__(self, register_name: str, index: int):
        self.register_name = str(register_name)
        self.index = int(index)

    def _cond(self, op: str, value: int) -> Condition:
        if value not in (0, 1):
            raise ValueError(f"位比较值必须是 0 或 1，收到 {value!r}")
        return Condition(self.register_name, self.index, op, value)

    def __eq__(self, value):  # type: ignore[override]
        return self._cond("==", value)

    def __ne__(self, value):  # type: ignore[override]
        return self._cond("!=", value)

    def __hash__(self):
        return hash((self.register_name, self.index))


class ClassicalRegister:
    """经典寄存器：size 个位，creg[0] 为 LSB（整数值 = Σ bit_i << i）。"""

    __slots__ = ("name", "size")

    def __init__(self, size: int, name: str):
        if int(size) <= 0:
            raise ValueError(f"size 必须为正，收到 {size}")
        if not str(name).strip():
            raise ValueError("name 不能为空")
        self.size = int(size)
        self.name = str(name)

    def __getitem__(self, index: int) -> Bit:
        idx = int(index)
        if idx < 0 or idx >= self.size:
            raise IndexError(f"位下标 {idx} 越界（size={self.size}）")
        return Bit(self.name, idx)

    def __len__(self) -> int:
        return self.size

    def __eq__(self, value):  # type: ignore[override]
        return Condition(self.name, None, "==", int(value))

    def __ne__(self, value):  # type: ignore[override]
        return Condition(self.name, None, "!=", int(value))

    def __hash__(self):
        return hash((self.name, self.size))
```

在 `aicir/core/__init__.py` 加 `from .classical import ClassicalRegister, Bit, Condition` 并加入 `__all__`；`aicir/__init__.py` 顶层再导出 `ClassicalRegister`（跟随该文件既有再导出风格）。

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. pytest tests/core/test_classical.py -v && PYTHONPATH=. python -c "from aicir import ClassicalRegister; print(ClassicalRegister(2,'c'))"`
Expected: PASS，打印寄存器。

- [ ] **Step 5: 提交**

```bash
git add aicir/core/classical.py aicir/core/__init__.py aicir/__init__.py tests/core/test_classical.py
git commit -m "feat(core): ClassicalRegister/Bit/Condition 经典条件模型"
```

---

### Task 2: measure → 经典寄存器

**Files:**
- Modify: `aicir/core/circuit.py`（`measure` 工厂）
- Test: `tests/core/test_measure_creg.py`（新建）

**Interfaces:**
- Consumes: Task 1 的 `Bit`/`ClassicalRegister`。
- Produces: `measure(qubits, *, basis="Z", id=None, creg=None, cbits=None) -> Measurement`。有经典目标时 `Measurement.classical_bits` 存位下标、`metadata["classical_register"]` 存寄存器名。

> 说明：`classical_register` 走 `Measurement.metadata`（`from_dict` 自动把未知键收进 metadata、`to_dict` 再发出），故无需改 `Measurement` 类内部，只改 `measure` 工厂。

- [ ] **Step 1: 写失败测试**

`tests/core/test_measure_creg.py`：

```python
import pytest
from aicir.core.circuit import measure
from aicir.core.classical import ClassicalRegister


def test_measure_creg_all_bits():
    reg = ClassicalRegister(2, "c")
    m = measure([0, 1], creg=reg)
    d = m.to_dict()
    assert d["classical_bits"] == [0, 1]
    assert d["classical_register"] == "c"


def test_measure_cbits_explicit():
    reg = ClassicalRegister(3, "c")
    m = measure([0, 1], cbits=[reg[2], reg[0]])
    d = m.to_dict()
    assert d["classical_bits"] == [2, 0]
    assert d["classical_register"] == "c"


def test_measure_no_classical_target_unchanged():
    d = measure([0, 1]).to_dict()
    assert "classical_register" not in d
    assert d.get("classical_bits", []) == []


def test_measure_creg_rejects_nonz_basis():
    reg = ClassicalRegister(1, "c")
    with pytest.raises(ValueError, match="Z"):
        measure(0, basis="X", creg=reg)


def test_measure_creg_and_cbits_mutually_exclusive():
    reg = ClassicalRegister(2, "c")
    with pytest.raises(ValueError):
        measure([0, 1], creg=reg, cbits=[reg[0], reg[1]])


def test_measure_cbits_cross_register_rejected():
    a, b = ClassicalRegister(1, "a"), ClassicalRegister(1, "b")
    with pytest.raises(ValueError, match="同一"):
        measure([0, 1], cbits=[a[0], b[0]])


def test_measure_creg_length_mismatch_rejected():
    reg = ClassicalRegister(1, "c")
    with pytest.raises(ValueError):
        measure([0, 1], creg=reg)  # 2 qubits, reg 只有 1 位
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/core/test_measure_creg.py -v`
Expected: FAIL（`measure() got an unexpected keyword argument 'creg'`）

- [ ] **Step 3: 改写 `measure` 工厂**

`aicir/core/circuit.py`，把 `measure` 替换为：

```python
def measure(qubits=None, *, basis="Z", id=None, creg=None, cbits=None):
    """线路内投影测量标记。

    无经典目标（creg/cbits 均 None）：联合 Pauli 投影，保留比特（原行为）。
    有经典目标：per-qubit Z 基投影，比特 |0>->0 / |1>->1 写入指定经典位。
      creg=ClassicalRegister：按序写入 0..k-1 号位（len(qubits)<=len(creg)）。
      cbits=[Bit, ...]：显式指定每比特写入的位（len==len(qubits)，须同一寄存器）。
    """
    qs = _normalize_marker_qubits(qubits)
    if creg is None and cbits is None:
        return Measurement(qs, basis=basis, id=id)

    from .classical import Bit, ClassicalRegister
    if creg is not None and cbits is not None:
        raise ValueError("creg 与 cbits 互斥，只能提供其一")
    if str(basis).strip().upper() != "Z":
        raise ValueError("有经典目标的 measure 仅支持 Z 基")

    if creg is not None:
        if not isinstance(creg, ClassicalRegister):
            raise ValueError("creg 必须是 ClassicalRegister")
        if len(qs) > len(creg):
            raise ValueError(f"比特数 {len(qs)} 超过寄存器位数 {len(creg)}")
        reg_name = creg.name
        clbits = tuple(range(len(qs)))
    else:
        bits = list(cbits)
        if len(bits) != len(qs):
            raise ValueError(f"cbits 数 {len(bits)} 与比特数 {len(qs)} 不符")
        names = {b.register_name for b in bits}
        if len(names) != 1:
            raise ValueError("cbits 必须属于同一寄存器")
        reg_name = names.pop()
        clbits = tuple(b.index for b in bits)

    return Measurement(qs, basis="Z", id=id, classical_bits=clbits,
                       metadata={"classical_register": reg_name})
```

- [ ] **Step 4: 跑测试确认通过 + 无回归**

Run: `PYTHONPATH=. pytest tests/core/test_measure_creg.py tests/measure -q`
Expected: PASS（现有 measure 测试无回归）

- [ ] **Step 5: 提交**

```bash
git add aicir/core/circuit.py tests/core/test_measure_creg.py
git commit -m "feat(core): measure 支持写入经典寄存器（creg/cbits，per-qubit Z）"
```

---

### Task 3: 控制流指令 `if_` / `while_`（IR + 工厂 + 路由）

**Files:**
- Create: `aicir/ir/control_flow.py`
- Modify: `aicir/ir/__init__.py`（导出 `ControlFlow`）、`aicir/ir/accessors.py`（`as_instruction`/`instruction_name` 路由）、`aicir/core/circuit.py`（`if_`/`while_` 工厂 + `n_qubits` 推断 + `__all__`）
- Test: `tests/core/test_control_flow_build.py`（新建）

**Interfaces:**
- Consumes: Task 1 的 `Condition`。
- Produces:
  - `ControlFlow`（LegacyGateView 子类）：`.name`("if"/"while")、`.condition`(Condition)、`.body_gates`(tuple[dict])、`.else_gates`(tuple[dict]|None)、`.max_iterations`(int|None)、`.n_qubits`(int)；`.body`/`.else_body` 属性懒重建 `Circuit`；`.to_dict()`/`ControlFlow.from_dict(d)`。
  - `if_(condition, body, else_body=None) -> ControlFlow`；`while_(condition, body, *, max_iterations) -> ControlFlow`（body/else_body 为 `Circuit`）。

- [ ] **Step 1: 写失败测试**

`tests/core/test_control_flow_build.py`：

```python
import pytest
from aicir.core.circuit import Circuit, if_, pauli_x, while_
from aicir.core.classical import ClassicalRegister
from aicir.ir import ControlFlow
from aicir.ir.accessors import as_instruction, instruction_name


def _body():
    return Circuit(pauli_x(1), n_qubits=2)


def test_if_build_and_fields():
    reg = ClassicalRegister(1, "c")
    cf = if_(reg[0] == 1, _body())
    assert isinstance(cf, ControlFlow)
    assert cf.name == "if" and cf.n_qubits == 2
    assert cf.condition.op == "==" and cf.else_gates is None
    assert cf.body.n_qubits == 2 and len(cf.body.gates) == 1


def test_if_else():
    reg = ClassicalRegister(1, "c")
    cf = if_(reg[0] == 1, _body(), else_body=Circuit(pauli_x(0), n_qubits=2))
    assert cf.else_body.gates[0]["target_qubit"] == 0


def test_while_requires_max_iterations():
    reg = ClassicalRegister(1, "c")
    with pytest.raises(TypeError):
        while_(reg[0] == 1, _body())  # max_iterations 必填
    cf = while_(reg[0] == 1, _body(), max_iterations=10)
    assert cf.name == "while" and cf.max_iterations == 10


def test_body_nqubits_must_match():
    reg = ClassicalRegister(1, "c")
    with pytest.raises(ValueError, match="n_qubits"):
        if_(reg[0] == 1, Circuit(pauli_x(1), n_qubits=3))  # 稍后放入 2 比特父线路时不符
    # 直接构造时以 body 自身 n_qubits 为准，父线路校验在 append 时


def test_roundtrip_dict():
    reg = ClassicalRegister(1, "c")
    cf = if_(reg[0] == 1, _body(), else_body=Circuit(pauli_x(0), n_qubits=2))
    d = cf.to_dict()
    assert d["type"] == "if"
    back = ControlFlow.from_dict(d)
    assert back.to_dict() == d
    assert back.condition.register_name == "c"


def test_as_instruction_routes_control_flow():
    reg = ClassicalRegister(1, "c")
    d = if_(reg[0] == 1, _body()).to_dict()
    inst = as_instruction(d)
    assert isinstance(inst, ControlFlow)
    assert instruction_name(inst) == "if"


def test_circuit_stores_control_flow():
    reg = ClassicalRegister(1, "c")
    c = Circuit(pauli_x(0), if_(reg[0] == 1, _body()), n_qubits=2)
    assert c.n_qubits == 2
    names = [as_instruction(g).name if not hasattr(g, "name") else g.name
             for g in c.gates]
    # 末条应是 if 指令 dict
    assert c.gates[-1]["type"] == "if"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/core/test_control_flow_build.py -v`
Expected: FAIL（`ImportError: cannot import name 'ControlFlow'`）

- [ ] **Step 3: 实现 `aicir/ir/control_flow.py`**

```python
"""控制流指令 ControlFlow：携带嵌套 body 的 if/while。

body 以 gate-dict 元组 + n_qubits 存储（不直接持有 Circuit，避免 ir<->core 循环
导入）；.body/.else_body 属性在访问时懒重建 Circuit。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.classical import Condition
from .operation import LegacyGateView

_CF_NAMES = {"if", "while"}


class ControlFlow(LegacyGateView):
    """if / while 控制流指令。"""

    def __init__(self, name, condition, body_gates, n_qubits,
                 else_gates=None, max_iterations=None):
        name = str(name).lower()
        if name not in _CF_NAMES:
            raise ValueError(f"控制流 name 必须是 if/while，收到 {name!r}")
        if not isinstance(condition, Condition):
            raise TypeError("condition 必须是 Condition")
        if name == "while" and max_iterations is None:
            raise ValueError("while 必须提供 max_iterations")
        self.name = name
        self.condition = condition
        self.body_gates = tuple(dict(g) for g in body_gates)
        self.else_gates = None if else_gates is None else tuple(dict(g) for g in else_gates)
        self.n_qubits = int(n_qubits)
        self.max_iterations = None if max_iterations is None else int(max_iterations)

    @property
    def body(self):
        from ..core.circuit import Circuit
        return Circuit(*self.body_gates, n_qubits=self.n_qubits)

    @property
    def else_body(self):
        if self.else_gates is None:
            return None
        from ..core.circuit import Circuit
        return Circuit(*self.else_gates, n_qubits=self.n_qubits)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.name,
            "condition": self.condition.to_dict(),
            "body": [dict(g) for g in self.body_gates],
            "n_qubits": self.n_qubits,
        }
        if self.else_gates is not None:
            d["else_body"] = [dict(g) for g in self.else_gates]
        if self.name == "while":
            d["max_iterations"] = self.max_iterations
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ControlFlow":
        return cls(
            name=d["type"],
            condition=Condition.from_dict(d["condition"]),
            body_gates=d["body"],
            n_qubits=int(d["n_qubits"]),
            else_gates=d.get("else_body"),
            max_iterations=d.get("max_iterations"),
        )
```

`aicir/ir/__init__.py`：加 `from .control_flow import ControlFlow` 并入 `__all__`。

`aicir/ir/accessors.py`：`as_instruction` 中，`Measurement`/`Operation` 分支之前加控制流路由；顶部 import `ControlFlow`：

```python
from .control_flow import ControlFlow
...
def as_instruction(value):
    if isinstance(value, (Operation, Measurement, ControlFlow)):
        return value
    if isinstance(value, Mapping):
        t = str(value.get("type", "")).lower()
        if t in {"if", "while"}:
            return ControlFlow.from_dict(value)
        if t in {"measure", "measurement", "reset"}:
            return Measurement.from_dict(value)
        return Operation.from_dict(value)
    raise TypeError("instruction must be Operation, Measurement, ControlFlow, or a gate mapping")
```

`instruction_name`：`ControlFlow` 返回 `.name`（加 `isinstance(inst, ControlFlow): return inst.name` 分支，放在 Measurement 分支旁）。

`aicir/core/circuit.py`：加工厂（放在 `reset` 附近）：

```python
def if_(condition, body, else_body=None):
    """条件执行 body（可选 else_body），依据测量写入的经典寄存器。"""
    from ..ir.control_flow import ControlFlow
    n = int(body.n_qubits)
    if else_body is not None and int(else_body.n_qubits) != n:
        raise ValueError("else_body 的 n_qubits 必须与 body 一致")
    return ControlFlow("if", condition, list(body.gates), n,
                       else_gates=(None if else_body is None else list(else_body.gates)))


def while_(condition, body, *, max_iterations):
    """条件循环执行 body，最多 max_iterations 次；超限仍满足条件抛 RuntimeError。"""
    from ..ir.control_flow import ControlFlow
    if int(max_iterations) <= 0:
        raise ValueError("max_iterations 必须为正")
    return ControlFlow("while", condition, list(body.gates), int(body.n_qubits),
                       max_iterations=int(max_iterations))
```

并把 `if_`, `while_` 加进 `aicir/core/circuit.py` 的 `__all__`，`aicir/core/__init__.py` 与 `aicir/__init__.py` 顶层再导出。

`_infer_n_qubits_from_gates`（circuit.py）：使其识别控制流指令的 `n_qubits`。找到该函数，在遍历时对 `dict` 且 `type in {"if","while"}` 的项取其 `["n_qubits"]` 纳入 max（读实现后按其模式加一分支；控制流 body 已知 n_qubits，父线路通常显式传 n_qubits，此分支保证纯控制流构造也能推断）。

`Circuit.append`/父线路 n_qubits 一致性：在 `append`/`__init__` 归一化后，若控制流指令 `n_qubits` 与父 `self.n_qubits` 不符则抛 `ValueError`（读 `__init__`/`append` 现有结构后加校验；`test_body_nqubits_must_match` 覆盖 append 场景——若直接构造 3 比特 body 放入 2 比特父线路应报错）。

> 实现提示：`test_body_nqubits_must_match` 第一断言（直接 `if_` 传 3-比特 body）当前工厂不报错（工厂以 body 自身 n_qubits 为准）。请把该断言改为「放入 2 比特父线路时报错」：`with pytest.raises(ValueError): Circuit(if_(reg[0]==1, Circuit(pauli_x(1), n_qubits=3)), n_qubits=2)`。即校验点在 Circuit 组装，不在工厂。实现时同步修正该测试。

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. pytest tests/core/test_control_flow_build.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add aicir/ir/control_flow.py aicir/ir/__init__.py aicir/ir/accessors.py aicir/core/circuit.py aicir/core/__init__.py aicir/__init__.py tests/core/test_control_flow_build.py
git commit -m "feat(ir): ControlFlow 指令 + if_/while_ 工厂 + as_instruction 路由"
```

---

### Task 4: 递归轨迹执行器

**Files:**
- Modify: `aicir/measure/trajectory.py`（递归执行器 + measure→creg + if/while + `TrajectoryResult.classical`）
- Modify: `aicir/measure/measure.py`（路由：含控制流/creg-measure 走 trajectory 分支）
- Test: `tests/measure/test_control_flow_exec.py`（新建）

**Interfaces:**
- Consumes: Task 1 `Condition`、Task 2 measure→creg 标记、Task 3 `ControlFlow`。
- Produces: `TrajectoryResult.classical: dict[str,list[int]]`；`run_trajectory` 正确执行控制流并写经典 store。

- [ ] **Step 1: 写失败测试**

`tests/measure/test_control_flow_exec.py`：

```python
import numpy as np
from aicir import Circuit, Measure, NumpyBackend, cnot, hadamard, if_, pauli_x, while_
from aicir.core.classical import ClassicalRegister
from aicir.measure.measure import measure as _mkmeasure  # noqa
from aicir.core.circuit import measure


def _run(circ, shots=400, seed=7):
    return Measure(NumpyBackend()).run(circ, shots=shots, seed=seed)


def test_measure_creg_deterministic():
    reg = ClassicalRegister(1, "c")
    c = Circuit(pauli_x(0), measure(0, creg=reg), n_qubits=1)
    res = _run(c, shots=20)
    cc = res.classical_counts(reg)
    assert cc == {1: 20}


def test_if_correlates_branch_with_measurement():
    reg = ClassicalRegister(1, "c")
    body = Circuit(pauli_x(1), n_qubits=2)
    c = Circuit(hadamard(0), measure(0, creg=reg), if_(reg[0] == 1, body), n_qubits=2)
    # 每 shot：q1 == c[0]。末端测 q1 与 classical c 完全关联。
    res = _run(c, shots=300)
    cc = res.classical_counts(reg)
    assert set(cc) == {0, 1} and abs(cc[0] - cc[1]) < 120  # ~50/50


def test_if_else_both_branches():
    reg = ClassicalRegister(1, "c")
    c = Circuit(
        hadamard(0), measure(0, creg=reg),
        if_(reg[0] == 1, Circuit(pauli_x(1), n_qubits=2),
            else_body=Circuit(pauli_x(2), n_qubits=2) if False else Circuit(hadamard(1), n_qubits=2)),
        n_qubits=2,
    )
    res = _run(c, shots=100)
    assert set(res.classical_counts(reg)) <= {0, 1}


def test_while_converges():
    # body：把 q0 无条件置 0 再测入 c[0]（一步内必收敛，条件 c[0]==1 变假）
    reg = ClassicalRegister(1, "c")
    body = Circuit(pauli_x(0), measure(0, creg=reg), n_qubits=1)  # X 后测：若初为1则变0
    c = Circuit(pauli_x(0), measure(0, creg=reg),   # c[0]=1
                while_(reg[0] == 1, body, max_iterations=5), n_qubits=1)
    res = _run(c, shots=10)
    # 循环体一次后 c[0]=0，退出
    assert res.classical_counts(reg) == {0: 10}


def test_while_overflow_raises():
    import pytest
    reg = ClassicalRegister(1, "c")
    # 条件恒真：body 不改变 c[0]（只作用 q1），c[0] 始终 1
    body = Circuit(pauli_x(1), n_qubits=2)
    c = Circuit(pauli_x(0), measure(0, creg=reg),
                while_(reg[0] == 1, body, max_iterations=3), n_qubits=2)
    with pytest.raises(RuntimeError, match="max_iterations"):
        _run(c, shots=1)


def test_nested_if_in_while():
    reg = ClassicalRegister(1, "c")
    inner = Circuit(if_(reg[0] == 1, Circuit(pauli_x(0), n_qubits=1)), measure(0, creg=reg), n_qubits=1)
    c = Circuit(pauli_x(0), measure(0, creg=reg),
                while_(reg[0] == 1, inner, max_iterations=5), n_qubits=1)
    res = _run(c, shots=10)
    assert res.classical_counts(reg) == {0: 10}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/measure/test_control_flow_exec.py -v`
Expected: FAIL（`AttributeError: ... 'classical'` 或路由未触发控制流）

- [ ] **Step 3: 重构 `trajectory.py`**

在 `TrajectoryResult` dataclass 加字段：

```python
    classical: Dict[str, list] = field(default_factory=dict)
```

新增 import 与逐比特测量 helper（复用 projector 的 Z 投影）。把 `run_trajectory` 主体逐操作循环抽成 `_exec_ops`，并加控制流/creg 分支：

```python
from ..ir import ControlFlow  # 顶部
from .projector import measure_joint_pauli, reset_channel  # 已有

def _measure_into_creg(state, qubits, reg_name, clbits, classical, rng):
    """per-qubit Z 投影，比特 |0>->0/|1>->1 写入 classical[reg_name][clbit]。"""
    from .projector import terminal_z_measure  # 逐比特 Z 采样
    # 对给定 qubits 逐个 Z 投影；terminal_z_measure 返回坍缩态与各比特 0/1
    state, bits = terminal_z_measure(state, list(qubits), rng)
    slot = classical.setdefault(reg_name, [0] * (max(clbits) + 1))
    if len(slot) < max(clbits) + 1:
        slot.extend([0] * (max(clbits) + 1 - len(slot)))
    for cb, b in zip(clbits, bits):
        slot[cb] = int(b)
    return state


def _exec_ops(ops, state, classical, backend, n, *, rng, noise_model,
              snap_ops, incircuit, snaps, op_index_ref):
    from ..ir import circuit_instructions
    for gate in circuit_instructions(ops):
        if isinstance(gate, ControlFlow):
            cond = gate.condition
            if gate.name == "if":
                if cond.evaluate(classical):
                    state = _exec_ops(gate.body.gates, state, classical, backend, n,
                                      rng=rng, noise_model=noise_model, snap_ops=set(),
                                      incircuit=incircuit, snaps=snaps, op_index_ref=op_index_ref)
                elif gate.else_gates is not None:
                    state = _exec_ops(gate.else_body.gates, state, classical, backend, n,
                                      rng=rng, noise_model=noise_model, snap_ops=set(),
                                      incircuit=incircuit, snaps=snaps, op_index_ref=op_index_ref)
            else:  # while
                iters = 0
                while cond.evaluate(classical):
                    iters += 1
                    if iters > gate.max_iterations:
                        raise RuntimeError(
                            f"while 超过 max_iterations={gate.max_iterations} 仍满足条件")
                    state = _exec_ops(gate.body.gates, state, classical, backend, n,
                                      rng=rng, noise_model=noise_model, snap_ops=set(),
                                      incircuit=incircuit, snaps=snaps, op_index_ref=op_index_ref)
            op_index_ref[0] += 1
            continue

        op_index = op_index_ref[0]
        if _is_measure(gate):
            reg_name = gate.get("classical_register")
            if reg_name is not None:
                clbits = list(getattr(gate, "classical_bits", ()) or gate.get("classical_bits", ()))
                qubits = _marker_qubits(gate, n)
                state = _measure_into_creg(state, qubits, reg_name, clbits, classical, rng)
            else:
                qubits = _marker_qubits(gate, n)
                basis = _gate_basis(gate)
                state, lam = measure_joint_pauli(state, qubits, basis, rng)
                incircuit[op_index] = lam
        elif _is_reset(gate):
            state = reset_channel(state, _marker_qubits(gate, n))
        else:
            state = _apply_unitary(gate, state, backend, n, noise_model)  # 抽出原酉门+噪声逻辑

        if op_index in snap_ops:
            snaps[op_index] = state
        op_index_ref[0] += 1
    return state
```

把原 `run_trajectory` 中「酉门演化 + 可选噪声」那段抽成 `_apply_unitary(gate, state, backend, n, noise_model) -> State`（原样搬运逻辑，见现文件 113-138 行）。`run_trajectory` 改为：

```python
def run_trajectory(circuit, init_state, backend, *, tm, measure_qubits, snap_ops, rng, noise_model=None):
    classical: Dict[str, list] = {}
    incircuit: Dict[int, int] = {}
    snaps: Dict[int, State] = {}
    n = init_state.n_qubits
    state = _exec_ops(circuit, init_state, classical, backend, n, rng=rng,
                      noise_model=noise_model, snap_ops=snap_ops, incircuit=incircuit,
                      snaps=snaps, op_index_ref=[0])
    pre = state
    if tm and measure_qubits is not None and len(measure_qubits) > 0:
        post, terminal = terminal_z_measure(state, measure_qubits, rng)
    elif tm and measure_qubits is None:
        post, terminal = terminal_z_measure(state, list(range(n)), rng)
    else:
        post, terminal = pre, None
    return TrajectoryResult(pre=pre, post=post, incircuit=incircuit,
                            terminal=terminal, snaps=snaps, classical=classical)
```

`aicir/measure/measure.py`：路由判断 `has_incircuit` 扩展为「含 in-circuit measure **或** 含控制流指令」：

```python
def _needs_trajectory(circuit) -> bool:
    from ..ir import ControlFlow
    for g in circuit_instructions(circuit):
        if isinstance(g, ControlFlow) or _is_measure(g):
            return True
    return False
```

把 `has_incircuit = any(...measure...)` 处改为 `has_incircuit = _needs_trajectory(circuit) if n_ops else False`。（含控制流时必须逐轨迹执行。）

> 注：`_measure_into_creg` 用 `terminal_z_measure` 做逐比特 Z 采样（该函数已实现按 qubits 顺序返回 0/1 比特并坍缩态），语义正确（线路内非末端调用不影响后续演化）。实现时确认其返回顺序与 `qubits` 入参一致；若不一致按其约定映射 clbits。

- [ ] **Step 4: 跑测试确认通过 + 无回归**

Run: `PYTHONPATH=. pytest tests/measure/test_control_flow_exec.py tests/measure -q`
Expected: PASS（现有 measure 轨迹测试无回归——无控制流/creg 时 `_exec_ops` 等价原循环）

> 注：`classical_counts` 由 Task 5 提供；本任务测试用到它，故 Task 4 与 Task 5 顺序上 Task 5 的 `classical_counts` 需先存在，或本任务测试临时改读 `res` 内部。**决定：把 `Result.classical_counts` 的最小实现并入本任务 Step 3**（读 `TrajectoryResult.classical`），Task 5 只做完善与独立测试。实现时在 `result.py` 加 `classical_counts`。

- [ ] **Step 5: 提交**

```bash
git add aicir/measure/trajectory.py aicir/measure/measure.py aicir/measure/result.py tests/measure/test_control_flow_exec.py
git commit -m "feat(measure): 递归轨迹执行器支持 if/else/while 与 measure→creg"
```

---

### Task 5: Result 经典读出完善

**Files:**
- Modify: `aicir/measure/result.py`（`classical_counts` 完善 + register 传名或对象）
- Test: `tests/measure/test_classical_counts.py`（新建）

**Interfaces:**
- Consumes: Task 4 的 `TrajectoryResult.classical`。
- Produces: `result.classical_counts(register) -> dict[int,int]`（register 可传 `str` 名或 `ClassicalRegister`）。

- [ ] **Step 1: 写失败测试**

`tests/measure/test_classical_counts.py`：

```python
from aicir import Circuit, Measure, NumpyBackend, hadamard, pauli_x
from aicir.core.circuit import measure
from aicir.core.classical import ClassicalRegister


def test_classical_counts_by_name_and_object():
    reg = ClassicalRegister(2, "c")
    c = Circuit(pauli_x(0), measure([0, 1], creg=reg), n_qubits=2)
    res = Measure(NumpyBackend()).run(c, shots=8, seed=1)
    # q0=1,q1=0 -> bits [1,0] -> int 1
    assert res.classical_counts(reg) == {1: 8}
    assert res.classical_counts("c") == {1: 8}


def test_classical_counts_distribution():
    reg = ClassicalRegister(1, "c")
    c = Circuit(hadamard(0), measure(0, creg=reg), n_qubits=1)
    res = Measure(NumpyBackend()).run(c, shots=400, seed=3)
    cc = res.classical_counts(reg)
    assert set(cc) == {0, 1} and sum(cc.values()) == 400


def test_classical_counts_unknown_register_empty():
    reg = ClassicalRegister(1, "c")
    c = Circuit(pauli_x(0), n_qubits=1)  # 无 measure→creg
    res = Measure(NumpyBackend()).run(c, shots=5, seed=1)
    assert res.classical_counts(reg) == {}
```

- [ ] **Step 2: 跑测试确认失败/通过**

Run: `PYTHONPATH=. pytest tests/measure/test_classical_counts.py -v`
Expected: 若 Task 4 已加最小 `classical_counts`，`unknown_register_empty` 等边界可能未覆盖 → 补齐。

- [ ] **Step 3: 完善 `classical_counts`**

`aicir/measure/result.py`，`Result` 内：

```python
    def classical_counts(self, register):
        """按 shot 统计某经典寄存器的整数值分布（creg[0]=LSB）。

        register 可传寄存器名(str)或 ClassicalRegister。从未写入 → 空 dict。
        """
        name = register if isinstance(register, str) else register.name
        counts: dict[int, int] = {}
        for tr in self._trajectories:  # 按 result.py 实际持有轨迹的属性名调整
            bits = tr.classical.get(name)
            if bits is None:
                continue
            value = sum(int(b) << i for i, b in enumerate(bits))
            counts[value] = counts.get(value, 0) + 1
        return counts
```

（读 `result.py` 的 `Result.__init__`/`_build_result` 确认轨迹列表实际存储位置与 exact/shots 归一；exact 模式下 M=1，counts 反映单代表轨迹——与 `output`/`counts` 现有语义一致处理。）

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. pytest tests/measure/test_classical_counts.py -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add aicir/measure/result.py tests/measure/test_classical_counts.py
git commit -m "feat(measure): Result.classical_counts 经典寄存器分布读出"
```

---

### Task 6: unitary / TN 守卫

**Files:**
- Modify: `aicir/core/circuit.py`（`unitary()` 遇控制流报错）、`aicir/simulator/network.py` 或 `aicir/simulator/__init__.py`（TN 遇控制流报错）
- Test: `tests/core/test_control_flow_guards.py`（新建）

**Interfaces:**
- Consumes: Task 3 `ControlFlow`。

- [ ] **Step 1: 写失败测试**

`tests/core/test_control_flow_guards.py`：

```python
import pytest
from aicir import Circuit, NumpyBackend, if_, pauli_x
from aicir.core.circuit import measure
from aicir.core.classical import ClassicalRegister
from aicir.simulator import tn_statevector


def _cf_circuit():
    reg = ClassicalRegister(1, "c")
    return Circuit(pauli_x(0), measure(0, creg=reg),
                   if_(reg[0] == 1, Circuit(pauli_x(0), n_qubits=1)), n_qubits=1)


def test_unitary_rejects_control_flow():
    with pytest.raises(ValueError, match="控制流"):
        _cf_circuit().unitary(backend=NumpyBackend())


def test_unitary_rejects_even_with_ignore_nonunitary():
    with pytest.raises(ValueError, match="控制流"):
        _cf_circuit().unitary(backend=NumpyBackend(), ignore_nonunitary=True)


def test_tn_rejects_control_flow():
    with pytest.raises(ValueError, match="控制流"):
        tn_statevector(_cf_circuit())
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. pytest tests/core/test_control_flow_guards.py -v`
Expected: FAIL（未报错或报错信息不符）

- [ ] **Step 3: 加守卫**

`Circuit.unitary()`：在逐门累乘循环开头（读现有 `unitary` 实现，约 508 行起）加：

```python
            if isinstance(g, ControlFlow) or (isinstance(g, Mapping) and str(g.get("type", "")).lower() in {"if", "while"}):
                raise ValueError("控制流指令无法表示为酉矩阵；请用 Measure.run 执行")
```

（`ControlFlow`/`Mapping` 从 ir/collections 导入；`ignore_nonunitary` 分支之前判断，使其也报错。）

`aicir/simulator/network.py` 的 `build_network`（张量网络构建遍历门处）：遇 `type in {"if","while"}` 抛 `ValueError("控制流指令不支持张量网络模拟；请用 Measure.run 执行")`。

- [ ] **Step 4: 跑测试确认通过 + 无回归**

Run: `PYTHONPATH=. pytest tests/core/test_control_flow_guards.py tests/simulator -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add aicir/core/circuit.py aicir/simulator/network.py tests/core/test_control_flow_guards.py
git commit -m "feat: unitary()/TN 引擎拒绝控制流指令"
```

---

### Task 7: JSON 整线路往返 + 执行等价

**Files:**
- Modify: `aicir/core/io/json_io.py`（若控制流嵌套 body 未自动往返则补）
- Test: `tests/core/test_control_flow_json.py`（新建）

**Interfaces:**
- Consumes: Task 3 序列化、Task 4 执行。

> 由于控制流指令 `to_dict` 已把 body 序列化为 gate-dict 列表、`circuit_from_json` → `Circuit(*gates)` → `normalize_gate`/`as_instruction` 路由回 `ControlFlow`，整线路往返可能已工作。本任务先写测试验证；若 `_jsonable_value`/`_restore_json_value` 对嵌套 body 或 condition 处理有缺失再补。

- [ ] **Step 1: 写失败测试**

`tests/core/test_control_flow_json.py`：

```python
from aicir import Circuit, Measure, NumpyBackend, hadamard, if_, pauli_x, while_
from aicir.core.circuit import measure
from aicir.core.classical import ClassicalRegister
from aicir.core.io.json_io import circuit_from_json, circuit_to_json


def _circ():
    reg = ClassicalRegister(1, "c")
    return Circuit(
        hadamard(0), measure(0, creg=reg),
        if_(reg[0] == 1, Circuit(pauli_x(1), n_qubits=2),
            else_body=Circuit(hadamard(1), n_qubits=2)),
        while_(reg[0] == 1, Circuit(pauli_x(0), measure(0, creg=reg), n_qubits=2), max_iterations=4),
        n_qubits=2,
    )


def test_json_roundtrip_structure():
    c = _circ()
    back = circuit_from_json(circuit_to_json(c))
    assert back.n_qubits == 2
    assert back.gates[2]["type"] == "if"
    assert back.gates[2]["else_body"][0]["type"] == "hadamard"
    assert back.gates[3]["type"] == "while" and back.gates[3]["max_iterations"] == 4
    assert back.gates[3]["condition"]["target"]["register"] == "c"


def test_json_roundtrip_execution_equivalence():
    c = _circ()
    back = circuit_from_json(circuit_to_json(c))
    reg = ClassicalRegister(1, "c")
    r1 = Measure(NumpyBackend()).run(c, shots=200, seed=5).classical_counts(reg)
    r2 = Measure(NumpyBackend()).run(back, shots=200, seed=5).classical_counts(reg)
    assert r1 == r2
```

- [ ] **Step 2: 跑测试确认失败或通过**

Run: `PYTHONPATH=. pytest tests/core/test_control_flow_json.py -v`
Expected: 若已自动往返则 PASS；否则修 `json_io` 的 `_jsonable_value`/`_restore_json_value` 使嵌套 body/condition dict 原样保留（它们本就是纯 dict/list/int/str，一般无需特殊处理）。

- [ ] **Step 3: （按需）修 json_io**

若失败，定位 `_jsonable_value`/`_restore_json_value`，确保嵌套 list[dict]（body）与 condition dict 递归保留、不被误当作门处理。改动限于序列化保真，不改语义。

- [ ] **Step 4: 跑测试确认通过 + 无回归**

Run: `PYTHONPATH=. pytest tests/core/test_control_flow_json.py tests/core -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add aicir/core/io/json_io.py tests/core/test_control_flow_json.py
git commit -m "test/feat(io): 控制流+creg 电路 JSON 往返与执行等价"
```

---

### Task 8: 文档 + compare.md + CHANGELOG + CLAUDE.md

**Files:**
- Modify: `aicir/measure/README.md`（补控制流用法节）、`CHANGELOG.md`、`compare.md`、`CLAUDE.md`

- [ ] **Step 1: measure README 补控制流节**

`aicir/measure/README.md` 加一节「经典控制流」：`ClassicalRegister` 创建、`measure(q, creg=)` 写入、`if_(cond, body, else_body=)`/`while_(cond, body, max_iterations=)`、条件 `reg[i]==v`/`reg==N`、`result.classical_counts(reg)`、以及「控制流仅走 `Measure.run`，不参与 `unitary()`/张量网络」。给一个完整可跑示例（H→measure→if）。

- [ ] **Step 2: CHANGELOG 条目**

`CHANGELOG.md` 2026-07-03 `### Added` 追加：`ClassicalRegister/Bit/Condition`、`measure(creg=/cbits=)`、`if_`/`while_`（含 else、必填 max_iterations、超限 RuntimeError）、`Result.classical_counts`、JSON 往返；控制流走测量轨迹路径、`unitary()`/TN 拒绝、QASM3 导出推迟。

- [ ] **Step 3: compare.md 更新**

- §3 能力矩阵：「经典寄存器」行 aicir 由 ⚠️/❌ 改 ✅（`ClassicalRegister`）；「经典控制流 if/while」行由 ❌ 改 ✅（`if_`/`while_`，测量轨迹路径）。
- §5 电路构建模型「控制流」行：aicir 由「无」改为「`if_`/`while_`（测量反馈，`Measure.run` 执行）」。
- §19「仅 WuYueSDK 具备」移除「经典控制流（qif/qwhile）+ 经典寄存器」条目；并入「两者都具备」。
- §20 选型表「需要经典控制流」行由 **WuYueSDK** 改为「均可（aicir `if_`/`while_` / WuYue `qif/qwhile`）」。
- 保持客观、不写 aicir 开发时间线（见近期 compare.md 客观化提交风格）。

- [ ] **Step 4: CLAUDE.md**

`aicir/measure/` 或 `aicir/core/` 子系统描述补：经典控制流（`ClassicalRegister`、`measure(creg=)`、`if_`/`while_`，运行于 `Measure.run` 测量轨迹，`unitary()`/TN 不支持）。

- [ ] **Step 5: 全量回归 + 提交**

Run: `PYTHONPATH=. pytest -q`
Expected: 全绿（基线 + 新增控制流测试，2 skipped 不变）

```bash
git add aicir/measure/README.md CHANGELOG.md compare.md CLAUDE.md
git commit -m "docs: 经典控制流用法、compare.md 差距更新、CHANGELOG/CLAUDE.md"
```

---

## Self-Review

**Spec coverage:** 组件1 classical.py→T1；组件2 measure→creg→T2(标记)+T4(执行写入)；组件3 if_/while_ 指令→T3；组件4 递归执行器→T4；组件5 Result 读出→T4(最小)+T5(完善)；组件6 unitary/TN 守卫→T6；组件7 JSON 往返→T3(指令序列化)+T7(整线路+执行等价)；文档+compare.md+CHANGELOG+CLAUDE.md→T8。错误处理表各项：while 缺 max_iterations(T3 TypeError)、while 超限 RuntimeError(T4)、body n_qubits 不符(T3)、Bit==非0/1(T1)、measure creg basis≠Z(T2)、cbits 跨寄存器(T2)、位数不符(T2)、unitary/TN(T6)、未写入寄存器默认0(T1)——全覆盖。

**Placeholder scan:** 各 step 有具体代码/命令；T3 有一处显式指出并要求同步修正 `test_body_nqubits_must_match` 断言（校验点在 Circuit 组装非工厂）；T4 显式合并 `classical_counts` 最小实现以解依赖顺序；T5/T7 标注「读实现后按需补」并给出具体定位——均为读现有代码的指令，非空泛占位。

**Type consistency:** `Condition(register_name,index,op,value)`+`.evaluate(store)`+`.to_dict/from_dict` T1 定义、T3 `ControlFlow` 消费、T4 求值一致；`ControlFlow` 字段（name/condition/body_gates/else_gates/n_qubits/max_iterations）T3 定义、T4/T6 消费一致；`measure(...,creg=,cbits=)` T2 定义、T4 读 `classical_register`/`classical_bits` 一致；`TrajectoryResult.classical` T4 产出、T5 消费一致；`classical_counts(register)` T4(最小)/T5(完善) 同签名。

**跨任务顺序风险:** T4 测试用到 `classical_counts` → 已决定 T4 Step3 并入其最小实现（T5 完善）。T7 可能零改动通过（body 本为纯 dict）→ 测试先行验证，失败才改 json_io。
