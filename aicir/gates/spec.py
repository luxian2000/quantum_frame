"""GateSpec：门元信息的单点描述。

NEXT.md 第 7 节的第一片落地：每个门的目标比特数、参数个数、别名与
QASM 名称只在此注册一次，供 `Operation` 构造期校验与 `ValidatePass`
等模块复用。``matrix``/``generator``/``decomposition`` 等字段留待
后续需要时扩展（当前矩阵构造仍由 ``gate_to_matrix`` 负责）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class GateSpec:
    """单个门类型的元信息。

    属性说明：

    - ``name``：规范门名，与门字典的 ``type`` 字段一致。
    - ``num_qubits``：目标比特数（不含控制位）；``None`` 表示可变
      （如 ``unitary``、``measure``，以及可作用于整个寄存器的 ``identity``）。
    - ``num_params``：参数个数；符号 ``Parameter`` 同样计为一个参数；
      ``None`` 表示可变（如 ``unitary`` 的矩阵参数在占位场景可缺省）。
    - ``aliases``：等价的 ``type`` 写法（如 ``"X"``、``"cnot"``）。
    - ``controlled``：是否必须携带至少一个控制位。
    - ``num_controls``：控制位数量；``controlled == (num_controls > 0)``。
    - ``qasm_name``：OpenQASM 导出名；``None`` 表示暂未约定。
    - ``symbol``：ASCII/绘图显示符号（受控门为目标位符号）；``None``
      表示特殊绘制（swap/rzz/rxx/measure）或退回通用 fallback。
    - ``generator``：单参数旋转门的 Pauli 生成元标签（``U = exp(-i θ G / 2)``），
      如 ``rx`` 为 ``"X"``、``rzz`` 为 ``"ZZ"``；受控旋转记其目标位生成元。
      ``None`` 表示非（标准 Pauli）参数化旋转。供 QML 自省能否用解析参数移位。
    - ``decomposition``：把该门分解到更基础门集的规则，签名
      ``(qubits, controls, control_states, params) -> list[dict] | None``
      （返回 ``None`` 表示当前入参形态不适用，如多控制位）。供 ``transpile``
      的 ``DecomposePass`` 驱动；``None`` 表示无内置分解规则。
    """

    name: str
    num_qubits: int | None
    num_params: int | None
    aliases: tuple[str, ...] = ()
    controlled: bool = False
    num_controls: int = 0
    qasm_name: str | None = None
    symbol: str | None = None
    generator: str | None = None
    decomposition: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("GateSpec name cannot be empty")
        if self.num_qubits is not None and int(self.num_qubits) < 0:
            raise ValueError("num_qubits must be non-negative or None")
        if self.num_params is not None and int(self.num_params) < 0:
            raise ValueError("num_params must be non-negative or None")
        if int(self.num_controls) < 0:
            raise ValueError("num_controls must be non-negative")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "aliases", tuple(str(alias) for alias in self.aliases))
