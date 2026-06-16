"""门元信息注册表（GateSpec）。

NEXT.md 第 7 节的第一片落地：门的目标比特数、参数个数、别名与 QASM
名称在此单点注册，`aicir.ir.Operation` 构造期与 `aicir.transpile` 的
`ValidatePass` 据此校验。未注册的门名保持宽松（自定义门不受限）。
"""

from .registry import (
    canonical_gate_name,
    get_gate_spec,
    register_gate,
    registered_gate_names,
    unregister_gate,
)
from .spec import GateSpec

__all__ = [
    "GateSpec",
    "canonical_gate_name",
    "get_gate_spec",
    "register_gate",
    "registered_gate_names",
    "unregister_gate",
]
