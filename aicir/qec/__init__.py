"""aicir.qec

量子纠错（Quantum Error Correction）实验平台。

面向**新型在线实时纠错/解码算法**：码、syndrome 提取调度、在线解码器
三处均可插拔。详见本包 README。
"""

from __future__ import annotations

from .code import StabilizerCode, gf2_to_pauli, pauli_to_gf2, symplectic_product
from .codes import CODES, get_code, register_code
from .detectors import Detector, DetectorLayout, Observable
from .schedules import (BareAncillaSchedule, register_schedule, resolve_schedule,
                        verify_schedule)

__all__ = ["StabilizerCode", "pauli_to_gf2", "gf2_to_pauli", "symplectic_product", "CODES", "get_code", "register_code", "Detector", "Observable", "DetectorLayout"]
__all__ += ["BareAncillaSchedule", "register_schedule", "resolve_schedule", "verify_schedule"]
