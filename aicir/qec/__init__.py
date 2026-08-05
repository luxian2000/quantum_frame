"""aicir.qec

量子纠错（Quantum Error Correction）实验平台。

面向**新型在线实时纠错/解码算法**：码、syndrome 提取调度、在线解码器三处均可
插拔，均不需要改模块内部代码。内置的五个码是参考实现与验证语料，不是产品本身。

当前为 M1（骨架）。M3 可视化与 M2 规模化/Stim 互操作见 README 的里程碑一节。
"""

from __future__ import annotations

from .code import StabilizerCode, gf2_to_pauli, pauli_to_gf2, symplectic_product
from .codes import CODES, get_code, register_code
from .decoders import DecodeStep, LookupDecoder, register_decoder, resolve_decoder
from .detectors import Detector, DetectorLayout, Observable
from .errors import ErrorEvent, PauliErrorModel
from .record import QECResult, QECShotRecord
from .runner import TimingModel, run
from .schedules import (BareAncillaSchedule, register_schedule, resolve_schedule,
                        verify_schedule)

__all__ = [
    "StabilizerCode", "pauli_to_gf2", "gf2_to_pauli", "symplectic_product",
    "CODES", "get_code", "register_code",
    "Detector", "Observable", "DetectorLayout",
    "BareAncillaSchedule", "register_schedule", "resolve_schedule", "verify_schedule",
    "ErrorEvent", "PauliErrorModel",
    "DecodeStep", "LookupDecoder", "register_decoder", "resolve_decoder",
    "QECShotRecord", "QECResult", "run", "TimingModel",
]
