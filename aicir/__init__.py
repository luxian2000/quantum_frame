# aicir — 量子模拟器顶层包
from __future__ import annotations

from importlib import import_module

__all__: list[str] = []


def _export(names: list[str]) -> None:
    __all__.extend(name for name in names if name not in __all__)


def _is_missing_torch(exc: ModuleNotFoundError) -> bool:
    return exc.name == "torch"


from .channel.backends.numpy_backend import NumpyBackend

_export(["NumpyBackend"])

try:
    from .channel.backends.gpu_backend import GPUBackend, TorchBackend
    from .channel.backends.npu_backend import NPUBackend, NPURuntimeContext, npu_runtime_context_from_env
except ModuleNotFoundError as exc:
    if not _is_missing_torch(exc):
        raise
else:
    _export(
        [
            "GPUBackend",
            "TorchBackend",
            "NPUBackend",
            "NPURuntimeContext",
            "npu_runtime_context_from_env",
        ]
    )

from .channel.operators import Hamiltonian, PauliOp, PauliString

_export(
    [
        "PauliOp",
        "PauliString",
        "Hamiltonian",
    ]
)

from .ir import CircuitIR, Measurement, Observable, Operation

_export(["CircuitIR", "Measurement", "Observable", "Operation"])

try:
    from .channel.noise import (
        AmplitudeDampingChannel,
        BitFlipChannel,
        DepolarizingChannel,
        NoiseChannel,
        NoiseModel,
        PhaseFlipChannel,
    )
except ModuleNotFoundError as exc:
    if not _is_missing_torch(exc):
        raise
else:
    _export(
        [
            "NoiseChannel",
            "NoiseModel",
            "DepolarizingChannel",
            "BitFlipChannel",
            "PhaseFlipChannel",
            "AmplitudeDampingChannel",
        ]
    )

try:
    from .core.state import State, StateVector
    from .core.density import DensityMatrix
    from .core import (
        Circuit,
        Parameter,
        ccnot,
        cnot,
        circuit,
        crx,
        cry,
        crz,
        cx,
        cy,
        cz,
        hadamard,
        measure,
        molmer_sorensen,
        ms_gate,
        pauli_x,
        pauli_y,
        pauli_z,
        rxx,
        rzz,
        rx,
        ry,
        rz,
        s_gate,
        swap,
        t_gate,
        toffoli,
        u2,
        u3,
    )
    from .core.io.json_io import (
        circuit_from_json,
        circuit_to_json,
        load_circuit_json,
        save_circuit_json,
    )
    from .core.io.qasm import (
        circuit_from_qasm,
        circuit_to_qasm,
        circuit_to_qasm3,
        load_circuit_qasm,
        save_circuit_qasm,
        save_circuit_qasm3,
    )
except ModuleNotFoundError as exc:
    if not _is_missing_torch(exc):
        raise
else:
    _export(
        [
            "State",
            "StateVector",
            "DensityMatrix",
            "Circuit",
            "Parameter",
            "circuit",
            "pauli_x",
            "pauli_y",
            "pauli_z",
            "rzz",
            "rxx",
            "ms_gate",
            "molmer_sorensen",
            "hadamard",
            "measure",
            "rx",
            "ry",
            "rz",
            "s_gate",
            "t_gate",
            "cx",
            "cnot",
            "cy",
            "cz",
            "crx",
            "cry",
            "crz",
            "swap",
            "toffoli",
            "ccnot",
            "u2",
            "u3",
            "circuit_to_json",
            "circuit_from_json",
            "save_circuit_json",
            "load_circuit_json",
            "circuit_to_qasm",
            "circuit_to_qasm3",
            "circuit_from_qasm",
            "save_circuit_qasm",
            "save_circuit_qasm3",
            "load_circuit_qasm",
        ]
    )

try:
    from .measure import PauliEstimateResult, PauliEstimator
    from .measure.measure import Measure
    from .measure.result import Result
except ModuleNotFoundError as exc:
    if not _is_missing_torch(exc):
        raise
else:
    _export(["Measure", "PauliEstimator", "PauliEstimateResult", "Result"])

for _module_name in [
    "chemistry",
    "devices",
    "encoder",
    "gates",
    "ir",
    "metrics",
    "optimization",
    "primitives",
    "qas",
    "qml",
    "transpile",
    "universal",
    "visual",
    "vqc",
    "wireless",
]:
    try:
        globals()[_module_name] = import_module(f".{_module_name}", __name__)
    except ModuleNotFoundError as exc:
        if not _is_missing_torch(exc):
            raise
    else:
        _export([_module_name])
