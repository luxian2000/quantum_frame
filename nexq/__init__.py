# nexq — 量子模拟器顶层包
from .channel.backends.numpy_backend import NumpyBackend
from .core.state import State, StateVector
from .core.density import DensityMatrix
from .channel.operators import PauliOp, PauliString, Hamiltonian
from .channel.noise import (
    AmplitudeDampingChannel,
    BitFlipChannel,
    DepolarizingChannel,
    NoiseChannel,
    NoiseModel,
    PhaseFlipChannel,
)
from .measure.measure import Measure
from .measure.result import Result
from .core import (
    Circuit,
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
    pauli_x,
    pauli_y,
    pauli_z,
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
from . import (
    chemistry,
    encoder,
    metrics,
    optimization,
    qas,
    qml,
    universal,
    vqc,
    wireless,
)

_OPTIONAL_BACKENDS: list[str] = []
try:
    from .channel.backends.torch_backend import TorchBackend
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    _OPTIONAL_BACKENDS.append("TorchBackend")

try:
    from .channel.backends.npu_backend import NPUBackend, NPURuntimeContext, npu_runtime_context_from_env
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    _OPTIONAL_BACKENDS.extend(["NPUBackend", "NPURuntimeContext", "npu_runtime_context_from_env"])

__all__ = [
    "NumpyBackend",
    "State",
    "StateVector",
    "DensityMatrix",
    "PauliOp",
    "PauliString",
    "Hamiltonian",
    "NoiseChannel",
    "NoiseModel",
    "DepolarizingChannel",
    "BitFlipChannel",
    "PhaseFlipChannel",
    "AmplitudeDampingChannel",
    "Measure",
    "Result",
    "Circuit",
    "circuit",
    "pauli_x",
    "pauli_y",
    "pauli_z",
    "hadamard",
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
    "chemistry",
    "encoder",
    "metrics",
    "optimization",
    "qas",
    "qml",
    "universal",
    "vqc",
    "wireless",
] + _OPTIONAL_BACKENDS
