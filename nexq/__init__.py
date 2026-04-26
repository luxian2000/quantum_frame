# nexq — 量子模拟器顶层包
from .channel.backends.torch_backend import TorchBackend
from .channel.backends.numpy_backend import NumpyBackend
from .channel.backends.npu_backend import NPUBackend, NPURuntimeContext, npu_runtime_context_from_env
from .circuit.state_vector import StateVector
from .circuit.density_matrix import DensityMatrix
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
from .circuit import (
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
from .circuit.io.json_io import (
    circuit_from_json,
    circuit_to_json,
    load_circuit_json,
    save_circuit_json,
)
from .circuit.io.qasm import (
    circuit_from_qasm,
    circuit_to_qasm,
    circuit_to_qasm3,
    load_circuit_qasm,
    save_circuit_qasm,
    save_circuit_qasm3,
)
from . import algorithms
from . import encoder

__all__ = [
    "TorchBackend",
    "NumpyBackend",
    "NPUBackend",
    "NPURuntimeContext",
    "npu_runtime_context_from_env",
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
    "algorithms",
    "encoder",
]
