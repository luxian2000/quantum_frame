import unittest

import numpy as np
import torch

from aicir import Circuit, Measure, cnot, hadamard, ry, measure
from aicir.channel.backends import NumpyBackend
from aicir.core.circuit import hadamard, ry, cnot, crx, rxx, swap, toffoli, measure
from aicir.core.gates import apply_gate_to_state, gate_to_matrix
from aicir.measure.result import Result


measurement = Measure(NumpyBackend())
cir = Circuit(
    hadamard(0),
    ry(0.5, 1),
    cnot(1, [0]),
    crx(0.3, 2, [0]),
    rxx(0.4, 1, 2),
    swap(0, 2),
    toffoli(0, [1, 2]),
    measure(0)
)

cir.plot()

result = measurement.run(cir)

print(result.output)

print(result.state)

print(result.counts)

print(result.final_state)