from aicir import Circuit, hadamard, cnot, pauli_x, cry, rz, rzz, s_gate, swap, u3, u2, rxx
import numpy as np

cir = Circuit(
    hadamard(0),
    u2(0.4, 0.5, 3),
    cry(np.pi / 3, 0, [1, 2]),
    rz(0.5, 1),
    rxx(np.pi / 4, 2, 3),
    u3(0.1, 0.2, 0.3, 2),
    cnot(1, [0, 2, 3]),
    cnot(2, [0]),
    swap(1, 2),
    pauli_x(3),
    s_gate(2),
    rzz(np.pi / 3, 1, 3)
)

cir.plot()
