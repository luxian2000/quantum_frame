from aicir import Circuit, hadamard, cnot, pauli_x, cx, ry, rz, rzz, s_gate
from aicir.visual import plot

cir = Circuit(
    hadamard(0),
    cnot(0, [1, 2]),
    rz(0.5, 1),
    cnot(1, [0, 2]),
    cnot(2, [0]),
    pauli_x(3),
    s_gate(2),
    rzz(0.2, 0, 2),
    rzz(0.3, 1, 3)
)

plot(cir)