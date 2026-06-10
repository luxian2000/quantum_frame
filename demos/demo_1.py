from aicir import (
    Circuit, Measure, NumpyBackend,
    hadamard, cnot, pauli_x, cry, rz, rzz, s_gate, swap, u3, u2, rxx, measure, cz
)
import numpy as np

cir = Circuit(
    hadamard(0),
    u2(0.4, 0.5, 3),
    cry(np.pi / 3, 0, [1, 3, 2]),
    rz(0.5, 1),
    cz(2, [3]),
    rxx(np.pi / 4, 2, 3),
    u3(0.1, 0.2, 0.3, 2),
    cnot(1, [0, 3]),
    cnot(2, [0]),
    swap(0, 3),
    pauli_x(3),
    s_gate(2),
    rzz(np.pi / 3, 0, 3),
    measure(1, 3)
)

cir.plot()

# Second measurement mechanism: the in-circuit measure() gate decides which
# qubits are read out, so Measure.run needs no separate measurement targets.
# Counts come back over qubits 1, 2, 3 only (3-bit strings).
result = Measure(NumpyBackend()).run(cir, shots=1024)
print(result.summary())
print("measured qubits:", result.metadata["measured_qubits"])
print("counts:", result.counts)
