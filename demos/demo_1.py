from aicir import (
    Circuit, Measure, NumpyBackend,
    hadamard, cnot, pauli_x, cry, rz, rzz, s_gate, swap, u3, u2, rxx, measure, cz, reset, t_gate
)
import numpy as np

# Gate factories keep their original signatures but now return typed
# Operation/Measurement IR objects (validated, immutable, dict-read compatible).
# The in-circuit measure/reset API is part of the same operation stream:
# - measure(*qubits, basis="Z", id=None) performs a projective Pauli measurement.
# - reset(*qubits) resets the target qubits to |0> with no prerequisite measure.
cir = Circuit(
    hadamard(0),
    u2(0.4, 0.5, 3),
    cry(np.pi / 3, 0, [2, 3]),
    rz(0.5, 1),
    cz(2, [1, 3]),
    rxx(np.pi / 4, 2, 3),
    u3(0.1, 0.2, 0.3, 2),
    cnot(1, [0, 3]),
    measure(0, 1, basis="Z", id="m0"),
    measure(2, basis="X", id="m1"),
    reset(2),
    pauli_x(1),
    t_gate(0),
    swap(0, 3),
    pauli_x(3),
    reset(0),
    s_gate(2),
    rzz(np.pi / 3, 0, 3),
)

cir.plot()

# In-circuit measure() and reset() can coexist with terminal measurement.
# Operation index 8 is the in-circuit measure above, and id="m0" gives the
# same result a stable name for result.output/counts/prob.
result = Measure(NumpyBackend()).run(cir, shots=10, seed=7)
print(result.summary())

print("measurement specs:", result.measurement_specs)
print("in-circuit output by op index:", result.output(8))
print("in-circuit output by id:", result.output("m0"))
print("in-circuit counts by id:", result.counts("m0"))
print("in-circuit probabilities by id:", result.prob("m0"))

# reset(0) is operation index 9. It is a channel in the circuit; there is no
# separate Result output for reset itself. The terminal measurement below is
# performed after reset and all following gates. -1 means terminal measurement.
print("terminal measured qubits:", result.terminal_qubits)
print("terminal output:", result.output(-1))
print("terminal counts:", result.counts(-1))
