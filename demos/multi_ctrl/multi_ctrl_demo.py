import numpy as np

from aicir import Circuit, Measure, NumpyBackend, hadamard, cnot, pauli_x

cir = Circuit(
    hadamard(0),
    pauli_x(2),
    hadamard(3),
    cnot([1, 3], [0, 2])
)

mm = Measure(NumpyBackend())

result = mm.run(cir, shots=1, snap=[1, 2])

# snap(1) 与 state 现在都是 State 对象，可直接打印狄拉克记号
print(result.snap(1).ket)
print(result.snap(2).ket)
print(result.state.ket)

# 多目标受控门的 16x16 酉矩阵（cnot([1,2],[0,3]) 跨越 q0..q3）
with np.errstate(all="ignore"):
    matrix = Circuit(cnot([1, 2], [0, 3]), n_qubits=4).unitary()
np.set_printoptions(linewidth=200)
print(np.real_if_close(matrix).astype(int))

cir.plot()