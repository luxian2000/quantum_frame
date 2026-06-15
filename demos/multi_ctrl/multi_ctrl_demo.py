from aicir import Circuit, Measure, NumpyBackend, hadamard, cnot

cir = Circuit(
    hadamard(0),
    hadamard(3),
    cnot(1, [0, 3], [0, 1])
)

mm = Measure(NumpyBackend())

result = mm.run(cir, shots=1, snap=[1])

# snap(1) 与 state 现在都是 State 对象，可直接打印狄拉克记号
print(result.snap(1).ket)

print(result.state.ket)

cir.plot()