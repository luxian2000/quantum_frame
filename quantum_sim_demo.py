#!/usr/bin/env python3
"""quantum_sim_demo.py

一个快速演示：从定义量子线路，到使用 `Measure` 运行（状态矢量 / 密度矩阵 + 噪声）、参数扫描、以及 JSON/QASM I/O。

运行：
    python quantum_sim_demo.py
"""

import math
import numpy as np

from quantum_sim import (
    Circuit,
    hadamard,
    cnot,
    ry,
    Measure,
    TorchBackend,
    NumpyBackend,
    circuit_to_json,
    circuit_to_qasm,
    NoiseModel,
    BitFlipChannel,
    Hamiltonian,
)


def sep(title: str):
    print("\n" + "=" * 10 + " " + title + " " + "=" * 10)


def bell_demo():
    sep("Bell state (state vector)")
    backend = TorchBackend(device="cpu")
    engine = Measure(backend)

    circ = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
    print("Circuit:", circ)

    res = engine.run(circ, shots=1024)
    print("Probabilities:", np.round(res.probabilities, 6))
    print("Counts (sampled):", res.counts)
    print("Most probable:", res.most_probable())
    return circ, engine, backend


def expectation_demo(circ, engine, backend):
    sep("Hamiltonian expectation (ZZ)")
    H = Hamiltonian(n_qubits=2).add_term(1.0, {"Z": [0, 1]})
    op = H.to_matrix(backend)
    res = engine.run(circ, shots=None, observables={"ZZ": op})
    print("<ZZ> =", res.expectation_values.get("ZZ"))
    print("Var(ZZ) =", res.expectation_variances.get("ZZ"))


def noise_demo(circ):
    sep("Density matrix + noise demo")
    backend = TorchBackend(device="cpu")
    engine = Measure(backend)
    noise = NoiseModel().add_channel(BitFlipChannel(target_qubit=0, p=0.9))
    res = engine.run_density_matrix(circ, shots=None, noise_model=noise)
    print("Probabilities with bit-flip(p=0.9):", np.round(res.probabilities, 6))


def scan_demo(backend):
    sep("Parameter scan (RY on q0)")
    engine = Measure(backend)
    HZ = Hamiltonian(n_qubits=2).add_term(1.0, {"Z": [0]})
    op = HZ.to_matrix(backend)

    params = [0.0, math.pi / 2.0, math.pi]

    def build(theta):
        return Circuit(ry(theta, 0), n_qubits=2)

    results = engine.scan_parameters(build, params, shots=None, observables={"Z0": op})
    for r in results:
        print(
            f"index={r.metadata['scan_index']}, theta={r.metadata['scan_param']:.6f}, <Z0>={r.expectation_values['Z0']:.6f}"
        )


def io_demo(circ):
    sep("JSON / QASM I/O demo")
    j = circuit_to_json(circ)
    q2 = circuit_to_qasm(circ, version="2.0")
    q3 = circuit_to_qasm(circ, version="3.0")
    print("JSON:\n", j)
    print("QASM2:\n", q2)
    print("QASM3:\n", q3)


def numpy_backend_demo(circ):
    sep("Numpy backend demo")
    backend = NumpyBackend()
    engine = Measure(backend)
    res = engine.run(circ, shots=512)
    print("Probabilities (NumpyBackend):", np.round(res.probabilities, 6))


def main():
    circ, engine, backend = bell_demo()
    expectation_demo(circ, engine, backend)
    noise_demo(circ)
    scan_demo(backend)
    io_demo(circ)
    numpy_backend_demo(circ)
    print("\nDemo finished.")


if __name__ == "__main__":
    main()
