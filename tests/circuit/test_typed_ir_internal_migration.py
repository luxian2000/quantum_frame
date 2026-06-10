import numpy as np
import pytest

from aicir import CircuitIR, Measurement, NumpyBackend, Operation
from aicir.channel.operators import Hamiltonian
from aicir.core.gates import gate_to_matrix
from aicir.core.io.json_io import circuit_from_json, circuit_to_json
from aicir.core.io.qasm import circuit_to_qasm
from aicir.measure import Measure
from aicir.metrics._utils import count_two_qubit_gates, depth_proxy
from aicir.optimizer.circuit import optimize_circuit
from aicir.qml import ad
from aicir.transpile import PassManager
from aicir.visual import gate_histogram, plot


@pytest.fixture()
def plt():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    yield pyplot
    pyplot.close("all")


def _typed_ir_example():
    return CircuitIR(
        (
            Operation("hadamard", qubits=(0,)),
            Operation("cx", qubits=(1,), controls=(0,)),
            Operation("rz", qubits=(1,), params=(0.25,)),
        ),
        n_qubits=2,
    )


def test_circuit_ir_is_accepted_by_qasm_visual_and_metrics(plt):
    ir = _typed_ir_example()

    qasm = circuit_to_qasm(ir)
    histogram = gate_histogram(ir)
    fig, ax = plot(ir, path=None, save=False)

    assert "h q[0];" in qasm
    assert "cx q[0],q[1];" in qasm
    assert histogram == {"cx": 1, "hadamard": 1, "rz": 1}
    assert count_two_qubit_gates(ir) == 1
    assert depth_proxy(ir) == 3.0
    assert fig is ax.figure


def test_circuit_ir_is_accepted_by_json_round_trip():
    ir = _typed_ir_example()

    rebuilt = circuit_from_json(circuit_to_json(ir))

    assert rebuilt.n_qubits == ir.n_qubits
    assert rebuilt.gates == ir.to_gate_dicts()


def test_circuit_ir_is_accepted_by_transpile_and_optimizer():
    ir = CircuitIR(
        (
            Operation("hadamard", qubits=(0,)),
            Operation("hadamard", qubits=(0,)),
            Operation("rx", qubits=(0,), params=(0.1,)),
            Operation("rx", qubits=(0,), params=(0.2,)),
        ),
        n_qubits=1,
    )

    optimized = PassManager(["cancel_inverse", "merge_rotations"]).run(ir)
    optimized_via_legacy = optimize_circuit(ir)

    assert optimized.gates == [{"type": "rx", "target_qubit": 0, "parameter": 0.30000000000000004}]
    assert optimized_via_legacy.gates == optimized.gates


def test_circuit_ir_is_accepted_by_measure_and_adjoint_gradient():
    backend = NumpyBackend()
    measure_ir = CircuitIR(
        (
            Operation("hadamard", qubits=(0,)),
            Operation("cx", qubits=(1,), controls=(0,)),
            Measurement((0, 1)),
        ),
        n_qubits=2,
    )

    result = Measure(backend).run(measure_ir, return_state=False)

    np.testing.assert_allclose(result.probabilities, [0.5, 0.0, 0.0, 0.5], atol=1e-6)

    grad_ir = CircuitIR(
        (Operation("ry", qubits=(0,), params=(0.4,)),),
        n_qubits=1,
    )

    gradient, value = ad(grad_ir, Hamiltonian([("Z", 1.0)]), backend=backend, return_value=True)

    np.testing.assert_allclose(value, np.cos(0.4), atol=1e-6)
    np.testing.assert_allclose(gradient, [-np.sin(0.4)], atol=1e-6)


def test_gate_matrix_accepts_typed_operation_directly():
    matrix = gate_to_matrix(Operation("rzz", qubits=(0, 1), params=(np.pi / 2,)), cir_qubits=2)

    assert matrix.shape == (4, 4)
