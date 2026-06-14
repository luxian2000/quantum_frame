import numpy as np
import pytest

from aicir import Circuit, NumpyBackend, State, ry
from aicir.qml import rotosolve


def test_rotosolve_minimizes_separable_sinusoidal_objective():
    def objective(theta):
        return (
            2.0 * np.sin(theta[0] + 0.3)
            + 0.5 * np.sin(theta[1] - 0.2)
            + 1.25
        )

    optimized, value = rotosolve(
        objective,
        np.array([0.7, -0.9]),
        return_value=True,
    )

    assert optimized.shape == (2,)
    assert value == pytest.approx(1.25 - 2.0 - 0.5, abs=1e-12)


def test_rotosolve_preserves_parameter_shape():
    def objective(theta):
        return np.sum(np.sin(theta + np.array([[0.1, -0.2], [0.3, -0.4]])))

    params = np.array([[0.2, 0.4], [-0.5, 0.1]])
    optimized = rotosolve(objective, params)

    assert optimized.shape == params.shape
    assert objective(optimized) == pytest.approx(-4.0, abs=1e-12)


def test_rotosolve_updates_only_selected_coordinates():
    def objective(theta):
        return np.sin(theta[0] + 0.1) + np.sin(theta[1] - 0.2)

    params = np.array([0.3, 0.4])
    optimized = rotosolve(objective, params, parameter_indices=[0])

    assert optimized[1] == pytest.approx(params[1])
    assert objective(optimized) == pytest.approx(
        -1.0 + np.sin(params[1] - 0.2),
        abs=1e-12,
    )


def test_rotosolve_leaves_flat_coordinate_unchanged():
    params = np.array([0.3])

    optimized = rotosolve(lambda theta: 1.0, params)

    assert np.allclose(optimized, params)


def test_rotosolve_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="n_sweeps"):
        rotosolve(lambda theta: np.sum(theta), np.array([0.1]), n_sweeps=0)
    with pytest.raises(ValueError, match="shift"):
        rotosolve(lambda theta: np.sum(theta), np.array([0.1]), shift=np.pi)
    with pytest.raises(ValueError, match="scalar"):
        rotosolve(lambda theta: np.array([theta[0], theta[0]]), np.array([0.1]))


def test_rotosolve_with_parameterized_circuit_template():
    backend = NumpyBackend()
    z = backend.cast(np.diag([1.0, -1.0]).astype(np.complex64))

    def objective(theta):
        circuit = Circuit(ry(theta[0], 0), n_qubits=1)
        state = State.zero_state(1, backend).evolve(circuit.unitary(backend=backend))
        return backend.expectation_sv(state.data, z)

    optimized, value = rotosolve(objective, np.array([0.5]), return_value=True)

    assert value == pytest.approx(-1.0, abs=1e-6)
    assert np.cos(optimized[0]) == pytest.approx(-1.0, abs=1e-6)


def test_rotosolve_npu_backend_path_does_not_move_tensors_to_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    from aicir.backends.npu_backend import NPUBackend

    backend = NPUBackend(device="cpu")
    z = backend.cast(np.diag([1.0, -1.0]).astype(np.complex64))

    def objective(theta):
        circuit = Circuit(ry(theta[0], 0), n_qubits=1)
        state = State.zero_state(1, backend).evolve(circuit.unitary(backend=backend))
        return backend.expectation_sv(state.data, z)

    def fail_cpu(self):
        raise AssertionError("rotosolve NPU path must not move tensors to CPU")

    monkeypatch.setattr(torch.Tensor, "cpu", fail_cpu)

    optimized, value = rotosolve(
        objective,
        np.array([0.5]),
        backend=backend,
        return_value=True,
    )

    assert isinstance(optimized, torch.Tensor)
    assert isinstance(value, torch.Tensor)
    assert tuple(optimized.shape) == (1,)
    assert torch.allclose(value, torch.as_tensor(-1.0, dtype=value.dtype), atol=1e-5)
