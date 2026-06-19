import numpy as np
import pytest

from aicir import Circuit, NumpyBackend, Parameter, State, cx, ry
from aicir.qml import auto, psr, spsr, spsa, mpsr, fd, ad, qng, bdqng, kqng, dqng


def test_psr_matches_analytic_gradient_for_vector_params():
    params = np.array([0.3, -0.4])

    def objective(theta):
        return np.cos(theta[0]) + np.sin(theta[1])

    grad = psr(objective, params)

    assert np.allclose(grad, [-np.sin(params[0]), np.cos(params[1])])


def test_psr_preserves_parameter_shape():
    params = np.array([[0.1, 0.2], [0.3, 0.4]])

    def objective(theta):
        return np.sum(np.cos(theta))

    grad = psr(objective, params)

    assert grad.shape == params.shape
    assert np.allclose(grad, -np.sin(params))


def test_psr_supports_scalar_param():
    theta = np.array(0.7)

    def objective(value):
        return np.cos(value)

    grad = psr(objective, theta)

    assert grad.shape == ()
    assert np.allclose(grad, -np.sin(theta))


def test_psr_with_parameterized_circuit_template():
    theta = Parameter("theta")
    template = Circuit(ry(theta, 0), n_qubits=1)
    backend = NumpyBackend()
    z = np.diag([1.0, -1.0])

    def objective(values):
        circuit = template.bind_parameters({"theta": values[0]})
        state = State.zero_state(1, backend).evolve(circuit.unitary()).to_numpy().reshape(-1)
        return np.real(np.vdot(state, z @ state))

    value = np.array([0.5])
    grad = psr(objective, value)

    assert np.allclose(grad, [-np.sin(value[0])])


def test_psr_rejects_vector_valued_objective():
    def objective(theta):
        return np.array([theta[0], theta[0]])

    with pytest.raises(ValueError, match="scalar"):
        psr(objective, np.array([0.1]))


def test_spsr_matches_psr_when_all_parameters_are_sampled():
    params = np.array([0.1, 0.2, 0.3])

    def objective(theta):
        return np.sum(np.cos(theta))

    grad = spsr(objective, params, n_samples=params.size, rng=123)

    assert np.allclose(grad, psr(objective, params))


def test_spsr_returns_unbiased_scaled_coordinate_estimate():
    params = np.array([0.1, 0.2, 0.3])

    def objective(theta):
        return np.sum(np.cos(theta))

    sampled = np.random.default_rng(7).choice(params.size, size=1, replace=False)
    grad = spsr(objective, params, n_samples=1, rng=7)
    expected = np.zeros_like(params)
    expected[sampled[0]] = params.size * (-np.sin(params[sampled[0]]))

    assert np.allclose(grad, expected)


def test_spsr_rejects_too_many_samples_without_replacement():
    with pytest.raises(ValueError, match="n_samples"):
        spsr(lambda theta: theta[0], np.array([0.1]), n_samples=2)


def test_spsa_matches_central_difference_for_scalar_param():
    theta = np.array(0.7)

    def objective(value):
        return np.cos(value)

    grad = spsa(objective, theta, eps=1e-6, perturbations=np.array(1.0))

    assert grad.shape == ()
    assert np.allclose(grad, -np.sin(theta), atol=1e-6)


def test_spsa_uses_supplied_perturbation_formula():
    params = np.array([0.2, -0.1])
    weights = np.array([2.0, -3.0])
    delta = np.array([1.0, -1.0])

    def objective(theta):
        return float(np.dot(weights, theta))

    grad = spsa(objective, params, eps=1e-6, perturbations=delta)
    expected = np.dot(weights, delta) / delta

    assert np.allclose(grad, expected, atol=1e-9)


def test_spsa_averages_multiple_perturbations():
    params = np.array([0.2, -0.1])
    weights = np.array([2.0, -3.0])
    deltas = np.array([
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
    ])

    def objective(theta):
        return float(np.dot(weights, theta))

    grad = spsa(objective, params, eps=1e-6, perturbations=deltas)

    assert np.allclose(grad, weights, atol=1e-9)


def test_spsa_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="eps"):
        spsa(lambda theta: np.sum(theta), np.array([0.1]), eps=0.0)
    with pytest.raises(ValueError, match="zero"):
        spsa(lambda theta: np.sum(theta), np.array([0.1, 0.2]), perturbations=np.array([1.0, 0.0]))


def test_mpsr_single_index_matches_psr_coordinate():
    params = np.array([0.4, -0.2])

    def objective(theta):
        return np.cos(theta[0]) + np.sin(theta[1])

    assert np.allclose(mpsr(objective, params, parameter_indices=1), psr(objective, params)[1])


def test_mpsr_computes_mixed_partial_derivative():
    params = np.array([0.4, -0.2])

    def objective(theta):
        return np.cos(theta[0]) * np.sin(theta[1])

    mixed = mpsr(objective, params, parameter_indices=[0, 1])

    assert np.allclose(mixed, -np.sin(params[0]) * np.cos(params[1]))


def test_mpsr_supports_tuple_indices_for_shaped_params():
    params = np.array([[0.4, 0.1], [-0.2, 0.3]])

    def objective(theta):
        return np.cos(theta[0, 0]) * np.sin(theta[1, 0])

    mixed = mpsr(objective, params, parameter_indices=[(0, 0), (1, 0)])

    assert np.allclose(mixed, -np.sin(params[0, 0]) * np.cos(params[1, 0]))


def test_mpsr_rejects_duplicate_indices():
    with pytest.raises(ValueError, match="duplicates"):
        mpsr(lambda theta: np.sum(theta), np.array([0.1, 0.2]), parameter_indices=[0, 0])


def test_fd_matches_analytic_gradient_for_vector_params():
    params = np.array([0.3, -0.4])

    def objective(theta):
        return np.cos(theta[0]) + np.sin(theta[1])

    grad = fd(objective, params, eps=1e-6)

    assert np.allclose(grad, [-np.sin(params[0]), np.cos(params[1])], atol=1e-6)


@pytest.mark.parametrize("mode", ["central", "forward", "backward"])
def test_fd_supports_all_modes(mode):
    params = np.array([0.3, -0.4])

    def objective(theta):
        return np.cos(theta[0]) + np.sin(theta[1])

    grad = fd(objective, params, eps=1e-6, mode=mode)

    assert np.allclose(grad, [-np.sin(params[0]), np.cos(params[1])], atol=1e-4)


def test_fd_preserves_parameter_shape():
    params = np.array([[0.1, 0.2], [0.3, 0.4]])

    def objective(theta):
        return np.sum(np.cos(theta))

    grad = fd(objective, params, eps=1e-6)

    assert grad.shape == params.shape
    assert np.allclose(grad, -np.sin(params), atol=1e-6)


def test_fd_supports_scalar_param():
    theta = np.array(0.7)

    def objective(value):
        return np.cos(value)

    grad = fd(objective, theta, eps=1e-6)

    assert grad.shape == ()
    assert np.allclose(grad, -np.sin(theta), atol=1e-6)


def test_fd_with_parameterized_circuit_template():
    theta = Parameter("theta")
    template = Circuit(ry(theta, 0), n_qubits=1)
    backend = NumpyBackend()
    z = np.diag([1.0, -1.0])

    def objective(values):
        circuit = template.bind_parameters({"theta": values[0]})
        state = State.zero_state(1, backend).evolve(circuit.unitary()).to_numpy().reshape(-1)
        return np.real(np.vdot(state, z @ state))

    value = np.array([0.5])
    grad = fd(objective, value)

    assert np.allclose(grad, [-np.sin(value[0])], atol=1e-4)


def test_fd_rejects_invalid_mode():
    with pytest.raises(ValueError, match="mode"):
        fd(lambda t: np.sum(t), np.array([0.1, 0.2]), mode="diagonal")


def test_fd_rejects_non_positive_eps():
    with pytest.raises(ValueError, match="eps"):
        fd(lambda t: np.sum(t), np.array([0.1, 0.2]), eps=0.0)


def test_fd_rejects_vector_valued_objective():
    with pytest.raises(ValueError):
        fd(lambda t: t, np.array([0.1, 0.2]))


def test_gradients_accept_backend_native_tensor_objective():
    """Objectives returning raw backend tensors (e.g. NPU/Torch expectation_sv,
    autograd-tracked or complex) must work across every gradient rule."""
    torch = pytest.importorskip("torch")
    from aicir.backends.gpu_backend import TorchBackend

    theta = Parameter("theta")
    template = Circuit(ry(theta, 0), n_qubits=1)
    backend = TorchBackend(device="cpu")
    z = backend.cast(np.diag([1.0, -1.0]).astype(np.complex64))

    def objective(values):
        circuit = template.bind_parameters({"theta": float(values[0])})
        state = State.zero_state(1, backend).evolve(circuit.unitary(backend=backend))
        # Return the raw backend tensor (not float()) to mimic device-resident output.
        return backend.expectation_sv(state.data, z)

    value = np.array([0.5])
    exact = -np.sin(value[0])

    assert np.allclose(psr(objective, value), [exact], atol=1e-5)
    assert np.allclose(fd(objective, value), [exact], atol=1e-4)
    assert np.allclose(spsr(objective, value, n_samples=1, rng=0), [exact], atol=1e-5)
    assert np.allclose(spsa(objective, value, perturbations=np.array([1.0])), [exact], atol=1e-4)
    assert np.allclose(mpsr(objective, value, parameter_indices=[0]), exact, atol=1e-5)


def test_as_scalar_handles_complex_and_autograd_tensors():
    torch = pytest.importorskip("torch")
    from aicir.qml.deriv import _as_scalar

    assert _as_scalar(torch.tensor(0.7 + 0j), label="x") == pytest.approx(0.7)
    assert _as_scalar(torch.tensor(0.5, requires_grad=True) * 2, label="x") == pytest.approx(1.0)


def _parity_z_observable(n_qubits):
    diag = [(-1) ** bin(i).count("1") for i in range(1 << n_qubits)]
    return np.diag(diag).astype(np.complex64)


def test_ad_matches_analytic_single_rotation():
    backend = NumpyBackend()
    circuit = Circuit(ry(0.5, 0), n_qubits=1)
    z = np.diag([1.0, -1.0]).astype(np.complex64)
    grad = ad(circuit, z, backend=backend)
    assert np.allclose(grad, [-np.sin(0.5)], atol=1e-6)


def test_ad_matches_psr_on_mixed_ansatz():
    from aicir.core.circuit import cx, crx, rx, rxx, rz, rzz, hadamard

    backend = NumpyBackend()
    obs = _parity_z_observable(3)

    def build(theta):
        return Circuit(
            hadamard(0),
            ry(theta[0], 0), rz(theta[1], 1), rx(theta[2], 2),
            cx(1, [0]),
            crx(theta[3], 2, [1]),
            rzz(theta[4], 0, 2),
            rxx(theta[5], 0, 1),
            ry(theta[6], 1),
            n_qubits=3,
        )

    theta = np.array([0.3, -0.5, 0.8, 0.4, -0.7, 0.6, 0.9])

    def objective(p):
        state = State.zero_state(3, backend).evolve(build(p).unitary(backend=backend))
        return backend.to_numpy(backend.expectation_sv(state.data, backend.cast(obs)))

    grad_ad = ad(build(theta), obs, backend=backend)
    grad_psr = psr(objective, theta)
    assert grad_ad.shape == (7,)
    assert np.allclose(grad_ad, grad_psr, atol=1e-5)


def test_ad_returns_expectation_value_when_requested():
    backend = NumpyBackend()
    circuit = Circuit(ry(0.5, 0), n_qubits=1)
    z = np.diag([1.0, -1.0]).astype(np.complex64)
    grad, value = ad(circuit, z, backend=backend, return_value=True)
    assert np.allclose(grad, [-np.sin(0.5)], atol=1e-6)
    assert value == pytest.approx(np.cos(0.5), abs=1e-6)


def test_ad_accepts_hamiltonian_observable():
    from aicir.core.operators import Hamiltonian

    backend = NumpyBackend()
    circuit = Circuit(ry(0.5, 0), n_qubits=1)
    hamiltonian = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])
    grad = ad(circuit, hamiltonian, backend=backend)
    assert np.allclose(grad, [-np.sin(0.5)], atol=1e-6)


def test_ad_matches_psr_on_torch_backend():
    torch = pytest.importorskip("torch")
    from aicir.backends.gpu_backend import TorchBackend
    from aicir.core.circuit import rz

    backend = TorchBackend(device="cpu")
    obs = _parity_z_observable(2)
    circuit = Circuit(ry(0.4, 0), cx(1, [0]), rz(0.7, 1), n_qubits=2)

    def objective(p):
        c = Circuit(ry(p[0], 0), cx(1, [0]), rz(p[1], 1), n_qubits=2)
        state = State.zero_state(2, backend).evolve(c.unitary(backend=backend))
        return backend.to_numpy(backend.expectation_sv(state.data, backend.cast(obs)))

    grad_ad = ad(circuit, obs, backend=backend)
    grad_psr = psr(objective, np.array([0.4, 0.7]))
    assert np.allclose(grad_ad, grad_psr, atol=1e-5)


def test_ad_rejects_unbound_parameters():
    circuit = Circuit(ry(Parameter("theta"), 0), n_qubits=1)
    z = np.diag([1.0, -1.0]).astype(np.complex64)
    with pytest.raises(ValueError, match="unbound"):
        ad(circuit, z)


def test_auto_matches_psr_on_mixed_ansatz():
    torch = pytest.importorskip("torch")
    from aicir.backends.gpu_backend import TorchBackend
    from aicir.core.circuit import crx, rx, rz, rzz, hadamard

    backend = TorchBackend(device="cpu")
    obs = _parity_z_observable(3)

    def build(theta):
        return Circuit(
            hadamard(0),
            ry(theta[0], 0), rz(theta[1], 1), rx(theta[2], 2),
            cx(1, [0]), crx(theta[3], 2, [1]), rzz(theta[4], 0, 2), ry(theta[5], 1),
            n_qubits=3,
        )

    theta = np.array([0.3, -0.5, 0.8, 0.4, -0.7, 0.9])

    def fn(t):  # differentiable: returns a grad-connected tensor
        state = State.zero_state(3, backend).evolve(build(t).unitary(backend=backend))
        return backend.expectation_sv(state.data, backend.cast(obs))

    def obj(p):  # black-box float objective for psr reference
        state = State.zero_state(3, backend).evolve(build(p).unitary(backend=backend))
        return backend.to_numpy(backend.expectation_sv(state.data, backend.cast(obs)))

    grad_auto = auto(fn, theta, backend=backend)
    assert grad_auto.shape == theta.shape
    assert np.allclose(grad_auto, psr(obj, theta), atol=1e-5)


def test_auto_supports_scalar_param():
    torch = pytest.importorskip("torch")
    from aicir.backends.gpu_backend import TorchBackend

    backend = TorchBackend(device="cpu")
    z = np.diag([1.0, -1.0]).astype(np.complex64)

    def fn(t):
        circuit = Circuit(ry(t, 0), n_qubits=1)
        state = State.zero_state(1, backend).evolve(circuit.unitary(backend=backend))
        return backend.expectation_sv(state.data, backend.cast(z))

    grad = auto(fn, np.array(0.5), backend=backend)
    assert grad.shape == ()
    assert np.allclose(grad, -np.sin(0.5), atol=1e-5)


def test_auto_honors_backend_dtype_and_device():
    torch = pytest.importorskip("torch")
    from aicir.backends.gpu_backend import TorchBackend

    backend = TorchBackend(dtype=torch.complex128, device="cpu")
    z = np.diag([1.0, -1.0]).astype(np.complex128)
    captured = {}

    def fn(t):
        captured["dtype"] = t.dtype
        captured["device"] = t.device
        captured["requires_grad"] = t.requires_grad
        circuit = Circuit(ry(t, 0), n_qubits=1)
        state = State.zero_state(1, backend).evolve(circuit.unitary(backend=backend))
        return backend.expectation_sv(state.data, backend.cast(z))

    grad = auto(fn, np.array(0.5), backend=backend)
    # double-precision backend -> float64 params placed on the backend device.
    assert captured["dtype"] == torch.float64
    assert captured["device"] == backend._device
    assert captured["requires_grad"] is True
    assert np.allclose(grad, -np.sin(0.5), atol=1e-9)


def test_auto_rejects_non_differentiable_objective():
    torch = pytest.importorskip("torch")
    from aicir.backends.gpu_backend import TorchBackend

    backend = TorchBackend(device="cpu")

    def fn(t):  # breaks the graph by returning a python float
        circuit = Circuit(ry(t, 0), n_qubits=1)
        state = State.zero_state(1, backend).evolve(circuit.unitary(backend=backend))
        return float(backend.to_numpy(backend.expectation_sv(state.data, backend.cast(np.diag([1.0, -1.0]).astype(np.complex64)))))

    with pytest.raises(ValueError, match="autograd graph"):
        auto(fn, np.array(0.5), backend=backend)


def test_qng_matches_single_rotation_natural_gradient():
    backend = NumpyBackend()
    z = np.diag([1.0, -1.0]).astype(np.complex64)

    def state_fn(theta):
        circuit = Circuit(ry(theta[0], 0), n_qubits=1)
        return State.zero_state(1, backend).evolve(circuit.unitary(backend=backend))

    def objective(theta):
        state = state_fn(theta)
        return backend.expectation_sv(state.data, backend.cast(z))

    theta = np.array([0.5])
    natural_grad, grad, qfim = qng(
        objective,
        state_fn,
        theta,
        return_gradient=True,
        return_qfim=True,
    )

    assert np.allclose(grad, [-np.sin(theta[0])], atol=1e-5)
    assert np.allclose(qfim, [[1.0]], atol=1e-5)
    assert np.allclose(natural_grad, grad, atol=1e-5)


def test_qng_preconditions_with_supplied_qfim():
    natural_grad = qng(
        None,
        None,
        np.array([0.1, 0.2]),
        grad=np.array([2.0, 4.0]),
        qfim=np.diag([2.0, 4.0]),
        damping=0.0,
    )

    assert np.allclose(natural_grad, [1.0, 1.0])


def test_qng_accepts_spsa_gradient_method():
    backend = NumpyBackend()
    z = np.diag([1.0, -1.0]).astype(np.complex64)

    def state_fn(theta):
        circuit = Circuit(ry(theta[0], 0), n_qubits=1)
        return State.zero_state(1, backend).evolve(circuit.unitary(backend=backend))

    def objective(theta):
        state = state_fn(theta)
        return backend.expectation_sv(state.data, backend.cast(z))

    theta = np.array([0.5])
    natural_grad, grad = qng(
        objective,
        state_fn,
        theta,
        gradient_method="spsa",
        gradient_kwargs={"eps": 1e-3, "perturbations": np.array([1.0])},
        return_gradient=True,
    )

    assert np.allclose(grad, [-np.sin(theta[0])], atol=1e-4)
    assert np.allclose(natural_grad, grad, atol=1e-4)


def test_qng_accepts_npu_backend_state_tensor():
    pytest.importorskip("torch")
    from aicir.backends.npu_backend import NPUBackend

    backend = NPUBackend(device="cpu")
    z = backend.cast(np.diag([1.0, -1.0]).astype(np.complex64))

    def state_fn(theta):
        circuit = Circuit(ry(theta[0], 0), n_qubits=1)
        state = State.zero_state(1, backend).evolve(circuit.unitary(backend=backend))
        return state.data

    def objective(theta):
        return backend.expectation_sv(state_fn(theta), z)

    theta = np.array([0.5])
    natural_grad, qfim = qng(objective, state_fn, theta, return_qfim=True)

    assert np.allclose(qfim, [[1.0]], atol=1e-5)
    assert np.allclose(natural_grad, [-np.sin(theta[0])], atol=1e-5)


def test_bdqng_preconditions_with_supplied_qfim_blocks():
    natural_grad = bdqng(
        None,
        None,
        np.array([0.1, 0.2, 0.3]),
        blocks=[[0, 1], [2]],
        grad=np.array([2.0, 4.0, 6.0]),
        qfim_blocks=[np.diag([2.0, 4.0]), np.array([[3.0]])],
        damping=0.0,
    )

    assert np.allclose(natural_grad, [1.0, 1.0, 2.0])


def test_bdqng_single_block_matches_full_qng():
    backend = NumpyBackend()
    z = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.complex64)

    def state_fn(theta):
        circuit = Circuit(ry(theta[0], 0), ry(theta[1], 1), n_qubits=2)
        return State.zero_state(2, backend).evolve(circuit.unitary(backend=backend))

    def objective(theta):
        state = state_fn(theta)
        return backend.expectation_sv(state.data, backend.cast(z))

    theta = np.array([0.4, -0.2])
    natural_full, qfim_full = qng(objective, state_fn, theta, return_qfim=True)
    natural_block, qfim_blocks = bdqng(
        objective,
        state_fn,
        theta,
        blocks=[[0, 1]],
        return_qfim_blocks=True,
    )

    assert len(qfim_blocks) == 1
    assert np.allclose(qfim_blocks[0], qfim_full, atol=1e-5)
    assert np.allclose(natural_block, natural_full, atol=1e-5)


def test_bdqng_accepts_npu_backend_state_tensor():
    pytest.importorskip("torch")
    from aicir.backends.npu_backend import NPUBackend

    backend = NPUBackend(device="cpu")
    z = backend.cast(np.diag([1.0, -1.0]).astype(np.complex64))

    def state_fn(theta):
        circuit = Circuit(ry(theta[0], 0), n_qubits=1)
        state = State.zero_state(1, backend).evolve(circuit.unitary(backend=backend))
        return state.data

    def objective(theta):
        return backend.expectation_sv(state_fn(theta), z)

    theta = np.array([0.5])
    natural_grad, qfim_blocks = bdqng(objective, state_fn, theta, return_qfim_blocks=True)

    assert len(qfim_blocks) == 1
    assert np.allclose(qfim_blocks[0], [[1.0]], atol=1e-5)
    assert np.allclose(natural_grad, [-np.sin(theta[0])], atol=1e-5)


def test_bdqng_rejects_blocks_that_do_not_cover_all_parameters():
    with pytest.raises(ValueError, match="cover every parameter"):
        bdqng(
            None,
            None,
            np.array([0.1, 0.2]),
            blocks=[[0]],
            grad=np.array([1.0, 1.0]),
            qfim_blocks=[np.array([[1.0]])],
        )


def test_kqng_preconditions_with_supplied_kfac_factors():
    left = np.diag([2.0, 4.0])
    right = np.diag([3.0, 5.0])
    grad = np.array([[6.0, 10.0], [12.0, 20.0]])

    natural_grad = kqng(
        None,
        None,
        np.zeros((2, 2)),
        grad=grad,
        kfac_factors=[(left, right)],
        damping=0.0,
    )

    assert np.allclose(natural_grad, np.ones((2, 2)))


def test_kqng_single_column_factorization_matches_full_qng():
    backend = NumpyBackend()
    z = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.complex64)

    def state_fn(theta):
        circuit = Circuit(ry(theta[0], 0), ry(theta[1], 1), n_qubits=2)
        return State.zero_state(2, backend).evolve(circuit.unitary(backend=backend))

    def objective(theta):
        state = state_fn(theta)
        return backend.expectation_sv(state.data, backend.cast(z))

    theta = np.array([0.4, -0.2])
    natural_full = qng(objective, state_fn, theta, damping=0.0)
    natural_kfac, factors = kqng(
        objective,
        state_fn,
        theta,
        factor_shapes=(2, 1),
        damping=0.0,
        return_kfac_factors=True,
    )

    assert len(factors) == 1
    assert factors[0][0].shape == (2, 2)
    assert factors[0][1].shape == (1, 1)
    assert np.allclose(natural_kfac, natural_full, atol=1e-5)


def test_kqng_npu_backend_path_does_not_move_tensors_to_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    from aicir.backends.npu_backend import NPUBackend

    backend = NPUBackend(device="cpu")
    z = backend.cast(np.diag([1.0, -1.0]).astype(np.complex64))

    def state_fn(theta):
        circuit = Circuit(ry(theta[0], 0), n_qubits=1)
        state = State.zero_state(1, backend).evolve(circuit.unitary(backend=backend))
        return state.data

    def objective(theta):
        return backend.expectation_sv(state_fn(theta), z)

    def fail_cpu(self):
        raise AssertionError("kqng NPU path must not move tensors to CPU")

    monkeypatch.setattr(torch.Tensor, "cpu", fail_cpu)

    theta = np.array([0.5])
    natural_grad, factors = kqng(
        objective,
        state_fn,
        theta,
        backend=backend,
        return_kfac_factors=True,
    )

    assert isinstance(natural_grad, torch.Tensor)
    assert len(factors) == 1
    assert isinstance(factors[0][0], torch.Tensor)
    assert isinstance(factors[0][1], torch.Tensor)
    assert tuple(natural_grad.shape) == (1,)


def test_dqng_preconditions_with_supplied_qfim_diag():
    natural_grad = dqng(
        None,
        None,
        np.array([0.1, 0.2, 0.3]),
        grad=np.array([2.0, 4.0, 6.0]),
        qfim_diag=np.array([2.0, 4.0, 3.0]),
        damping=0.0,
    )

    assert np.allclose(natural_grad, [1.0, 1.0, 2.0])


def test_dqng_matches_bdqng_with_singleton_blocks():
    backend = NumpyBackend()
    z = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.complex64)

    def state_fn(theta):
        circuit = Circuit(ry(theta[0], 0), ry(theta[1], 1), n_qubits=2)
        return State.zero_state(2, backend).evolve(circuit.unitary(backend=backend))

    def objective(theta):
        state = state_fn(theta)
        return backend.expectation_sv(state.data, backend.cast(z))

    theta = np.array([0.4, -0.2])
    natural_diag, qfim_diag = dqng(objective, state_fn, theta, return_qfim_diag=True)
    natural_blocks, qfim_blocks = bdqng(objective, state_fn, theta, block_size=1, return_qfim_blocks=True)

    assert np.allclose(qfim_diag, [block[0, 0] for block in qfim_blocks], atol=1e-5)
    assert np.allclose(natural_diag, natural_blocks, atol=1e-5)


def test_dqng_npu_backend_path_does_not_move_tensors_to_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    from aicir.backends.npu_backend import NPUBackend

    backend = NPUBackend(device="cpu")
    z = backend.cast(np.diag([1.0, -1.0]).astype(np.complex64))

    def state_fn(theta):
        circuit = Circuit(ry(theta[0], 0), n_qubits=1)
        state = State.zero_state(1, backend).evolve(circuit.unitary(backend=backend))
        return state.data

    def objective(theta):
        return backend.expectation_sv(state_fn(theta), z)

    def fail_cpu(self):
        raise AssertionError("dqng NPU path must not move tensors to CPU")

    monkeypatch.setattr(torch.Tensor, "cpu", fail_cpu)

    theta = np.array([0.5])
    natural_grad, qfim_diag = dqng(
        objective,
        state_fn,
        theta,
        backend=backend,
        return_qfim_diag=True,
    )

    assert isinstance(natural_grad, torch.Tensor)
    assert isinstance(qfim_diag, torch.Tensor)
    assert tuple(natural_grad.shape) == (1,)
    assert tuple(qfim_diag.shape) == (1,)


def test_dqng_npu_backend_spsa_path_does_not_move_tensors_to_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    from aicir.backends.npu_backend import NPUBackend

    backend = NPUBackend(device="cpu")
    z = backend.cast(np.diag([1.0, -1.0]).astype(np.complex64))

    def state_fn(theta):
        circuit = Circuit(ry(theta[0], 0), n_qubits=1)
        state = State.zero_state(1, backend).evolve(circuit.unitary(backend=backend))
        return state.data

    def objective(theta):
        return backend.expectation_sv(state_fn(theta), z)

    def fail_cpu(self):
        raise AssertionError("dqng NPU SPSA path must not move tensors to CPU")

    monkeypatch.setattr(torch.Tensor, "cpu", fail_cpu)

    theta = np.array([0.5])
    natural_grad, qfim_diag = dqng(
        objective,
        state_fn,
        theta,
        backend=backend,
        gradient_method="spsa",
        gradient_kwargs={"eps": 1e-3, "perturbations": np.array([1.0])},
        return_qfim_diag=True,
    )

    assert isinstance(natural_grad, torch.Tensor)
    assert isinstance(qfim_diag, torch.Tensor)
    assert tuple(natural_grad.shape) == (1,)
    assert tuple(qfim_diag.shape) == (1,)
