import numpy as np
import pytest

from aicir.optimizer import (
    Adam,
    COBYLA,
    GD,
    LBFGSB,
    NelderMead,
    OptimizationResult,
    SPSA,
    ScipyMinimize,
    minimize,
    scipy_minimize,
)


def _quadratic(target):
    target = np.asarray(target, dtype=float)

    def fn(params):
        delta = np.asarray(params, dtype=float) - target
        return float(np.dot(delta.reshape(-1), delta.reshape(-1)))

    def grad(params):
        return 2.0 * (np.asarray(params, dtype=float) - target)

    return fn, grad


def test_adam_optimizer_minimizes_with_explicit_gradient():
    fn, grad = _quadratic(np.array([1.0, -2.0]))
    optimizer = Adam(max_iters=250, learning_rate=0.08, tol=1e-4)

    result = optimizer.minimize(fn, np.array([0.0, 0.0]), gradient_fn=grad)

    assert isinstance(result, OptimizationResult)
    assert result.success
    assert result.fun < 1e-8
    assert np.allclose(result.x, [1.0, -2.0], atol=1e-3)
    assert result.best_fun <= result.fun + 1e-12
    assert result.history


def test_gradient_descent_accepts_psr_gradient_method():
    def fn(params):
        return float(np.cos(params[0]))

    optimizer = GD(max_iters=80, learning_rate=0.15, gradient_method="psr")
    result = optimizer.minimize(fn, np.array([0.1]))

    assert result.fun < -0.99


def test_spsa_optimizer_minimizes_one_dimensional_objective():
    fn, _ = _quadratic(np.array([2.0]))
    optimizer = SPSA(max_iters=80, learning_rate=0.08, perturbation=1e-3, rng=3)

    result = optimizer.minimize(fn, np.array([0.0]))

    assert result.success
    assert result.fun < 1e-6
    assert np.allclose(result.x, [2.0], atol=1e-3)


def test_spsa_optimizer_uses_provided_gradient_fn():
    fn, grad = _quadratic(np.array([2.0]))
    calls = []

    def counting_grad(params):
        calls.append(params.copy())
        return grad(params)

    optimizer = SPSA(max_iters=80, learning_rate=0.08, perturbation=1e-3, rng=3)
    result = optimizer.minimize(fn, np.array([0.0]), gradient_fn=counting_grad)

    assert result.success
    assert result.fun < 1e-8
    assert np.allclose(result.x, [2.0], atol=1e-4)
    assert len(calls) == result.nit


def test_optimizer_callbacks_receive_copied_params():
    fn, grad = _quadratic(np.array([1.0]))
    seen = []

    def callback(step, value, params):
        params[...] = 999.0
        seen.append((step, value, params.copy()))

    optimizer = Adam(max_iters=3, learning_rate=0.1, save_history=False)
    result = optimizer.minimize(fn, np.array([0.0]), gradient_fn=grad, callback=callback)

    assert result.nit == 3
    assert len(seen) == 3
    assert result.x[0] != 999.0


def test_scipy_optimizer_minimizes_with_l_bfgs_b_and_gradient():
    pytest.importorskip("scipy")
    fn, grad = _quadratic(np.array([1.5, -0.5]))
    optimizer = LBFGSB(options={"maxiter": 50})

    result = optimizer.minimize(fn, np.array([0.0, 0.0]), gradient_fn=grad)

    assert result.success
    assert result.fun < 1e-10
    assert np.allclose(result.x, [1.5, -0.5], atol=1e-5)
    assert result.raw_result is not None


def test_cobyla_optimizer_and_scipy_minimize_wrapper():
    pytest.importorskip("scipy")
    fn, _ = _quadratic(np.array([0.25]))

    result = COBYLA(options={"maxiter": 80, "rhobeg": 0.5}).minimize(fn, np.array([2.0]))
    wrapped = scipy_minimize(fn, np.array([2.0]), method="COBYLA", options={"maxiter": 80, "rhobeg": 0.5})
    generic = minimize(fn, np.array([2.0]), optimizer=ScipyMinimize("COBYLA", options={"maxiter": 80}))

    assert result.fun < 1e-6
    assert wrapped.fun < 1e-6
    assert generic.fun < 1e-5


def test_nelder_mead_optimizer_minimizes_black_box_objective():
    pytest.importorskip("scipy")
    fn, _ = _quadratic(np.array([1.0, -2.0]))

    result = NelderMead(options={"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-8}).minimize(
        fn,
        np.array([3.0, 4.0]),
    )

    assert result.success
    assert result.fun < 1e-8
    assert np.allclose(result.x, [1.0, -2.0], atol=1e-4)
