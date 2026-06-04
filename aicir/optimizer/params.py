"""Classical parameter optimizers for VQE/VQA objectives."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..qml.deriv import fd, psr, spsa


ScalarFn = Callable[[np.ndarray], Any]
GradientFn = Callable[[np.ndarray], Any]
Schedule = float | Callable[[int], float]


def _as_scalar(value: Any, *, label: str) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{label} must return a scalar value")
    if np.iscomplexobj(array):
        array = array.real
    return float(array)


def _as_params(params: Any) -> np.ndarray:
    theta = np.asarray(params, dtype=float)
    if theta.size == 0:
        raise ValueError("init_params must contain at least one parameter")
    return theta.copy()


def _schedule_value(value: Schedule, iteration: int, *, label: str) -> float:
    raw = value(iteration) if callable(value) else value
    scalar = float(raw)
    if not scalar > 0.0:
        raise ValueError(f"{label} must be positive")
    return scalar


def _call_callback(callback, step: int, value: float, params: np.ndarray) -> None:
    if callback is not None:
        callback(step, value, params.copy())


class _ObjectiveCounter:
    def __init__(self, fn: ScalarFn) -> None:
        self.fn = fn
        self.nfev = 0

    def __call__(self, params: np.ndarray) -> float:
        self.nfev += 1
        return _as_scalar(self.fn(np.asarray(params, dtype=float)), label="fn(params)")


@dataclass
class OptimizationResult:
    """Result returned by classical parameter optimizers."""

    x: np.ndarray
    fun: float
    nit: int
    nfev: int
    success: bool
    message: str
    best_x: np.ndarray | None = None
    best_fun: float | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    raw_result: Any = None

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.fun = float(self.fun)
        if self.best_x is None:
            self.best_x = self.x.copy()
        else:
            self.best_x = np.asarray(self.best_x, dtype=float)
        if self.best_fun is None:
            self.best_fun = self.fun
        else:
            self.best_fun = float(self.best_fun)

    @property
    def parameters(self) -> np.ndarray:
        """Alias for VQE-style result access."""

        return self.x

    @property
    def value(self) -> float:
        """Alias for objective value access."""

        return self.fun


def _gradient_from_method(
    fn: ScalarFn,
    params: np.ndarray,
    gradient_method: str | GradientFn,
    gradient_kwargs: Mapping[str, Any] | None,
) -> np.ndarray:
    if callable(gradient_method):
        grad = gradient_method(params)
        return np.asarray(grad, dtype=float).reshape(params.shape)

    kwargs = dict(gradient_kwargs or {})
    method = str(gradient_method).strip().lower()
    if method == "psr":
        return psr(fn, params, **kwargs)
    if method == "fd":
        return fd(fn, params, **kwargs)
    if method == "spsa":
        return spsa(fn, params, **kwargs)
    raise ValueError("gradient_method must be 'psr', 'fd', 'spsa', or a callable")


class GD:
    """Fixed-step gradient descent for VQE/VQA objectives."""

    def __init__(
        self,
        *,
        max_iters: int = 200,
        learning_rate: Schedule = 0.1,
        gradient_method: str | GradientFn = "fd",
        gradient_kwargs: Mapping[str, Any] | None = None,
        tol: float | None = None,
        save_history: bool = True,
    ) -> None:
        self.max_iters = int(max_iters)
        if self.max_iters <= 0:
            raise ValueError("max_iters must be positive")
        self.learning_rate = learning_rate
        self.gradient_method = gradient_method
        self.gradient_kwargs = dict(gradient_kwargs or {})
        self.tol = None if tol is None else float(tol)
        self.save_history = bool(save_history)

    def minimize(
        self,
        fn: ScalarFn,
        init_params: Any,
        *,
        gradient_fn: GradientFn | None = None,
        callback: Callable[[int, float, np.ndarray], None] | None = None,
    ) -> OptimizationResult:
        params = _as_params(init_params)
        objective = _ObjectiveCounter(fn)
        gradient_method = gradient_fn if gradient_fn is not None else self.gradient_method
        history: list[dict[str, Any]] = []
        best_x = params.copy()
        best_fun = np.inf
        message = "Reached max_iters"
        success = False
        iterations_done = 0

        for step in range(self.max_iters):
            iterations_done = step + 1
            value = objective(params)
            if value < best_fun:
                best_fun = value
                best_x = params.copy()

            grad = _gradient_from_method(objective, params, gradient_method, self.gradient_kwargs)
            grad_norm = float(np.linalg.norm(grad.reshape(-1)))
            lr = _schedule_value(self.learning_rate, step, label="learning_rate")

            if self.save_history:
                history.append({"step": step, "fun": value, "grad_norm": grad_norm, "learning_rate": lr})
            _call_callback(callback, step, value, params)

            if self.tol is not None and grad_norm <= self.tol:
                message = "Gradient norm reached tol"
                success = True
                break

            params = params - lr * grad

        final_value = objective(params)
        if final_value < best_fun:
            best_fun = final_value
            best_x = params.copy()

        if not success and self.tol is None:
            success = True
        return OptimizationResult(
            x=params,
            fun=final_value,
            nit=iterations_done,
            nfev=objective.nfev,
            success=success,
            message=message,
            best_x=best_x,
            best_fun=best_fun,
            history=history,
        )


class Adam:
    """Adam optimizer over NumPy parameter arrays."""

    def __init__(
        self,
        *,
        max_iters: int = 200,
        learning_rate: Schedule = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        gradient_method: str | GradientFn = "fd",
        gradient_kwargs: Mapping[str, Any] | None = None,
        tol: float | None = None,
        save_history: bool = True,
    ) -> None:
        self.max_iters = int(max_iters)
        if self.max_iters <= 0:
            raise ValueError("max_iters must be positive")
        self.learning_rate = learning_rate
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        if not 0.0 <= self.beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0.0 <= self.beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        if not self.eps > 0.0:
            raise ValueError("eps must be positive")
        self.gradient_method = gradient_method
        self.gradient_kwargs = dict(gradient_kwargs or {})
        self.tol = None if tol is None else float(tol)
        self.save_history = bool(save_history)

    def minimize(
        self,
        fn: ScalarFn,
        init_params: Any,
        *,
        gradient_fn: GradientFn | None = None,
        callback: Callable[[int, float, np.ndarray], None] | None = None,
    ) -> OptimizationResult:
        params = _as_params(init_params)
        objective = _ObjectiveCounter(fn)
        gradient_method = gradient_fn if gradient_fn is not None else self.gradient_method
        m = np.zeros_like(params, dtype=float)
        v = np.zeros_like(params, dtype=float)
        history: list[dict[str, Any]] = []
        best_x = params.copy()
        best_fun = np.inf
        message = "Reached max_iters"
        success = False
        iterations_done = 0

        for step in range(self.max_iters):
            iterations_done = step + 1
            value = objective(params)
            if value < best_fun:
                best_fun = value
                best_x = params.copy()

            grad = _gradient_from_method(objective, params, gradient_method, self.gradient_kwargs)
            grad_norm = float(np.linalg.norm(grad.reshape(-1)))
            lr = _schedule_value(self.learning_rate, step, label="learning_rate")

            if self.save_history:
                history.append({"step": step, "fun": value, "grad_norm": grad_norm, "learning_rate": lr})
            _call_callback(callback, step, value, params)

            if self.tol is not None and grad_norm <= self.tol:
                message = "Gradient norm reached tol"
                success = True
                break

            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad * grad)
            t = step + 1
            m_hat = m / (1.0 - self.beta1 ** t)
            v_hat = v / (1.0 - self.beta2 ** t)
            params = params - lr * m_hat / (np.sqrt(v_hat) + self.eps)

        final_value = objective(params)
        if final_value < best_fun:
            best_fun = final_value
            best_x = params.copy()

        if not success and self.tol is None:
            success = True
        return OptimizationResult(
            x=params,
            fun=final_value,
            nit=iterations_done,
            nfev=objective.nfev,
            success=success,
            message=message,
            best_x=best_x,
            best_fun=best_fun,
            history=history,
        )


class SPSA:
    """SPSA-gradient optimizer for noisy or expensive VQE objectives."""

    def __init__(
        self,
        *,
        max_iters: int = 200,
        learning_rate: Schedule = 0.1,
        perturbation: Schedule = 1e-2,
        n_samples: int = 1,
        rng: Any = None,
        tol: float | None = None,
        save_history: bool = True,
    ) -> None:
        self.max_iters = int(max_iters)
        if self.max_iters <= 0:
            raise ValueError("max_iters must be positive")
        self.learning_rate = learning_rate
        self.perturbation = perturbation
        self.n_samples = int(n_samples)
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        self.rng = rng
        self.tol = None if tol is None else float(tol)
        self.save_history = bool(save_history)

    def minimize(
        self,
        fn: ScalarFn,
        init_params: Any,
        *,
        callback: Callable[[int, float, np.ndarray], None] | None = None,
    ) -> OptimizationResult:
        params = _as_params(init_params)
        objective = _ObjectiveCounter(fn)
        rng = np.random.default_rng(self.rng)
        history: list[dict[str, Any]] = []
        best_x = params.copy()
        best_fun = np.inf
        message = "Reached max_iters"
        success = False
        iterations_done = 0

        for step in range(self.max_iters):
            iterations_done = step + 1
            value = objective(params)
            if value < best_fun:
                best_fun = value
                best_x = params.copy()

            lr = _schedule_value(self.learning_rate, step, label="learning_rate")
            eps_value = _schedule_value(self.perturbation, step, label="perturbation")
            grad = spsa(objective, params, eps=eps_value, n_samples=self.n_samples, rng=rng)
            grad_norm = float(np.linalg.norm(grad.reshape(-1)))

            if self.save_history:
                history.append(
                    {
                        "step": step,
                        "fun": value,
                        "grad_norm": grad_norm,
                        "learning_rate": lr,
                        "perturbation": eps_value,
                    }
                )
            _call_callback(callback, step, value, params)

            if self.tol is not None and grad_norm <= self.tol:
                message = "Gradient norm reached tol"
                success = True
                break

            params = params - lr * grad

        final_value = objective(params)
        if final_value < best_fun:
            best_fun = final_value
            best_x = params.copy()

        if not success and self.tol is None:
            success = True
        return OptimizationResult(
            x=params,
            fun=final_value,
            nit=iterations_done,
            nfev=objective.nfev,
            success=success,
            message=message,
            best_x=best_x,
            best_fun=best_fun,
            history=history,
        )


class ScipyMinimize:
    """Wrapper around ``scipy.optimize.minimize`` preserving parameter shape."""

    def __init__(
        self,
        method: str = "COBYLA",
        *,
        bounds: Any = None,
        constraints: Any = (),
        options: Mapping[str, Any] | None = None,
        gradient_method: str | GradientFn | None = None,
        gradient_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.method = method
        self.bounds = bounds
        self.constraints = constraints
        self.options = dict(options or {})
        self.gradient_method = gradient_method
        self.gradient_kwargs = dict(gradient_kwargs or {})

    def minimize(
        self,
        fn: ScalarFn,
        init_params: Any,
        *,
        gradient_fn: GradientFn | None = None,
        callback: Callable[[int, float, np.ndarray], None] | None = None,
    ) -> OptimizationResult:
        try:
            from scipy.optimize import minimize as scipy_minimize_impl
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ScipyMinimize requires scipy; install scipy or use Adam/SPSA."
            ) from exc

        init = _as_params(init_params)
        shape = init.shape
        objective = _ObjectiveCounter(lambda x: fn(np.asarray(x, dtype=float).reshape(shape)))
        history: list[dict[str, Any]] = []

        def flat_objective(x_flat):
            return objective(np.asarray(x_flat, dtype=float))

        jac_method = gradient_fn if gradient_fn is not None else self.gradient_method
        jac = None
        if jac_method is not None:
            def jac(x_flat):
                theta = np.asarray(x_flat, dtype=float).reshape(shape)
                grad = _gradient_from_method(
                    lambda value: objective(np.asarray(value, dtype=float).reshape(shape)),
                    theta,
                    jac_method,
                    self.gradient_kwargs,
                )
                return np.asarray(grad, dtype=float).reshape(-1)

        step_counter = {"step": 0}

        def scipy_callback(xk, *args):
            theta = np.asarray(xk, dtype=float).reshape(shape)
            value = objective(theta.reshape(-1))
            step = step_counter["step"]
            step_counter["step"] += 1
            history.append({"step": step, "fun": value})
            _call_callback(callback, step, value, theta)

        result = scipy_minimize_impl(
            flat_objective,
            init.reshape(-1),
            method=self.method,
            jac=jac,
            bounds=self.bounds,
            constraints=self.constraints,
            callback=scipy_callback if callback is not None else None,
            options=self.options,
        )
        x = np.asarray(result.x, dtype=float).reshape(shape)
        return OptimizationResult(
            x=x,
            fun=float(result.fun),
            nit=int(getattr(result, "nit", step_counter["step"])),
            nfev=objective.nfev,
            success=bool(result.success),
            message=str(result.message),
            best_x=x.copy(),
            best_fun=float(result.fun),
            history=history,
            raw_result=result,
        )


class COBYLA(ScipyMinimize):
    """Derivative-free COBYLA optimizer."""

    def __init__(self, *, options: Mapping[str, Any] | None = None, constraints: Any = ()) -> None:
        super().__init__("COBYLA", constraints=constraints, options=options)


class LBFGSB(ScipyMinimize):
    """L-BFGS-B optimizer with optional explicit gradient."""

    def __init__(
        self,
        *,
        bounds: Any = None,
        options: Mapping[str, Any] | None = None,
        gradient_method: str | GradientFn | None = None,
        gradient_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            "L-BFGS-B",
            bounds=bounds,
            options=options,
            gradient_method=gradient_method,
            gradient_kwargs=gradient_kwargs,
        )


def scipy_minimize(
    fn: ScalarFn,
    init_params: Any,
    *,
    method: str = "COBYLA",
    **kwargs: Any,
) -> OptimizationResult:
    """Convenience wrapper for :class:`ScipyMinimize`."""

    optimizer_kwargs = {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key in {"bounds", "constraints", "options", "gradient_method", "gradient_kwargs"}
    }
    optimizer = ScipyMinimize(method=method, **optimizer_kwargs)
    return optimizer.minimize(fn, init_params, **kwargs)


def minimize(
    fn: ScalarFn,
    init_params: Any,
    *,
    optimizer: Any | None = None,
    method: str = "COBYLA",
    **kwargs: Any,
) -> OptimizationResult:
    """Minimize ``fn`` with an optimizer object or SciPy method name."""

    if optimizer is None:
        return scipy_minimize(fn, init_params, method=method, **kwargs)
    if not hasattr(optimizer, "minimize"):
        raise TypeError("optimizer must expose a minimize(fn, init_params, ...) method")
    return optimizer.minimize(fn, init_params, **kwargs)


__all__ = [
    "Adam",
    "COBYLA",
    "GD",
    "LBFGSB",
    "OptimizationResult",
    "SPSA",
    "ScipyMinimize",
    "minimize",
    "scipy_minimize",
]
