import inspect

from aicir.optimization.qubo import Binary, Model, run_model_qaoa, run_qubo_qaoa
from aicir.optimizer import GD
from aicir.vqc.QAOA import BasicQAOA


def test_run_model_qaoa_keywords_are_superset_of_basic_qaoa_run() -> None:
    run_params = set(inspect.signature(BasicQAOA.run).parameters) - {"self"}
    run_model_qaoa_params = set(inspect.signature(run_model_qaoa).parameters)

    assert run_params <= run_model_qaoa_params


def test_run_model_qaoa_forwards_optimizer_to_basic_qaoa_run() -> None:
    x = Binary("qaoa_kwargs_x")
    y = Binary("qaoa_kwargs_y")
    model = Model(2.0 * x + 4.0 * x * y)

    calls: list[int] = []

    def counting_callback(step, value, theta):
        calls.append(step)

    optimizer = GD(max_iters=3)
    result = run_model_qaoa(
        model,
        p=1,
        optimizer=optimizer,
        callback=counting_callback,
        seed=0,
    )

    assert result.optimizer_result is not None
    assert len(calls) > 0


def test_run_qubo_qaoa_alias_still_works() -> None:
    x = Binary("qaoa_kwargs_alias_x")
    model = Model(2.0 * x)

    result = run_qubo_qaoa(model, p=1, max_iters=2, seed=0)

    assert result.energy is not None
