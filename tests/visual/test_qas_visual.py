import pytest

from aicir.backends.numpy_backend import NumpyBackend
from aicir.qas import ArchitectureSearch, SearchConfig, build_common_architectures, evaluate_architectures
from aicir.visual import (
    compare_architectures,
    plot_architecture_metrics,
    plot_qas_summary,
    plot_search_history,
    qas_scores_to_rows,
)


@pytest.fixture(scope="module")
def qas_scores():
    backend = NumpyBackend()
    architectures = build_common_architectures(
        n_qubits=2,
        layers=1,
        backend=backend,
        names=["hea_linear", "qaoa_chain"],
    )
    return evaluate_architectures(architectures, backend=backend, n_samples=4)


@pytest.fixture()
def plt():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    yield pyplot
    pyplot.close("all")


def test_qas_scores_to_rows_accepts_architecture_scores(qas_scores):
    rows = qas_scores_to_rows(qas_scores)

    assert len(rows) == 2
    assert rows[0]["rank"] == 1
    assert "weighted_score" in rows[0]
    assert "expressibility_metric" in rows[0]


def test_qas_scores_to_rows_accepts_search_result():
    backend = NumpyBackend()
    search = ArchitectureSearch(backend=backend)
    result = search.run(
        SearchConfig(
            n_qubits=2,
            candidate_layers=1,
            n_samples=4,
            include_common_candidates=True,
        ),
        extra_candidates=[],
    )

    rows = qas_scores_to_rows(result)

    assert len(rows) == len(result.scores)
    assert rows[0]["rank"] == 1


def test_qas_scores_to_rows_accepts_architecture_specs():
    backend = NumpyBackend()
    architectures = build_common_architectures(
        n_qubits=2,
        layers=1,
        backend=backend,
        names=["hea_linear"],
    )

    [row] = qas_scores_to_rows(architectures)

    assert row["name"] == "hea_linear_cx_L1"
    assert row["n_gates"] > 0
    assert "hardware_efficiency" in row


def test_plot_search_history(qas_scores, plt):
    fig, ax = plot_search_history(qas_scores, metrics=["weighted_score", "trainability"])

    assert fig is ax.figure
    assert len(ax.lines) == 2
    assert ax.get_xlabel() == "rank"


def test_plot_search_history_accepts_dict_records(plt):
    records = [
        {"episode": 0, "reward": 0.1, "weighted_score": 0.2},
        {"episode": 1, "reward": 0.3, "weighted_score": 0.4},
    ]

    fig, ax = plot_search_history(records, x="episode")

    assert fig is ax.figure
    assert len(ax.lines) == 2
    assert ax.get_xlabel() == "episode"


def test_plot_architecture_metrics(qas_scores, plt):
    fig, ax = plot_architecture_metrics(qas_scores[0], metrics=["weighted_score", "hardware_efficiency"])

    assert fig is ax.figure
    assert len(ax.patches) == 2


def test_compare_architectures(qas_scores, plt):
    fig, ax = compare_architectures(qas_scores, metrics=["weighted_score", "n_gates"])

    assert fig is ax.figure
    assert len(ax.patches) == 4


def test_plot_qas_summary(qas_scores, plt):
    fig, axes = plot_qas_summary(qas_scores[0], metrics=["weighted_score", "trainability"])

    assert fig is axes[0].figure
    assert len(axes) == 2
    assert len(axes[1].patches) == 2
