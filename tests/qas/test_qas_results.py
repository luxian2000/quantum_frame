"""QASResult：满足 aicir.protocols.AlgorithmResult 协议，默认值合理。"""

from aicir.protocols import AlgorithmResult
from aicir.qas import QASResult
from aicir.qas.core.results import QASResult as QASResultFromCore


def test_qas_result_is_exported_from_top_level_qas_package():
    assert QASResult is QASResultFromCore


def test_qas_result_satisfies_algorithm_result_protocol():
    result = QASResult(method="crlqas", value=1.5)

    assert isinstance(result, AlgorithmResult)
    assert result.method == "crlqas"
    assert result.value == 1.5


def test_qas_result_defaults_are_sane():
    result = QASResult(method="dqas", value=None)

    assert result.circuit is None
    assert result.parameters is None
    assert result.history == []
    assert result.metadata == {}
    assert result.raw is None


def test_qas_result_default_containers_are_independent_across_instances():
    a = QASResult(method="a", value=0.0)
    b = QASResult(method="b", value=0.0)

    a.history.append({"step": 1})
    a.metadata["k"] = "v"

    assert b.history == []
    assert b.metadata == {}


def test_qas_result_carries_raw_method_specific_object():
    class _Dummy:
        pass

    raw = _Dummy()
    result = QASResult(method="supernet", value=-1.0, raw=raw)

    assert result.raw is raw
