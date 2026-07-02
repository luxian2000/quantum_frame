import pytest

from aicir.chemistry import pipeline


def test_require_qiskit_nature_raises_helpful_error_when_missing():
    if pipeline._qiskit_nature_available():
        pytest.skip("qiskit-nature 已安装，跳过缺失分支")
    with pytest.raises(ImportError, match=r"\[chem\]"):
        pipeline._require_qiskit_nature()


def test_build_molecule_is_exported():
    from aicir.chemistry import build_molecule  # noqa: F401
