"""QAS config 工厂对未知字段的报错内容测试。"""

import pytest

torch = pytest.importorskip("torch")

from aicir.qas import config as qas_config


def test_unknown_config_field_error_names_the_bad_field():
    with pytest.raises(TypeError) as excinfo:
        qas_config.supernet(not_a_real_field=1)

    message = str(excinfo.value)
    assert "not_a_real_field" in message
