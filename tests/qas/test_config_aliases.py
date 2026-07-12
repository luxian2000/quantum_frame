"""aicir.qas.core.config：字段别名（旧字段名 -> 规范字段名）与未知字段报错。"""

import warnings

import pytest

torch = pytest.importorskip("torch")

from aicir.qas import config as qas_config


def test_pprdql_episode_num_alias_lands_on_max_episodes_with_warning():
    with pytest.warns(DeprecationWarning, match="episode_num"):
        cfg = qas_config.pprdql(episode_num=7)

    assert cfg.max_episodes == 7
    assert not hasattr(cfg, "episode_num")


def test_pporb_episode_num_alias_lands_on_max_episodes_with_warning():
    with pytest.warns(DeprecationWarning, match="episode_num"):
        cfg = qas_config.pporb(episode_num=9)

    assert cfg.max_episodes == 9


def test_crlqas_q_hidden_dim_alias_lands_on_hidden_dim_with_warning():
    with pytest.warns(DeprecationWarning, match="q_hidden_dim"):
        cfg = qas_config.crlqas(q_hidden_dim=64)

    assert cfg.hidden_dim == 64
    assert not hasattr(cfg, "q_hidden_dim")


def test_canonical_kwarg_emits_no_deprecation_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        cfg = qas_config.pprdql(max_episodes=11)

    assert cfg.max_episodes == 11


def test_alias_and_canonical_both_given_raises_type_error():
    with pytest.raises(TypeError):
        qas_config.pprdql(episode_num=1, max_episodes=2)


def test_unknown_field_still_raises_type_error_naming_the_field():
    with pytest.raises(TypeError) as excinfo:
        qas_config.pprdql(not_a_real_field=1)

    assert "not_a_real_field" in str(excinfo.value)


def test_unknown_field_after_alias_application_still_raises():
    with pytest.raises(TypeError) as excinfo:
        qas_config.crlqas(hidden_dim=32, still_not_real=1)

    assert "still_not_real" in str(excinfo.value)
