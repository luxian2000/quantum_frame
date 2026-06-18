import pytest

pytest.importorskip("torch")

from aicir.qas.algorithms.supernet import SupernetConfig, Supernet, supernet_qas
from aicir.operators import Hamiltonian


def test_shard_mode_defaults_to_safe():
    assert SupernetConfig().shard_mode == "safe"


def test_shard_mode_rejects_unknown_value():
    with pytest.raises(ValueError):
        Supernet(SupernetConfig(shard_mode="turbo"))


def test_supernet_qas_forwards_mode():
    ham = Hamiltonian(n_qubits=2, terms=[("ZZ", -1.0), ("XI", 0.2)])
    # Run a tiny search; just assert it accepts mode and returns a result.
    result = supernet_qas(
        ham, layers=1, supernet_num=1, supernet_steps=1,
        finetune_steps=1, ranking_num=1, seed=1, mode="aggressive",
    )
    assert result.best_circuit is not None
