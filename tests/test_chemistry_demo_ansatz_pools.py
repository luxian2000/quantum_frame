from demos.BeH2.BeH2 import beh2_vqe_qas_kwargs
from demos.CH4.CH4 import ch4_vqe_qas_kwargs
from demos.H2O.H2O_dqas import _validate_requested_device, h2o_dqas_config
from demos.H2O.H2O import h2o_vqe_qas_config
from demos.LiH.LiH import lih_vqe_qas_config


def _assert_chemistry_pool(config) -> None:
    values = config if isinstance(config, dict) else config.__dict__

    assert values["single_qubit_gates"] == ("i",)
    assert values["two_qubit_gates"] == ("single_excitation",)
    assert values["four_qubit_gates"] == ("double_excitation",)
    assert values["hf_occupied_qubits"]
    assert values["two_qubit_pairs"]
    assert values["four_qubit_groups"]


def test_small_molecule_demos_use_hf_excitation_supernet_pools():
    _assert_chemistry_pool(lih_vqe_qas_config())
    _assert_chemistry_pool(h2o_vqe_qas_config())


def test_large_molecule_demos_use_hf_excitation_supernet_pools():
    _assert_chemistry_pool(beh2_vqe_qas_kwargs())
    _assert_chemistry_pool(ch4_vqe_qas_kwargs())


def test_h2o_dqas_demo_uses_excitation_pool():
    cfg = h2o_dqas_config(search_epochs=0, finetune_steps=0)

    assert cfg.gate_pool == "excitation"
    assert cfg.single_excitations
    assert cfg.double_excitations
    assert cfg.hf_occupied_qubits


def test_h2o_dqas_rejects_out_of_range_npu_device():
    message = _validate_requested_device("npu:2", npu_count=2)

    assert message is not None
    assert "npu:2" in message
    assert "npu:0..npu:1" in message


def test_h2o_dqas_accepts_cpu_and_valid_npu_device():
    assert _validate_requested_device("cpu", npu_count=0) is None
    assert _validate_requested_device("npu:1", npu_count=2) is None
    assert _validate_requested_device("npu", npu_count=1) is None
