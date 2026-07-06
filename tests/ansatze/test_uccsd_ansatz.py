import numpy as np
import pytest

from aicir import Circuit
from aicir.ansatze import uccsd, uccsd_parameter_count

_EXC = (("single", (0, 2)), ("single", (1, 3)), ("double", (0, 1, 2, 3)))


def test_parameter_count():
    assert uccsd_parameter_count(_EXC) == 3
    assert uccsd_parameter_count(_EXC, reps=2) == 6


def test_returns_circuit_with_symbolic_parameters():
    circ = uccsd(4, (1, 1, 0, 0), _EXC)
    assert isinstance(circ, Circuit)
    assert circ.n_qubits == 4
    assert len(circ.parameters) == 3


def test_hf_reference_applies_x_on_occupied_qubits():
    # 无激发 → 电路只含 HF 的 pauli_x，落在占据位
    circ = uccsd(4, (1, 1, 0, 0), ())
    x_targets = sorted(
        g.get("target_qubit") for g in circ.gates if g.get("type") == "pauli_x"
    )
    assert x_targets == [0, 1]


def test_bound_parameter_count_matches():
    circ = uccsd(4, (1, 1, 0, 0), _EXC, parameters=np.zeros(3))
    assert len(circ.parameters) == 0  # 全绑定为数值


def test_rejects_none_metadata():
    with pytest.raises(ValueError, match="hf_occupation"):
        uccsd(4, None, _EXC)
    with pytest.raises(ValueError, match="excitations"):
        uccsd(4, (1, 1, 0, 0), None)


def test_rejects_bad_occupation_length():
    with pytest.raises(ValueError, match="hf_occupation"):
        uccsd(4, (1, 1, 0), _EXC)


def test_rejects_out_of_range_excitation():
    with pytest.raises(ValueError, match="越界|out of range|索引"):
        uccsd(4, (1, 1, 0, 0), (("single", (0, 9)),))
