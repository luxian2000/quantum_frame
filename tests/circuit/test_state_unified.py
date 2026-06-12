import numpy as np
import pytest

from aicir.core import State


def test_from_array_infers_n_qubits_and_defaults_numpy_backend():
    s = State.from_array([1, 0, 0, 1])  # 无 n_qubits / 无 backend
    assert s.n_qubits == 2
    assert s.is_density is False
    assert s.backend is not None


def test_from_matrix_builds_density_form():
    rho = np.array([[0.5, 0], [0, 0.5]], dtype=np.complex64)
    s = State.from_matrix(rho)  # 推断 1 比特
    assert s.n_qubits == 1
    assert s.is_density is True


def test_from_array_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        State.from_array([1, 0, 0])


def test_from_array_rejects_empty():
    with pytest.raises(ValueError):
        State.from_array([])
