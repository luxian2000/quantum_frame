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


def test_matrix_form_methods_dispatch():
    rho = np.array([[0.5, 0], [0, 0.5]], dtype=np.complex64)
    s = State.from_matrix(rho)
    assert s.purity() == pytest.approx(0.5)
    assert s.is_pure() is False
    assert s.von_neumann_entropy() == pytest.approx(np.log(2))
    probs = np.asarray(s.probabilities())
    np.testing.assert_allclose(probs, [0.5, 0.5], atol=1e-6)


def test_vector_form_purity_is_one_and_partial_trace_returns_density():
    s = State.from_array([1, 0, 0, 1], n_qubits=2)  # 贝尔态
    assert s.purity() == pytest.approx(1.0)
    red = s.partial_trace(keep=[0])
    assert red.is_density is True
    assert red.n_qubits == 1
    assert red.purity() == pytest.approx(0.5, abs=1e-6)


def test_to_density_matrix_returns_matrix_form_state():
    s = State.from_array([1, 0], n_qubits=1)
    rho = s.to_density_matrix()
    assert isinstance(rho, State)
    assert rho.is_density is True


def test_pure_state_three_representations():
    s = State.from_array([1, 0, 0, 1], n_qubits=2)
    np.testing.assert_allclose(
        s.array, np.array([1, 0, 0, 1]) / np.sqrt(2), atol=1e-6
    )
    assert s.matrix.shape == (4, 4)
    assert s.ket == "1/\\sqrt{2}|00>+1/\\sqrt{2}|11>"


def test_mixed_state_array_is_none_and_ket_is_operator_form():
    rho = np.array([[0.5, 0], [0, 0.5]], dtype=np.complex64)
    s = State.from_matrix(rho)
    assert s.array is None
    assert s.matrix.shape == (2, 2)
    assert s.ket == "0.5|0><0|+0.5|1><1|"


def test_matrix_form_pure_state_array_extracted_via_eigenvector():
    s = State.from_array([0, 1], n_qubits=1).to_density_matrix()  # |1><1|
    assert s.is_density is True
    np.testing.assert_allclose(np.abs(s.array), [0, 1], atol=1e-6)
    assert s.ket == "1|1>"


def test_representations_are_printable():
    s = State.from_array([1, 0], n_qubits=1)
    assert isinstance(str(s.ket), str)
    assert "1" in np.array2string(s.array)
    assert s.matrix.shape == (2, 2)


def test_matrix_form_state_is_printable_via_str():
    # 回归：matrix 形态 State 直接 print 不应崩溃
    rho = np.array([[0.5, 0], [0, 0.5]], dtype=np.complex64)
    s = State.from_matrix(rho)
    text = str(s)  # 不应抛异常
    assert "><" in text
