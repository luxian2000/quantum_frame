import numpy as np

from aicir import NumpyBackend
from aicir.optimization.qubo import (
    Binary,
    Model,
    QuboBuilder,
    builder_to_hamiltonian,
    ising_to_hamiltonian,
    model_to_hamiltonian,
    qubo_to_ising_indices,
)


def _diagonal(matrix) -> list[float]:
    return [float(np.real(value)) for value in np.diag(matrix)]


def test_ising_to_hamiltonian_preserves_terms_and_offset() -> None:
    ising = qubo_to_ising_indices(
        {(0, 0): 2.0, (0, 1): 4.0},
        offset=0.0,
        variable_names=["x", "y"],
    )

    hamiltonian = ising_to_hamiltonian(ising)

    assert hamiltonian.n_qubits == 2
    assert [(term.qubit_labels, term.coefficient) for term in hamiltonian.terms] == [
        (["Z", "I"], 2.0 + 0.0j),
        (["I", "Z"], 1.0 + 0.0j),
        (["Z", "Z"], 1.0 + 0.0j),
        (["I", "I"], 2.0 + 0.0j),
    ]


def test_model_to_hamiltonian_matrix_matches_qubo_energy() -> None:
    x = Binary("adapter_x")
    y = Binary("adapter_y")
    model = Model(2.0 * x + 4.0 * x * y)

    hamiltonian = model_to_hamiltonian(model)
    matrix = hamiltonian.to_matrix(NumpyBackend())

    # The Ising convention uses x=(1+Z)/2, so computational |0> maps to
    # binary value 1 and |1> maps to binary value 0.
    assert _diagonal(matrix) == [6.0, 2.0, 0.0, 0.0]


def test_builder_to_hamiltonian_can_drop_global_offset() -> None:
    builder = QuboBuilder()
    x = builder.registry.get_or_create("x")
    builder.add_linear(x, 2.0)

    with_offset = builder_to_hamiltonian(builder)
    without_offset = builder_to_hamiltonian(builder, include_offset=False)

    assert len(with_offset.terms) == 2
    assert len(without_offset.terms) == 1
    assert without_offset.terms[0].qubit_labels == ["Z"]
