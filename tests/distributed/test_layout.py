import pytest

from aicir import Circuit, cx, hadamard, pauli_x
from aicir.distributed.layout import _Layout, _ShardSpec


def test_shard_spec_vector_and_matrix_shapes():
    layout = _Layout.explicit(
        (0, 1, 2, 3),
        n_qubits=4,
        distributed_axes=2,
    )

    vector = _ShardSpec.build(
        n_qubits=4,
        world_size=4,
        rank=1,
        kind="vector",
        layout=layout,
    )
    matrix = _ShardSpec.build(
        n_qubits=4,
        world_size=4,
        rank=1,
        kind="matrix",
        layout=layout,
    )

    assert vector.local_shape == (4, 1)
    assert vector.global_shape == (16, 1)
    assert matrix.local_shape == (4, 16)
    assert matrix.global_shape == (16, 16)
    assert vector.global_start == 4
    assert vector.global_stop == 8


@pytest.mark.parametrize(
    ("world_size", "message"),
    [(3, "2 的幂"), (8, "n_qubits")],
)
def test_shard_spec_rejects_invalid_distribution(world_size, message):
    layout = _Layout.explicit(
        (0, 1),
        n_qubits=2,
        distributed_axes=1,
    )

    with pytest.raises(ValueError, match=message):
        _ShardSpec.build(
            n_qubits=2,
            world_size=world_size,
            rank=0,
            kind="vector",
            layout=layout,
        )


def test_explicit_layout_requires_complete_bijection():
    with pytest.raises(ValueError, match="完整双射"):
        _Layout.explicit(
            (0, 0, 2),
            n_qubits=3,
            distributed_axes=1,
        )


def test_automatic_layout_keeps_frequently_used_qubits_local():
    circuit = Circuit(
        hadamard(0),
        pauli_x(0),
        cx(target_qubit=1, control_qubits=(0,)),
        hadamard(0),
        n_qubits=3,
    )

    layout = _Layout.auto(circuit, n_qubits=3, distributed_axes=1)

    assert layout.distributed_logical_qubits == (2,)
    assert layout.logical_to_storage[2] == 0
    assert layout.digest() == _Layout.auto(
        circuit,
        n_qubits=3,
        distributed_axes=1,
    ).digest()


def test_layout_inverse_round_trip():
    layout = _Layout.explicit(
        (2, 0, 3, 1),
        n_qubits=4,
        distributed_axes=2,
    )

    assert layout.storage_to_logical == (1, 3, 0, 2)
    for logical, storage in enumerate(layout.logical_to_storage):
        assert layout.storage_to_logical[storage] == logical
