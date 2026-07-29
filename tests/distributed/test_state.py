import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from aicir.distributed import DistNPUBackend, DistState
from aicir.distributed.layout import _Layout, _ShardSpec


class _RejectComplexIndex(TorchDispatchMode):
    """Model Ascend's aclnnIndex dtype restriction on a CPU test run."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.index.Tensor and args[0].is_complex():
            raise RuntimeError("complex indexing is not supported")
        return func(*args, **(kwargs or {}))


def _backend(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    return DistNPUBackend.from_env(
        fallback_to_cpu=True,
        init_process_group=False,
    )


def _spec(kind="vector"):
    layout = _Layout.explicit(
        (0, 1, 2),
        n_qubits=3,
        distributed_axes=0,
    )
    return _ShardSpec.build(3, 1, 0, kind, layout)


def test_dist_state_rejects_wrong_local_shape(monkeypatch):
    backend = _backend(monkeypatch)

    with pytest.raises(ValueError, match="local_shape"):
        DistState.from_local(
            torch.zeros(4, 1, dtype=torch.complex64),
            spec=_spec(),
            backend=backend,
        )


def test_dist_state_rejects_automatic_differentiation(monkeypatch):
    backend = _backend(monkeypatch)
    local = torch.zeros(
        8,
        1,
        dtype=torch.complex64,
        requires_grad=True,
    )

    with pytest.raises(ValueError, match="requires_grad"):
        DistState.from_local(local, spec=_spec(), backend=backend)


def test_zero_state_and_properties(monkeypatch):
    backend = _backend(monkeypatch)
    layout = _spec().layout

    state = DistState.zero(3, backend=backend, layout=layout)

    assert state.local_shape == (8, 1)
    assert state.global_shape == (8, 1)
    assert state.n_qubits == 3
    assert state.kind == "vector"
    assert not state.is_density
    assert state.rank == 0
    assert state.world_size == 1
    assert not hasattr(state, "data")
    torch.testing.assert_close(
        state.local_probabilities(),
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )


def test_density_local_probabilities_use_global_diagonal(monkeypatch):
    backend = _backend(monkeypatch)
    rho = torch.diag(
        torch.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0],
            dtype=torch.complex64,
        )
    )
    state = DistState.from_local(
        rho,
        spec=_spec("matrix"),
        backend=backend,
    )

    np.testing.assert_allclose(
        state.local_probabilities().numpy(),
        [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0],
        atol=1e-6,
    )


def test_density_local_probabilities_do_not_index_complex_tensor(monkeypatch):
    backend = _backend(monkeypatch)
    rho = torch.diag(
        torch.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0],
            dtype=torch.complex64,
        )
    )
    state = DistState.from_local(
        rho,
        spec=_spec("matrix"),
        backend=backend,
    )

    with _RejectComplexIndex():
        probabilities = state.local_probabilities()

    torch.testing.assert_close(
        probabilities,
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0]),
    )


def test_single_rank_gather_returns_existing_state(monkeypatch):
    backend = _backend(monkeypatch)
    local = torch.arange(8, dtype=torch.float32).to(torch.complex64).reshape(8, 1)
    state = DistState.from_local(local, spec=_spec(), backend=backend)

    gathered = state.gather(root=0)

    np.testing.assert_array_equal(gathered.to_numpy(), np.arange(8))
    np.testing.assert_array_equal(state.to_numpy(root=0), np.arange(8))
