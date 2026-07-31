"""Gradient-preserving result boundaries for paired-real statevectors."""

from __future__ import annotations

import numpy as np
import torch

from aicir import PauliString
from aicir.distributed import DistNPUBackend, DistResult, DistState
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.autograd._reducers import _PairReducer
from aicir.distributed.layout import _Layout, _ShardSpec


def _backend(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    return DistNPUBackend.from_env(
        fallback_to_cpu=True,
        init_process_group=False,
    )


def test_paired_result_scalars_keep_graph_and_materializers_detach(monkeypatch):
    backend = _backend(monkeypatch)
    layout = _Layout.explicit((0,), n_qubits=1, distributed_axes=0)
    spec = _ShardSpec.build(1, 1, 0, "vector", layout)
    real = torch.tensor([[1.0], [0.5]], dtype=torch.float32, requires_grad=True)
    imag = torch.tensor([[0.25], [0.0]], dtype=torch.float32, requires_grad=True)
    state = DistState.from_pair(_Pair(real, imag), spec=spec, backend=backend)
    probabilities = state.local_probabilities()
    energy = _PairReducer(backend).expectation(
        state._pair,
        spec,
        PauliString("Z", n_qubits=1),
    )
    result = DistResult(
        state=state,
        local_probabilities=probabilities,
        expectations={"energy": energy},
        counts=None,
        rank=0,
        world_size=1,
    )

    assert result.expectations["energy"].requires_grad
    assert result.local_probabilities.requires_grad
    assert isinstance(float(result.expectations["energy"]), float)
    (result.expectations["energy"] + result.local_probabilities[0]).backward()
    assert real.grad is not None
    assert imag.grad is not None

    gathered = result.state.gather()
    assert not gathered.data.requires_grad
    materialized_state = result.state.to_numpy()
    materialized_probabilities = result.gather_probabilities()
    assert isinstance(materialized_state, np.ndarray)
    assert isinstance(materialized_probabilities, np.ndarray)
    np.testing.assert_allclose(materialized_probabilities, probabilities.detach().numpy())
