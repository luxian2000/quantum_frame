"""Gradient-preserving result boundaries for paired-real statevectors."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aicir import Circuit, PauliString, hadamard
from aicir.distributed import (
    DistNPUBackend,
    DistResult,
    DistSimulator,
    DistState,
    PureStateParam,
)
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.autograd._reducers import _PairReducer
from aicir.distributed.gates import _GatePlanner, _VectorKernel
from aicir.distributed.layout import _Layout, _ShardSpec
from aicir.distributed.reducers import _Reducer
from scripts.npu import distributed_autograd_probe as probe


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
    assert isinstance(gathered.data, np.ndarray)
    materialized_state = result.state.to_numpy()
    materialized_probabilities = result.gather_probabilities()
    assert isinstance(materialized_state, np.ndarray)
    assert isinstance(materialized_probabilities, np.ndarray)
    np.testing.assert_allclose(materialized_probabilities, probabilities.detach().numpy())


def test_probe_initial_state_sections_cover_private_trainable_contracts(monkeypatch):
    backend = _backend(monkeypatch)

    statevector = probe._statevector_section(backend)
    contract = probe._contract_section(backend)

    assert statevector["root_owned_initial_state_gradient_finite"]
    assert statevector["sharded_initial_state_gradient_finite"]
    assert statevector["root_owned_layout_value_error"] <= 1e-6
    assert statevector["root_owned_layout_gradient_error"] <= 1e-6
    assert statevector["sharded_layout_value_error"] <= 1e-6
    assert statevector["sharded_layout_gradient_error"] <= 1e-6
    assert contract["direct_complex_leaf_rejected"]
    assert contract["rank_requires_grad_mismatch_rejected"] is None
    assert contract["public_routing_enabled"]


def test_legacy_vector_kernel_rejects_pair_before_complex_boundary(monkeypatch):
    backend = _backend(monkeypatch)
    layout = _Layout.explicit((0,), n_qubits=1, distributed_axes=0)
    spec = _ShardSpec.build(1, 1, 0, "vector", layout)
    state = DistState.from_pair(
        _Pair(torch.ones((2, 1)), torch.zeros((2, 1))),
        spec=spec,
        backend=backend,
    )
    plan = _GatePlanner(backend, layout, 1).plan(hadamard(0), 0)

    def fail_complex(*_args, **_kwargs):
        raise AssertionError("legacy torch.complex boundary reached")

    monkeypatch.setattr(torch, "complex", fail_complex)
    with pytest.raises(ValueError, match="paired-real DistState"):
        _VectorKernel(backend).apply(state, plan)


def test_paired_local_data_is_a_detached_cpu_diagnostic(monkeypatch):
    backend = _backend(monkeypatch)
    layout = _Layout.explicit((0,), n_qubits=1, distributed_axes=0)
    spec = _ShardSpec.build(1, 1, 0, "vector", layout)
    real = torch.ones((2, 1), requires_grad=True)
    state = DistState.from_pair(
        _Pair(real, torch.zeros((2, 1), requires_grad=True)),
        spec=spec,
        backend=backend,
    )

    diagnostic = state.local_data

    assert diagnostic.device.type == "cpu"
    assert diagnostic.is_complex()
    assert not diagnostic.requires_grad


def test_legacy_reducer_rejects_pair_before_complex_boundary(monkeypatch):
    backend = _backend(monkeypatch)
    layout = _Layout.explicit((0,), n_qubits=1, distributed_axes=0)
    spec = _ShardSpec.build(1, 1, 0, "vector", layout)
    state = DistState.from_pair(
        _Pair(torch.ones((2, 1)), torch.zeros((2, 1))),
        spec=spec,
        backend=backend,
    )

    def fail_complex(*_args, **_kwargs):
        raise AssertionError("legacy torch.complex boundary reached")

    monkeypatch.setattr(torch, "complex", fail_complex)
    with pytest.raises(ValueError, match="paired-real DistState"):
        _Reducer(backend).expectation(state, PauliString("Z", n_qubits=1))


@pytest.mark.parametrize("initial_kind", ("pair", "pure"))
def test_public_run_routes_paired_initial_inputs_without_complex_boundary(
    monkeypatch,
    initial_kind,
):
    backend = _backend(monkeypatch)
    layout = _Layout.explicit((0,), n_qubits=1, distributed_axes=0)
    spec = _ShardSpec.build(1, 1, 0, "vector", layout)
    initial_state = (
        DistState.from_pair(
            _Pair(torch.ones((2, 1)), torch.zeros((2, 1))),
            spec=spec,
            backend=backend,
        )
        if initial_kind == "pair"
        else PureStateParam(torch.ones(2), torch.zeros(2))
    )

    def fail_complex(*_args, **_kwargs):
        raise AssertionError("public complex boundary reached")

    monkeypatch.setattr(torch, "complex", fail_complex)
    result = DistSimulator(backend).run(Circuit(n_qubits=1), initial_state=initial_state)
    assert result.state._pair is not None
