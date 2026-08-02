"""Checkpoint policy and paired-real replay contracts on one CPU rank."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from aicir import Circuit, PauliString
from aicir.core.circuit import ry
from aicir.distributed import DistNPUBackend, DistSimulator
from aicir.distributed.autograd._checkpoint import (
    _CheckpointPlanner,
    _CheckpointPolicy,
    _available_memory_bytes,
    _recompute_segment,
)
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.autograd._reducers import _PairReducer
from aicir.distributed.gates import _AutogradExecutionContext, _GatePlanner
from aicir.distributed.layout import _Layout, _ShardSpec
from aicir.distributed.state import DistState
from aicir.noise import BitFlipChannel, NoiseModel


def _backend(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    return DistNPUBackend.from_env(fallback_to_cpu=True, init_process_group=False)


@pytest.mark.parametrize(
    ("value", "expected"),
    (("none", "none"), ("auto", "auto"), (1, 1), (4, 4), (16, 16)),
)
def test_checkpoint_policy_parses_supported_values(value, expected):
    assert _CheckpointPolicy.parse(value).value == expected


@pytest.mark.parametrize("value", (0, -1, True, False, "every", None))
def test_checkpoint_policy_rejects_invalid_values_exactly(value):
    with pytest.raises(ValueError, match="^grad_checkpoint 必须是 'none'、'auto' 或正整数$"):
        _CheckpointPolicy.parse(value)


def test_checkpoint_planner_uses_local_paired_bytes_and_80_percent_budget(monkeypatch):
    backend = _backend(monkeypatch)
    layout = _Layout.explicit((0, 1), n_qubits=2, distributed_axes=0)
    spec = _ShardSpec.build(2, 1, 0, "matrix", layout)
    planner = _CheckpointPlanner(spec, circuit_depth=16, available_bytes=1350)

    # One paired float32 density state takes 2 * 4 * 4 * 4 = 128 bytes.
    # The planner includes two paired forward temporaries and a 20% margin;
    # the smallest interval that keeps the saved boundary budget <= 1080 is 4.
    assert planner.interval() == 4
    assert planner.state_bytes == 128
    assert planner.available_bytes == 1350
    assert planner.budget_bytes == 1080


def test_recompute_segment_replays_original_plan_indices_without_mutation(monkeypatch):
    backend = _backend(monkeypatch)
    layout = _Layout.explicit((0,), n_qubits=1, distributed_axes=0)
    spec = _ShardSpec.build(1, 1, 0, "vector", layout)
    theta = torch.tensor(0.2, dtype=torch.float32, requires_grad=True)
    plans = tuple(
        _GatePlanner(backend, layout, 1, execution_context=_AutogradExecutionContext()).plan(
            ry(theta, 0), index
        )
        for index in range(3)
    )
    state = DistState.from_pair(
        _Pair(torch.tensor([[1.0], [0.0]], dtype=torch.float32), torch.zeros((2, 1))),
        spec=spec,
        backend=backend,
    )

    class RecordingEngine:
        def __init__(self):
            self.indices = []

        def apply(self, current, plan, *, operation_index):
            self.indices.append((operation_index, plan.instruction_index))
            return current

    engine = RecordingEngine()
    result = _recompute_segment(state, plans, 1, 3, engine)

    assert result is state
    assert engine.indices == [(1, 1), (2, 2)]


@pytest.mark.parametrize("policy", ("none", "auto", 1, 4, 16))
def test_paired_real_checkpoint_modes_match_output_and_gradient(monkeypatch, policy):
    backend = _backend(monkeypatch)
    simulator = DistSimulator(backend)
    layout = _Layout.explicit((1, 0), n_qubits=2, distributed_axes=0)
    spec = _ShardSpec.build(2, 1, 0, "vector", layout)
    theta = torch.tensor(0.31, dtype=torch.float32, requires_grad=True)
    circuit = Circuit(n_qubits=2)
    for _ in range(5):
        circuit.append(ry(theta, 0))
    state = DistState.from_pair(
        _Pair(
            torch.tensor([[0.5], [0.5], [0.5], [0.5]], dtype=torch.float32),
            torch.tensor([[0.1], [-0.2], [0.3], [-0.4]], dtype=torch.float32),
        ),
        spec=spec,
        backend=backend,
    )

    evolved, checkpoint = simulator._run_paired_real(
        circuit,
        initial_state=state,
        layout=layout,
        grad_checkpoint=policy,
        available_memory_bytes=1 << 30,
    )
    value = _PairReducer(backend).expectation(
        evolved._pair, evolved.spec, PauliString("ZI", n_qubits=2)
    )
    value.backward()

    reference_theta = torch.tensor(0.31, dtype=torch.float32, requires_grad=True)
    reference = DistState.from_pair(state._pair, spec=spec, backend=backend)
    reference_circuit = Circuit(n_qubits=2)
    for _ in range(5):
        reference_circuit.append(ry(reference_theta, 0))
    expected, _ = simulator._run_paired_real(
        reference_circuit,
        initial_state=reference,
        layout=layout,
        grad_checkpoint="none",
    )
    expected_value = _PairReducer(backend).expectation(
        expected._pair, expected.spec, PauliString("ZI", n_qubits=2)
    )
    expected_value.backward()

    torch.testing.assert_close(value, expected_value)
    torch.testing.assert_close(theta.grad, reference_theta.grad, atol=1e-6, rtol=1e-6)
    assert checkpoint.interval == ({"none": 0, "auto": 1, 1: 1, 4: 4, 16: 16}[policy])
    assert checkpoint.saved_state_count >= 1


def test_public_run_validates_checkpoint_and_opens_autograd_route(monkeypatch):
    backend = _backend(monkeypatch)
    simulator = DistSimulator(backend)
    circuit = Circuit(n_qubits=1)
    circuit.append(ry(torch.tensor(0.2, dtype=torch.float32, requires_grad=True), 0))

    with pytest.raises(ValueError, match="^grad_checkpoint 必须是 'none'、'auto' 或正整数$"):
        simulator.run(circuit, grad_checkpoint="bad")
    result = simulator.run(circuit, grad_checkpoint="none")
    assert result.state._pair is not None


@pytest.mark.parametrize("policy", ("none", "auto", 1, 4, 16))
def test_public_forward_and_sampling_are_unchanged_for_valid_checkpoint_policies(monkeypatch, policy):
    backend = _backend(monkeypatch)
    simulator = DistSimulator(backend)
    circuit = Circuit(n_qubits=1)
    circuit.append(ry(0.31, 0))
    result = simulator.run(circuit, shots=20, seed=7, grad_checkpoint=policy)
    reference = simulator.run(circuit, shots=20, seed=7, grad_checkpoint="auto")
    torch.testing.assert_close(result.local_probabilities, reference.local_probabilities)
    assert result.counts == reference.counts


def test_npu_memory_discovery_never_uses_host_memory(monkeypatch):
    monkeypatch.setattr(torch, "npu", object(), raising=False)
    monkeypatch.setattr("aicir.distributed.autograd._checkpoint.os.sysconf", lambda *_: 123)
    assert _available_memory_bytes("npu:0") == (None, "conservative")


def test_measurement_boundary_synchronizes_then_isolates_each_policy_peak(monkeypatch):
    backend = _backend(monkeypatch)
    simulator = DistSimulator(backend)
    events = []
    peaks = iter((101, 202, 303))

    monkeypatch.setattr(
        "aicir.distributed.simulator._reset_peak_memory_stats",
        lambda _device: events.append("reset") or "cpu",
    )
    monkeypatch.setattr(
        "aicir.distributed.simulator._synchronize_device",
        lambda _device: events.append("sync"),
    )
    monkeypatch.setattr(
        "aicir.distributed.simulator._peak_allocation_bytes",
        lambda _device: events.append("peak") or next(peaks),
    )

    def run_workload(*_args, **kwargs):
        events.append(f"workload:{kwargs['grad_checkpoint']}")
        return kwargs["grad_checkpoint"], SimpleNamespace()

    monkeypatch.setattr(simulator, "_run_paired_real", run_workload)
    observed = []
    for policy in ("none", "auto", 16):
        state, metrics = simulator._measure_paired_real(
            object(),
            initial_state=object(),
            layout=object(),
            grad_checkpoint=policy,
        )
        assert state == policy
        observed.append(metrics.peak_allocation_bytes)

    assert observed == [101, 202, 303]
    assert events == [
        "sync", "reset", "workload:none", "sync", "peak",
        "sync", "reset", "workload:auto", "sync", "peak",
        "sync", "reset", "workload:16", "sync", "peak",
    ]


def test_measurement_boundary_reports_blocked_without_allocator_support(monkeypatch):
    backend = _backend(monkeypatch)
    simulator = DistSimulator(backend)
    events = []

    monkeypatch.setattr(
        "aicir.distributed.simulator._reset_peak_memory_stats",
        lambda _device: events.append("reset") or None,
    )
    monkeypatch.setattr(
        "aicir.distributed.simulator._synchronize_device",
        lambda _device: events.append("sync"),
    )
    monkeypatch.setattr(
        "aicir.distributed.simulator._peak_allocation_bytes",
        lambda _device: pytest.fail("blocked allocator must not read a peak"),
    )
    monkeypatch.setattr(
        simulator,
        "_run_paired_real",
        lambda *_args, **_kwargs: ("state", SimpleNamespace()),
    )

    state, metrics = simulator._measure_paired_real(object())

    assert state == "state"
    assert metrics.peak_allocation_status == "UNAVAILABLE"
    assert metrics.peak_allocation_bytes is None
    assert events == ["sync", "reset"]


def test_normal_paired_real_execution_never_resets_allocator_peak(monkeypatch):
    backend = _backend(monkeypatch)
    simulator = DistSimulator(backend)
    resets = []
    monkeypatch.setattr(
        "aicir.distributed.simulator._reset_peak_memory_stats",
        lambda _device: resets.append("reset"),
    )
    layout = _Layout.explicit((0,), n_qubits=1, distributed_axes=0)
    spec = _ShardSpec.build(1, 1, 0, "vector", layout)
    state = DistState.from_pair(
        _Pair(torch.tensor([[1.0], [0.0]]), torch.zeros((2, 1))),
        spec=spec,
        backend=backend,
    )

    simulator._run_paired_real(
        Circuit(n_qubits=1), initial_state=state, layout=layout, grad_checkpoint="none"
    )

    assert resets == []


@pytest.mark.parametrize("policy", ("none", "auto", 1, 4, 16))
def test_checkpoint_keeps_analytic_noise_value_and_probability_gradient(monkeypatch, policy):
    backend = _backend(monkeypatch)
    layout = _Layout.explicit((0,), n_qubits=1, distributed_axes=0)
    spec = _ShardSpec.build(1, 1, 0, "matrix", layout)
    probability = torch.tensor(0.23, dtype=torch.float32, requires_grad=True)
    circuit = Circuit(n_qubits=1)
    circuit.append(ry(torch.tensor(0.0, dtype=torch.float32), 0))
    circuit.noise_model = NoiseModel().add_channel(BitFlipChannel(0, probability))
    state = DistState.from_pair(_Pair(torch.tensor([[0.0, 0.0], [0.0, 1.0]]), torch.zeros((2, 2))), spec=spec, backend=backend)
    evolved, metrics = DistSimulator(backend)._run_paired_real(circuit, initial_state=state, layout=layout, grad_checkpoint=policy, available_memory_bytes=1 << 30)
    value = _PairReducer(backend).expectation(evolved._pair, evolved.spec, PauliString("Z", n_qubits=1))
    value.backward()
    assert float(value) == pytest.approx(-1.0 + 2.0 * 0.23, abs=1e-6)
    assert float(probability.grad) == pytest.approx(2.0, abs=1e-6)
    if policy == "none":
        assert metrics.recomputed_gate_count == 0
        assert metrics.saved_state_count == 2
    else:
        assert metrics.recomputed_gate_count > 0
        assert metrics.saved_state_count == 2
