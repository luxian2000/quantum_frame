"""Built-in paired-real channel gradients against independent float64 oracles."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aicir import PauliString
from aicir.core.circuit import rx
from aicir.distributed import DistNPUBackend
from aicir.distributed.autograd._channels import _selected_noise_rules
from aicir.distributed.autograd._density import _PairMatrixKernel
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.autograd._reducers import _PairReducer
from aicir.distributed.layout import _Layout, _ShardSpec
from aicir.distributed.state import DistState
from aicir.noise import (
    AmplitudeDampingChannel,
    BitFlipChannel,
    DepolarizingChannel,
    NoiseModel,
    PhaseFlipChannel,
)


def _backend(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    return DistNPUBackend.from_env(fallback_to_cpu=True, init_process_group=False)


def _one_density(backend):
    layout = _Layout.explicit((0,), n_qubits=1, distributed_axes=0)
    spec = _ShardSpec.build(1, 1, 0, "matrix", layout)
    pair = _Pair(
        torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        torch.zeros((2, 2), dtype=torch.float32),
    )
    return DistState.from_pair(pair, spec=spec, backend=backend)


def _plus_density(backend):
    layout = _Layout.explicit((0,), n_qubits=1, distributed_axes=0)
    spec = _ShardSpec.build(1, 1, 0, "matrix", layout)
    pair = _Pair(
        torch.full((2, 2), 0.5, dtype=torch.float32),
        torch.zeros((2, 2), dtype=torch.float32),
    )
    return DistState.from_pair(pair, spec=spec, backend=backend)


def _reference(channel_factory, probability, *, plus=False):
    rho = np.full((2, 2), 0.5, dtype=np.complex128) if plus else np.diag([0.0, 1.0]).astype(np.complex128)
    channel = channel_factory(float(probability))
    if isinstance(channel, BitFlipChannel):
        matrices = (np.sqrt(1.0 - probability) * np.eye(2), np.sqrt(probability) * np.array([[0.0, 1.0], [1.0, 0.0]]))
    elif isinstance(channel, PhaseFlipChannel):
        matrices = (np.sqrt(1.0 - probability) * np.eye(2), np.sqrt(probability) * np.diag([1.0, -1.0]))
    elif isinstance(channel, DepolarizingChannel):
        matrices = (
            np.sqrt(1.0 - probability) * np.eye(2),
            np.sqrt(probability / 3.0) * np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.sqrt(probability / 3.0) * np.array([[0.0, -1j], [1j, 0.0]]),
            np.sqrt(probability / 3.0) * np.diag([1.0, -1.0]),
        )
    else:
        matrices = (
            np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - probability)]], dtype=np.complex128),
            np.array([[0.0, np.sqrt(probability)], [0.0, 0.0]], dtype=np.complex128),
        )
    return sum((matrix @ rho @ matrix.conj().T for matrix in matrices))


@pytest.mark.parametrize(
    ("channel_factory", "plus"),
    (
        (lambda p: BitFlipChannel(0, p), False),
        (lambda p: PhaseFlipChannel(0, p), True),
        (lambda p: DepolarizingChannel(0, p), False),
        (lambda p: AmplitudeDampingChannel(0, p), False),
    ),
)
def test_builtin_channel_probability_and_pauli_gradients_match_float64(monkeypatch, channel_factory, plus):
    backend = _backend(monkeypatch)
    probability = torch.tensor(0.23, dtype=torch.float32, requires_grad=True)
    state = _plus_density(backend) if plus else _one_density(backend)
    evolved = _PairMatrixKernel(backend).apply_channel(
        state, channel_factory(probability), instruction_index=31
    )
    probabilities = _PairReducer(backend).probabilities(evolved._pair, evolved.spec)
    pauli = _PairReducer(backend).expectation(
        evolved._pair, evolved.spec, PauliString("X" if plus else "Z", n_qubits=1)
    )
    loss = probabilities[0] + 0.37 * pauli
    loss.backward()

    epsilon = 1e-6
    def objective(value):
        density = _reference(channel_factory, value, plus=plus)
        observable = np.array([[0.0, 1.0], [1.0, 0.0]]) if plus else np.diag([1.0, -1.0])
        return float(density[0, 0].real + 0.37 * np.trace(density @ observable).real)

    finite_difference = (objective(0.23 + epsilon) - objective(0.23 - epsilon)) / (2.0 * epsilon)
    assert probability.grad is not None
    assert abs(float(probability.grad) - finite_difference) <= 1e-4
    assert abs(float(probabilities.sum().detach()) - 1.0) <= 1e-5


def test_channel_sequence_keeps_all_probability_leaves_differentiable(monkeypatch):
    backend = _backend(monkeypatch)
    first = torch.tensor(0.17, dtype=torch.float32, requires_grad=True)
    second = torch.tensor(0.31, dtype=torch.float32, requires_grad=True)
    kernel = _PairMatrixKernel(backend)
    state = kernel.apply_channel(_one_density(backend), BitFlipChannel(0, first), instruction_index=41)
    state = kernel.apply_channel(state, AmplitudeDampingChannel(0, second), instruction_index=42)
    value = _PairReducer(backend).expectation(state._pair, state.spec, PauliString("Z", n_qubits=1))
    value.backward()
    assert first.grad is not None and second.grad is not None
    assert torch.isfinite(first.grad) and torch.isfinite(second.grad)


def test_noise_rule_selection_honors_gate_filter_and_excluded_target():
    bit = BitFlipChannel(0, 0.1)
    phase = PhaseFlipChannel(1, 0.2)
    model = NoiseModel().add_channel(bit, after_gates=["rx"], exclude_gate_qubits=True).add_channel(phase)
    gate = rx(0.2, 1)
    assert _selected_noise_rules(model, gate) == (bit, phase)
    assert _selected_noise_rules(model, rx(0.2, 0)) == (phase,)
