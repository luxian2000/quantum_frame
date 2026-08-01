"""One collective-safe gradient bucket for replicated native parameters."""

from __future__ import annotations

import pytest
import torch

from aicir import Circuit
from aicir.core.circuit import ry
from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.autograd._parameters import (
    _bind_trainable_aliases,
    _bucket_parameters,
    _parameter_structure_digest,
    _preflight_parameter_structure,
    _replicated_parameter_entries,
    StinespringParam,
)
from aicir.ir import instruction_parameter
from aicir.noise import NoiseModel


class _RecordingCommunicator:
    """CPU stand-in that exposes the exact collective surface used by buckets."""

    device = torch.device("cpu")

    def __init__(self, world_size=2, gathered=None):
        self.world_size = world_size
        self.calls = []
        self._gathered = gathered
        self.barriers = 0

    def all_reduce_sum_real(self, value):
        self.calls.append(value.detach().clone())
        return value * self.world_size

    def all_gather_real(self, value):
        if self._gathered is not None:
            return self._gathered(value)
        return [value.clone() for _ in range(self.world_size)]

    def barrier(self):
        self.barriers += 1


@pytest.mark.parametrize("count", (32, 128))
def test_bucket_reduces_all_scalar_gradients_in_one_float32_collective(count):
    communicator = _RecordingCommunicator()
    leaves = tuple(torch.tensor(float(index), dtype=torch.float32, requires_grad=True) for index in range(count))

    aliases = _bucket_parameters(leaves, communicator=communicator)
    sum((index + 1) * value for index, value in enumerate(aliases)).backward()

    assert len(communicator.calls) == 1
    assert communicator.calls[0].dtype is torch.float32
    assert communicator.calls[0].numel() == count
    for index, leaf in enumerate(leaves):
        assert leaf.grad.item() == pytest.approx(2.0 * (index + 1))


def test_bucket_preserves_paired_real_ranges_and_fills_missing_alias_gradients():
    communicator = _RecordingCommunicator()
    real = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
    imag = torch.tensor([3.0, 4.0], dtype=torch.float32, requires_grad=True)

    real_alias, imag_alias = _bucket_parameters(
        (("state.real", real, "real"), ("state.imag", imag, "imag")),
        communicator=communicator,
    )
    # ``imag_alias`` is intentionally unused: backward must still pack its zero range.
    real_alias.square().sum().backward()

    assert len(communicator.calls) == 1
    torch.testing.assert_close(communicator.calls[0], torch.tensor([2.0, 4.0, 0.0, 0.0]))
    torch.testing.assert_close(real.grad, torch.tensor([4.0, 8.0]))
    torch.testing.assert_close(imag.grad, torch.zeros_like(imag))


def test_parameter_structure_digest_includes_order_shape_dtype_requires_grad_and_component():
    value = torch.zeros(2, dtype=torch.float32, requires_grad=True)
    first = _parameter_structure_digest((("theta", value, "real"),))
    reordered = _parameter_structure_digest((("theta", value, "imag"),))
    changed_dtype = _parameter_structure_digest((("theta", value.double(), "real"),))
    changed_requires_grad = _parameter_structure_digest((("theta", value.detach(), "real"),))

    assert len(first) == 32
    assert len({first, reordered, changed_dtype, changed_requires_grad}) == 4


@pytest.mark.parametrize("kind", ("order", "shape", "dtype", "requires_grad"))
def test_parameter_structure_mismatch_is_synchronized_before_data_collectives(kind):
    local = torch.zeros(2, dtype=torch.float32, requires_grad=True)
    local_entries = (("theta", local, "real"),)
    if kind == "order":
        remote = (("other", local, "real"),)
    elif kind == "shape":
        remote = (("theta", torch.zeros(3, dtype=torch.float32, requires_grad=True), "real"),)
    elif kind == "dtype":
        remote = (("theta", torch.zeros(2, dtype=torch.float64, requires_grad=True), "real"),)
    else:
        remote = (("theta", torch.zeros(2, dtype=torch.float32), "real"),)
    remote_digest = _parameter_structure_digest(remote)
    communicator = _RecordingCommunicator(gathered=lambda current: [current, remote_digest])

    with pytest.raises(ValueError, match="^各 rank 的可训练参数结构不一致$"):
        _preflight_parameter_structure(local_entries, communicator=communicator)
    assert communicator.barriers == 1
    assert communicator.calls == []


def test_alias_binding_rebuilds_typed_circuit_without_mutating_caller():
    theta = torch.tensor(0.2, dtype=torch.float32, requires_grad=True)
    real = torch.eye(2, dtype=torch.float32, requires_grad=True)
    imag = torch.zeros((2, 2), dtype=torch.float32, requires_grad=True)
    circuit = Circuit(ry(theta, 0), {"type": "unitary", "parameter": _Pair(real, imag), "n_qubits": 1}, n_qubits=1)
    aliases = _bucket_parameters((theta, real, imag), communicator=_RecordingCommunicator(world_size=1))

    rebound = _bind_trainable_aliases(circuit, {id(theta): aliases[0], id(real): aliases[1], id(imag): aliases[2]})

    assert rebound is not circuit
    assert rebound.operations[0] is not circuit.operations[0]
    assert instruction_parameter(circuit.operations[0]) is theta
    assert instruction_parameter(rebound.operations[0]) is aliases[0]
    original_pair = instruction_parameter(circuit.operations[1])
    rebound_pair = instruction_parameter(rebound.operations[1])
    assert original_pair.real is real and original_pair.imag is imag
    assert rebound_pair.real is aliases[1] and rebound_pair.imag is aliases[2]


def test_replicated_entries_include_circuit_and_stinespring_leaves_but_not_noise_cache():
    theta = torch.tensor(0.2, dtype=torch.float32, requires_grad=True)
    real = torch.eye(2, dtype=torch.float32, requires_grad=True)
    imag = torch.zeros((2, 2), dtype=torch.float32, requires_grad=True)
    channel = StinespringParam(2, 2, 1, real, imag)
    circuit = Circuit(ry(theta, 0), n_qubits=1)
    circuit.noise_model = NoiseModel().add_channel(channel)
    cached_non_parameter = torch.ones(1, dtype=torch.float32, requires_grad=True)
    circuit.noise_model._kraus_cache["stale"] = cached_non_parameter

    entries = _replicated_parameter_entries(circuit)

    assert [id(entry.value) for entry in entries] == [id(theta), id(real), id(imag)]
    assert all(entry.value is not cached_non_parameter for entry in entries)


def test_alias_binding_clears_rebuilt_noise_runtime_cache():
    theta = torch.tensor(0.2, dtype=torch.float32, requires_grad=True)
    circuit = Circuit(ry(theta, 0), n_qubits=1)
    circuit.noise_model = NoiseModel()
    cached_non_parameter = torch.ones(1, dtype=torch.float32, requires_grad=True)
    circuit.noise_model._kraus_cache["stale"] = cached_non_parameter
    (alias,) = _bucket_parameters((theta,), communicator=_RecordingCommunicator(world_size=1))

    rebound = _bind_trainable_aliases(circuit, {id(theta): alias})

    assert circuit.noise_model._kraus_cache == {"stale": cached_non_parameter}
    assert rebound.noise_model._kraus_cache == {}
