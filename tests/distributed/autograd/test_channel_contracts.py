"""Collective-safe channel validation and Stinespring construction contracts."""

from __future__ import annotations

import pytest
import torch

from aicir.distributed.autograd._channels import _preflight_channel
from aicir.distributed.autograd._parameters import StinespringParam


def _raw(size=4):
    return torch.ones((size, size), dtype=torch.float32), torch.zeros((size, size), dtype=torch.float32)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"input_dim": 2.0, "output_dim": 2, "environment_dim": 2},
        {"input_dim": 2, "output_dim": 4, "environment_dim": 2},
        {"input_dim": 3, "output_dim": 3, "environment_dim": 2},
        {"input_dim": 2, "output_dim": 2, "environment_dim": 0},
        {"input_dim": 2, "output_dim": 2, "environment_dim": 2, "target_qubits": (0, 0)},
        {"input_dim": 4, "output_dim": 4, "environment_dim": 2, "target_qubits": (0,)},
    ),
)
def test_stinespring_constructor_rejects_invalid_dimensions_and_targets(kwargs):
    real, imag = _raw()
    with pytest.raises((TypeError, ValueError)):
        StinespringParam(real=real, imag=imag, **kwargs)


def test_stinespring_constructor_accepts_nonzero_logical_target_and_exact_shape():
    real, imag = _raw()
    parameter = StinespringParam(2, 2, 2, real, imag, target_qubits=(3,))
    assert parameter.target_qubits == (3,)


def test_channel_preflight_rejects_hostile_metadata_before_planning(monkeypatch):
    """The helper must return a synchronized validation result, not parse eagerly."""

    class Hostile:
        def __int__(self):
            raise RuntimeError("must not escape preflight")

    monkeypatch.setenv("WORLD_SIZE", "1")
    result = _preflight_channel(Hostile(), n_qubits=1, communicator=None)
    assert result is False
