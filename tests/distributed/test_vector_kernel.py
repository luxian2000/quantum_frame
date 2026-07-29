from unittest.mock import Mock

import torch

from aicir import pauli_x
from aicir.distributed import DistNPUBackend, DistState
from aicir.distributed.gates import _GatePlanner, _VectorKernel
from aicir.distributed.layout import _Layout, _ShardSpec


def test_local_gate_does_not_call_communicator(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    backend = DistNPUBackend.from_env(
        fallback_to_cpu=True,
        init_process_group=False,
    )
    backend._communicator = Mock()
    layout = _Layout.explicit(
        (0, 1),
        n_qubits=2,
        distributed_axes=1,
    )
    spec = _ShardSpec.build(2, 2, 0, "vector", layout)
    state = DistState.from_local(
        torch.tensor([[1.0], [0.0]], dtype=torch.complex64),
        spec=spec,
        backend=backend,
    )
    plan = _GatePlanner(backend, layout, 2).plan(
        pauli_x(1),
        instruction_index=0,
    )

    result = _VectorKernel(backend).apply(state, plan)

    torch.testing.assert_close(
        result.local_data,
        torch.tensor([[0.0], [1.0]], dtype=torch.complex64),
    )
    backend.communicator.exchange.assert_not_called()

