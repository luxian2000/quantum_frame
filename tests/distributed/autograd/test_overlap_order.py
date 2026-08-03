from types import SimpleNamespace

import torch

from aicir.distributed.autograd._pair import _Pair
from aicir.distributed.autograd._vector import _PairVectorKernel
import aicir.distributed.autograd._vector as vector_module


def test_overlap_launches_before_local_matrix_and_waits_before_remote(monkeypatch):
    """The overlap path is a kernel schedule, not an immediate async wait."""

    events = []
    source = _Pair(torch.ones(1), torch.zeros(1))
    backend = SimpleNamespace(
        rank=0,
        world_size=2,
        communicator=SimpleNamespace(autograd_communication_mode="overlap"),
    )
    plan = SimpleNamespace(
        local_matrix=torch.eye(1, dtype=torch.float32),
        partner_masks=(1,),
        partner_for=lambda *, rank, mask: 1,
    )

    class _Launch:
        def wait(self):
            events.append("wait")
            return source

    def launch(pair, **kwargs):
        events.append("launch")
        return _Launch()

    def apply_source(pair, plan, matrix, *, source_rank):
        events.append("local" if source_rank == 0 else "remote")
        return pair

    monkeypatch.setattr(vector_module, "_launch_pair_exchange", launch)
    kernel = _PairVectorKernel(backend)
    monkeypatch.setattr(kernel, "_apply_source", apply_source)
    kernel.apply(source, plan, operation_index=0)

    assert events == ["launch", "local", "wait", "remote"]
