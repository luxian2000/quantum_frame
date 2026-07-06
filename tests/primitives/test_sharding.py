import pytest

torch = pytest.importorskip("torch")

from aicir.qas.core.sharding import (
    ShardContext,
    owned_indices,
    shard_context,
    all_gather,
    all_reduce_mean,
    broadcast_parameters,
)
from aicir.backends.npu_backend import NPUBackend
from aicir.backends.gpu_backend import GPUBackend


@pytest.mark.parametrize("n,world", [(10, 4), (8, 2), (5, 3), (3, 4)])
def test_owned_indices_partition_covers_range_once(n, world):
    seen = []
    for rank in range(world):
        seen.extend(owned_indices(n, rank, world))
    assert sorted(seen) == list(range(n))


def test_owned_indices_is_strided():
    assert owned_indices(10, 1, 4) == [1, 5, 9]


def test_shard_context_off_distributed_is_inactive():
    # No process group initialized in this test process.
    ctx = shard_context(NPUBackend(device="npu:0"))
    assert ctx == ShardContext(is_sharded=False, rank=0, world_size=1)


def test_shard_context_non_npu_backend_is_inactive():
    ctx = shard_context(GPUBackend(device="cpu"))
    assert ctx.is_sharded is False


def test_all_gather_without_process_group_returns_singleton():
    assert all_gather({"x": 1}) == [{"x": 1}]


class _FakeDevice:
    def __init__(self, device_type):
        self.type = device_type


class _FakeTensor:
    def __init__(self, device_type="npu"):
        self.device = _FakeDevice(device_type)
        self.dtype = torch.float32
        self.divisor = None
        self.copied = None

    def detach(self):
        return self

    def cpu(self):
        return _FakeTensor("cpu")

    def div_(self, value):
        self.divisor = value
        return self

    def to(self, *, device=None, dtype=None):
        device_type = getattr(device, "type", None) or getattr(self.device, "type", "cpu")
        return _FakeTensor(device_type)

    def copy_(self, other):
        self.copied = other
        return self


def _patch_gloo_dist(monkeypatch):
    import aicir.qas.core.sharding as sharding

    monkeypatch.setattr(sharding.dist, "is_available", lambda: True)
    monkeypatch.setattr(sharding.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(sharding.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(sharding.dist, "get_backend", lambda: "gloo")
    return sharding


def test_all_reduce_mean_stages_non_cpu_tensors_for_gloo(monkeypatch):
    sharding = _patch_gloo_dist(monkeypatch)
    calls = []

    def fake_all_reduce(tensor, op):
        calls.append(tensor)

    monkeypatch.setattr(sharding.dist, "all_reduce", fake_all_reduce)
    tensor = _FakeTensor("npu")

    all_reduce_mean([tensor])

    assert calls[0].device.type == "cpu"
    assert calls[0].divisor == 2
    assert tensor.copied.device.type == "npu"


def test_broadcast_parameters_stages_non_cpu_tensors_for_gloo(monkeypatch):
    sharding = _patch_gloo_dist(monkeypatch)
    calls = []

    def fake_broadcast(tensor, src):
        calls.append((tensor, src))

    monkeypatch.setattr(sharding.dist, "broadcast", fake_broadcast)
    tensor = _FakeTensor("npu")

    broadcast_parameters({"theta": tensor}, src=0)

    staged, src = calls[0]
    assert staged.device.type == "cpu"
    assert src == 0
    assert tensor.copied.device.type == "npu"
