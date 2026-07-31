import torch

from aicir.distributed.communication import _Communicator


def test_single_rank_reductions_do_not_require_process_group():
    communicator = _Communicator(
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )
    value = torch.tensor([2.0], dtype=torch.float32)

    reduced = communicator.all_reduce_sum(value)
    gathered = communicator.gather_to_root(value, root=0)

    assert reduced is not value
    torch.testing.assert_close(reduced, value)
    assert len(gathered) == 1
    torch.testing.assert_close(gathered[0], value)


def test_complex_exchange_falls_back_to_ordered_real_imag_transport(monkeypatch):
    communicator = _Communicator(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        supports_complex=False,
    )
    calls = []

    def fake_exchange(tensor, peer, tag):
        calls.append((tensor.dtype, peer, tag))
        offset = 10.0 if tag == 14 else 20.0
        return tensor + offset

    monkeypatch.setattr(communicator, "_exchange_tensor", fake_exchange)

    result = communicator.exchange(
        torch.tensor([1.0 + 2.0j], dtype=torch.complex64),
        peer=1,
        tag=7,
    )

    torch.testing.assert_close(
        result,
        torch.tensor([11.0 + 22.0j], dtype=torch.complex64),
    )
    assert calls == [
        (torch.float32, 1, 14),
        (torch.float32, 1, 15),
    ]


def test_complex_exchange_uses_one_transport_when_supported(monkeypatch):
    communicator = _Communicator(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        supports_complex=True,
    )
    calls = []

    def fake_exchange(tensor, peer, tag):
        calls.append((tensor.dtype, peer, tag))
        return tensor * 2

    monkeypatch.setattr(communicator, "_exchange_tensor", fake_exchange)

    result = communicator.exchange(
        torch.tensor([1.0 + 2.0j], dtype=torch.complex64),
        peer=1,
        tag=7,
    )

    torch.testing.assert_close(
        result,
        torch.tensor([2.0 + 4.0j], dtype=torch.complex64),
    )
    assert calls == [(torch.complex64, 1, 7)]


def test_real_transport_bytes_are_per_rank_endpoint_logical_payload(monkeypatch):
    class _FakeDist:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def gather(tensor, gather_list, **_kwargs):
            if gather_list is not None:
                for item in gather_list:
                    item.copy_(tensor)

        @staticmethod
        def scatter(receive, scatter_list, **_kwargs):
            if scatter_list is None:
                receive.zero_()
            else:
                receive.copy_(scatter_list[0])

    communicator = _Communicator(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        dist_module=_FakeDist(),
    )
    monkeypatch.setattr(
        communicator,
        "_exchange_tensor",
        lambda tensor, _peer, _tag: tensor.clone(),
    )
    payload = torch.ones(2, dtype=torch.float32)

    communicator.exchange_real(payload, peer=1, tag=3)
    communicator.gather_to_root_real(payload, root=0)
    communicator.scatter_from_root_real([payload, payload], root=0, shape=(2,))

    records = communicator.communication_records
    assert [record["bytes"] for record in records] == [16, 16, 16]
    assert communicator.communication_counters["bytes"] == 48

    nonroot = _Communicator(
        rank=1,
        world_size=2,
        device=torch.device("cpu"),
        dist_module=_FakeDist(),
    )
    nonroot.gather_to_root_real(payload, root=0)
    nonroot.scatter_from_root_real(None, root=0, shape=(2,))
    assert [record["bytes"] for record in nonroot.communication_records] == [8, 8]
