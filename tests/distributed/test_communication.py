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
