"""跨 rank 一致性摘要必须与张量所在设备无关。

可训练门的参数是 torch tensor。非 CPU tensor 的 ``repr`` 会带上设备号
（``device='npu:0'`` / ``device='npu:1'``），若摘要基于 ``repr``，各 rank 必然
算出不同的值，真机多卡上任何可训练 ``DistSimulator.run()`` 都会被误判成
"各 rank 的线路、布局或运行选项不一致"。CPU/gloo 的 repr 不含设备名，所以
这个缺陷在本地多进程测试里永远复现不出来——这里用 repr 注入来锁住不变量。
"""

import torch

from aicir import Circuit, ry
from aicir.distributed.simulator import (
    DistSimulator,
    _contract_digest,
    _contract_value,
)
from aicir.ir import circuit_instructions, instruction_to_gate_dict


def _agreement_payload(circuit):
    """与 _assert_process_agreement 相同的 payload 结构。"""

    return (
        int(circuit.n_qubits),
        tuple(
            instruction_to_gate_dict(instruction)
            for instruction in circuit_instructions(circuit)
        ),
        "layout-digest",
        None,
        (),
        False,
        True,
        True,
    )


def _trainable_circuit(value=0.31):
    theta = torch.tensor(value, dtype=torch.float32, requires_grad=True)
    return Circuit(ry(theta, 0), n_qubits=2)


class _CapturingCommunicator:
    """记录 _assert_process_agreement 实际提交给 all_gather 的摘要。"""

    def __init__(self):
        self.captured = []

    def all_gather(self, tensor):
        self.captured.append(bytes(tensor.detach().cpu().tolist()))
        return [tensor]


class _StubBackend:
    def __init__(self, communicator):
        self._device = torch.device("cpu")
        self.communicator = communicator


def _digest_under_device_repr(rank, circuit):
    """走真实的 _assert_process_agreement，取它算出的本 rank 摘要。"""

    communicator = _CapturingCommunicator()
    simulator = DistSimulator.__new__(DistSimulator)
    simulator._backend = _StubBackend(communicator)

    original = torch.Tensor.__repr__
    try:
        torch.Tensor.__repr__ = (
            lambda self: original(self)[:-1] + f", device='npu:{rank}')"
        )
        simulator._assert_process_agreement(
            circuit=circuit,
            layout=_StubLayout(),
            shots=None,
            measure_qubits=(),
            collapse=False,
            return_state=True,
            return_probabilities=True,
        )
    finally:
        torch.Tensor.__repr__ = original
    return communicator.captured[0]


class _StubLayout:
    def digest(self):
        return "layout-digest"


def test_assert_process_agreement_digest_ignores_tensor_device():
    """真机复现：同一线路在 npu:0 与 npu:1 上必须得到相同摘要。

    修复前 _assert_process_agreement 用 repr(payload)，两者不同即抛
    "各 rank 的线路、布局或运行选项不一致"。
    """

    circuit = _trainable_circuit()
    assert _digest_under_device_repr(0, circuit) == _digest_under_device_repr(1, circuit)


def test_repr_payload_really_does_differ_across_devices():
    """记录缺陷成因：基于 repr 的摘要确实会随设备号变化。"""

    import hashlib

    payload = _agreement_payload(_trainable_circuit())
    original = torch.Tensor.__repr__
    digests = []
    try:
        for rank in (0, 1):
            torch.Tensor.__repr__ = (
                lambda self, _rank=rank: original(self)[:-1] + f", device='npu:{_rank}')"
            )
            digests.append(hashlib.sha256(repr(payload).encode("utf-8")).digest())
    finally:
        torch.Tensor.__repr__ = original
    assert digests[0] != digests[1]


def test_agreement_digest_still_detects_different_parameter_values():
    """设备无关不等于放宽校验：参数取值不同仍必须被发现。"""

    first = _contract_digest(_contract_value(_agreement_payload(_trainable_circuit(0.31))))
    second = _contract_digest(_contract_value(_agreement_payload(_trainable_circuit(0.32))))
    assert first != second


def test_agreement_digest_detects_different_circuits():
    first = _contract_digest(_contract_value(_agreement_payload(_trainable_circuit())))
    other = Circuit(ry(torch.tensor(0.31, requires_grad=True), 1), n_qubits=2)
    assert first != _contract_digest(_contract_value(_agreement_payload(other)))


def test_agreement_digest_detects_requires_grad_mismatch():
    """rank_route_mismatch 契约依赖这一点：只有 rank0 requires_grad 必须被发现。"""

    trainable = _contract_digest(_contract_value(_agreement_payload(_trainable_circuit())))
    frozen = Circuit(
        ry(torch.tensor(0.31, dtype=torch.float32, requires_grad=False), 0), n_qubits=2
    )
    assert trainable != _contract_digest(_contract_value(_agreement_payload(frozen)))
