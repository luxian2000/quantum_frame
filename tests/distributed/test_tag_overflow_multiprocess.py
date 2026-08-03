"""P2P tag 必须落在传输层的 32 位有符号区间内。

上层 tag 由 (指令序号, 噪声规则, Kraus 项, partner mask) 组合而成，随门数
线性膨胀；``ProcessGroup::send`` 的 tag 是 32 位 C++ int，越界会直接抛
``TypeError: send(): incompatible function arguments``。真机上
``supports_complex=False``，``exchange`` 还会把复数拆成 ``2t``/``2t+1`` 两次
实数传输，可用空间再减半，因此这里两种传输契约都要覆盖。
"""

import os
import socket

import numpy as np
import torch
import torch.multiprocessing as mp

from aicir import AmplitudeDampingChannel, Circuit, NoiseModel, hadamard
from aicir.distributed import DistSimulator
from aicir.distributed.communication import _TAG_MODULUS, _wrap_tag


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _noisy_circuit(gate_count):
    # 噪声信道作用在逻辑比特 1；显式 layout 把它放到分布式存储轴，
    # 保证每个 Kraus 项都真的走 P2P 交换。
    circuit = Circuit(*[hadamard(0) for _ in range(gate_count)], n_qubits=2)
    circuit.noise_model = NoiseModel().add_channel(
        AmplitudeDampingChannel(target_qubit=1, gamma=0.2)
    )
    return circuit


def _worker(rank, world_size, port, supports_complex, output_path):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    simulator = DistSimulator.from_env(
        fallback_to_cpu=True,
        process_group_backend="gloo",
    )
    # supports_complex=False 复现 Ascend 的传输契约：复数拆成实部/虚部两传。
    simulator.backend.communicator.supports_complex = bool(supports_complex)

    # 六个门远超过原先的溢出阈值（CPU 契约第 3 个门、NPU 契约第 2 个门）。
    result = simulator.run(_noisy_circuit(6), layout=(1, 0))
    density = result.state.to_numpy(root=0)
    if rank == 0:
        np.save(output_path, density)
    else:
        assert density is None
    torch.distributed.destroy_process_group()


def _run(supports_complex, output_path):
    mp.spawn(
        _worker,
        args=(2, _free_port(), supports_complex, output_path),
        nprocs=2,
        join=True,
    )
    return np.load(output_path)


def _reference():
    """六次 hadamard(0) 后跟随每门触发的振幅阻尼，用稠密 Kraus 独立推导。"""

    from aicir.backends.numpy_backend import NumpyBackend

    kraus = [
        np.asarray(operator, dtype=np.complex128)
        for operator in AmplitudeDampingChannel(
            target_qubit=1, gamma=0.2
        ).kraus_operators(2, NumpyBackend())
    ]
    unitary = np.asarray(Circuit(hadamard(0), n_qubits=2).unitary(), dtype=np.complex128)
    rho = np.zeros((4, 4), dtype=np.complex128)
    rho[0, 0] = 1.0
    for _ in range(6):
        rho = unitary @ rho @ unitary.conj().T
        rho = sum(operator @ rho @ operator.conj().T for operator in kraus)
    return rho


def test_wrap_tag_keeps_paired_real_imag_tags_adjacent():
    # exchange 用 (2t, 2t+1) 承载实部/虚部；取模基数是偶数，
    # 因此配对关系和奇偶性在折叠后保持不变。
    for tag in (0, 1, 7, _TAG_MODULUS - 1, _TAG_MODULUS, 3_145_732_097):
        real_tag = _wrap_tag(tag * 2)
        imag_tag = _wrap_tag(tag * 2 + 1)
        assert real_tag % 2 == 0
        assert imag_tag % 2 == 1
        assert imag_tag == real_tag + 1
        assert real_tag // 2 == imag_tag // 2
        assert 0 <= real_tag <= 2**31 - 1
        assert 0 <= imag_tag <= 2**31 - 1


def test_wrap_tag_rejects_negative():
    import pytest

    with pytest.raises(ValueError):
        _wrap_tag(-1)


def test_multi_gate_noise_survives_tag_folding_on_cpu_transport(tmp_path):
    actual = _run(True, str(tmp_path / "cpu.npy"))
    np.testing.assert_allclose(actual, _reference(), atol=1e-6)
    np.testing.assert_allclose(np.trace(actual), 1.0, atol=1e-6)


def test_multi_gate_noise_survives_tag_folding_on_paired_real_transport(tmp_path):
    actual = _run(False, str(tmp_path / "paired.npy"))
    np.testing.assert_allclose(actual, _reference(), atol=1e-6)
    np.testing.assert_allclose(np.trace(actual), 1.0, atol=1e-6)
