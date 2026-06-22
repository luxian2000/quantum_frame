"""Supernet 布局空间懒采样/懒参数（BeH2 16 比特 SIGKILL 根因回归）。

旧实现在构造时枚举 `product(single_qubit_gates, repeat=n_qubits)` = gates**n_qubits，
16 比特即 5**16 ≈ 1.5e11，撑爆主机内存。现改为按槽采样 + 首次访问懒建共享参数。
"""

import pytest

pytest.importorskip("torch")

from aicir.qas.algorithms.supernet import SupernetConfig, Supernet


def _cfg(n_qubits, **kw):
    base = dict(
        n_qubits=n_qubits,
        layers=1,
        supernet_num=1,
        supernet_steps=0,
        ranking_num=1,
        finetune_steps=0,
        two_qubit_pairs=(),
        task="vqe",
        seed=1,
    )
    base.update(kw)
    return SupernetConfig(**base)


def test_shared_parameters_not_prefilled_at_construction():
    qas = Supernet(_cfg(4))
    # 懒创建：构造不枚举布局、不预建任何共享参数。
    assert qas.shared_parameters == {}


def test_params_bounded_by_visited_architecture():
    qas = Supernet(_cfg(4))
    arch = qas.sample_architecture()
    qas.build_circuit(arch, supernet_id=0)
    # 仅访问到的一个架构的槽，远小于 5**4 个布局。
    assert len(qas.shared_parameters) <= qas.config.layers * qas.config.n_qubits


def test_large_n_qubits_constructs_without_enumeration():
    # 5**16 枚举会 OOM；懒路径必须秒级构造 + 采样 + 建路。
    qas = Supernet(_cfg(16))
    arch = qas.sample_architecture()
    circuit, keys, tensors = qas.build_circuit(arch, supernet_id=0)
    assert circuit.n_qubits == 16
    assert len(qas.shared_parameters) <= qas.config.layers * qas.config.n_qubits
