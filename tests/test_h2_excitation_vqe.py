"""H2 关联验证：HF + double_excitation 经 autograd 优化达化学精度。

证明激发门能产生真实电子关联（仅单激发/无关联无法达此精度）。

JW 映射约定说明：
  HF 行列式 = |0101⟩（全局索引 5，比特 1,3 占据）
  关联行列式 = |1010⟩（全局索引 10，比特 0,2 占据）

double_excitation(theta, 0, 2, 1, 3) 的局部坐标
  局部比特顺序: [0→实际 q0, 1→实际 q2, 2→实际 q1, 3→实际 q3]
  局部 |0011⟩ = 局部位 2,3 置 1 = 实际 q1,q3 → HF
  局部 |1100⟩ = 局部位 0,1 置 1 = 实际 q0,q2 → 关联
  从而单个 double_excitation 门恰好耦合 HF 与主关联行列式。
"""

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from aicir import double_excitation, NumpyBackend
from aicir.backends.gpu_backend import GPUBackend
from aicir.chemistry import molecule_hamiltonian, molecule_matrix
from aicir.core.circuit import Circuit, pauli_x
from aicir.core.gates import apply_gate_to_state

CHEMICAL_ACCURACY = 1.6e-3
EXACT_GROUND = -1.8572750091552734


def _hf_occupied_qubits(ham_matrix):
    # 在 2 电子扇区里选与精确基态重叠最大的 HF 行列式（避开 JW 排序约定的歧义）
    w, v = np.linalg.eigh(ham_matrix)
    ground = v[:, 0]
    weight2 = [i for i in range(16) if bin(i).count("1") == 2]
    best = max(weight2, key=lambda i: abs(ground[i]))
    return [q for q in range(4) if (best >> (3 - q)) & 1]


def _correlating_qubits(ham_matrix):
    # 找第二大振幅的 2 电子行列式（主关联行列式）
    w, v = np.linalg.eigh(ham_matrix)
    ground = v[:, 0]
    weight2 = [i for i in range(16) if bin(i).count("1") == 2]
    sorted_w2 = sorted(weight2, key=lambda i: abs(ground[i]), reverse=True)
    second = sorted_w2[1]
    return [q for q in range(4) if (second >> (3 - q)) & 1]


def test_h2_double_excitation_reaches_chemical_accuracy():
    ham = molecule_hamiltonian("h2_jw")
    Hmat = molecule_matrix("h2_jw", backend=NumpyBackend())
    occupied = _hf_occupied_qubits(Hmat)        # JW 下 HF 比特：[1, 3]
    correlating = _correlating_qubits(Hmat)     # 主关联行列式比特：[0, 2]

    # 构造 double_excitation 的比特参数使局部 |0011⟩↔|1100⟩ 对应 HF↔关联
    # 局部位序 [corr[0], corr[1], occ[0], occ[1]] 满足此要求
    q0, q1, q2, q3 = correlating[0], correlating[1], occupied[0], occupied[1]

    backend = GPUBackend(device="cpu")
    Ht = backend.cast(Hmat.astype(np.complex64))

    def energy(theta_t):
        state = backend.zeros_state(4)
        # 从 HF 行列式出发
        for q in occupied:
            state = apply_gate_to_state(pauli_x(q).to_dict(), state, 4, backend)
        # 单个 double_excitation 门耦合 HF ↔ 关联行列式
        for g in Circuit(double_excitation(theta_t, q0, q1, q2, q3), n_qubits=4).gates:
            state = apply_gate_to_state(g, state, 4, backend)
        return backend.expectation_sv(state, Ht).real

    theta = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.1)
    for _ in range(400):
        opt.zero_grad()
        e = energy(theta)
        e.backward()
        opt.step()

    final = float(energy(theta).detach())
    assert abs(final - EXACT_GROUND) < CHEMICAL_ACCURACY, f"E={final}, exact={EXACT_GROUND}"
