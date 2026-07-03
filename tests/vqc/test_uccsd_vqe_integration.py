"""UCCSD + VQE 端到端集成测试。

驱动链路：``build_molecule``（现算 H2 JW qubit Hamiltonian，含 HF 占据/激发元数据）
-> ``uccsd``（吃纯数据的 UCCSD 模板）-> ``BasicVQE``（走 Circuit 模板编排路径）。

不使用手写的 HF 占据/激发常量：Task 5 的复查确认了比特序约定下 H2（4 qubit JW）
正确的 HF 占据是 ``(0, 1, 0, 1)``（occupied qubits 1、3），而 ``build_molecule``
产出的 ``hf_occupation``/``excitations``/``terms`` 互相一致（已独立验证
⟨HF|H|HF⟩ = -1.8369680643 Ha）。本测试从 ``build_molecule`` 现算，天然保证一致性，
且不依赖任何可能过时的手写比特序猜测。
"""

from __future__ import annotations

import numpy as np
import pytest

qiskit_nature = pytest.importorskip("qiskit_nature")
pytest.importorskip("pyscf")

from aicir import NumpyBackend
from aicir.chemistry import build_molecule
from aicir.optimizer import GD
from aicir.vqc import BasicVQE
from aicir.ansatze import uccsd, uccsd_parameter_count


def test_uccsd_vqe_reaches_h2_ground_energy():
    mol = build_molecule("H 0 0 0; H 0 0 0.735", basis="sto-3g", mapping="jordan_wigner")
    hamiltonian = mol.to_hamiltonian()

    n_params = uccsd_parameter_count(mol.excitations)
    ansatz = uccsd(mol.n_qubits, mol.hf_occupation, mol.excitations)
    assert len(ansatz.parameters) == n_params

    exact_ground_energy = float(
        np.linalg.eigvalsh(hamiltonian.to_matrix(NumpyBackend()))[0]
    )

    solver = BasicVQE(
        hamiltonian,
        ansatz=ansatz,
        backend=NumpyBackend(),
        optimizer=GD(max_iters=200, learning_rate=0.3, gradient_method="psr"),
    )
    result = solver.run(init_params=np.zeros(n_params))

    # 变分原理：VQE 能量不应低于精确基态能量（允许浮点误差）。
    assert result.energy >= exact_ground_energy - 1e-6
    # 真正收敛到基态附近，而非空泛的上界。
    assert result.energy == pytest.approx(exact_ground_energy, abs=5e-3)
