import numpy as np
import pytest

from aicir.ansatze import uccsd, uccsd_parameter_count
from aicir.backends import NumpyBackend
from aicir.chemistry import MoleculeHamiltonian, get_molecule
from aicir.core.circuit import Circuit, pauli_x
from aicir.core.state import State

# ≤6-qubit、基态能量已验证的 preset：Phase 2 静态补齐了 n_electrons/hf_occupation/
# excitations（见 aicir/chemistry/molecules/H2.py、LiH.py、H2O.py 的补齐注释）。
_BACKFILLED_PRESETS = ("h2", "h2_jw", "lih", "h2o")

# 未补齐（保持 None）：h2_tapered 用 TaperedQubitMapper，不在 build_molecule 支持的
# 三种 mapper 之列，且 tapering 后单比特表示不再与自旋轨道占据数一一对应；
# nh3/n2/beh2 是 12–16 qubit 结构守卫 preset，本仓不做 dense 对角化验证。
_UNBACKFILLED_PRESETS = ("h2_tapered", "nh3", "n2", "beh2")


def test_existing_preset_metadata_defaults_to_none_for_unbackfilled_presets():
    for name in _UNBACKFILLED_PRESETS:
        mol = get_molecule(name)
        assert mol.n_electrons is None
        assert mol.hf_occupation is None
        assert mol.excitations is None


def test_metadata_fields_are_settable():
    mol = MoleculeHamiltonian(
        name="toy",
        formula="H2",
        n_qubits=4,
        terms=((-1.0, "IIII"),),
        basis="STO-3G",
        mapping="JordanWignerMapper",
        geometry="toy",
        source="toy",
        n_electrons=2,
        hf_occupation=(1, 1, 0, 0),
        excitations=(("single", (0, 2)), ("double", (0, 1, 2, 3))),
    )
    assert mol.n_electrons == 2
    assert mol.hf_occupation == (1, 1, 0, 0)
    assert mol.excitations == (("single", (0, 2)), ("double", (0, 1, 2, 3)))


@pytest.mark.parametrize("name", _BACKFILLED_PRESETS)
def test_backfilled_preset_metadata_is_non_none(name):
    mol = get_molecule(name)
    assert mol.n_electrons is not None
    assert mol.hf_occupation is not None
    assert mol.excitations is not None


@pytest.mark.parametrize("name", _BACKFILLED_PRESETS)
def test_backfilled_hf_occupation_length_matches_n_qubits(name):
    mol = get_molecule(name)
    assert len(mol.hf_occupation) == mol.n_qubits
    assert set(mol.hf_occupation).issubset({0, 1})


# h2 用 ParityMapper(two_qubit_reduction)：电子数/自旋宇称已被吸收进被约化掉的两个
# 比特，剩余 bitstring 的比特和不再等于电子数，"sum == n_electrons" 只对
# Jordan-Wigner（未做对称约化）preset 成立。
_JW_BACKFILLED_PRESETS = ("h2_jw", "lih", "h2o")


@pytest.mark.parametrize("name", _JW_BACKFILLED_PRESETS)
def test_backfilled_jw_hf_occupation_sum_matches_n_electrons(name):
    mol = get_molecule(name)
    assert sum(mol.hf_occupation) == mol.n_electrons


@pytest.mark.parametrize("name", _BACKFILLED_PRESETS)
def test_backfilled_excitations_indices_are_in_range(name):
    mol = get_molecule(name)
    for kind, indices in mol.excitations:
        assert kind in ("single", "double")
        assert len(indices) == (2 if kind == "single" else 4)
        assert all(0 <= i < mol.n_qubits for i in indices)


def _hf_energy(mol: MoleculeHamiltonian) -> float:
    """用 hf_occupation 摆计算基态 HF 态，返回 ⟨HF|H|HF⟩（纯 numpy，无可选依赖）。"""

    backend = NumpyBackend()
    circuit = Circuit(
        *[pauli_x(i) for i, bit in enumerate(mol.hf_occupation) if bit == 1],
        n_qubits=mol.n_qubits,
    )
    state = State.zero_state(mol.n_qubits, backend).evolve(circuit.unitary(backend))
    psi = state.to_numpy()
    h_matrix = backend.to_numpy(mol.to_hamiltonian().to_matrix(backend))
    return complex(np.conj(psi) @ h_matrix @ psi).real


# 独立参照：qiskit_nature 可用时现算的 ⟨HF|H|HF⟩（不含核排斥能），用于离线交叉验证
# 补齐值。此处直接固化为常量，测试本身不依赖 qiskit_nature/pyscf。preset 系数四舍
# 五入到 8 位小数，故与 build_molecule 全精度现算值有 <1e-6 的微小偏差。
_KNOWN_HF_ENERGY = {
    "h2": -1.8369680643081665,  # 与 qiskit reference_energy - nuclear_repulsion 一致（<1e-7）
    "h2_jw": -1.836967945098877,
    "lih": -1.0590503215789795,
    "h2o": -6.155271530151367,
}


@pytest.mark.parametrize("name", sorted(_KNOWN_HF_ENERGY))
def test_backfilled_hf_occupation_reproduces_known_hf_energy(name):
    """回归：hf_occupation 的比特序须与 terms 对齐，否则 ⟨HF|H|HF⟩ 会明显偏离。"""

    mol = get_molecule(name)
    energy = _hf_energy(mol)
    assert energy == pytest.approx(_KNOWN_HF_ENERGY[name], abs=1e-6)


def test_uccsd_bridge_from_backfilled_h2_preset():
    """aicir.ansatze.uccsd 消费补齐后的 get_molecule("h2") 元数据（纯 numpy 路径）。

    钉住 chemistry <-> ansatze 桥接契约：``uccsd(mol.n_qubits, mol.hf_occupation,
    mol.excitations)`` 能正常构造，参数个数与激发数一致，且 HF 计算基态的
    ⟨HF|H|HF⟩ 与已知 HF 能量吻合（证明 hf_occupation 与 excitations 比特序自洽）。
    """

    mol = get_molecule("h2")

    n_params = uccsd_parameter_count(mol.excitations)
    ansatz = uccsd(mol.n_qubits, mol.hf_occupation, mol.excitations)
    assert len(ansatz.parameters) == n_params == len(mol.excitations)

    energy = _hf_energy(mol)
    assert energy == pytest.approx(_KNOWN_HF_ENERGY["h2"], abs=1e-6)
