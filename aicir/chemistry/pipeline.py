"""电子结构流水线：给定分子几何/基组/映射，现算 qubit Hamiltonian。

底层调 Qiskit Nature（内部包 PySCF）。属可选能力，需 ``pip install -e ".[chem]"``。
与固定预置并列——预置是快速、零依赖的常用分子；本流水线支持任意分子。
"""

from __future__ import annotations

from . import _qiskit_bridge
from .molecules._base import MoleculeHamiltonian

_CHEM_INSTALL_HINT = (
    "电子结构流水线需要 qiskit-nature 与 pyscf；请安装可选依赖："
    'pip install -e ".[chem]"'
)


def _qiskit_nature_available() -> bool:
    return _qiskit_bridge.qiskit_nature_available()


def _require_qiskit_nature():
    if not _qiskit_nature_available():
        raise ImportError(_CHEM_INSTALL_HINT)


def build_molecule(
    geometry,
    *,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    mapping: str = "jordan_wigner",
    active_electrons=None,
    active_orbitals=None,
    two_qubit_reduction: bool = False,
    name: str = "custom",
) -> MoleculeHamiltonian:
    """给定分子几何/基组/映射，现算 qubit Hamiltonian。

    ``jordan_wigner``/``parity``/``bravyi_kitaev`` 均填充
    ``n_electrons``/``hf_occupation``/``excitations``。其中 parity/BK 的
    ``hf_occupation`` 是 mapper 变换后的 bitstring，``excitations`` 是结构索引；
    二者不声明 mapper-correct 化学 UCCSD 激发。
    """

    _require_qiskit_nature()

    driver = _qiskit_bridge.build_driver(geometry, basis=basis, charge=charge, spin=spin)
    problem = driver.run()
    problem = _qiskit_bridge.apply_active_space(
        problem, active_electrons=active_electrons, active_orbitals=active_orbitals
    )

    second_q_op = problem.hamiltonian.second_q_op()
    mapper = _qiskit_bridge.select_mapper(
        mapping,
        num_particles=problem.num_particles,
        two_qubit_reduction=two_qubit_reduction,
    )

    qubit_op = mapper.map(second_q_op)
    terms = _qiskit_bridge.sparse_pauli_to_terms(qubit_op)  # 转 aicir PauliTerm，对齐比特序
    n_qubits = qubit_op.num_qubits

    n_electrons = sum(problem.num_particles)
    hf_occupation = _qiskit_bridge.hf_occupation_from_mapper(problem, mapper, n_qubits)
    excitations = _qiskit_bridge.structural_excitations(problem, n_qubits)

    return MoleculeHamiltonian(
        name=name,
        formula=name.upper(),
        n_qubits=n_qubits,
        terms=terms,
        basis=basis.upper(),
        mapping=mapping,
        geometry=str(geometry),
        source="aicir.chemistry.build_molecule (Qiskit Nature/PySCF)",
        n_electrons=n_electrons,
        hf_occupation=hf_occupation,
        excitations=excitations,
    )
