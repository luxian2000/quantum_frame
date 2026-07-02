"""电子结构流水线：给定分子几何/基组/映射，现算 qubit Hamiltonian。

底层调 Qiskit Nature（内部包 PySCF）。属可选能力，需 ``pip install -e ".[chem]"``。
与固定预置并列——预置是快速、零依赖的常用分子；本流水线支持任意分子。
"""

from __future__ import annotations

from .molecules._base import MoleculeHamiltonian

_CHEM_INSTALL_HINT = (
    "电子结构流水线需要 qiskit-nature 与 pyscf；请安装可选依赖："
    'pip install -e ".[chem]"'
)


def _qiskit_nature_available() -> bool:
    try:
        import qiskit_nature  # noqa: F401
    except ImportError:
        return False
    return True


def _require_qiskit_nature():
    if not _qiskit_nature_available():
        raise ImportError(_CHEM_INSTALL_HINT)


def build_molecule(*args, **kwargs) -> MoleculeHamiltonian:
    """现算分子 qubit Hamiltonian（Task 5 实现）。"""

    _require_qiskit_nature()
    raise NotImplementedError("build_molecule 将在 Task 5 实现")
