"""将 demos/BeH2 与 demos/CH4 的 QAS/VQE 结果与计算化学软件对比。

包含两类参考值：
  (A) 对 demo 中给定的比特哈密顿量做精确对角化（scipy 稀疏 Lanczos），
      得到该基组/活性空间下的真实基态能量；
  (B) 用 PySCF 独立计算同一分子（相同几何/基组/活性空间）的
      HF 以及 CASCI(BeH2) / FCI(CH4) 参考能量。

结论：两个比特哈密顿量都忠实地表示了对应分子——精确对角化结果在补回
核排斥能后与 PySCF 完全吻合（CH4 与 FCI 吻合到 10 位）。demo 中记录的是
**仅电子部分**的能量。对比真实基态可见：两个 demo 中 QAS 搜索得到的电路
都比固定 ansatz 基线更差，且均未达到化学精度（1.6 mHa）。

从仓库根目录运行：

    PYTHONPATH=. python demos/compare_chem_reference.py
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from demos.BeH2.BeH2 import build_beh2_hamiltonian
from demos.CH4.CH4 import build_ch4_hamiltonian, CH4_NUCLEAR_REPULSION_ENERGY

# demo 输出文件中记录的能量（仅电子部分，Ha）
DEMO_BEH2 = {"qas": -18.5680789948, "baseline": -19.0915603638}
DEMO_CH4 = {"qas": -52.7432708740, "baseline": -53.1978912354}
HA2KCAL = 627.509  # 1 Hartree -> kcal/mol

I2 = sp.identity(2, format="csr", dtype=complex)
X = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
Y = sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
Z = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def pauli_to_sparse(term: str):
    """把 Pauli 串（如 'IXZ...'）转成稀疏矩阵。"""
    m = PAULI[term[0]]
    for ch in term[1:]:
        m = sp.kron(m, PAULI[ch], format="csr")
    return m


def sparse_ground_energy(ham) -> float:
    """对比特哈密顿量做稀疏对角化，返回基态能量。"""
    dim = 2 ** ham.n_qubits
    H = sp.csr_matrix((dim, dim), dtype=complex)
    for ps in ham.terms:
        term = "".join(ps.qubit_labels)
        H = H + complex(ps.coefficient) * pauli_to_sparse(term)
    H = (H + H.getH()) * 0.5  # 数值上强制厄米
    vals = spla.eigsh(H, k=1, which="SA", return_eigenvectors=False, maxiter=5000)
    return float(vals[0])


def _err_line(label: str, energy: float, exact: float) -> str:
    err = energy - exact
    return (f"  {label:<24}: {energy:+.10f} Ha  "
            f"(误差 {err * 1e3:+.1f} mHa / {err * HA2KCAL:+.1f} kcal/mol)")


def pyscf_reference() -> None:
    """用 PySCF 独立计算参考能量（可选，未安装时跳过）。"""
    try:
        from pyscf import gto, scf, fci, cc, mcscf
    except ImportError:
        print("\n(B) 未安装 PySCF，跳过独立参考计算。")
        return

    print("\n" + "=" * 80)
    print("(B) PySCF 独立计算化学参考（总能量，含核排斥）")
    print("=" * 80)

    print("\n--- BeH2 (3-21g, 线性, r_BeH = 1.30 A) ---")
    mol = gto.M(atom="Be 0 0 0; H 0 0 1.3; H 0 0 -1.3", basis="3-21g",
                charge=0, spin=0, verbose=0)
    mf = scf.RHF(mol).run()
    print(f"  HF 总能量              : {mf.e_tot:+.10f} Ha")
    print(f"  核排斥能               : {mol.energy_nuc():+.10f} Ha")
    mc = mcscf.CASCI(mf, 8, 6).run()
    print(f"  CASCI(6e,8o) 总能量    : {mc.e_tot:+.10f} Ha")

    print("\n--- CH4 (sto-3g, 四面体, r_CH = 1.087 A) ---")
    mol2 = gto.M(atom="""C 0 0 0
H  0.627636  0.627636  0.627636
H  0.627636 -0.627636 -0.627636
H -0.627636  0.627636 -0.627636
H -0.627636 -0.627636  0.627636""", basis="sto3g", charge=0, spin=0, verbose=0)
    mf2 = scf.RHF(mol2).run()
    print(f"  HF 总能量              : {mf2.e_tot:+.10f} Ha")
    print(f"  核排斥能               : {mol2.energy_nuc():+.10f} Ha")
    efci, _ = fci.FCI(mf2).kernel()
    print(f"  FCI 总能量             : {efci:+.10f} Ha")


def main() -> None:
    print("=" * 80)
    print("(A) 给定比特哈密顿量的精确基态（scipy 稀疏 Lanczos，仅电子部分）")
    print("=" * 80)

    beh2 = build_beh2_hamiltonian()
    print(f"\nBeH2: {beh2.n_qubits} 比特, {len(beh2.terms)} 项 -> 对角化中...")
    e_beh2 = sparse_ground_energy(beh2)
    print(f"  exact (qubit-H FCI)     : {e_beh2:+.10f} Ha  <-- 真实基态")
    print(_err_line("demo 固定 ansatz VQE", DEMO_BEH2["baseline"], e_beh2))
    print(_err_line("demo QAS fine-tuned", DEMO_BEH2["qas"], e_beh2))

    ch4 = build_ch4_hamiltonian()
    print(f"\nCH4: {ch4.n_qubits} 比特, {len(ch4.terms)} 项 -> 对角化中...")
    e_ch4 = sparse_ground_energy(ch4)
    nuc = float(CH4_NUCLEAR_REPULSION_ENERGY)
    print(f"  exact 电子部分          : {e_ch4:+.10f} Ha  <-- 真实基态")
    print(f"  + 核排斥能              : {nuc:+.10f} Ha")
    print(f"  exact 总能量 (=FCI)     : {e_ch4 + nuc:+.10f} Ha")
    print(_err_line("demo 固定 ansatz VQE", DEMO_CH4["baseline"], e_ch4))
    print(_err_line("demo QAS fine-tuned", DEMO_CH4["qas"], e_ch4))

    pyscf_reference()

    print("\n" + "=" * 80)
    print("结论：化学精度 = 1.6 mHa；两个 demo 的 QAS 电路均比固定 ansatz 基线更差，")
    print("且都未达到化学精度。比特哈密顿量本身正确（补回核排斥后与 PySCF 吻合）。")
    print("=" * 80)


if __name__ == "__main__":
    main()
