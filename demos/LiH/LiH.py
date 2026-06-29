"""Construct a PySCF/Qiskit Nature Hamiltonian for lithium hydride.

Run from the repository root:

    python -m demos.LiH

The coefficients below were generated with PySCF/Qiskit Nature for neutral
singlet LiH at 1.595 angstrom, STO-3G basis, a 2-electron/2-spatial-orbital
active space, and Jordan-Wigner mapping. The constant term includes the nuclear
repulsion energy and active-space inactive-energy offset reported by Qiskit
Nature.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np

from aicir import Hamiltonian, NumpyBackend
from aicir.measure import hamiltonian_pauli_terms
from aicir.qas import Supernet, SupernetConfig, SupernetResult
from demos.chemistry_ansatz import closed_shell_excitation_pools, save_qasm3_if_supported


LIH_GEOMETRY_ANGSTROM = (  # 以埃为单位给出 LiH 分子的几何构型。
    ("Li", (0.000000, 0.000000, 0.000000)),  # 将锂原子放在坐标原点。
    ("H", (0.000000, 0.000000, 1.595000)),  # 将氢原子放在 z 轴正方向 1.595 埃处。
)

# LiH 有效空间模型的基础化学参数。
LIH_BASIS = "sto3g"  # 采用 STO-3G 最小基组进行轨道展开。
LIH_CHARGE = 0  # 体系总电荷为 0，对应中性 LiH 分子。
LIH_SPIN = 0  # 总自旋量子数为 0，对应单重态。
LIH_ACTIVE_ELECTRONS = 2  # 有效空间中保留 2 个活性电子。
LIH_ACTIVE_SPATIAL_ORBITALS = 2  # 有效空间中保留 2 个空间轨道。
LIH_QUBIT_MAPPER = "JordanWignerMapper"  # 使用 Jordan-Wigner 变换映射到量子比特。

# 记录该哈密顿量在外部量化化学工具中的生成链路，便于复现。
LIH_GENERATION = (  # 将电子结构问题转换成 4 比特哈密顿量的处理流程。
    "PySCFDriver(atom='Li 0 0 0; H 0 0 1.595', basis='sto3g', "  # 用 PySCFDriver 构造 LiH 分子并求积分。
    "charge=0, spin=0) -> "  # 指定中性单重态的电荷和自旋参数。
    "ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=2) -> "  # 截取 2 电子 2 轨道的活性空间。
    "JordanWignerMapper"  # 最后用 Jordan-Wigner 编码映射到 Pauli 算符。
)

LIH_NUCLEAR_REPULSION_ENERGY = 0.995317638094044  # 原子核间库仑排斥能常数项。
LIH_ACTIVE_SPACE_OFFSET = -7.798291188105942  # 有效空间截断后保留下来的能量偏移项。


def build_lih_hamiltonian() -> Hamiltonian:
    """Return the 4-qubit LiH active-space Hamiltonian.

    The term format mirrors Qiskit's sparse-list ordering:
    ``("ZZ", [0, 3], coeff)`` means ``coeff * Z_0 Z_3``.
    """

    # 这里直接写入由 PySCF/Qiskit Nature 导出的 Pauli 展开系数，
    # 避免运行示例时再依赖量化化学求解流程。
    return Hamiltonian(n_qubits=4, terms=[
        ("IIII", -0.7059409881285760),
        ("IIIZ", 0.1561395312006330),
        ("IIZI", -0.0149911864321253),
        ("IIZZ", 0.0526847769863971),
        ("IZII", 0.1561395312006330),
        ("IZIZ", 0.1219144565378520),
        ("YYII", 0.0139782944733026),
        ("YYIZ", 0.0121237381452783),
        ("XXII", 0.0139782944733026),
        ("XXIZ", 0.0121237381452783),
        ("ZIII", -0.0149911864321253),
        ("ZIIZ", 0.0559382693417738),
        ("IIYY", 0.0139782944733026),
        ("IZYY", 0.0121237381452783),
        ("IIXX", 0.0139782944733026),
        ("IZXX", 0.0121237381452783),
        ("YYYY", 0.0032534923553767),
        ("XXYY", 0.0032534923553767),
        ("YYXX", 0.0032534923553767),
        ("XXXX", 0.0032534923553767),
        ("ZIYY", -0.0018545501857935),
        ("ZIXX", -0.0018545501857935),
        ("IZZI", 0.0559382693417738),
        ("YYZI", -0.0018545501857935),
        ("XXZI", -0.0018545501857935),
        ("ZIZI", 0.0844837493973667),
        ("ZZII", 0.0526847769863971),
    ])


def exact_ground_energy(hamiltonian: Hamiltonian) -> float:
    """Return the minimum eigenvalue of the dense Hamiltonian matrix."""

    # 用稠密矩阵精确对角化得到基态能量，作为 QAS/VQE 的参考真值。
    backend = NumpyBackend()
    matrix = hamiltonian.to_matrix(backend)
    matrix_np = np.asarray(backend.to_numpy(matrix), dtype=np.complex64)
    return float(np.linalg.eigvalsh(matrix_np).min())


def lih_vqe_qas_config(**overrides) -> SupernetConfig:
    """Build the supernet search config for the LiH ground-state VQE.

    The search starts from the closed-shell Hartree-Fock determinant and uses a
    chemistry-motivated excitation pool: spin-preserving ``single_excitation``
    (Givens) gates plus paired ``double_excitation`` gates.
    """

    hf_qubits, single_excitations, double_excitations = closed_shell_excitation_pools(
        LIH_ACTIVE_ELECTRONS,
        LIH_ACTIVE_SPATIAL_ORBITALS,
    )
    params: dict = dict(
        n_qubits=4,
        layers=3,
        single_qubit_gates=("i",),
        two_qubit_gates=("single_excitation",),
        two_qubit_pairs=single_excitations,
        four_qubit_gates=("double_excitation",),
        four_qubit_groups=double_excitations,
        hf_occupied_qubits=hf_qubits,
        # W>1 supernets relieve the single-supernet "fierce competition" that
        # otherwise mis-ranks architectures and stalls the energy ~0.04 Ha high.
        supernet_num=3,
        supernet_steps=200,
        ranking_num=60,
        finetune_steps=120,
        learning_rate=0.1,
        finetune_learning_rate=0.05,
        seed=1,
        device="cpu",
        task="vqe",
    )
    params.update(overrides)
    return SupernetConfig(**params)


def search_ground_state_qas(hamiltonian: Hamiltonian, **overrides) -> SupernetResult:
    """Search a ground-state-preparing circuit for ``hamiltonian`` with supernet.

    Wraps :mod:`aicir.qas.algorithms.supernet`: it sets up a weight-shared supernet, samples
    and optimises ansatze in one stage, ranks them, and fine-tunes the best one.
    The returned result exposes the fine-tuned energy, the fixed-ansatz VQE
    baseline, and the selected circuit in ``final_metrics``.
    """

    # 若调用方未显式指定量子比特数，则与输入哈密顿量保持一致。
    overrides.setdefault("n_qubits", hamiltonian.n_qubits)
    config = lih_vqe_qas_config(**overrides)
    return Supernet(config).train(hamiltonian=hamiltonian)


# Map each aicir gate dict onto the circuit builder that reconstructs it. Each
# entry returns ``(builder_name, call_arguments)`` from the gate dict.
def _gate_to_python_call(gate: dict) -> tuple[str, str] | None:
    gate_type = gate["type"]

    def angle(value) -> str:
        return repr(float(value))

    if gate_type in ("identity", "I"):
        # 占位恒等门不需要写回生成代码。
        return None  # no-op placeholder: nothing to emit

    # 无参数的单比特基础门直接映射到对应构造器。
    single_fixed = {
        "pauli_x": "pauli_x", "X": "pauli_x",
        "pauli_y": "pauli_y", "Y": "pauli_y",
        "pauli_z": "pauli_z", "Z": "pauli_z",
        "hadamard": "hadamard", "H": "hadamard",
        "s_gate": "s_gate", "S": "s_gate",
        "t_gate": "t_gate", "T": "t_gate",
    }
    if gate_type in single_fixed:
        return single_fixed[gate_type], f"{int(gate['target_qubit'])}"

    # 参数化单比特旋转门需要把角度一并序列化。
    single_rot = {"rx": "rx", "ry": "ry", "rz": "rz"}
    if gate_type in single_rot:
        return single_rot[gate_type], f"{angle(gate['parameter'])}, {int(gate['target_qubit'])}"

    if gate_type == "u3":
        p = gate["parameter"]
        return "u3", f"{angle(p[0])}, {angle(p[1])}, {angle(p[2])}, {int(gate['target_qubit'])}"
    if gate_type == "u2":
        p = gate["parameter"]
        return "u2", f"{angle(p[0])}, {angle(p[1])}, {int(gate['target_qubit'])}"

    controlled = {"cx": "cx", "cnot": "cx", "cy": "cy", "cz": "cz"}
    controlled_rot = {"crx": "crx", "cry": "cry", "crz": "crz"}
    if gate_type in controlled or gate_type in controlled_rot:
        target = int(gate["target_qubit"])
        controls = [int(q) for q in gate["control_qubits"]]
        states = [int(s) for s in gate.get("control_states", [1] * len(controls))]
        # 只有非常规控制态时才显式写出 control_states，避免生成代码冗余。
        tail = "" if states == [1] * len(controls) else f", {states}"
        if gate_type in controlled_rot:
            return controlled_rot[gate_type], f"{angle(gate['parameter'])}, {target}, {controls}{tail}"
        return controlled[gate_type], f"{target}, {controls}{tail}"

    if gate_type in ("toffoli", "ccnot"):
        controls = [int(q) for q in gate["control_qubits"]]
        return "toffoli", f"{int(gate['target_qubit'])}, {controls}"

    if gate_type == "swap":
        return "swap", f"{int(gate['qubit_1'])}, {int(gate['qubit_2'])}"
    if gate_type in ("rzz", "rxx"):
        return gate_type, f"{angle(gate['parameter'])}, {int(gate['qubit_1'])}, {int(gate['qubit_2'])}"
    if gate_type in ("single_excitation", "givens"):
        return "single_excitation", f"{angle(gate['parameter'])}, {int(gate['qubit_1'])}, {int(gate['qubit_2'])}"
    if gate_type == "double_excitation":
        qubits = [int(q) for q in gate["qubits"]]
        return "double_excitation", f"{angle(gate['parameter'])}, {', '.join(str(q) for q in qubits)}"

    raise ValueError(f"Cannot serialize gate type {gate_type!r} to Python source")


def circuit_to_python_source(
    circuit,
    func_name: str = "build_lih_qas_circuit",
    figure_name: str = "LiH_cir.png",
    title: str = "LiH supernet ground-state ansatz",
) -> str:
    """Render ``circuit`` as importable Python that rebuilds and plots it.

    The generated module exposes ``func_name()`` and a module-level ``circuit``,
    and—when run as ``__main__``—calls :func:`aicir.visual.plot` to save the
    diagram next to itself.
    """

    # 先把门对象翻译成可重放的构造调用，再拼成独立 Python 模块。
    calls: list[tuple[str, str]] = []
    for gate in circuit.gates:
        rendered = _gate_to_python_call(gate)
        if rendered is not None:
            calls.append(rendered)

    used_builders = sorted({name for name, _ in calls})
    import_list = ", ".join(["Circuit", *used_builders])

    lines = [
        '"""Auto-generated by demos/LiH.py.',
        "",
        "supernet-searched ansatz that prepares the LiH active-space ground state.",
        "Regenerate with: ``python -m demos.LiH``.",
        "Plot with:       ``python -m demos.LiH_cir``.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from pathlib import Path",
        "",
        f"from aicir.core.circuit import {import_list}",
        "from aicir.visual import plot",
        "",
        "",
        f"def {func_name}():",
        '    """Return the supernet-searched LiH ground-state circuit."""',
        "    gates = [",
    ]
    for name, args in calls:
        lines.append(f"        {name}({args}),")
    lines.append("    ]")
    lines.append(f"    return Circuit(*gates, n_qubits={int(circuit.n_qubits)})")
    lines.append("")
    lines.append("")
    lines.append(f"circuit = {func_name}()")
    lines.append("")
    lines.append("")
    lines.append('if __name__ == "__main__":')
    lines.append(f'    figure_path = Path(__file__).with_name("{figure_name}")')
    lines.append(f"    plot(circuit, figure_path, title={title!r})")
    lines.append('    print(f"Saved circuit figure to {figure_path}")')
    lines.append("")
    return "\n".join(lines)


def save_circuit_python(circuit, file_path, func_name: str = "build_lih_qas_circuit") -> None:
    """Write ``circuit`` to ``file_path`` as an importable Python module."""

    Path(file_path).write_text(circuit_to_python_source(circuit, func_name), encoding="utf-8")


def main() -> None:
    """Print the LiH geometry, Pauli terms, and exact dense-matrix energy."""

    # 1. 构造哈密顿量并打印问题定义。
    hamiltonian = build_lih_hamiltonian()
    terms = hamiltonian_pauli_terms(hamiltonian)

    print("LiH PySCF/Qiskit Nature active-space Hamiltonian")
    print("Geometry (angstrom):")
    for atom, coords in LIH_GEOMETRY_ANGSTROM:
        x, y, z = coords
        print(f"  {atom:>2s}: ({x:+.6f}, {y:+.6f}, {z:+.6f})")

    print(f"\nBasis : {LIH_BASIS}")
    print(f"Charge: {LIH_CHARGE}")
    print(f"Spin  : {LIH_SPIN}")
    print(
        "Active space: "
        f"{LIH_ACTIVE_ELECTRONS} electrons, "
        f"{LIH_ACTIVE_SPATIAL_ORBITALS} spatial orbitals"
    )
    print(f"Mapper: {LIH_QUBIT_MAPPER}")
    print(f"Nuclear repulsion energy: {LIH_NUCLEAR_REPULSION_ENERGY:+.10f}")
    print(f"Active-space offset     : {LIH_ACTIVE_SPACE_OFFSET:+.10f}")
    print(f"\nQubits: {hamiltonian.n_qubits}")
    print(f"Terms : {len(terms)}")
    for term in terms:
        print(f"  {term.coefficient:+.10f} * {term.pauli}")

    # 2. 先计算精确基态能量，作为后续近似算法的对照基准。
    exact = exact_ground_energy(hamiltonian)
    print(f"\nDense-matrix exact ground energy: {exact:+.10f}")

    # 3. 运行 supernet 搜索浅层 ansatz，并与固定 ansatz 的 VQE 基线比较。
    print("\nSearching a ground-state circuit with supernet (task='vqe')...")
    result = search_ground_state_qas(hamiltonian)
    metrics = result.final_metrics
    qas_energy = float(metrics["fine_tuned_energy"])
    baseline_energy = float(metrics["baseline_vqe_energy"])
    print(f"  exact ground energy       : {exact:+.10f}")
    print(f"  QAS fine-tuned energy     : {qas_energy:+.10f}")
    print(f"  fixed-ansatz VQE baseline : {baseline_energy:+.10f}")
    print(f"  |QAS - exact|             : {abs(qas_energy - exact):.3e} Ha")
    print(
        "  selected excitations      : "
        f"{metrics['selected_excitation_count']} "
        f"(single={metrics['selected_two_qubit_count']}, double={metrics['selected_four_qubit_count']})"
    )
    print("  selected ansatz circuit:")
    print(metrics["selected_circuit_ascii"])

    qasm_path = Path(__file__).parent / "LiH_cir.qasm"
    saved_qasm, qasm_message = save_qasm3_if_supported(result.best_circuit, qasm_path)
    if saved_qasm:
        print(f"\n  OpenQASM 3.0 saved to: {qasm_message}")
    else:
        print(f"\n  OpenQASM 3.0 skipped: {qasm_message}")

    # 4. 同时导出 Python 重建脚本，方便后续画图和复用线路。
    circuit_py_path = Path(__file__).parent / "LiH_cir.py"
    save_circuit_python(result.best_circuit, circuit_py_path)
    print(f"  Python circuit saved to: {circuit_py_path}")

    # Hand off to the generated module: running it as __main__ plots the figure.
    print("  Plotting via LiH_cir.py ...")
    runpy.run_path(str(circuit_py_path), run_name="__main__")


if __name__ == "__main__":
    main()
