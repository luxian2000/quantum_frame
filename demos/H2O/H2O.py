"""Construct a PySCF/Qiskit Nature Hamiltonian for a water molecule.

Run from the repository root:

    python -m demos.H2O.H2O

The coefficients below were generated with PySCF/Qiskit Nature for neutral
singlet H2O, STO-3G basis, a 4-electron/3-spatial-orbital active space, and
Jordan-Wigner mapping. The constant term includes the nuclear repulsion energy
and active-space inactive-energy offset reported by Qiskit Nature.

The script then searches a ground-state-preparing circuit with the supernet
QAS method (``aicir.qas.algorithms.supernet``), in the same style as ``demos/LiH``, and
records it to ``H2O_cir.py`` / ``H2O_cir.qasm`` / ``H2O_cir.png``.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np

from aicir import Hamiltonian, NumpyBackend
from aicir.measure import hamiltonian_pauli_terms
from aicir.qas import Supernet, SupernetConfig, SupernetResult
from demos.chemistry_ansatz import closed_shell_excitation_pools, save_qasm3_if_supported


H2O_GEOMETRY_ANGSTROM = (
    ("O", (0.000000, 0.000000, 0.000000)),
    ("H", (0.757160, 0.586260, 0.000000)),
    ("H", (-0.757160, 0.586260, 0.000000)),
)

H2O_BASIS = "sto3g"
H2O_CHARGE = 0
H2O_SPIN = 0
H2O_ACTIVE_ELECTRONS = 4
H2O_ACTIVE_SPATIAL_ORBITALS = 3
H2O_QUBIT_MAPPER = "JordanWignerMapper"

H2O_GENERATION = (
    "PySCFDriver("
    "atom='O 0 0 0; H 0.757160 0.586260 0; H -0.757160 0.586260 0', "
    "basis='sto3g', charge=0, spin=0) -> "
    "ActiveSpaceTransformer(num_electrons=4, num_spatial_orbitals=3) -> "
    "JordanWignerMapper"
)

H2O_NUCLEAR_REPULSION_ENERGY = 9.191200742618042
H2O_ACTIVE_SPACE_OFFSET = -77.99892062901534


def build_h2o_hamiltonian() -> Hamiltonian:
    """Return the 6-qubit H2O active-space Hamiltonian."""

    return Hamiltonian(n_qubits=6, terms=[
        ("IIIIII", -4.5241061234101245),
        ("IIIIIZ", 0.5153159797373158),
        ("IIIYZY", -0.0777577585041152),
        ("IIIXZX", -0.0777577585041152),
        ("IIIIZI", 0.4813205218163351),
        ("IIIZII", 0.0902426376429984),
        ("IIZIII", 0.5153159797373158),
        ("YZYIII", -0.0777577585041153),
        ("XZXIII", -0.0777577585041153),
        ("IZIIII", 0.4813205218163352),
        ("ZIIIII", 0.0902426376429983),
        ("IIIIZZ", 0.1682539303981457),
        ("IIIZIZ", 0.1200923081829575),
        ("IIZIIZ", 0.1956895651366700),
        ("YZYIIZ", -0.0303380764625682),
        ("XZXIIZ", -0.0303380764625682),
        ("IZIIIZ", 0.1822330598869542),
        ("ZIIIIZ", 0.1372661146010317),
        ("IIIYIY", -0.0295021027117906),
        ("IIIXIX", -0.0295021027117906),
        ("IYYIYY", 0.0139791294888085),
        ("IXXIYY", 0.0139791294888085),
        ("IYYIXX", 0.0139791294888085),
        ("IXXIXX", 0.0139791294888085),
        ("YYIIYY", -0.0004375325735518),
        ("XXIIYY", -0.0004375325735518),
        ("YYIIXX", -0.0004375325735518),
        ("XXIIXX", -0.0004375325735518),
        ("IIZYZY", -0.0303380764625682),
        ("IIZXZX", -0.0303380764625682),
        ("YZYYZY", 0.0171738064180742),
        ("XZXYZY", 0.0171738064180742),
        ("YZYXZX", 0.0171738064180742),
        ("XZXXZX", 0.0171738064180742),
        ("IZIYZY", -0.0290645701382388),
        ("IZIXZX", -0.0290645701382388),
        ("ZIIYZY", -0.0111466920449807),
        ("ZIIXZX", -0.0111466920449807),
        ("IIIZZI", 0.1375870149850004),
        ("IIZIZI", 0.1822330598869542),
        ("YZYIZI", -0.0290645701382388),
        ("XZXIZI", -0.0290645701382388),
        ("IZIIZI", 0.2200397733437616),
        ("ZIIIZI", 0.1472359927649237),
        ("IYYYYI", -0.0004375325735518),
        ("IXXYYI", -0.0004375325735518),
        ("IYYXXI", -0.0004375325735518),
        ("IXXXXI", -0.0004375325735518),
        ("YYIYYI", 0.0096489777799233),
        ("XXIYYI", 0.0096489777799233),
        ("YYIXXI", 0.0096489777799233),
        ("XXIXXI", 0.0096489777799233),
        ("IIZZII", 0.1372661146010317),
        ("YZYZII", -0.0111466920449807),
        ("XZXZII", -0.0111466920449807),
        ("IZIZII", 0.1472359927649237),
        ("ZIIZII", 0.1492816648983666),
        ("IZZIII", 0.1682539303981457),
        ("ZIZIII", 0.1200923081829575),
        ("YIYIII", -0.0295021027117906),
        ("XIXIII", -0.0295021027117906),
        ("ZZIIII", 0.1375870149850004),
    ])


def exact_ground_energy(hamiltonian: Hamiltonian) -> float:
    """Return the minimum eigenvalue of the dense Hamiltonian matrix."""

    backend = NumpyBackend()
    matrix = hamiltonian.to_matrix(backend)
    matrix_np = np.asarray(backend.to_numpy(matrix), dtype=np.complex64)
    return float(np.linalg.eigvalsh(matrix_np).min())


H2O_HF_OCCUPIED_QUBITS, H2O_SINGLE_EXCITATIONS, H2O_DOUBLE_EXCITATIONS = (
    closed_shell_excitation_pools(
        H2O_ACTIVE_ELECTRONS,
        H2O_ACTIVE_SPATIAL_ORBITALS,
    )
)


def h2o_vqe_qas_config(**overrides) -> SupernetConfig:
    """Build the supernet search config for the H2O ground-state VQE.

    Same style as ``demos/LiH``: start from the closed-shell Hartree-Fock
    determinant and search spin-preserving single excitations plus paired double
    excitations.
    """

    params: dict = dict(
        n_qubits=6,
        layers=6,
        single_qubit_gates=("i",),
        two_qubit_gates=("single_excitation",),
        two_qubit_pairs=H2O_SINGLE_EXCITATIONS,
        four_qubit_gates=("double_excitation",),
        four_qubit_groups=H2O_DOUBLE_EXCITATIONS,
        hf_occupied_qubits=H2O_HF_OCCUPIED_QUBITS,
        # W>1 supernets relieve the single-supernet "fierce competition"; H2O
        # needs more of everything than LiH to approach chemical accuracy.
        supernet_num=5,
        supernet_steps=250,
        ranking_num=80,
        finetune_steps=250,
        learning_rate=0.1,
        finetune_learning_rate=0.05,
        seed=2,
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

    overrides.setdefault("n_qubits", hamiltonian.n_qubits)
    config = h2o_vqe_qas_config(**overrides)
    return Supernet(config).train(hamiltonian=hamiltonian)


# Map each aicir gate dict onto the circuit builder that reconstructs it. Each
# entry returns ``(builder_name, call_arguments)`` from the gate dict.
def _gate_to_python_call(gate: dict) -> tuple[str, str] | None:
    gate_type = gate["type"]

    def angle(value) -> str:
        return repr(float(value))

    if gate_type in ("identity", "I"):
        return None  # no-op placeholder: nothing to emit

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
    func_name: str = "build_h2o_qas_circuit",
    figure_name: str = "H2O_cir.png",
    title: str = "H2O supernet ground-state ansatz",
) -> str:
    """Render ``circuit`` as importable Python that rebuilds and plots it.

    The generated module exposes ``func_name()`` and a module-level ``circuit``,
    and—when run as ``__main__``—calls :func:`aicir.visual.plot` to save the
    diagram next to itself.
    """

    calls: list[tuple[str, str]] = []
    for gate in circuit.gates:
        rendered = _gate_to_python_call(gate)
        if rendered is not None:
            calls.append(rendered)

    used_builders = sorted({name for name, _ in calls})
    import_list = ", ".join(["Circuit", *used_builders])

    lines = [
        '"""Auto-generated by demos/H2O/H2O.py.',
        "",
        "supernet-searched ansatz that prepares the H2O active-space ground state.",
        "Regenerate with: ``python -m demos.H2O.H2O``.",
        "Plot with:       ``python -m demos.H2O.H2O_cir``.",
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
        '    """Return the supernet-searched H2O ground-state circuit."""',
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


def save_circuit_python(circuit, file_path, func_name: str = "build_h2o_qas_circuit") -> None:
    """Write ``circuit`` to ``file_path`` as an importable Python module."""

    Path(file_path).write_text(circuit_to_python_source(circuit, func_name), encoding="utf-8")


def main() -> None:
    """Print the H2O Hamiltonian, exact energy, and search a ground-state circuit."""

    hamiltonian = build_h2o_hamiltonian()
    terms = hamiltonian_pauli_terms(hamiltonian)

    print("H2O PySCF/Qiskit Nature active-space Hamiltonian")
    print("Geometry (angstrom):")
    for atom, coords in H2O_GEOMETRY_ANGSTROM:
        x, y, z = coords
        print(f"  {atom:>2s}: ({x:+.6f}, {y:+.6f}, {z:+.6f})")

    print(f"\nBasis : {H2O_BASIS}")
    print(f"Charge: {H2O_CHARGE}")
    print(f"Spin  : {H2O_SPIN}")
    print(
        "Active space: "
        f"{H2O_ACTIVE_ELECTRONS} electrons, "
        f"{H2O_ACTIVE_SPATIAL_ORBITALS} spatial orbitals"
    )
    print(f"Mapper: {H2O_QUBIT_MAPPER}")
    print(f"Nuclear repulsion energy: {H2O_NUCLEAR_REPULSION_ENERGY:+.10f}")
    print(f"Active-space offset     : {H2O_ACTIVE_SPACE_OFFSET:+.10f}")
    print(f"\nQubits: {hamiltonian.n_qubits}")
    print(f"Terms : {len(terms)}")
    for term in terms:
        print(f"  {term.coefficient:+.10f} * {term.pauli}")

    exact = exact_ground_energy(hamiltonian)
    print(f"\nDense-matrix exact ground energy: {exact:+.10f}")

    # Search a shallow ground-state circuit with the supernet QAS method and
    # compare against the exact energy and the fixed-ansatz VQE baseline.
    print("\nSearching a ground-state circuit with supernet (task='vqe')...")
    print("  (6-qubit H2O is much harder than LiH; this takes ~1 minute.)")
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

    qasm_path = Path(__file__).parent / "H2O_cir.qasm"
    saved_qasm, qasm_message = save_qasm3_if_supported(result.best_circuit, qasm_path)
    if saved_qasm:
        print(f"\n  OpenQASM 3.0 saved to: {qasm_message}")
    else:
        print(f"\n  OpenQASM 3.0 skipped: {qasm_message}")

    circuit_py_path = Path(__file__).parent / "H2O_cir.py"
    save_circuit_python(result.best_circuit, circuit_py_path)
    print(f"  Python circuit saved to: {circuit_py_path}")

    # Hand off to the generated module: running it as __main__ plots the figure.
    print("  Plotting via H2O_cir.py ...")
    runpy.run_path(str(circuit_py_path), run_name="__main__")


if __name__ == "__main__":
    main()
