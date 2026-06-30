"""QDRATS (QuantumDARTS) QAS demo for the H2O active-space ground state.

Run from the repository root:

    python -m demos.H2O.H2O_qdrats

Reuses the 6-qubit H2O active-space Hamiltonian from ``demos/H2O/H2O.py`` and
searches a ground-state-preparing circuit with the differentiable QuantumDARTS
method (``aicir.qas`` method name ``"qdrats"``) over an **excitation gate pool**
(``gate_pool="excitation"``): starting from the closed-shell Hartree-Fock reference,
each layer slot chooses between applying a parameterised ``single_excitation`` /
``double_excitation`` operator or identity; Gumbel-Softmax + a straight-through
estimator update the architecture weights while the excitation angles are optimised
in alternation. This is the particle-number/spin-preserving ansatz used by the
supernet demo (``demos/H2O/H2O.py``), now searched with QDRATS.

The discretised circuit's energy is compared against the dense-matrix exact value,
and the searched circuit is written to ``H2O_qdrats_cir.py`` (a self-plotting module
in the same style as ``H2O_cir.py``); running that module transpile-optimises the
circuit via ``aicir.transpile.optimize`` and plots the optimised result.

Requires ``torch`` (QDRATS runs on a Torch/NPU backend).
"""

from __future__ import annotations

import runpy
from pathlib import Path

from aicir.qas import config, run
from demos.H2O.H2O import (
    H2O_DOUBLE_EXCITATIONS,
    H2O_HF_OCCUPIED_QUBITS,
    H2O_SINGLE_EXCITATIONS,
    build_h2o_hamiltonian,
    exact_ground_energy,
    save_circuit_python,
)


def h2o_qdrats_config(**overrides) -> object:
    """Build the QDRATS excitation-pool search config for the 6-qubit H2O ground state.

    Searches the spin-preserving single/double excitation ansatz from the
    closed-shell HF reference. Modest defaults keep the demo runnable on CPU in a
    few minutes; raise ``search_epochs`` / ``layers`` / ``finetune_steps`` to push
    toward chemical accuracy.
    """

    params: dict = dict(
        n_qubits=6,
        layers=2,
        gate_pool="excitation",
        single_excitations=H2O_SINGLE_EXCITATIONS,
        double_excitations=H2O_DOUBLE_EXCITATIONS,
        hf_occupied_qubits=H2O_HF_OCCUPIED_QUBITS,
        search_epochs=80,
        theta_steps=2,
        finetune_steps=150,
        architecture_learning_rate=0.05,
        theta_learning_rate=0.05,
        finetune_learning_rate=0.03,
        temperature=1.0,
        seed=2,
        device="cpu",
    )
    params.update(overrides)
    return config.qdrats(**params)


def main() -> None:
    """Search an H2O ground-state circuit with QDRATS and report against exact."""

    hamiltonian = build_h2o_hamiltonian()
    exact = exact_ground_energy(hamiltonian)

    print("H2O active-space Hamiltonian (PySCF/Qiskit Nature, STO-3G, JW)")
    print(f"  qubits: {hamiltonian.n_qubits}")
    print(f"  dense-matrix exact ground energy: {exact:+.10f} Ha")

    print("\nSearching a ground-state circuit with QDRATS (QuantumDARTS)...")
    print("  (6-qubit H2O; modest config, expect a few minutes on CPU.)")
    result = run("qdrats", hamiltonian=hamiltonian, config=h2o_qdrats_config())

    qdrats_energy = float(result.minimum_energy)
    print(f"\n  exact ground energy   : {exact:+.10f}")
    print(f"  QDRATS minimum energy : {qdrats_energy:+.10f}")
    print(f"  |QDRATS - exact|      : {abs(qdrats_energy - exact):.3e} Ha")
    print(f"  search layers         : {result.config.layers}")

    print("\n  discretised architecture (gate per qubit-layer position):")
    for layer_index, labels in enumerate(result.architecture_labels):
        print(f"    layer {layer_index}: {labels}")

    print("\n  searched circuit:")
    result.circuit.show()

    circuit_py_path = Path(__file__).parent / "H2O_qdrats_cir.py"
    save_circuit_python(
        result.circuit,
        circuit_py_path,
        func_name="build_h2o_qdrats_circuit",
        figure_name="H2O_qdrats_cir.png",
        title="H2O QDRATS ground-state ansatz",
        generated_by="demos/H2O/H2O_qdrats.py",
        description="QDRATS (QuantumDARTS) searched ansatz preparing the H2O active-space ground state.",
        regen_cmd="python -m demos.H2O.H2O_qdrats",
        plot_cmd="python -m demos.H2O.H2O_qdrats_cir",
        optimize=True,
    )
    print(f"  Python circuit saved to: {circuit_py_path}")

    # Hand off to the generated module: running it as __main__ plots the figure.
    print("  Plotting via H2O_qdrats_cir.py ...")
    runpy.run_path(str(circuit_py_path), run_name="__main__")


if __name__ == "__main__":
    main()
