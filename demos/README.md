# aicir demos

These scripts demonstrate the `aicir.visual` module for circuits, state vectors,
density matrices, QAS/metrics results, and a small VQE workflow.

Run from the repository root:

```bash
python -m demos.vqe_h2_demo
python -m demos.H2O
python -m demos.BeH2.BeH2
python -m demos.BeH2.BeH2_npu
python -m demos.LiH
python -m demos.visual_circuit_demo
python -m demos.visual_state_demo
python -m demos.visual_density_demo
python -m demos.visual_qas_demo
```

H2 VQE with shot-based Pauli-term energy estimation:

```bash
python -m demos.vqe_h2_demo --estimator pauli --optimizer spsa --shots 4096
```

Construct and inspect a PySCF/Qiskit Nature H2O active-space Hamiltonian:

```bash
python -m demos.H2O
```

Construct and inspect a PySCF/Qiskit Nature BeH2 active-space Hamiltonian
(default 6e/8o -> 16 qubits):

```bash
python -m demos.BeH2.BeH2
```

Construct and inspect a PySCF/Qiskit Nature LiH active-space Hamiltonian:

```bash
python -m demos.LiH
```

By default, figures are saved to:

```text
demos/visual_outputs/
```

Use a custom output directory:

```bash
python -m demos.visual_qas_demo --output-dir /tmp/aicir_visual_demo
```

Show figures interactively after saving:

```bash
python -m demos.visual_state_demo --show
```
