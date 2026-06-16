# aicir demos

These scripts demonstrate the `aicir.visual` module for circuits, state vectors,
density matrices, QAS/metrics results, small VQE workflows, and QUBO-to-QAOA
integration.

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
python -m demos.reset_demo
python demos/qubo_qaoa_demo.py
python demos/qubo_tsp_debug_demo.py
```

H2 VQE with shot-based Pauli-term energy estimation:

```bash
python -m demos.vqe_h2_demo --estimator pauli --optimizer spsa --shots 4096
```

Train a 6-qubit QCNN classifier on the local MNIST dataset in `/Volumes/Right/DataSpace/MNIST`:

```bash
python -m demos.mnist.qcnn_demo
```

This demo uses a 6-qubit QCNN feature extractor + 10-class softmax head
trained with cross-entropy. It saves the trained model and all figures to
`demos/mnist/`, and writes a 10-sample random test prediction panel.

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

Build a small QUBO model, run the dense-matrix BasicQAOA helper, and decode the
most likely QUBO assignment:

```bash
python demos/qubo_qaoa_demo.py
```

Debug the QUBO modeling API on a small TSP instance. This verbose demo exposes
`VariableRegistry`, `ModelContext`, `Polynomial`, constraints, `Model`,
`QuboBuilder`, QUBO/Ising/QAOA exports, brute-force validation, and solution
decoding:

```bash
python demos/qubo_tsp_debug_demo.py
```

Run the optional dense BasicQAOA step for the same TSP model:

```bash
python demos/qubo_tsp_debug_demo.py --run-qaoa --qaoa-iters 10
```

Verify in-circuit `reset` execution with `result.snap(...)` snapshots:

```bash
python -m demos.reset_demo
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
