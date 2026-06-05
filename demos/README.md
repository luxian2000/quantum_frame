# aicir demos

These scripts demonstrate the `aicir.visual` module for circuits, state vectors,
density matrices, QAS/metrics results, and a small VQE workflow.

Run from the repository root:

```bash
python -m demos.vqe_h2_demo
python -m demos.visual_circuit_demo
python -m demos.visual_state_demo
python -m demos.visual_density_demo
python -m demos.visual_qas_demo
```

H2 VQE with shot-based Pauli-term energy estimation:

```bash
python -m demos.vqe_h2_demo --estimator pauli --optimizer spsa --shots 4096
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
