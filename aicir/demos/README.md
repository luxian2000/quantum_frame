# aicir visual demos

These scripts demonstrate the `aicir.visual` module for circuits, state vectors,
density matrices, and QAS/metrics results.

Run from the repository root:

```bash
python -m aicir.demos.visual_circuit_demo
python -m aicir.demos.visual_state_demo
python -m aicir.demos.visual_density_demo
python -m aicir.demos.visual_qas_demo
```

By default, figures are saved to:

```text
aicir/demos/visual_outputs/
```

Use a custom output directory:

```bash
python -m aicir.demos.visual_qas_demo --output-dir /tmp/aicir_visual_demo
```

Show figures interactively after saving:

```bash
python -m aicir.demos.visual_state_demo --show
```
