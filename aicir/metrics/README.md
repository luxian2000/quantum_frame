# Metrics

This package contains reusable, task-agnostic metrics for scoring quantum
circuits and ansatz templates. The metrics are designed to be lightweight
building blocks for QAS, VQE ansatz screening, and other architecture-level
analysis workflows.

## Trainability Metrics

`trainability.py` provides two levels of trainability proxies.

### Structure Proxy

```python
from aicir.metrics.trainability import structure_proxy, structure_proxy_details
```

`structure_proxy(circuit)` is a cheap structural score based on:

- circuit depth proxy
- two-qubit gate density
- parameter / single-qubit gate density

It is useful for fast filtering when simulation-based probes are too expensive.

### Local-Probe Gradient Metrics

```python
from aicir.metrics.trainability import (
    gradient_norm_score,
    gradient_variance_score,
    local_probe_gradient_statistics,
)
```

`local_probe_gradient_statistics(circuit, ...)` estimates task-agnostic
parameter-shift gradients on a simple local probe objective. It reports:

- `mean_gradient_norm`
- `gradient_variance`
- `mean_abs_gradient`
- `zero_gradient_fraction`
- `n_parameters`
- `n_gradient_samples`

The score helpers map those statistics into `[0, 1]`:

- `gradient_norm_score(circuit, ...)`
- `gradient_variance_score(circuit, ...)`

These metrics are zero-cost in the NAS/QAS sense: they do not require training
the ansatz on a downstream task.

## Hardware Metrics

`hardware.py` provides hardware-efficiency metrics for architecture-level
screening.

### Native-Gate / Depth / Two-Qubit Proxy

```python
from aicir.metrics.hardware import native_depth_twoq_efficiency
```

`native_depth_twoq_efficiency(circuit, ...)` scores a circuit using:

- native gate compatibility
- depth proxy
- two-qubit gate density

### Topology Mapping Efficiency

```python
from aicir.metrics.hardware import HardwareProfile, topology_mapping_efficiency
```

`HardwareProfile` describes a target device:

```python
profile = HardwareProfile(
    native_gates=("hadamard", "rx", "ry", "rz", "cx", "cnot"),
    coupling_map=[(0, 1), (1, 2), (2, 3)],
    edge_fidelity=None,
    gate_durations={},
    max_depth=12,
)
```

`topology_mapping_efficiency(circuit, profile)` scores the circuit using:

- native gate ratio
- coupling-map compatibility
- routing distance
- depth
- two-qubit gate density

Use `topology_mapping_efficiency_details(circuit, profile)` to inspect the
underlying components.

`edge_fidelity` can be supplied as a scalar or per-edge dictionary. It is
reported as `mapping_fidelity_score`, but it is not included in the primary
hardware-efficiency score. This keeps hardware efficiency separate from noise
robustness metrics.

## Public Exports

The main exports are available from `aicir.metrics`:

```python
from aicir.metrics import (
    HardwareProfile,
    gradient_norm_score,
    gradient_variance_score,
    local_probe_gradient_statistics,
    native_depth_twoq_efficiency,
    topology_mapping_efficiency,
)
```

## Notes

- Scores are clipped to `[0, 1]`, where larger is better.
- These metrics do not depend on QAS demos or experiment runners.
- The trainability gradient probes use the NumPy backend by default.
