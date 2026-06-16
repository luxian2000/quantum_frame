# aicir

A from-scratch quantum circuit simulator and quantum-algorithm framework for Python. Supports state vectors, density matrices, noise models, variational algorithms (VQE/QAOA/VQD/SSVQE), quantum architecture search, QML gradients, and OpenQASM I/O — with pluggable backends for CPU, GPU, and Ascend NPU.

## Features

- **Unified quantum state** — `State` class handles pure states (amplitude vector) and mixed states (density matrix) with a consistent API
- **Rich gate library** — single-qubit, rotation, controlled, multi-target, and multi-control gates; typed `Operation` IR with construction-time validation
- **Flexible measurement** — in-circuit Pauli projection, terminal readout, shot sampling, exact mode, state snapshots, and partial traces
- **Variational algorithms** — `BasicVQE`, `run_vqe`, QAOA, VQD, SSVQE with built-in ansatz templates (HEA, trapped-ion HEA-TI)
- **QML gradients** — parameter-shift (`psr`, `spsr`, `multipsr`), finite-difference, SPSA, quantum natural gradient, and PyTorch `autograd`
- **Quantum architecture search** — weight-shared supernet, CRLQAS, PPR\_DQL, PPO\_RB (requires PyTorch)
- **Noise simulation** — depolarizing, bit/phase flip, amplitude damping, ion-trap noise via density-matrix evolution
- **OpenQASM I/O** — round-trip import/export for OpenQASM 2.0 and 3.0; Qiskit, PennyLane, and WuYue interop
- **Pluggable backends** — `NumpyBackend` (CPU), `GPUBackend` (PyTorch / CUDA), `NPUBackend` (Ascend) — swap with one line

## Installation

```bash
# Core (NumPy only)
pip install aicir

# With optional extras
pip install "aicir[torch]"   # GPU/NPU backend + QAS
pip install "aicir[viz]"     # Circuit visualization (matplotlib)
pip install "aicir[sci]"     # Classical optimizers (scipy)
pip install "aicir[all]"     # Everything above
```

Or install editable from source:

```bash
git clone https://github.com/luxian2000/quantum_frame.git
cd quantum_frame
pip install -e ".[all]"
```

> `torch`, `matplotlib`, and `scipy` are optional. Core simulation works with NumPy alone.

## Quick Start

```python
from aicir import Circuit, Measure, NumpyBackend, hadamard, cnot
from aicir.core import State

backend = NumpyBackend()

# Build a Bell-state circuit
cir = Circuit(
    hadamard(0),
    cnot(1, [0]),
    n_qubits=2,
    backend=backend,
)

# Run with 1024 shots
result = Measure(backend).run(cir, shots=1024)
print(result.counts)   # {'00': ~512, '11': ~512}

# Or evolve a state directly
psi = State.zero_state(2, backend)
psi1 = psi.evolve(cir.unitary())
print(psi1.ket)        # 1/√2|00> + 1/√2|11>
```

## Backends

| Backend | Library | Device | Autograd |
| --- | --- | --- | --- |
| `NumpyBackend` | NumPy | CPU | No |
| `GPUBackend` | PyTorch | CPU / CUDA | Yes |
| `NPUBackend` | PyTorch + torch\_npu | Ascend NPU | Yes |

Switch backends by changing one line — no other code changes needed:

```python
from aicir import GPUBackend, NPUBackend

backend = GPUBackend(device="cuda:0")
# or
backend = NPUBackend.from_distributed_env(fallback_to_cpu=True)
```

## Variational Algorithms

```python
from aicir.vqc import run_vqe
from aicir.chemistry import h2_hamiltonian

result = run_vqe(h2_hamiltonian(), n_qubits=4)
print(f"Ground state energy: {result.energy:.6f} Ha")
```

## QML Gradients

```python
from aicir.qml import psr   # parameter-shift rule

grad = psr(circuit, params, hamiltonian, backend)
```

## Demos

```bash
PYTHONPATH=. python demos/vqe_h2_demo.py
PYTHONPATH=. python demos/grover_demo.py
PYTHONPATH=. python demos/qft_demo.py
PYTHONPATH=. python demos/qubo_qaoa_demo.py
PYTHONPATH=. python demos/qnn_gradient_demo.py
```

For Ascend NPU:

```bash
python demos/demo_npu.py
python demos/demo_npu.py --allow-cpu-fallback
torchrun --nproc_per_node=4 demos/demo_npu.py
```

## Running Tests

```bash
PYTHONPATH=. pytest          # full suite
PYTHONPATH=. pytest tests/vqc/test_vqe_orchestration.py  # single file
```

## Project Structure

```text
aicir/
  backends/     # NumpyBackend, GPUBackend, NPUBackend
  core/         # State, Circuit, gates, I/O
  measure/      # Measure, Result, Estimator
  vqc/          # VQE, QAOA, VQD, SSVQE, ansatz templates
  qas/          # Quantum architecture search (requires torch)
  qml/          # Gradient methods
  noise/        # Noise channels and models
  ir/           # Typed Operation IR
  gates/        # GateSpec registry
  transpile/    # Pass-manager pipeline
  primitives/   # ShotSampler, StatevectorEstimator, ShotEstimator
  optimizer/    # Classical optimizers (Adam, COBYLA, LBFGS, …)
  chemistry/    # Preset qubit Hamiltonians (H2, H2-JW, H2-tapered)
  encoder/      # AmplitudeEncoder, AngleEncoder, BasisEncoder
  universal/    # Reusable primitives (QFT, …)
  visual/       # Circuit visualization
demos/          # Runnable end-to-end examples
tests/          # pytest test suite
```

## Requirements

- Python ≥ 3.11
- NumPy (required)
- PyTorch (optional — GPU/NPU backend, QAS, autograd)
- matplotlib (optional — visualization)
- scipy (optional — classical optimizers)
