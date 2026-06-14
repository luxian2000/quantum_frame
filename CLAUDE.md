# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`aicir` is a from-scratch quantum circuit simulator and quantum-algorithm framework (state vectors, density matrices, noise, VQE/QAOA/VQD/SSVQE, quantum architecture search, QML gradients, OpenQASM I/O). It ships a standard `pyproject.toml` (setuptools, flat layout, package `aicir`); install editable with `pip install -e ".[all]"`, or just run from the source tree via `PYTHONPATH=.` — both work. Most docs and code comments are in Chinese; match that when editing comments/docstrings.

## Commands

```bash
# Install editable (optional; tests/demos also work via PYTHONPATH=.)
pip install -e ".[all]"        # extras: torch / viz / sci / all / dev

# Run the full test suite (run from repo root)
PYTHONPATH=. pytest            # or: python -m pytest -q (no PYTHONPATH needed if installed)

# Run a single test file / test
PYTHONPATH=. pytest tests/vqc/test_vqe_orchestration.py
PYTHONPATH=. pytest tests/qml/test_gradient.py -q
PYTHONPATH=. pytest tests/vqc/test_vqe_orchestration.py::test_name

# Run a demo (demos are modules; run from repo root)
PYTHONPATH=. python demos/vqe_h2_demo.py
python -m demos.qnn_gradient_demo

# Ascend NPU verification entrypoints (demo lives under demos/)
python demos/demo_npu.py                          # strict NPU check
python demos/demo_npu.py --allow-cpu-fallback     # allow CPU fallback for dev
torchrun --nproc_per_node=2 demos/demo_npu.py     # task-parallel (NOT state sharding)
```

`torch`, `matplotlib`, and `scipy` are **optional**. Tests that need them call `pytest.importorskip(...)` and skip cleanly when absent — don't add hard imports of these to core modules.

## Architecture

Everything is built on a backend abstraction so upper layers (`StateVector`, `Circuit`, `Measure`, VQE, QAS…) never touch the underlying numeric framework.

- **Backends** (`aicir/channel/backends/`): all implement the abstract `Backend` (`base.py`). `NumpyBackend` (default, CPU), `GPUBackend` (Torch, autograd-capable; `TorchBackend` is a deprecated alias), `NPUBackend` (Ascend). Conventions: state vector shape `(2^n, 1)`, density matrix `(2^n, 2^n)`, complex dtype; methods return native tensors, `to_numpy()` is the single conversion exit. To add a backend, subclass `Backend` and implement all abstract methods — nothing else should need changing.

- **Gates are plain dicts**, not objects. Gate factories (`pauli_x`, `cnot`, `rx`, `rzz`, `rxx`/`ms_gate`, `toffoli`, `u3`, …) return dicts consumed by `gate_to_matrix` (`aicir/core/gates.py`). Controlled gates take `(target, control_qubits_list)`; dicts may carry `control_states` (0/1) and `control_qubits`. A typed `Operation` IR (`aicir/ir/`) is being layered on top: `Circuit.__init__`/`append`/`extend` now route every gate through `normalize_gate`, so they accept `Operation` *or* dict, but the circuit still **stores dicts internally** (`Operation.to_dict()`). Don't assume `circuit.gates` holds `Operation`s.

- **Circuit** (`aicir/core/circuit.py`): a `dataclass`-based container of gate dicts plus `n_qubits`. `Parameter` is a *symbolic* placeholder (not an autograd tensor). Template circuits store only gate dicts until `bind_parameters(...)` (returns a new circuit by default; `inplace=True`, `allow_partial=True` available). `unitary()` errors on unbound parameters. For Torch autograd, pass Torch scalar tensors directly as gate params and use a `GPUBackend` — the computation graph is preserved for rotation/controlled-rotation/`rzz`/`rxx`/custom-unitary gates.

- **Two measurement mechanisms** (`aicir/measure/`): (1) call `Measure.run(..., measure_qubits=[...])` to pick read-out qubits explicitly; (2) embed `measure(*qubits)` gate markers in the `Circuit` and `Measure.run()` auto-detects them. The two are **mutually exclusive** — embedding `measure()` gates and also passing `measure_qubits` raises `ValueError`. `aicir.measure.estimator` provides shot-based Pauli-term energy estimation (qubit-wise commuting grouping, basis-change measurement, shot allocation).

- **Top-level re-exports** (`aicir/__init__.py`): gates, `Circuit`/`Parameter`, `Operation`, backends, `Measure`/`Result`, `PauliOp`/`PauliString`/`Hamiltonian`, noise channels, and QASM/JSON I/O are all importable from `aicir`. `StateVector`/`DensityMatrix` canonical path is `aicir.core` (also re-exported at top level).

### Subsystems (each has its own README — read it before working there)

- `aicir/vqc/` — `BasicVQE`/`run_vqe` orchestration; ansatz templates in `ansatz/` (`hea`, `hea_ti` for trapped-ion). Also QAOA/VQD/SSVQE.
- `aicir/qas/` — Quantum Architecture Search. Unified entry: `run(method, **kwargs)` + `config.<method>(...)`. Methods: `supernet` (weight-shared VQA ansatz search), `CRLQAS`, `PPR_DQL`, `PPO_RB` (RL-based). **Requires `torch`.**
- `aicir/qml/deriv.py` — gradient/preconditioning methods (`psr`, `spsr`, `multipsr`/`mpsr`, `fd`, `auto`, `ad`, `spsa`, `qng`, …). All backend-agnostic; return values may be raw backend tensors. VQE/SSVQE/VQD/VQA-QAS all funnel parameter-shift through `qml.deriv.psr` — keep that single source of truth.
- `aicir/chemistry/` — fixed preset qubit Hamiltonians only (`h2`, `h2_jw`, `h2_tapered`); NOT an electronic-structure pipeline. Only presets with confirmed coefficients are registered.
- `aicir/core/io/` — OpenQASM 2.0/3.0 + JSON round-trip. Multi-control `crx/cry/crz` auto-decompose (ancilla + `ccx` chains) **only under QASM 3.0**; `control_states=0` inserts surrounding `x` gates. Bind all symbolic params before QASM export.
- `aicir/channel/noise/` — `NoiseModel`/`NoiseChannel` (depolarizing, bit/phase flip, amplitude damping), ion-trap noise, metrics. Noise paths run through density-matrix simulation.
- `aicir/metrics/` — circuit expressibility, trainability, hardware metrics (incl. noisy variants).
- `aicir/visual/` — plotting (`Circuit.plot(...)`/`Circuit.show()`); needs `matplotlib`.
- `aicir/optimizer/` — two distinct concerns: `circuit.py` rewrites circuits (`optimize_circuit`/`optimize_basic`: rotation-gate merging, safe bounded reordering); `params.py` is the classical optimizer layer (`minimize`, `Adam`, `GD`, `SPSA`, `COBYLA`, `LBFGSB`, `ScipyMinimize`, `OptimizationResult`) that VQA loops drive.
- `aicir/optimization/` — `qubo/` provides QUBO modeling + Ising mapping and a QAOA adapter (`qubo.qaoa`); `sb/` is a placeholder for sample/subspace strategies. Distinct from `aicir/optimizer/` (don't confuse the two).
- `aicir/encoder/` — classical→quantum state-preparation encoders: `AmplitudeEncoder`, `AngleEncoder`, `BasisEncoder` (all subclass `BaseEncoder`).
- `aicir/universal/` — reusable circuit primitives, e.g. `qft.py` (QFT).
- `aicir/ir/` — typed `Operation` IR (`from_dict`/`to_dict`/`normalize_gate`) that interoperates with the dict gate surface. **This is the live edge of an in-progress architecture migration** (see `NEXT.md` for the target tree); the following siblings are intentional **placeholders** with no public API yet — fill them in as that work lands, don't assume they're functional: `aicir/gates/` (GateSpec registry), `aicir/devices/` (Target/hardware capability), `aicir/primitives/` (Sampler/Estimator), `aicir/transpile/` + `transpile/passes/` (pass-manager pipeline).

## Conventions

- Run everything from the repo root with `PYTHONPATH=.` (no installed package).
- Comments/docstrings/READMEs are Chinese — follow the surrounding style.
- Preset/config names use short canonical forms; old long aliases are intentionally not kept (see CHANGELOG).
- `CHANGELOG.md` is actively maintained with dated entries — add to it for notable interface changes.
