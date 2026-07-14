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
scripts/npu/typed_ir_deriv_probe.sh --section all # strict typed-IR/deriv hardware probe
scripts/npu/typed_ir.sh --strict-npu --pytest-arg -q
scripts/npu/deriv.sh --strict-npu --pytest-arg -q
scripts/npu/run_all.sh --strict-npu --fail-fast --pytest-arg -q
scripts/npu/multi_card.sh --nproc-per-node 4 --section all
scripts/npu/qnn_4card.sh --nproc-per-node 4
```

`torch`, `matplotlib`, and `scipy` are **optional**. Tests that need them call `pytest.importorskip(...)` and skip cleanly when absent — don't add hard imports of these to core modules.

## Architecture

Everything is built on a backend abstraction so upper layers (`StateVector`, `Circuit`, `Measure`, VQE, QAS…) never touch the underlying numeric framework.

- **Backends** (`aicir/backends/`): all implement the abstract `Backend` (`base.py`). `NumpyBackend` (default, CPU), `GPUBackend` (Torch, autograd-capable; `TorchBackend` is a deprecated alias), `NPUBackend` (Ascend). Conventions: state vector shape `(2^n, 1)`, density matrix `(2^n, 2^n)`, complex dtype; methods return native tensors, `to_numpy()` is the single conversion exit. Ascend note: `NPUBackend` uses `torch.complex64` tensors, but `torch_npu` does **not** support every complex64 operator/backward path. Avoid raw torch complex matmul/trace/fan-out gradient accumulation on real NPU; failures include `aclnnMatmul` or `aclnnInplaceAdd` for `DT_COMPLEX64`. Route NPU code through `NPUBackend` methods/custom autograd (real/imag decompositions such as `matmul`, `expectation_sv`, local-gate apply, Hamiltonian gradients). Generic full-matrix complex autograd is a CPU/fallback contract; real-NPU deriv is covered by `scripts/npu/typed_ir_deriv_probe.sh --section deriv` and `scripts/npu/deriv.sh --strict-npu`. To add a backend, subclass `Backend` and implement all abstract methods — nothing else should need changing.

  Ascend NPU usage rules to remember:
  - Single-card validation should use the strict NPU scripts, especially `scripts/npu/typed_ir_deriv_probe.sh --section all`, `scripts/npu/typed_ir.sh --strict-npu --pytest-arg -q`, `scripts/npu/deriv.sh --strict-npu --pytest-arg -q`, and the full sweep `scripts/npu/run_all.sh --strict-npu --fail-fast --pytest-arg -q`. Use `--allow-cpu-fallback` only for local development, never as evidence of real NPU correctness.
  - Multi-card validation follows `demos/BeH2/BeH2_npu.py`: launch with `torchrun`/`python -m torch.distributed.run`, let `LOCAL_RANK=0..N-1` map to `npu:{LOCAL_RANK}` through `NPUBackend.from_distributed_env(...)`, and require HCCL in strict real-NPU runs.
  - Do **not** set `ASCEND_RT_VISIBLE_DEVICES=0,5,6,7` for the current multi-card probes/demos. On the verified Ascend runtime this made `torch.npu.set_device(...)` reject both logical ids (`npu:1/2/3`) and physical ids (`npu:5/6/7`). The `--devices` option in `scripts/npu/multi_card.sh` and `scripts/npu/qnn_4card.sh` is compatibility-only and intentionally ignored.
  - Current multi-NPU support is task/data parallel, not single-statevector sharding. Each rank keeps a full circuit/statevector on its local NPU. Cross-rank communication uses Python objects and real tensors (`broadcast_parameters`, `all_reduce_mean`); do not introduce complex dtype collectives.
  - Verified real-hardware baselines: single-card strict sweep passed through `scripts/npu/run_all.sh`; 4-card HCCL passed `scripts/npu/multi_card.sh --nproc-per-node 4 --section all`; 4-card typed-IR QNN training passed `scripts/npu/qnn_4card.sh --nproc-per-node 4` and `scripts/npu/qnn_4card.sh --nproc-per-node 4 --steps 24 --samples 64`; measure 多 shot 轻量聚合（`return_state=False` 跳过密度矩阵）passed `scripts/npu/measure_agg.sh --n-qubits 24`（3/3 cases，capacity 档 633s——耗时主项是逐 shot 末端全态投影 O(shots·n·2^n)，非聚合）.

- **Gates are typed instructions at runtime**. Gate factories (`pauli_x`, `cnot`, `rx`, `rzz`, `rxx`/`ms_gate`, `toffoli`, `u3`, …) and `Circuit.__init__`/`append`/`extend` normalize through `aicir.ir.as_instruction(...)`; `Circuit.gates`, iteration, and `Circuit.operations` expose typed `Operation`/measurement/control-flow objects. Controlled gates take `(target, control_qubits_list)` and may carry `control_states` (0/1). Dicts are retained only as JSON/QASM/legacy interop via `Circuit.legacy_gates`, `Circuit.to_gate_dicts()`, `Operation.to_dict()`, and `aicir.ir.instruction_to_gate_dict(...)`. Runtime code should use instruction helpers (`instruction_name`, `instruction_qubits`, `instruction_controls`, `instruction_parameter`, etc.) instead of `gate["..."]` / `gate.get(...)`.

- **Circuit** (`aicir/core/circuit.py`): container of typed instructions plus `n_qubits`. `Parameter` is a *symbolic* placeholder (not an autograd tensor). Template circuits store typed instructions until `bind_parameters(...)` (returns a new circuit by default; `inplace=True`, `allow_partial=True` available). `unitary()` errors on unbound parameters. For Torch autograd, pass Torch scalar tensors directly as gate params and use a `GPUBackend`/`NPUBackend` path that avoids unsupported raw complex64 NPU ops — the computation graph is preserved for rotation/controlled-rotation/`rzz`/`rxx`/custom-unitary gates when routed through backend-safe operations.

- **Two measurement mechanisms** (`aicir/measure/`): (1) call `Measure.run(..., measure_qubits=[...])` to pick read-out qubits explicitly; (2) embed `measure(*qubits)` gate markers in the `Circuit` and `Measure.run()` auto-detects them. The two are **mutually exclusive** — embedding `measure()` gates and also passing `measure_qubits` raises `ValueError`. `aicir.measure.estimator` provides shot-based Pauli-term energy estimation (qubit-wise commuting grouping, basis-change measurement, shot allocation).

- **Classical control flow** (`aicir/core/classical.py`, `aicir.core.circuit.if_`/`while_`): `ClassicalRegister`/`Bit`/`Condition` model measurement-fed classical state (`reg[i] == v` / `reg == N`); `measure(qubits, creg=reg)`/`measure(qubits, cbits=[...])` write Z-basis outcomes into a register. `if_(condition, body, else_body=None)` and `while_(condition, body, *, max_iterations)` (required, `RuntimeError` on overrun) produce `ControlFlow` IR nodes (`aicir/ir/control_flow.py`). Control flow only runs on the `Measure.run` measurement-trajectory path (per-trajectory condition evaluation/recursive execution); `Circuit.unitary()` and the tensor-network engine (`aicir/simulator`) reject `ControlFlow` with `ValueError`. `Result.classical_counts(reg)` reads the per-trajectory classical register distribution.

- **Top-level re-exports** (`aicir/__init__.py`): gates, `Circuit`/`Parameter`, `Operation`, backends, `Measure`/`Result`, `PauliOp`/`PauliString`/`Hamiltonian`, noise channels, and QASM/JSON I/O are all importable from `aicir`. `StateVector`/`DensityMatrix` canonical path is `aicir.core` (also re-exported at top level).

### Subsystems (each has its own README — read it before working there)

- `aicir/vqc/` — `BasicVQE`/`run_vqe` orchestration. Also QAOA/VQD/SSVQE.
- `aicir/ansatze/` — parameterized circuit templates: `hea` / `hardware_efficient_ansatz` (local rotations + entangler topology), `hea_ti` for trapped-ion (per-layer RxRyRx + global TFIM/XY evolution), `uccsd` for chemistry (pure-data coupled fermion excitations, decoupled from `aicir.chemistry`, fSWAP network for non-adjacent pairs). All return `Circuit` with symbolic `Parameter`s.
- `aicir/qas/` — Quantum Architecture Search. Unified entry: `run(method, **kwargs)` + `config.<method>(...)`. Methods: `supernet` (weight-shared VQA ansatz search), `CRLQAS`, `PPR_DQL`, `PPO_RB` (RL-based). **Requires `torch`.**
- `aicir/qml/deriv.py` — gradient/preconditioning methods (`psr`, `spsr`, `multipsr`/`mpsr`, `fd`, `auto`, `ad`, `spsa`, `qng`, …). All backend-agnostic; return values may be raw backend tensors. VQE/SSVQE/VQD/VQA-QAS all funnel parameter-shift through `qml.deriv.psr` — keep that single source of truth.
- `aicir/chemistry/` — fixed preset qubit Hamiltonians, one module per molecule under `chemistry/molecules/`, **filenames in formula case** (`H2.py`/`LiH.py`/`H2O.py`/`NH3.py`/`N2.py`/`BeH2.py`; canonical names lowercase). Presets: `h2`/`h2_jw`/`h2_tapered`, `lih`, `h2o` (≤6q, ground-energy tested) and `nh3`/`n2`/`beh2` (12–16q, structural-guard only — dense diagonalization infeasible). Each self-registers into `MOLECULES` via `register_molecule`; coefficients come from PySCF/Qiskit Nature. Also has an optional `build_molecule` current-computation pipeline (`chem` extra: `qiskit-nature`+`pyscf`) for arbitrary molecules — Jordan-Wigner mapping fills `hf_occupation`/`excitations` metadata that bridges to `aicir.ansatze.uccsd`. Public API unchanged (`get_molecule`/`molecule_hamiltonian`/`molecule_matrix`/… from `aicir.chemistry`).
- `aicir/core/io/` — OpenQASM 2.0/3.0 + JSON round-trip. Multi-control `crx/cry/crz` auto-decompose (ancilla + `ccx` chains) **only under QASM 3.0**; `control_states=0` inserts surrounding `x` gates. Bind all symbolic params before QASM export.
- `aicir/noise/` — `NoiseModel`/`NoiseChannel` (depolarizing, bit/phase flip, amplitude damping), ion-trap noise, metrics. Noise paths run through density-matrix simulation.
- `aicir/metrics/` — circuit expressibility, trainability, hardware metrics (incl. noisy variants).
- `aicir/visual/` — plotting (`Circuit.plot(...)`/`Circuit.show()`); needs `matplotlib`.
- `aicir/optimizer/` — **classical parameter optimizer only**: `params.py` (`minimize`, `Adam`, `GD`, `SPSA`, `COBYLA`, `LBFGSB`, `ScipyMinimize`, `OptimizationResult`) that VQA loops drive. Circuit *structure* rewriting moved to `aicir/transpile/` (`optimizer/circuit.py` was removed).
- `aicir/optimization/` — `qubo/` provides QUBO modeling + Ising mapping and a QAOA adapter (`qubo.qaoa`); `sb/` is a placeholder for sample/subspace strategies. Distinct from `aicir/optimizer/` (don't confuse the two).
- `aicir/encoder/` — classical→quantum state-preparation encoders: `AmplitudeEncoder`, `AngleEncoder`, `BasisEncoder` (all subclass `BaseEncoder`).
- `aicir/universal/` — reusable circuit primitives, e.g. `qft.py` (QFT).
- `aicir/qrc/` — Quantum Reservoir Computing. **Placeholder skeleton** (package entry + README only, no public API yet); fixed quantum-dynamics reservoir + trained classical readout for time-series tasks. Don't assume it's functional.
- `aicir/ir/` — typed `Operation` IR (`from_dict`/`to_dict`/`normalize_gate`) that interoperates with the dict gate surface. **This is the live edge of an in-progress architecture migration** (see `NEXT.md` for the target tree); the following siblings are intentional **placeholders** with no public API yet — fill them in as that work lands, don't assume they're functional: `aicir/gates/` (GateSpec registry), `aicir/devices/` (Target/hardware capability). `aicir/primitives/` is now **functional** (Sampler/Estimator: `BaseSampler`/`BaseEstimator`, `Statevector`/`Shot`/`Noisy` `Sampler`+`Estimator`, `BackendSampler`/`BackendEstimator` injected-runner extension point, `SampleResult`/`EstimateResult`; `run(...)` takes `parameter_values=` for late template binding; `Shot`/`NoisyEstimator` expose `estimate(...)` so they inject into `BasicVQE(energy_estimator=...)`). `aicir/transpile/` + `transpile/passes/` are **functional** (pass-manager pipeline + `optimize`/`optimize_basic`/`optimize_circuit`; circuit-rewrite consolidated here from the old `aicir/optimizer/circuit.py`).

## Conventions

- Never add `Co-Authored-By: Claude ...` (or any Anthropic/Claude attribution) trailers to commit messages in this repo — the user wants no AI attribution appearing in commit history or the GitHub Contributors graph.
- Run everything from the repo root with `PYTHONPATH=.` (no installed package).
- Comments/docstrings/READMEs are Chinese — follow the surrounding style.
- Preset/config names use short canonical forms; old long aliases are intentionally not kept (see CHANGELOG).
- `CHANGELOG.md` is actively maintained with dated entries — add to it for notable interface changes.
