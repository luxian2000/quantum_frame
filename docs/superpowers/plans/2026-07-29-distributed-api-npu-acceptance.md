# Distributed API NPU Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strict, sectioned 2-/4-NPU probe that validates the complete forward-only `aicir.distributed` API and closes the explicit automatic-differentiation rejection contract.

**Architecture:** Keep `distributed_state_probe.py` as the stable smoke test. Add `distributed_api_probe.py` as a rank-synchronous acceptance runner with isolated sections, NumPy references, probe-local P2P counters, one rank-0 JSON report, and collective failure propagation. Production changes are limited to a shared autograd rejection contract and two NPU-safe density diagonal reads.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, torch_npu/HCCL, pytest, aicir typed IR and distributed API.

---

## File Map

- Create `aicir/distributed/_contracts.py`: shared forward-only autograd detection and error.
- Modify `aicir/distributed/state.py`: use shared contract for `DistState`.
- Modify `aicir/distributed/gates.py`: use shared contract for trainable gate matrices.
- Modify `aicir/distributed/simulator.py`: collectively reject trainable root inputs and gate payloads before execution.
- Modify `aicir/distributed/reducers.py`: avoid `aclnnIndex(DT_COMPLEX64)` in density expectation and collapse.
- Create `scripts/npu/distributed_api_probe.py`: complete sectioned hardware acceptance runner.
- Modify `tests/distributed/test_simulator_validation.py`: autograd and complex-index regression tests.
- Modify `tests/distributed/test_state.py`: shared autograd error contract.
- Modify `tests/distributed/test_probe_contract.py`: full-probe CLI/report/static contract.
- Modify `aicir/distributed/README.md`: launch commands, section meanings, hardware acceptance record.

### Task 1: Unified forward-only autograd rejection

**Files:**
- Create: `aicir/distributed/_contracts.py`
- Modify: `aicir/distributed/state.py:13-35`
- Modify: `aicir/distributed/gates.py:120-129`
- Modify: `aicir/distributed/simulator.py:309-350`
- Test: `tests/distributed/test_simulator_validation.py`
- Test: `tests/distributed/test_state.py`

- [ ] **Step 1: Write failing tests for root-owned trainable inputs and custom unitary**

Add tests:

```python
AUTOGRAD_ERROR = "DistSimulator 首期仅支持前向模拟，不支持自动微分"


@pytest.mark.parametrize("argument", ["initial_state", "initial_density_matrix"])
def test_rejects_root_owned_trainable_state_inputs(monkeypatch, argument):
    simulator = _simulator(monkeypatch)
    value = (
        torch.tensor([1.0, 0.0], dtype=torch.complex64, requires_grad=True)
        if argument == "initial_state"
        else torch.eye(2, dtype=torch.complex64, requires_grad=True)
    )

    with pytest.raises(ValueError, match=AUTOGRAD_ERROR):
        simulator.run(Circuit(n_qubits=1), **{argument: value})


def test_rejects_trainable_custom_unitary(monkeypatch):
    simulator = _simulator(monkeypatch)
    matrix = torch.eye(2, dtype=torch.complex64, requires_grad=True)

    with pytest.raises(ValueError, match=AUTOGRAD_ERROR):
        simulator.run(
            Circuit(
                {
                    "type": "unitary",
                    "parameter": matrix,
                    "n_qubits": 1,
                },
                n_qubits=1,
            )
        )
```

Change existing `test_rejects_trainable_gate_parameter` to match
`AUTOGRAD_ERROR`.

Change `test_dist_state_rejects_automatic_differentiation` in
`tests/distributed/test_state.py` to match `AUTOGRAD_ERROR`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=. pytest \
  tests/distributed/test_simulator_validation.py::test_rejects_root_owned_trainable_state_inputs \
  tests/distributed/test_simulator_validation.py::test_rejects_trainable_custom_unitary \
  tests/distributed/test_simulator_validation.py::test_rejects_trainable_gate_parameter -q
```

Expected: root-owned input cases fail because `DistSimulator` silently detaches;
existing gate errors fail message match.

- [ ] **Step 3: Add shared contract helper**

Create `aicir/distributed/_contracts.py`:

```python
"""Shared contracts for the forward-only distributed release."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


AUTOGRAD_ERROR = (
    "DistSimulator 首期仅支持前向模拟，不支持自动微分"
)


def contains_requires_grad(value) -> bool:
    if value is None:
        return False
    if bool(getattr(value, "requires_grad", False)):
        return True
    local_data = getattr(value, "local_data", None)
    if local_data is not None:
        return bool(getattr(local_data, "requires_grad", False))
    data = getattr(value, "data", None)
    if data is not None and not isinstance(value, (str, bytes)):
        return bool(getattr(data, "requires_grad", False))
    if isinstance(value, Mapping):
        return any(
            contains_requires_grad(item)
            for item in value.values()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(contains_requires_grad(item) for item in value)
    return False


def reject_requires_grad(value) -> None:
    if contains_requires_grad(value):
        raise ValueError(AUTOGRAD_ERROR)
```

- [ ] **Step 4: Route `DistState` and gate planning through shared error**

In `state.py`, import `reject_requires_grad`, then replace the current
`requires_grad` block with:

```python
reject_requires_grad(local_data)
```

In `gates.py`, import `reject_requires_grad`, then replace the current matrix
check with:

```python
reject_requires_grad(matrix)
```

- [ ] **Step 5: Add rank-synchronous rejection in `DistSimulator`**

Import:

```python
from ..ir import instruction_params
from ._contracts import AUTOGRAD_ERROR, contains_requires_grad
```

Add:

```python
def _assert_forward_only(
    self,
    circuit,
    initial_state,
    initial_density_matrix,
) -> None:
    local_rejected = any(
        (
            contains_requires_grad(initial_state),
            contains_requires_grad(initial_density_matrix),
            contains_requires_grad(
                tuple(
                    instruction_params(instruction)
                    for instruction in circuit_instructions(circuit)
                )
            ),
        )
    )
    flag = torch.tensor(
        int(local_rejected),
        dtype=torch.long,
        device=self._backend._device,
    )
    rejected_count = self._backend.communicator.all_reduce_sum(flag)
    if int(rejected_count.detach().cpu()) > 0:
        raise ValueError(AUTOGRAD_ERROR)
```

Call it at the start of `run`, before `_preflight`:

```python
self._assert_forward_only(
    circuit,
    initial_state,
    initial_density_matrix,
)
```

- [ ] **Step 6: Run tests and verify GREEN**

Run:

```bash
PYTHONPATH=. pytest tests/distributed/test_simulator_validation.py \
  tests/distributed/test_state.py tests/distributed/test_gate_planner.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add aicir/distributed/_contracts.py \
  aicir/distributed/state.py aicir/distributed/gates.py \
  aicir/distributed/simulator.py \
  tests/distributed/test_simulator_validation.py \
  tests/distributed/test_state.py
git commit -m "fix(distributed): reject autograd inputs consistently"
```

### Task 2: Remove remaining complex advanced indexing from density reducers

**Files:**
- Modify: `aicir/distributed/reducers.py:137-158,253-262`
- Test: `tests/distributed/test_simulator_validation.py`

- [ ] **Step 1: Add CPU emulation of Ascend complex-index restriction**

Add:

```python
from torch.utils._python_dispatch import TorchDispatchMode


class _RejectComplexIndex(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.index.Tensor and args[0].is_complex():
            raise RuntimeError("complex indexing is not supported")
        return func(*args, **(kwargs or {}))
```

Add tests:

```python
def test_density_expectation_does_not_index_complex_tensor(monkeypatch):
    simulator = _simulator(monkeypatch)
    rho = np.array(
        [[0.75, 0.0], [0.0, 0.25]],
        dtype=np.complex64,
    )
    observable = Observable.matrix(
        np.array([[1, 0], [0, -1]], dtype=np.complex64),
        metadata={"qubits": [0]},
    )

    with _RejectComplexIndex():
        result = simulator.run(
            Circuit(n_qubits=1),
            initial_density_matrix=rho,
            observables={"z": observable},
        )

    assert abs(result.expectations["z"] - 0.5) < 1e-6


def test_density_collapse_does_not_index_complex_tensor(monkeypatch):
    simulator = _simulator(monkeypatch)
    rho = np.diag([0.75, 0.25]).astype(np.complex64)

    with _RejectComplexIndex():
        result = simulator.run(
            Circuit(n_qubits=1),
            initial_density_matrix=rho,
            shots=1,
            collapse=True,
            seed=7,
        )

    assert sum(result.counts.values()) == 1
    assert abs(np.trace(result.state.to_numpy()).real - 1.0) < 1e-6
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=. pytest \
  tests/distributed/test_simulator_validation.py::test_density_expectation_does_not_index_complex_tensor \
  tests/distributed/test_simulator_validation.py::test_density_collapse_does_not_index_complex_tensor -q
```

Expected: both fail at `aten.index.Tensor` on complex input.

- [ ] **Step 3: Split real and imaginary diagonal reads**

Add to `reducers.py`:

```python
def _complex_diagonal_sum(matrix, rows, columns):
    real = torch.real(matrix)[rows, columns].sum()
    imag = torch.imag(matrix)[rows, columns].sum()
    return torch.complex(real, imag)
```

Replace density expectation diagonal read with:

```python
local = _complex_diagonal_sum(product, rows, columns)
```

Replace density-collapse trace read with a real trace:

```python
diagonal = torch.real(data)[
    rows,
    rows + state.spec.global_start,
].sum()
trace = self._backend.communicator.all_reduce_sum(
    diagonal.reshape(())
)
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
PYTHONPATH=. pytest tests/distributed/test_simulator_validation.py \
  tests/distributed/test_reducers_multiprocess.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aicir/distributed/reducers.py \
  tests/distributed/test_simulator_validation.py
git commit -m "fix(distributed): make density reductions NPU safe"
```

### Task 3: Add probe CLI, section registry, and report contract

**Files:**
- Create: `scripts/npu/distributed_api_probe.py`
- Modify: `tests/distributed/test_probe_contract.py`

- [ ] **Step 1: Write failing probe contract test**

Add:

```python
API_PROBE = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "npu"
    / "distributed_api_probe.py"
)

EXPECTED_SECTIONS = {
    "state",
    "layout",
    "continuation",
    "noise",
    "observable",
    "measure",
    "result",
    "communication",
    "contract",
}


def test_full_api_probe_has_sectioned_strict_contract():
    source = API_PROBE.read_text(encoding="utf-8")

    assert "fallback_to_cpu=False" in source
    assert "failed_invariants" in source
    assert "EXPECTED_SECTIONS" in source
    assert '"communicating_gate": True' not in source
    for section in EXPECTED_SECTIONS:
        assert f'"{section}"' in source


def test_full_api_probe_launch_preserves_cann_pythonpath():
    source = API_PROBE.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith("PYTHONPATH=")
        and "torchrun" in line
    ]
    assert lines
    assert all(
        line.startswith("PYTHONPATH=.:${PYTHONPATH} torchrun")
        for line in lines
    )
```

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
PYTHONPATH=. pytest tests/distributed/test_probe_contract.py -q
```

Expected: FAIL because `distributed_api_probe.py` does not exist.

- [ ] **Step 3: Create strict sectioned scaffold**

Create `scripts/npu/distributed_api_probe.py` with:

```python
"""Strict full-API acceptance probe for 2 or 4 Ascend NPUs.

Run from repository root:

    source /usr/local/Ascend/cann/set_env.sh
    PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=2 scripts/npu/distributed_api_probe.py --section all
    PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=4 scripts/npu/distributed_api_probe.py --section all
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import torch

from aicir.distributed import DistSimulator


STATE_ATOL = 1e-6
REDUCTION_ATOL = 1e-5
EXPECTED_SECTIONS = (
    "state",
    "layout",
    "continuation",
    "noise",
    "observable",
    "measure",
    "result",
    "communication",
    "contract",
)


def _section(passed, *, metrics=None, status=None):
    return {
        "passed": bool(passed),
        "status": status or ("PASS" if passed else "FAIL"),
        "metrics": dict(metrics or {}),
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section",
        choices=("all", *EXPECTED_SECTIONS),
        default="all",
    )
    return parser.parse_args()


def _validate_runtime(backend):
    device = backend._device
    if backend.world_size not in {2, 4}:
        raise ValueError("完整 API 探针只接受 world_size=2 或 world_size=4")
    if device.type != "npu":
        raise RuntimeError(f"严格探针要求 NPU，实际设备为 {device}")
    if backend.rank != backend.local_rank or device.index != backend.local_rank:
        raise RuntimeError("rank、local_rank 与 NPU device 不一致")


def _run_selected(simulator, selected):
    sections = {}
    names = EXPECTED_SECTIONS if selected == "all" else (selected,)
    for name in names:
        sections[name] = SECTION_RUNNERS[name](simulator)
    failed = [
        name for name, result in sections.items()
        if not result["passed"]
    ]
    return sections, failed


def _run_probe(selected):
    simulator = DistSimulator.from_env(fallback_to_cpu=False)
    backend = simulator.backend
    _validate_runtime(backend)
    sections, failed = _run_selected(simulator, selected)
    passed_count = torch.tensor(
        [int(not failed)],
        dtype=torch.long,
        device=backend._device,
    )
    passed_count = backend.communicator.all_reduce_sum(passed_count)
    passed = int(passed_count[0].detach().cpu()) == backend.world_size
    report = None
    if backend.rank == 0:
        report = {
            "passed": passed,
            "world_size": backend.world_size,
            "fallback_to_cpu": False,
            "sections": sections,
            "failed_invariants": failed,
        }
        print(json.dumps(report, sort_keys=True))
    return passed


def main():
    args = _parse_args()
    try:
        ok = _run_probe(args.section)
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

Before `main`, define:

```python
def _pending_section(_simulator):
    return _section(False, status="NOT_IMPLEMENTED")


SECTION_RUNNERS = {
    name: _pending_section
    for name in EXPECTED_SECTIONS
}
```

Later tasks replace entries one section at a time.

- [ ] **Step 4: Run contract test and verify GREEN**

Run:

```bash
PYTHONPATH=. pytest tests/distributed/test_probe_contract.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/npu/distributed_api_probe.py \
  tests/distributed/test_probe_contract.py
git commit -m "test(distributed): scaffold full NPU API probe"
```

### Task 4: Implement state, layout, and continuation sections

**Files:**
- Modify: `scripts/npu/distributed_api_probe.py`
- Test: `tests/distributed/test_probe_contract.py`

- [ ] **Step 1: Extend contract test with required evidence keys**

Add assertions:

```python
for token in (
    "initial_density_matrix",
    "logical_to_storage",
    "continuation_vector_error",
    "continuation_density_error",
    "local_tensor_sizes",
):
    assert token in source
```

- [ ] **Step 2: Run contract test and verify RED**

Run:

```bash
PYTHONPATH=. pytest tests/distributed/test_probe_contract.py -q
```

Expected: FAIL on missing evidence keys.

- [ ] **Step 3: Add common gather and metric helpers**

Add:

```python
def _root_value(backend, value):
    return value if backend.rank == 0 else None


def _gather_array(state):
    value = state.to_numpy(root=0)
    return None if value is None else np.asarray(value)


def _max_error(actual, expected):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def _root_scalars(backend, values):
    tensor = torch.tensor(
        values,
        dtype=torch.long,
        device=backend._device,
    )
    gathered = backend.communicator.gather_to_root(tensor, root=0)
    if backend.rank != 0:
        return None
    return [item.detach().cpu().tolist() for item in gathered]
```

- [ ] **Step 4: Implement `state`, `layout`, and `continuation`**

Use `n_qubits = log2(world_size) + 1`. Build a normalized vector with nonzero
real and imaginary entries and a positive semidefinite density matrix
`rho = |psi><psi|`. Root rank supplies full inputs; other ranks pass `None`.

For layout, use:

```python
logical_to_storage = tuple(
    list(range(1, n_qubits)) + [0]
)
layout = logical_to_storage
```

Run identical circuits under automatic and explicit layouts. Compare gathered
logical-order state, probabilities, and Pauli expectations.

For continuation, run prefix and suffix with:

```python
prefix = Circuit(
    hadamard(0),
    cx(target_qubit=1, control_qubits=(0,)),
    n_qubits=n_qubits,
)
suffix = Circuit(
    pauli_z(n_qubits - 1),
    pauli_x(0),
    n_qubits=n_qubits,
)
combined_circuit = Circuit(
    *prefix.gates,
    *suffix.gates,
    n_qubits=n_qubits,
)
prefix_result = simulator.run(prefix, layout=layout)
continued = simulator.run(
    suffix,
    initial_state=prefix_result.state,
    layout=layout,
)
combined = simulator.run(
    combined_circuit,
    layout=layout,
)
density_seed = None
if backend.rank == 0:
    density_seed = np.zeros(
        (1 << n_qubits, 1 << n_qubits),
        dtype=np.complex64,
    )
    density_seed[0, 0] = 1.0
prefix_density = simulator.run(
    prefix,
    initial_density_matrix=density_seed,
    layout=layout,
)
continued_dm = simulator.run(
    suffix,
    initial_density_matrix=prefix_density.state,
    layout=layout,
)
combined_dm = simulator.run(
    combined_circuit,
    initial_density_matrix=density_seed,
    layout=layout,
)
continued_vector = _gather_array(continued.state)
combined_vector = _gather_array(combined.state)
continued_density = _gather_array(continued_dm.state)
combined_density = _gather_array(combined_dm.state)
local_tensor_sizes = _root_scalars(
    backend,
    [
        continued.state.local_data.numel(),
        continued_dm.state.local_data.numel(),
    ],
)
metrics = None
if backend.rank == 0:
    metrics = {
        "continuation_vector_error": _max_error(
            continued_vector,
            combined_vector,
        ),
        "continuation_density_error": _max_error(
            continued_density,
            combined_density,
        ),
        "logical_to_storage": list(logical_to_storage),
        "local_tensor_sizes": local_tensor_sizes,
    }
```

Every non-root rank returns the same section pass bit received through
`backend.communicator.broadcast`.

- [ ] **Step 5: Run local contract and distributed regression tests**

Run:

```bash
PYTHONPATH=. pytest tests/distributed/test_probe_contract.py \
  tests/distributed/test_state_multiprocess.py \
  tests/distributed/test_simulator_multiprocess.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/npu/distributed_api_probe.py \
  tests/distributed/test_probe_contract.py
git commit -m "test(distributed): probe state layout and continuation"
```

### Task 5: Implement noise and observable sections

**Files:**
- Modify: `scripts/npu/distributed_api_probe.py`
- Test: `tests/distributed/test_probe_contract.py`

- [ ] **Step 1: Add failing evidence assertions**

Require:

```python
for token in (
    "BitFlipChannel",
    "PhaseFlipChannel",
    "DepolarizingChannel",
    "Hamiltonian",
    "Observable.matrix",
    "noise_density_error",
    "hamiltonian_error",
    "local_dense_error",
):
    assert token in source
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
PYTHONPATH=. pytest tests/distributed/test_probe_contract.py -q
```

Expected: FAIL on missing noise/observable evidence.

- [ ] **Step 3: Implement NumPy Kraus reference**

Add:

```python
from aicir.backends.numpy_backend import NumpyBackend


def _apply_kraus_reference(rho, channel, n_qubits):
    backend = NumpyBackend()
    output = np.zeros_like(rho, dtype=np.complex64)
    for operator in channel.kraus_operators(n_qubits, backend):
        matrix = np.asarray(backend.to_numpy(operator))
        output += matrix @ rho @ matrix.conj().T
    return output
```

- [ ] **Step 4: Implement noise case**

Use one gate and a `NoiseModel` containing:

```python
channels = (
    BitFlipChannel(target_qubit=0, p=0.2),
    PhaseFlipChannel(target_qubit=0, p=0.3),
    DepolarizingChannel(target_qubit=0, p=0.15),
)
noise_model = NoiseModel()
for channel in channels:
    noise_model.add_channel(channel, after_gates=("x",))
```

Calculate reference by applying X to the root density matrix, then applying
channels in rule order with `_apply_kraus_reference`. Compare full density,
trace, and probabilities.

- [ ] **Step 5: Implement observable case**

Build:

```python
hamiltonian = Hamiltonian(
    n_qubits=n_qubits,
    terms=[
        ("Z" + "I" * (n_qubits - 1), 0.4),
        ("X" + "I" * (n_qubits - 1), -0.2),
    ],
)
local_dense = Observable.matrix(
    np.array([[0, 1], [1, 0]], dtype=np.complex64),
    metadata={"qubits": [n_qubits - 1]},
)
```

Evaluate PauliString, Hamiltonian, and dense observable in one simulator call.
On rank 0 build full matrices with `NumpyBackend`, calculate
`np.vdot(psi, operator @ psi)`, and record each absolute error.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
PYTHONPATH=. pytest tests/distributed/test_probe_contract.py \
  tests/distributed/test_noise_multiprocess.py \
  tests/distributed/test_reducers_multiprocess.py -q
```

Expected: PASS.

Commit:

```bash
git add scripts/npu/distributed_api_probe.py \
  tests/distributed/test_probe_contract.py
git commit -m "test(distributed): probe noise and observables"
```

### Task 6: Implement measurement, collapse, and result sections

**Files:**
- Modify: `scripts/npu/distributed_api_probe.py`
- Test: `tests/distributed/test_probe_contract.py`

- [ ] **Step 1: Add failing evidence assertions**

Require:

```python
for token in (
    "collapse=True",
    "measure_qubits",
    "collapsed_support_error",
    "return_state",
    "return_probabilities",
    "four_return_combinations",
):
    assert token in source
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
PYTHONPATH=. pytest tests/distributed/test_probe_contract.py -q
```

Expected: FAIL on missing measure/result evidence.

- [ ] **Step 3: Implement measurement section**

Prepare a GHZ state. Run full and subset sampling with fixed seeds. For collapse:

```python
collapsed_result = simulator.run(
    ghz_circuit,
    shots=1,
    measure_qubits=(0,),
    collapse=True,
    seed=37,
)
```

Gather collapsed state and probabilities. Extract measured bit from the single
count key. Sum probability outside basis states whose logical qubit 0 differs
from that bit. Record this as `collapsed_support_error`; also record norm error
and total-shot checks.

- [ ] **Step 4: Implement result section**

Run all combinations:

```python
combinations = (
    (True, True),
    (True, False),
    (False, True),
    (False, False),
)
```

For each result, assert:

```python
(result.state is not None) == return_state
(result.local_probabilities is not None) == return_probabilities
```

Call `gather_probabilities(root=0)` only when probabilities are enabled.
Record `"four_return_combinations": True` only after all four checks.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
PYTHONPATH=. pytest tests/distributed/test_probe_contract.py \
  tests/distributed/test_reducers_multiprocess.py \
  tests/distributed/test_result.py \
  tests/distributed/test_simulator_validation.py -q
```

Expected: PASS.

Commit:

```bash
git add scripts/npu/distributed_api_probe.py \
  tests/distributed/test_probe_contract.py
git commit -m "test(distributed): probe collapse and result flags"
```

### Task 7: Implement P2P evidence and error-contract sections

**Files:**
- Modify: `scripts/npu/distributed_api_probe.py`
- Test: `tests/distributed/test_probe_contract.py`

- [ ] **Step 1: Add failing assertions against literal communication claims**

Require:

```python
for token in (
    "local_p2p_delta",
    "distributed_p2p_delta",
    "peer_mask",
    "paired_transport_tags",
    "UNSUPPORTED_AS_DESIGNED",
    "EXPECTED_ERROR",
):
    assert token in source
assert '"local_gate": True' not in source
assert '"communicating_gate": True' not in source
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
PYTHONPATH=. pytest tests/distributed/test_probe_contract.py -q
```

Expected: FAIL on missing measured communication evidence.

- [ ] **Step 3: Implement probe-local exchange recorder**

Add:

```python
class _ExchangeRecorder:
    def __init__(self, communicator):
        self.communicator = communicator
        self.original = communicator._exchange_tensor
        self.records = []

    def install(self):
        def counted(tensor, peer, tag):
            self.records.append((int(peer), int(tag)))
            return self.original(tensor, peer, tag)
        self.communicator._exchange_tensor = counted

    def restore(self):
        self.communicator._exchange_tensor = self.original

    def snapshot(self):
        return len(self.records)

    def delta(self, start):
        return self.records[int(start):]
```

Install with `try/finally`.

- [ ] **Step 4: Measure local and distributed gates**

Use identity layout with:

```python
distributed_axes = int(math.log2(backend.world_size))
n_qubits = distributed_axes + 1
local_qubit = n_qubits - 1
```

Run X on `local_qubit`; record zero delta. Then run X separately on every
logical qubit in `range(distributed_axes)`; record positive deltas and peer
union. Encode fixed-size per-rank evidence:

```python
[
    local_p2p_delta,
    distributed_p2p_delta,
    peer_mask,
    even_tag_count,
    odd_tag_count,
]
```

Gather through `_root_scalars`. Acceptance requires zero local delta, positive
distributed delta, no self peer, at least `distributed_axes` distinct peers per
rank, and equal even/odd tag counts.

- [ ] **Step 5: Implement expected-error runner**

Add:

```python
def _expected_error(call, match):
    try:
        call()
    except Exception as error:
        return match in str(error)
    return False
```

Run invalid layout, invalid root shape, inconsistent rank input mode,
mid-circuit measurement, reset, `if_`, `while_`, and trainable
state/gate/custom-unitary inputs. Construct control flow with:

```python
register = ClassicalRegister(1, "probe")
unsupported_circuits = (
    Circuit(measure(0), n_qubits=n_qubits),
    Circuit(reset(0), n_qubits=n_qubits),
    Circuit(
        if_(
            register[0] == 1,
            Circuit(pauli_x(0), n_qubits=n_qubits),
        ),
        n_qubits=n_qubits,
    ),
    Circuit(
        while_(
            register[0] == 1,
            Circuit(pauli_x(0), n_qubits=n_qubits),
            max_iterations=1,
        ),
        n_qubits=n_qubits,
    ),
)
```

Every rank participates in each collective case. Report autograd as
`UNSUPPORTED_AS_DESIGNED`; other expected failures as `EXPECTED_ERROR`.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
PYTHONPATH=. pytest tests/distributed/test_probe_contract.py \
  tests/distributed/test_communication.py \
  tests/distributed/test_simulator_validation.py -q
```

Expected: PASS.

Commit:

```bash
git add scripts/npu/distributed_api_probe.py \
  tests/distributed/test_probe_contract.py
git commit -m "test(distributed): measure P2P and error contracts"
```

### Task 8: Documentation and local verification

**Files:**
- Modify: `aicir/distributed/README.md`
- Test: `tests/distributed/test_probe_contract.py`

- [ ] **Step 1: Document full probe**

Add a hardware-acceptance subsection containing:

```bash
source /usr/local/Ascend/cann/set_env.sh
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=2 \
  scripts/npu/distributed_api_probe.py --section all
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=4 \
  scripts/npu/distributed_api_probe.py --section all
```

Document every section, JSON status meanings, strict no-fallback rule, and that
NumPy references do not constitute simulator fallback.

- [ ] **Step 2: Run focused suite**

```bash
PYTHONPATH=. pytest tests/distributed -q
```

Expected: all distributed tests pass.

- [ ] **Step 3: Run repository-wide suite**

```bash
PYTHONPATH=. pytest -q
```

Expected: exit code 0; existing warnings allowed.

- [ ] **Step 4: Check source and documentation**

```bash
python -m py_compile scripts/npu/distributed_api_probe.py
git diff --check
```

Expected: both exit 0.

- [ ] **Step 5: Commit**

```bash
git add aicir/distributed/README.md \
  scripts/npu/distributed_api_probe.py \
  tests/distributed/test_probe_contract.py
git commit -m "docs(distributed): add full NPU acceptance probe"
```

### Task 9: Real 2-/4-NPU acceptance and future-plan handoff

**Files:**
- Modify after hardware evidence: `aicir/distributed/README.md`
- Create after both runs: `docs/superpowers/plans/2026-07-29-distributed-future-roadmap.md`

- [ ] **Step 1: Run 2-NPU complete acceptance**

```bash
source /usr/local/Ascend/cann/set_env.sh
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=2 \
  scripts/npu/distributed_api_probe.py --section all
```

Expected: rank-0 JSON has `"passed": true`, every section has
`"status": "PASS"`, and `"failed_invariants": []`.

- [ ] **Step 2: Run 4-NPU complete acceptance**

```bash
PYTHONPATH=.:${PYTHONPATH} torchrun --nproc-per-node=4 \
  scripts/npu/distributed_api_probe.py --section all
```

Expected: same pass contract; communication section reports at least two
distinct peers per rank.

- [ ] **Step 3: Record exact evidence**

Update the manual with CANN/PyTorch/torch_npu versions, device count,
world sizes, section results, numerical maxima, and known unsupported scope.
Do not claim unmeasured performance or scalability.

- [ ] **Step 4: Write future distributed roadmap**

Create a separate plan covering:

1. distributed parameter-shift gradients;
2. native distributed autograd only after NPU-safe backward kernels exist;
3. communication/computation overlap and buffer reuse;
4. multi-node HCCL topology and launch contracts;
5. checkpoint/restart and collective failure handling;
6. larger world sizes and state-size stress tests;
7. deterministic distributed RNG;
8. performance and memory benchmarks against single-NPU and replicated modes.

Base priorities on actual failed or high-cost probe evidence. Keep automatic
differentiation, performance, multi-node, and fault tolerance as separate
implementation slices.

- [ ] **Step 5: Final verification and commit**

```bash
PYTHONPATH=. pytest tests/distributed -q
git diff --check
git add aicir/distributed/README.md \
  docs/superpowers/plans/2026-07-29-distributed-future-roadmap.md
git commit -m "docs(distributed): record NPU acceptance and roadmap"
```

Expected: tests and diff check pass.
