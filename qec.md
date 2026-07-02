# aicir Quantum Error Correction Module Design

Date: 2026-07-02
Status: design only; implementation deferred

## Goal

Add a future `aicir.qec` package whose primary aim is **high-throughput syndrome
loop simulation on Ascend NPU and decoder design on NPU**. The module should sit
on top of the existing `aicir` syndrome_circuit, IR, backend, measurement, and
`aicir.noise` APIs, but its main execution surface is not dense state
simulation. It is batched, tensorized syndrome dynamics and decoder kernels.

The first useful target is an NPU-oriented QEC research/runtime layer that can:

- describe common stabilizer codes;
- represent topological and LDPC code families without flattening away their
  geometry or parity-check structure;
- generate syndrome-extraction circuits as ordinary `Circuit` objects;
- compile code/layout/noise data into batched syndrome-loop tensors;
- simulate many shots and many rounds of syndrome dynamics on NPU;
- run_loop NPU-native decoders, including repetition, matching-style, BP, BP-OSD
  components, and neural decoders where appropriate;
- reuse `aicir.noise.NoiseModel` and future `aicir.noise` samplers rather than
  defining a parallel QEC noise system;
- decode measured syndromes or detection events into Pauli-frame/recovery data;
- report logical failure and logical observable statistics.

This keeps the module physically meaningful without over-claiming. Quantum error
correction is not just "add a few gates around the syndrome_circuit"; the design must
make stabilizers, syndrome data, noise, recovery, and decoder execution explicit.

## API naming rule

Public QEC APIs should be concise and meaningful, but not cryptic. They should
also be easy to distinguish from existing aicir APIs such as `aicir.circuit`,
`Circuit`, `Measure.run`, and primitive `.run(...)` methods.

Preferred public names:

| Name | Meaning |
| --- | --- |
| `compile_loop` | compile code/layout/noise into an NPU syndrome loop |
| `run_loop` | run a compiled loop and return a syndrome batch |
| `run_cycle` | run loop simulation plus decoding and summarize logical results |
| `run_rounds` | run direct syndrome rounds for LDPC/topological studies |
| `syndrome_circuit` | build a syndrome-extraction circuit |
| `encoding_circuit` | build an encoding circuit |
| `SyndromeBatch` | batched syndrome/detection tensors |
| `LoopSpec` | compiled loop specification |
| `DetectionEvents` | detection-event record |
| `RepetitionNPU`, `BeliefPropagationNPU`, `MatchingNPU`, `LookupDecoder` | decoder names |

Avoid one-word verbs like `run` or `compile` in the public QEC surface because
they are too easy to confuse with existing aicir execution APIs.

## Non-goals for the first implementation

- No claim of scalable threshold simulation. Dense density-matrix validation is
  exponential and should be used for small codes only.
- No dense quantum-state simulation as the main QEC engine. The production path
  is syndrome/detection-event simulation on NPU.
- No hardware pulse scheduling, dynamical decoupling, or calibrated timing
  model.
- No full fault-tolerant logical-gate compiler.
- No mandatory compiled surface-code matching backend. Heavy decoders should
  remain optional dependencies with clear fallbacks or clear import errors.
- No mutation of the core `Circuit` semantics unless QEC exposes a concrete gap
  that cannot be solved by compiling to existing gates, `measure`, and `reset`.

## Existing aicir facts the design depends on

- `Circuit` stores normalized operations and exposes typed IR through
  `Circuit.ir` / `Circuit.operations`.
- `measure(qubits, basis="Z", id=...)` and `reset(qubits)` already exist as
  nonunitary in-circuit operations.
- `Circuit.unitary()` rejects nonunitary operations unless
  `ignore_nonunitary=True`, which is the correct behavior for QEC circuits.
- `State` supports both state-vector and density-matrix representations.
- `NoiseModel`, standard noise channels, and `NoisyEstimator` already provide a
  route for noisy simulation.
- `NPUBackend` is torch-based through the GPU backend shape and is the intended
  backend for tensorized syndrome loops and differentiable decoder experiments.
- `PauliString` and `Hamiltonian` already represent Pauli observables, but
  `measure(..., basis=...)` currently handles same-basis joint Pauli checks
  directly. General mixed Pauli strings such as `XZZXI` need a compiled
  measurement circuit or a future generalized Pauli measurement primitive.

## Recommended approach

Use a stabilizer/layout-first architecture, then compile QEC procedures into two
products:

- ordinary aicir `Circuit` objects for inspection, small-code validation, and
  compatibility with existing `Measure`/`NoiseModel`;
- batched tensor programs for NPU syndrome-loop simulation and NPU decoder
  research.

The QEC package should be a layer above core IR, not a replacement for it.

Alternative approaches considered:

| Approach | Description | Pros | Cons | Recommendation |
| --- | --- | --- | --- | --- |
| NPU syndrome-loop first | Provide code/layout descriptors, NPU loop tensors, and NPU decoder kernels | Matches the real aim; scales beyond density matrices; uses NPU strength | Requires careful tensor conventions and CPU validation path | Use this |
| Code-library first | Provide named codes, syndrome circuits, and decoders that emit `Circuit` | Fits current aicir APIs; testable; useful quickly | Does not solve large-code simulation | Keep as support layer |
| New QEC IR | Add first-class logical qubits, stabilizers, and syndrome ops into core IR | More expressive long-term | Large blast radius; premature; risks breaking existing tools | Defer |
| Generic decoder/simulator only | Focus on syndrome data and decoding independent of circuits | Good for surface-code studies | Weak integration with aicir's circuit/noise stack | Avoid as the only path |

## Package layout

```text
aicir/qec/
  __init__.py
  pauli.py          # Pauli-check helpers around existing PauliString
  codes.py          # code definitions and built-in code factories
  layouts.py        # lattice / Tanner-graph / qubit-coordinate layouts
  circuits.py       # encoding and syndrome-extraction circuit builders
  syndrome.py       # batched syndrome/detection-event tensors
  loop.py           # NPU syndrome-loop simulation
  decoders.py       # CPU reference decoders and decoder protocols
  decoders_npu.py   # NPU decoder kernels / tensorized decoder modules
  frame.py          # Pauli-frame updates and logical failure checks
  cycle.py          # correction-cycle orchestration
  results.py        # structured QEC experiment results
  analysis.py       # logical error and observable analysis
```

The public API should be small:

```python
from aicir.qec import (
    StabilizerCode,
    repetition_code,
    bit_flip_code,
    phase_flip_code,
    steane_code,
    surface_code,
    color_code,
    qldpc_code,
    syndrome_circuit,
    encoding_circuit,
    compile_loop,
    run_loop,
    run_cycle,
    run_rounds,
    RepetitionNPU,
    BeliefPropagationNPU,
    LookupDecoder,
)
```

## Core data model

### `StabilizerCode`

`StabilizerCode` is an immutable code descriptor.

Fields:

- `name: str`
- `n_data: int`
- `k_logical: int`
- `stabilizers: tuple[PauliString, ...]`
- `logical_x: tuple[PauliString, ...]`
- `logical_z: tuple[PauliString, ...]`
- `distance: int | None`
- `family: str`
- `layout: CodeLayout | None`
- `metadata: dict`

Validation:

- all stabilizers have the same `n_qubits == n_data`;
- stabilizers commute pairwise;
- logical operators commute with stabilizers;
- each logical `X_i` anticommutes with logical `Z_i` and commutes with other
  logical operators as expected;
- no duplicate stabilizers up to phase;
- coefficients of stabilizer checks must be real signs `+1` or `-1`.

The code descriptor should not own physical ancilla allocation. That belongs to
the syndrome_circuit builder because different extraction strategies use different
ancilla layouts.

### `CodeLayout`

`CodeLayout` records geometry when geometry matters. This is essential for
surface, color, heavy-hex, and many LDPC constructions because decoding and
noise are graph-dependent.

Fields:

- `data_coordinates: dict[int, tuple[float, ...]]`
- `check_coordinates: dict[int, tuple[float, ...]]`
- `check_support: tuple[tuple[int, ...], ...]`
- `check_type: tuple[str, ...]` such as `"X"`, `"Z"`, or `"mixed"`
- `boundaries: dict[str, tuple[int, ...]]`
- `edges: tuple[tuple[int, int], ...]`
- `metadata: dict`

For non-geometric codes this can be `None`. For QLDPC codes, the same slot can
carry Tanner-graph coordinates or just graph adjacency.

### `QLDPCCode`

Quantum LDPC codes need parity-check-matrix access in addition to Pauli-string
access. Represent them as a specialization of `StabilizerCode`, not as a
separate incompatible API.

Additional fields:

- `hx: np.ndarray | scipy.sparse.spmatrix | None`
- `hz: np.ndarray | scipy.sparse.spmatrix | None`
- `check_matrix: np.ndarray | scipy.sparse.spmatrix | None` for non-CSS codes
- `row_weight: int | tuple[int, int] | None`
- `col_weight: int | tuple[int, int] | None`
- `rate: float | None`
- `construction: str`

Rules:

- CSS QLDPC codes should provide `hx` and `hz`;
- non-CSS QLDPC codes should provide a symplectic binary check matrix;
- sparse matrices are preferred, but SciPy must remain optional;
- commutation must be checked by binary symplectic products, not dense matrix
  multiplication.

### Built-in codes

The module should cover the important QEC families, but it should stage them by
how safely they can be implemented on top of current aicir primitives.

First wave: small stabilizer and CSS codes:

- `repetition_code(distance, basis="Z")`
- `bit_flip_code()` as repetition distance 3 in Z-check form
- `phase_flip_code()` as repetition distance 3 in X-check form
- `steane_code()`
- `shor_code()`

Second wave: topological codes:

- `surface_code(distance, *, rotated=True, boundaries="rough_smooth")`
- `planar_surface_code(distance)`
- `toric_code(distance)`
- `color_code(distance, lattice="triangular")`
- `triangular_color_code(distance)`
- `honeycomb_color_code(distance)` as a layout/code descriptor, with syndrome_circuit
  extraction only when a safe schedule is specified.

Third wave: compact and non-CSS stabilizer codes:

- `five_qubit_code()`
- `bacon_shor_code(dx, dz)` as a subsystem-code descriptor
- `cat_code(length)` and `binomial_code(...)` as future non-Pauli/no-bosonic
  placeholders only if aicir grows bosonic primitives; until then they should
  be documented as out of scope.

Fourth wave: quantum LDPC and high-rate codes:

- `hypergraph_product_code(h1, h2)`
- `balanced_product_code(...)`
- `lifted_product_code(...)`
- `bivariate_bicycle_code(...)`
- `gross_code()` only if the exact construction data is embedded and tested
- `qldpc_code(hx=..., hz=...)` for user-supplied CSS LDPC checks
- `qldpc_code(check_matrix=...)` for user-supplied non-CSS symplectic checks

Surface/color-code constructors must return `StabilizerCode` objects with a
non-null `CodeLayout`. QLDPC constructors must return `QLDPCCode` objects and
avoid dense representations for large checks.

The first wave is enough to prove the API across same-basis and CSS stabilizer
families. Surface-code support should follow quickly because it is the standard
fault-tolerance baseline. Color-code and QLDPC support require more careful
decoder and layout abstractions, so their constructors can arrive before their
full circuit-level decoding stack.

### `Syndrome`

Represent syndrome outcomes as explicit structured data rather than raw bit
strings.

Fields:

- `code: StabilizerCode`
- `round_index: int`
- `values: tuple[int, ...]` where each value is `+1` or `-1`
- `ids: tuple[str, ...]`
- `metadata: dict`

Provide conversions:

- `to_bits(zero_for_plus=True) -> tuple[int, ...]`
- `from_measure_result(result, ids=...) -> Syndrome`

### `DetectionEvents`

Repeated-round topological codes need detection events, not only raw syndrome
values. A detection event is the parity change of a check outcome between
rounds, plus boundary events at initialization/final readout when applicable.

Fields:

- `code: StabilizerCode`
- `rounds: int`
- `events: tuple[tuple[int, int], ...]` as `(round_index, check_index)`
- `boundary_events: tuple[str, ...]`
- `metadata: dict`

This object is the natural input to matching and union-find decoders. It should
be derived from per-round `Syndrome` records, not replace them.

### `SyndromeBatch`

`SyndromeBatch` is the primary runtime object for NPU work. It stores many shots
and many rounds in backend-native tensor form.

Fields:

- `syndrome: tensor[shots, rounds, n_checks]` with binary values `{0, 1}`;
- `detection_events: tensor[shots, rounds, n_detectors] | None`;
- `observable_flips: tensor[shots, n_logical] | None`;
- `erasure_flags: tensor[shots, rounds, n_locations] | None`;
- `layout: CodeLayout`;
- `backend`;
- `metadata: dict`.

Conventions:

- binary `0` means stabilizer value `+1`; binary `1` means `-1`;
- tensors should remain on NPU during simulation and decoding;
- conversion to Python `Syndrome`/`DetectionEvents` is for debugging and small
  tests only.

### `LoopSpec`

`LoopSpec` is the compiled NPU workload.

Fields:

- `code: StabilizerCode`
- `rounds: int`
- `check_matrix_x / check_matrix_z` or non-CSS symplectic check tensor;
- `detector_matrix`;
- `logical_observable_matrix`;
- `fault_locations`;
- `noise_model: NoiseModel | PhenomenologicalNoise | None`;
- `layout: CodeLayout`;
- `metadata: dict`.

This spec should be created by `compile_loop(code, noise_model, ...)`
and consumed by `run_loop(...)`.

### Recovery operations

Represent recoveries as Pauli corrections before compiling them to gates.

```python
@dataclass(frozen=True)
class Recovery:
    pauli: PauliString
    syndrome: tuple[int, ...]
    confidence: float | None = None
```

Applying a recovery should return either:

- a list of physical gates (`x`, `y`, `z`) appended to a `Circuit`; or
- a Pauli-frame update, if the caller wants virtual correction.

The first implementation should support both modes. Pauli-frame correction is
the default because it is the scalable representation for NPU simulation.
Physical recovery gates are useful for small syndrome_circuit validation.

## Circuit generation

### Encoding circuits

`encoding_circuit(code, logical_state=None, *, data_qubits=None) -> Circuit`

For the first pass, only built-in codes need hand-written encoders. The generic
stabilizer-state encoder can be deferred because it is easy to get subtly wrong.

Required encoders:

- bit-flip repetition: encoding_circuit `alpha|0> + beta|1>` to
  `alpha|000> + beta|111>`;
- phase-flip repetition: basis-transformed bit-flip code;
- Steane: explicit known encoder, tested against stabilizer expectations.

### Syndrome extraction

`syndrome_circuit(code, *, data_qubits=None, ancilla_qubits=None,
reset_ancillas=True, measure_basis="compiled", id_prefix="s") -> Circuit`

The builder emits a syndrome_circuit that measures all stabilizer generators.

For a Z-type stabilizer:

1. prepare ancilla in `|0>`;
2. apply CNOTs from each data qubit in the check support to the ancilla;
3. measure ancilla in Z with id `"{id_prefix}{i}"`;
4. optionally reset the ancilla.

For an X-type stabilizer:

1. prepare ancilla in `|+>` using H;
2. apply CNOTs from ancilla to each data qubit in the check support;
3. measure ancilla in X or rotate then measure Z;
4. optionally reset the ancilla.

For mixed Pauli checks:

- First implementation: raise `NotImplementedError` with a clear message unless
  an explicit mixed-Pauli extraction strategy is implemented.
- Later implementation: use basis rotations around the data qubits, extract Z
  parity with an ancilla, then undo rotations. This must preserve the intended
  stabilizer measurement and avoid over-measuring individual data qubits.

Direct use of `measure(qubits, basis=...)` is acceptable for same-basis joint
checks in simulator-only paths, but the ancilla-extraction syndrome_circuit should be the
default because it mirrors hardware and gives the noise model places to act.

### Surface-code extraction

Surface-code builders must generate repeated X- and Z-check rounds over an
explicit lattice.

Requirements:

- expose data, X-check, and Z-check qubit coordinates;
- support rotated planar patches first, then unrotated planar and toric codes;
- provide a deterministic CNOT ordering schedule to avoid accidental hook-error
  orientation changes;
- label every measurement by round and check id, for example
  `"r3_z12"` and `"r3_x7"`;
- support repeated rounds where detection events are computed from parity
  changes between consecutive rounds;
- keep boundary checks explicit, not hidden as special cases.

The design must distinguish syndrome bits from detection events. Surface-code
matching decoders consume detection events; small stabilizer lookup decoders
consume one-round syndrome values.

### Color-code extraction

Color-code builders need a layout with face colors and face stabilizer supports.

Requirements:

- triangular 4.8.8 or 6.6.6 color-code patches should be the first target;
- each face has both X and Z stabilizers over the same support;
- measurement schedules must be explicit because high-weight face checks can
  create correlated hook errors;
- transversal logical gates can be represented as metadata initially, not
  compiled into a fault-tolerant gate set in the first implementation.

### QLDPC extraction

QLDPC builders should start from parity-check matrices or Tanner graphs.

Requirements:

- never build dense `2^n x 2^n` operators to validate checks;
- use sparse binary symplectic validation;
- provide Tanner-graph access for decoders;
- support schedule objects because parallel extraction of high-degree checks is
  a graph-coloring/scheduling problem;
- support phenomenological simulation before full circuit-level extraction for
  large QLDPC instances.

## NPU syndrome-loop simulation

The main runtime API should be:

```python
spec = compile_loop(
    code,
    noise_model=noise_model,
    rounds=rounds,
    backend=npu_backend,
    mode="phenomenological",
)

batch = run_loop(
    spec,
    shots=shots,
    seed=seed,
    backend=npu_backend,
)
```

`run_loop` should not construct dense quantum states. It should
sample faults and propagate them through parity-check, detector, and logical
observable tensors.

### Tensor data flow

```text
CodeLayout / QLDPC checks
  -> LoopSpec
  -> sample fault tensor on NPU
  -> syndrome tensor [shots, rounds, checks]
  -> detection-event tensor [shots, rounds, detectors]
  -> NPU decoder
  -> Pauli-frame / correction tensor
  -> logical failure tensor
  -> QECResult
```

### Required NPU properties

- Use backend-native tensors and keep hot arrays on NPU.
- Batch over `shots`; avoid Python loops over shots.
- Prefer binary tensor operations, scatter/gather, sparse adjacency, and batched
  matmul-like operations.
- Store sparse checks as compact edge lists or CSR-like tensors because NPU
  sparse support may be limited.
- Keep CPU conversion at API boundaries only: summaries, debugging, and tests.
- Expose a CPU/Numpy reference path with identical tensor conventions.

### Simulation modes

- `mode="phenomenological"`: sample data and measurement faults directly from
  `aicir.noise` phenomenological samplers; primary path for surface/color/QLDPC
  scale studies.
- `mode="circuit_level"`: use a scheduled syndrome-extraction circuit and
  sample gate/readout/reset/idle fault locations without simulating amplitudes;
  preferred once `aicir.noise` has the necessary channel metadata.
- `mode="density_matrix"`: small-code validation only, routed through existing
  `Measure`/`NoiseModel` semantics.

The NPU loop should be deterministic for a fixed seed and should record enough
metadata to replay the same sampled fault stream on CPU for debugging.

### Correction cycles

`run_cycle(code, circuit=None, *, decoder, noise_model=None,
backend=None, rounds=1, shots=1024, correction_mode="pauli_frame",
mode="phenomenological") -> QECResult`

Data flow:

```text
code/layout/noise
  -> compile syndrome-loop spec
  -> simulate syndrome/detection-event batch on backend
  -> decoder maps batch to Pauli-frame updates
  -> measure logical observables
  -> aggregate QECResult
```

For small validation runs, `run_cycle` may also expose the generated
syndrome circuit. `QECResult` must expose the compiled loop spec, optional
compiled circuit, and per-shot aggregate syndrome/recovery records for debugging.

## Decoders

### Decoder protocol

```python
class Decoder(Protocol):
    def decode(self, data) -> Recovery:
        ...

class NPUDecoder(Protocol):
    def decode_batch(self, batch: SyndromeBatch):
        ...
```

Concrete decoders should narrow `data` to `Syndrome`, `DetectionEvents`, or a
binary LDPC syndrome vector. NPU decoders should consume `SyndromeBatch` and
return backend-native correction/logical-failure tensors.

### Decoder families

- `LookupDecoder`: CPU reference exhaustive syndrome table for small stabilizer
  codes.
- `RepetitionDecoder` / `RepetitionNPU`: majority/minimum-weight decoder
  for repetition codes, vectorized over shots on NPU.
- `IdentityDecoder`: returns no correction, useful as a baseline.
- `MWPMDecoder`: reference matching decoder for surface-code detection events;
  compiled matching libraries remain optional.
- `MatchingNPU`: NPU-friendly matching approximation or local
  clustering decoder for large batched studies.
- `UnionFindDecoder` / `UnionFindNPU`: surface-code decoder where the
  NPU version uses batched graph primitives.
- `ColorCodeDecoder`: projection-to-surface-code or lookup decoder for small
  color-code patches.
- `BeliefPropagationDecoder` / `BeliefPropagationNPU`: belief-propagation decoder for QLDPC codes;
  this is a core NPU target because message passing is naturally tensorized.
- `BPOSDDecoder`: BP plus ordered-statistics post-processing. BP should run on
  NPU; OSD may initially fall back to CPU for small residual systems unless an
  NPU batched OSD kernel is designed.
- `NeuralDecoder`: optional torch module trained/evaluated on NPU using
  `SyndromeBatch` tensors.

`LookupDecoder.from_code(code, max_weight=1)` should enumerate all Pauli errors
up to `max_weight`, compute their syndromes, and choose a minimum-weight
representative. Ambiguous syndromes must be recorded, not silently ignored.

Compiled matching decoders should be optional. If unavailable, `MWPMDecoder`
should either use a documented pure-Python fallback for tiny patches or raise a
clear import error before a long simulation starts.

Decoder inputs should be typed:

- lookup/repetition decoders consume `Syndrome`;
- matching decoders consume `DetectionEvents`;
- QLDPC decoders consume binary syndrome vectors plus code metadata.
- NPU decoders consume `SyndromeBatch` and must not require per-shot Python
  callbacks.

Do not force all decoders through one bit-string convention. That hides
important differences between block-code syndromes, repeated-round detection
events, and LDPC parity-check syndromes.

### NPU decoder design rules

- Keep decoder state as tensors on NPU.
- Batch over shots and, where possible, over decoder iterations.
- Avoid Python priority queues in the hot path; use tensorized relaxations,
  batched local clustering, union-find primitives, or BP-style message passing.
- Make CPU reference decoders bit-for-bit comparable on small examples.
- For trainable decoders, use torch autograd normally; do not invent a separate
  gradient path.
- Return both correction tensors and logical-failure estimates so benchmarking
  can avoid converting all per-shot data back to Python.

## Results and analysis

### `QECResult`

Fields:

- `code: StabilizerCode`
- `loop_spec: LoopSpec | None`
- `circuit: Circuit`
- `shots: int`
- `syndromes: tuple[Syndrome, ...]`
- `syndrome_batch: SyndromeBatch | None`
- `detection_events: tuple[DetectionEvents, ...] | None`
- `recoveries: tuple[Recovery, ...]`
- `correction_tensor: object | None`
- `logical_failure_tensor: object | None`
- `logical_counts: dict[str, int]`
- `logical_error_rate: float | None`
- `noise_model: NoiseModel | None`
- `metadata: dict`

Methods:

- `summary() -> str`
- `logical_expectation(pauli: PauliString) -> float`
- `syndrome_counts() -> dict[tuple[int, ...], int]`
- `event_counts() -> dict[tuple[tuple[int, int], ...], int]`
- `frame_counts() -> dict[str, int]`

### Metrics

Initial analysis functions:

- `logical_error_rate(result, expected_logical=None)`
- `postselection_rate(result, accept_syndrome=None)`
- `syndrome_entropy(result)`
- `compare_with_uncorrected(corrected, uncorrected)`

These should live in `aicir/qec/analysis.py` and return plain dataclasses or
numbers, not plots.

## Noise and backend integration

`aicir` already has `aicir.noise.NoiseModel` and standard channels such as
bit-flip, phase-flip, depolarizing, and amplitude damping. QEC should reuse
that layer. It should not define a parallel QEC noise model.

QEC validation should prefer the existing density-matrix/noisy paths for small
codes because they capture nonunitary measurement, reset, and decoherence
consistently.

For larger surface/color/QLDPC studies, dense density-matrix simulation is not
the right tool. The repo should still keep noise ownership in `aicir.noise`,
but support two simulation noise levels:

- **circuit-level noise**: inject channels after actual aicir gates,
  measurement, reset, idle windows, and optional leakage events through
  `aicir.noise.NoiseModel`;
- **phenomenological noise**: flip syndrome bits, detection events, or data
  Pauli errors directly without simulating the full quantum circuit. This is
  required for larger surface-code and QLDPC experiments, but it should still be
  represented by an `aicir.noise` object or sampler, not by `aicir.qec`.

### Noise ownership

All physical and phenomenological noise objects belong in `aicir.noise`.
`aicir.qec` only consumes them.

Existing objects to reuse:

- `NoiseModel`
- `BitFlipChannel`
- `PhaseFlipChannel`
- `DepolarizingChannel`
- `AmplitudeDampingChannel`

Future additions, if QEC needs them, should be implemented under `aicir.noise`:

- `ReadoutError` or `MeasurementFlipChannel`
- `ResetError`
- `IdleError`
- `TwoQubitDepolarizingChannel`
- `BiasedPauliChannel`
- `ErasureChannel`
- `LeakageEventModel`
- `CoherentOverrotationChannel`
- `CorrelatedPauliChannel`
- `PhenomenologicalNoise` for direct syndrome/detection-event sampling

### Noise channels to support

Physical Pauli noise:

- single-qubit X, Y, Z, depolarizing;
- two-qubit depolarizing after `cx`, `cz`, `rxx`, `rzz`;
- biased dephasing with `bias_eta = p_z / (p_x + p_y)`;
- asymmetric X/Z error rates for repetition, surface, and XZZX-style studies.

Measurement and reset noise:

- pre-measurement bit flip;
- classical readout flip on the recorded syndrome/result;
- reset-to-`|1>` failure;
- measurement loss/erasure flag.

Idle and scheduling noise:

- idle depolarizing/dephasing per round;
- memory T1/T2 during syndrome extraction;
- schedule-aware idle windows for surface/color-code circuits.

Correlated and coherent noise:

- nearest-neighbor correlated Pauli faults such as `ZZ` or `XX`;
- crosstalk from simultaneous two-qubit gates;
- coherent overrotation on selected gates;
- slowly varying shot-to-shot parameter drift.

Leakage and erasure:

- leakage out of the computational subspace should be represented as a flagged
  classical event first, not by pretending the existing qubit Hilbert space is
  three-dimensional;
- heralded erasure should feed decoders that can use location information;
- unheralded leakage can be approximated by Pauli or depolarizing noise until
  aicir has qutrit support.

Rules:

- if `noise_model` is provided, require a density-matrix-capable backend or use
  `NumpyBackend` as the safe default;
- if a phenomenological sampler is needed for large codes, it should be an
  `aicir.noise` object consumed by `run_rounds`, not a QEC-owned
  model;
- circuit-level QEC noise must include measurement and reset faults; gate-only
  noise is not enough for honest QEC benchmarks, so `aicir.noise` should be
  extended where current channels are insufficient;
- if the generated circuit contains `measure` or `reset`, never call
  `Circuit.unitary()` except with `ignore_nonunitary=True` for explicit
  diagnostic use;
- preserve all syndrome measurement ids so `Measure.run(...).output(id)` can be
  used directly;
- keep shots explicit; QEC statistics without shot count are easy to
  misinterpret.

NPU/GPU support can be inherited through existing backends, but the first QEC
implementation should make NPU syndrome-loop simulation the primary optimized
path. Dense density-matrix simulation on NPU is not the target for large codes.

### Noise-model outputs

Noise simulations should report enough metadata to reproduce the experiment:

- physical error rates by channel class;
- syndrome measurement error rate;
- reset and idle error rates;
- number of rounds;
- code distance and layout name;
- decoder name and decoder options;
- random seed;
- whether the run was circuit-level or phenomenological.

## API examples

### Surface code syndrome loop on NPU

```python
from aicir.backends import NPUBackend
from aicir.noise import PhenomenologicalNoise
from aicir.qec import (
    MatchingNPU,
    compile_loop,
    run_loop,
    surface_code,
)

backend = NPUBackend()
code = surface_code(distance=7, rotated=True)
noise = PhenomenologicalNoise(p_data=0.001, p_meas=0.001)

spec = compile_loop(code, noise_model=noise, rounds=7, backend=backend)
batch = run_loop(spec, shots=100000, seed=1234, backend=backend)

decoder = MatchingNPU.from_code(code, backend=backend)
decoded = decoder.decode_batch(batch)
print(decoded.logical_error_rate())
```

`PhenomenologicalNoise` is a proposed future extension under
`aicir.noise`, not a QEC-owned noise model.

### Bit-flip code

```python
from aicir.qec import bit_flip_code, LookupDecoder, run_cycle
from aicir.noise import BitFlipChannel, NoiseModel

code = bit_flip_code()
decoder = LookupDecoder.from_code(code, max_weight=1)

noise = NoiseModel()
for q in range(code.n_data):
    noise.add_channel(BitFlipChannel(target_qubit=q, p=0.001), after_gates=["x", "cx"])

result = run_cycle(
    code,
    decoder=decoder,
    noise_model=noise,
    rounds=1,
    shots=2048,
)

print(result.logical_error_rate)
print(result.syndrome_counts())
```

### Syndrome circuit inspection

```python
from aicir.qec import steane_code, syndrome_circuit

code = steane_code()
cir = syndrome_circuit(code, id_prefix="steane_s")
cir.show()
```

### Surface code with phenomenological noise

```python
from aicir.backends import NPUBackend
from aicir.qec import surface_code, MatchingNPU, run_cycle
# Future extension under aicir.noise, not aicir.qec:
from aicir.noise import PhenomenologicalNoise

backend = NPUBackend()
code = surface_code(distance=5, rotated=True)
decoder = MatchingNPU.from_code(code, backend=backend)
noise = PhenomenologicalNoise(p_data=0.001, p_meas=0.001)

result = run_cycle(
    code,
    decoder=decoder,
    noise_model=noise,
    backend=backend,
    rounds=5,
    shots=10000,
)
```

### User-supplied QLDPC code

```python
from aicir.backends import NPUBackend
from aicir.noise import PhenomenologicalNoise
from aicir.qec import BPOSDDecoder, qldpc_code, run_rounds

backend = NPUBackend()
code = qldpc_code(hx=hx_sparse, hz=hz_sparse, name="experiment_ldpc")
decoder = BPOSDDecoder.from_code(code, max_iter=50, osd_order=2, backend=backend)
noise = PhenomenologicalNoise(p_data=0.003, p_meas=0.001)
stats = run_rounds(
    code,
    decoder=decoder,
    noise_model=noise,
    backend=backend,
    shots=2000,
)
```

## Error handling

- Noncommuting stabilizer generators: `ValueError`.
- Invalid logical operators: `ValueError`.
- Unsupported mixed-Pauli extraction strategy: `NotImplementedError`.
- Decoder receives a syndrome from a different code: `ValueError`.
- Ambiguous lookup table entries: keep the minimum-weight correction, record the
  ambiguity in decoder metadata, and expose it in `repr`/`summary`.
- Insufficient ancilla qubits for the requested extraction strategy:
  `ValueError`.
- Noise requested on a backend path that cannot represent nonunitary evolution:
  `ValueError`.
- Phenomenological noise requested with a circuit-level-only decoder:
  `ValueError`.
- NPU decoder receives CPU tensors or incompatible backend tensors:
  `ValueError`.
- NPU loop would require dense state simulation for the requested mode:
  `ValueError`.
- QLDPC constructor receives noncommuting `hx`/`hz`:
  `ValueError`.
- Surface/color-code layout has inconsistent check support or boundary labels:
  `ValueError`.

## Testing strategy

Unit tests:

- Pauli commutation and syndrome calculation.
- `StabilizerCode` validation failures.
- Built-in code stabilizers and logical operators.
- Surface-code layout: data/check counts, boundaries, check supports, logical
  strings for distances 3 and 5.
- Color-code layout: face colors, X/Z face checks, logical operators.
- QLDPC validation: sparse `hx @ hz.T mod 2 == 0`, row/column weights, rate.
- Syndrome syndrome_circuit ids and qubit allocation.
- LookupDecoder decoder tables for bit-flip, phase-flip, and Steane codes.
- `LoopSpec` tensor shapes and binary conventions.

Simulation tests:

- bit-flip code corrects all single X errors;
- phase-flip code corrects all single Z errors;
- Steane code corrects all single-qubit X/Z/Y errors at the Pauli-frame level;
- distance-3 rotated surface code produces consistent detection events under a
  single inserted data fault;
- small color-code patch detects single-qubit Pauli faults;
- small CSS QLDPC fixture decodes low-weight sampled Pauli errors with BP or
  lookup fallback;
- noisy bit-flip correction improves logical error rate below the uncorrected
  baseline for small physical error probability;
- phenomenological surface-code logical error decreases when distance increases
  below a conservative test error rate;
- NPU and CPU syndrome-loop reference paths produce identical results for a
  fixed sampled fault tensor;
- NPU repetition and BP decoders match CPU references on small batches;
- correction cycles preserve reproducibility with fixed seeds.

Regression tests:

- generated QEC circuits containing `measure`/`reset` are rejected by
  `Circuit.unitary()` by default;
- generated circuits remain valid under `Circuit.ir`;
- result parsing by measurement id works across repeated rounds.
- NPU APIs do not move hot tensors to CPU except at explicit summary/export
  boundaries.

## Staging plan for later implementation

1. Add data-only `aicir.qec` package: Pauli helpers, `StabilizerCode`,
   `QLDPCCode`, `CodeLayout`, `Syndrome`, `DetectionEvents`,
   `SyndromeBatch`, `Recovery`, and built-in repetition codes.
2. Add `compile_loop` and CPU reference `run_loop` for
   repetition and small CSS codes.
3. Add NPU `run_loop` with batched fault sampling and detection
   event generation.
4. Reuse existing `aicir.noise.NoiseModel`; add missing readout/reset/idle or
   phenomenological noise support to `aicir.noise` only when QEC requires it.
5. Add NPU repetition decoder and BP decoder kernels, with CPU reference tests.
6. Add `run_cycle` and `QECResult` as orchestration over loop specs,
   NPU batches, and decoders.
7. Add rotated surface-code layouts, detection events, and NPU-friendly
   matching/union-find approximations.
8. Add QLDPC constructors, Tanner-graph access, BeliefPropagationNPU/BPOSDDecoder
   interfaces, and phenomenological simulations.
9. Add color-code layouts and NPU batch simulations.
10. Add small-code circuit/density-matrix validation paths through existing
    `Measure` and `NoiseModel`.
11. Add mixed-Pauli extraction support where needed by non-CSS codes.

## Open design choices

- Whether to support virtual Pauli-frame correction in the first implementation
  or defer it after physical recovery gates are tested.
- Which NPU matching approximation should be the first surface-code decoder:
  batched union-find, local clustering, or a differentiable neural decoder.
- How much of BP-OSD should stay on NPU versus using CPU fallback for the OSD
  residual step.
- Whether mixed-Pauli stabilizer measurement should become a general core
  measurement primitive or remain compiled by `aicir.qec`.

The conservative answer is to keep mixed-Pauli measurement inside `aicir.qec`
until at least two independent users need it outside QEC.
