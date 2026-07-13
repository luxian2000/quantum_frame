# VQE-QAS Closed Loop

`aicir.qas.vqe_loop` implements the current VQE-QAS evaluation spine:

```text
P0 bootstrap candidate generation
  -> optional zero-cost structural screen
  -> fair VQE labels
  -> P1 mutation/oracle/fallback planning
  -> fair-label feedback
```

The central rule is unchanged: cheap evaluators and zero-cost filters only
screen candidates.  Final comparisons use the shared COBYLA fair-label protocol
and the primary metric remains `fair_best_energy`.

## Entry Points

Run P0 bootstrap plus fair labels through the package entry point:

```bash
python -m aicir.qas.vqe_loop \
  --hamiltonian lih_hamiltonian.json \
  --initial-labels 0 \
  --rounds 0 \
  --use-chemistry-excitation-pool \
  --active-electrons 2 \
  --active-spatial-orbitals 3 \
  --chemistry-excitation-count 12
```

The Hamiltonian JSON can also be passed positionally:

```bash
python -m aicir.qas.vqe_loop lih_hamiltonian.json --initial-labels 0 --rounds 0
```

`--disable-p0-zero-cost` turns off the optional P0 structural annotation path.
When enabled, bootstrap summaries include a `training_free` block with pass,
soft-flag, and hard-reject counts.

Run one or more current P1 rounds after bootstrap labels exist:

```bash
python -m aicir.qas.demos.run_p1_round_demo \
  --bootstrap-labels-csv outputs/qas_4q/bootstrap_labels.csv \
  --output-dir outputs/qas_4q/round1 \
  --batch-id round1 \
  --mutation-types gate_mutation,connectivity_mutation,layer_mutation,depth_mutation
```

The P1 summary reports oracle/fallback accounting, `zero_cost_status`, fair
calls, and cheap-eval calls.  The no-regret-lite policy keeps fallback topK as
the guardrail while using trusted oracle predictions to widen the candidate pool
or skip selected cheap evaluations.

## Hamiltonian Input

`--hamiltonian` accepts either a legacy Pauli-term list:

```json
[
  [-0.09706626816762543, "IIII"],
  [0.17141282644776914, "ZIII"],
  [0.12039548242542646, "XXXX"]
]
```

or a structured molecule specification:

```json
{
  "molecule": "LiH",
  "distance": 0.1
}
```

That shorthand expands to Li at `(0, 0, 0)` and H at `(0, 0, distance)`.
The omitted fields default to `kind=molecular`, `basis=sto3g`, `charge=0`,
`spin=0`, `unit=angstrom`, `driver=pyscf`, and `mapping=jordan_wigner`.
For non-diatomic molecules or custom layouts, pass explicit `geometry` instead.

Molecular inputs are resolved once through optional PySCF/Qiskit Nature
dependencies; fair VQE labels still run on the selected aicir backend.

If `--hamiltonian` is omitted, pass `--n-qubits`; the run then uses the TFIM
defaults in the built-in fair protocol in `benchmark_table.py`.

## Main Flow

`python -m aicir.qas.vqe_loop` calls `run_vqe_qas_closed_loop()` in
`p0_bootstrap_fair.py`.

```text
p0_bootstrap_fair.py
  -> p0_chemistry_excitation.py or p0_supernet_native.py
  -> training_free.py  # optional P0 zero-cost structural annotations
  -> stamp_literal_hamiltonian_terms()
  -> fair_labeling.py
```

P1 is intentionally routed through `demos/run_p1_round_demo.py` while the
chemistry excitation path and no-regret-lite summaries settle:

```text
demos/run_p1_round_demo.py
  -> p1_round.py
       -> p1_selection.py
       -> p1_evolution.py
       -> oracle.py
       -> training_free.py
  -> fair_labeling.py
```

## Mainline Module Boundaries

- `p1_evolution.py` is the public P1 evolution entry point: parent selection, supernet mutation, chemistry excitation mutation, operator-sequence mutation, optional A1/A2 operator growth, and mutation-row conversion.
- `p1_round.py` is the public P1 orchestration entry point.
- `p1_selection.py` owns fallback ranking, baseline selector queues, and P1 queue-row selection helpers.
- `p0_bootstrap_fair.py` remains a compatibility runner for P0 bootstrap plus fair labels.

## Module Responsibilities

- `p0_bootstrap_fair.py`: P0 bootstrap queue writer plus fair-label one-call API. Also owns `resolve_closed_loop_defaults()` (fills in `rounds`/`initial_labels`/batch sizing when left as `"auto"`), `default_initial_labels_for_qubits()`, `default_max_rounds_for_qubits()`, and `stamp_literal_hamiltonian_terms()` (writes the resolved Pauli-term list onto queue rows before fair labeling).
- `__main__.py`: package command-line entry for `python -m aicir.qas.vqe_loop`.
- `p0_chemistry_excitation.py`: chemistry excitation candidate rows and mutation-space metadata.
- `p0_supernet_native.py`: generic supernet-native bootstrap rows.
- `training_free.py`: optional zero-cost structural annotations for P0 and P1 rows.
- `fair_labeling.py`: turns queue rows into fair VQE benchmark labels.
- `demos/run_p1_round_demo.py`: current P1 entry point for mutation/oracle/fallback planning, optional fair labeling, and multi-round label feedback.
- `p1_round.py`: plans one P1 queue plus equal-budget baselines from labeled parents, mutation children, oracle predictions, and fallback E2/E5 scoring.
- `p1_evolution.py`: P1 parent selection, supernet/chemistry/operator-sequence mutation, optional operator growth, layer crossover, and mutation result rows.
- `p1_selection.py`: fallback ranking, baseline selector queues, and P1 queue-row selection helpers, including `decide_next_round_quotas()`'s row-level policy inputs.
- `oracle.py`: local/trusted oracle prediction and abstention helpers.
- `task_proxy.py`: VQE-specific task-aware cheap proxy evaluators used as P1 selectors; never replaces fair COBYLA labels.
- `graph_predictor.py`: dependency-light fair-energy predictor for P1 (fixed graph/structure features plus ridge regression), same fit/predict shape as a future GNN-based predictor.
- `ansatz_family.py`: per-ansatz-family capability declarations used by P0/P1 planning.
- `growth_routes.py`: executable defaults for the two supported P1 growth routes (`line_a_operator_sequence`, `line_b_chemistry_excitation`).
- `cheap_eval_experiment.py`: row schema, manifest, and injectable runner scaffold for E1-E5 cheap-evaluation diagnostic experiments; does not itself run VQE or supernet training.
- `benchmark_table.py`: benchmark-table schema, CSV IO, row-object parsing, built-in fair-label protocol defaults, and row-level P1 policies, including `decide_next_round_quotas()`.
- `fair_vqe.py`: evaluates and optimizes one architecture on one VQE problem using the frozen fair-label policy. `evaluate_vqe_energy()`/`optimize_vqe_energy()` run on the exact statevector Pauli-term path (`vqe_engine="statevector_pauli_terms"` in `optimize_vqe_energy()`'s result metadata), not `BasicVQE`. `evaluate_vqe_energy(..., estimator=...)` accepts an `aicir.primitives` `BaseEstimator`-compatible object as an optional injection point; `optimize_vqe_energy()`'s COBYLA loop does not take `estimator=` and always uses the built-in Pauli-term evaluation.
- `shard_scheduler.py`: splits a fair-label queue into independent shards for multi-NPU or multi-process runs.
- `p0_problem_aware.py`: optional P0 diagnostic problem-aware supernet sampler; not part of the core P0/P1/fair path.

## Protocol Boundaries

`benchmark_table.py` answers:

```text
How are fair-label protocols, benchmark rows, row parsing, and row-level P1 policies represented?
```

It defines `DEFAULT_FAIR_LABEL_PROTOCOL`, `BENCHMARK_TABLE_FIELDS`, `LabelStatus`,
`LabelSource`, CSV read/write helpers, ansatz/Hamiltonian row parsers, append
rules, retry transitions, and P1 row ranking/dedup/quota helpers. The internal
`protocol_version` remains the data-compatibility key.

`p1_round.py` answers:

```text
Which mutation children enter fair labeling, and how are oracle/fallback choices summarized?
```

## Useful Commands

Run fair labels directly:

```bash
python -m aicir.qas.vqe_loop.fair_labeling \
  --queue outputs/qas_4q/stage1_5_initial_label_queue.csv \
  --output outputs/qas_4q/benchmark_table_4q_v2.csv \
  --protocol default \
  --backend numpy \
  --dtype complex128
```

Run fair labels as independent shards:

```bash
python -m aicir.qas.vqe_loop.shard_scheduler \
  --queue outputs/qas_stage2/round1/round1_queue.csv \
  --output outputs/qas_stage2/round1/round1_labels.csv \
  --work-dir outputs/qas_stage2/round1/label_shards \
  --protocol default \
  --backend npu \
  --dtype complex64 \
  --num-shards 4
```

## Notes

- `shard_scheduler.py` is a task-parallel queue runner; it does not change the fair-label protocol.
- `fair_vqe.py` is not a general VQE frontend; it is the fair-label execution layer used by VQE-QAS.
- The old preparation/next-batch/trust-region chain has been removed from this package path.



