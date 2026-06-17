# VQE-QAS trust-region oracle TODO

This document freezes the implementable protocol for the <=10q VQE-QAS oracle
path after the `fair_vqe_protocol_v2` reset. The oracle is a fast
interpolation/routing tool, not a final judge and not a global reward function.

## 2026-06-16 v2 reset status

The old `benchmark_table_v1`, shifted-TFIM target labels, and oracle
calibration results are invalidated for oracle evidence. They were generated
with a measured/collapsed final state, which removed X/Y expectation values and
turned TFIM labels into a diagonal classical proxy. Keep those files only as
historical bug records.

The following pieces remain valid and should be reused:

- Stage-0 candidate grammar and candidate pool construction.
- Stage-1 illegal/duplicate/over-budget removal.
- Stage-1b zero-cost hard-reject/soft-flag logic.
- The local trust-region oracle framework.
- Stage-2 EA/oracle control flow and abstain mechanism.

The current label protocol is `fair_vqe_protocol_v2`:

```text
energy_evaluation = unmeasured_state_exact
backend = numpy
dtype = complex128
protocol_version = fair_vqe_protocol_v2
```

Smoke evidence for the v2 repair:

```text
balanced 4q 12-row smoke:
  fair_std_energy nonzero = 12/12
  median std = 0.126
  delta_ref range = 0.295 .. 0.967

expressive 4q 12-row smoke:
  fair_std_energy nonzero = 12/12
  median std = 0.148

extra 12-seed check:
  architecture = 4q_hea_mask_L1_ry_cz_linear_ry
  best_delta = 0.178
```

This proves that v2 labels are no longer flat, but it does not yet justify a
full 96-row relabel. Before rebuilding the benchmark table, run one budget
diagnostic on `4q_hea_mask_L1_ry_cz_linear_ry` with roughly doubled `nfev`.
If `delta_ref` continues toward `< 0.1`, the blocker is optimizer budget and
the protocol budget should be adjusted before the full run. If it does not
move, treat that structure as near its expressibility ceiling and keep the
current budget logic.

## Core flow

```text
Stage 0   Candidate grammar + coverage gate
Protocol  Freeze fair VQE protocol before any labels are generated
Stage 1a  Remove only illegal, duplicate, and over-budget candidates
Stage 1b  Zero-cost feasibility filter: hard reject extremes, soft-flag boundary
Stage 1.5 Offline benchmark table + retrieval/surrogate oracle
Stage 2A  Trust-region local beam search
Stage 2B  Boundary/OOD farthest-first expansion
Stage 3   Batched multi-seed fair VQE verification
Update    Versioned oracle rebuild; do not mix protocol versions
```

## Do not do

- Do not use B1/B2/B2-B1 or short-step VQE as a decision signal.
- Do not mix medium-fair labels into the oracle table.
- Do not let the oracle optimize arbitrary global candidate space.
- Do not allow low oracle score to permanently delete a candidate.
- Do not use expressibility/trainability as a fixed weighted final ranker in
  Stage 1b.
- Do not use supernet inherited score as a final ranker unless separately
  audited and versioned as a feature.
- Do not generate labels before `fair_vqe_protocol_v2` is frozen.
- Do not mix labels from different `protocol_version` values.
- Do not use `benchmark_table_v1` or shifted-TFIM v1 target labels for oracle
  training, validation, or calibration claims.
- Do not rerun the full 96-row benchmark until the 4q budget-vs-ceiling
  diagnostic has passed.

## Stage 0 gates

Stage 0 must produce a coverage report before labels are generated.

Required coverage axes:

- `family`
- `entangler_type`
- `depth_group`
- `topology`
- `n_params`
- `two_q_count`
- `hamiltonian_coverage`

No-go conditions:

- a required family, entangler type, depth group, or topology is absent;
- duplicate/canonical collision rate is unexpectedly high;
- parameter or two-qubit gate distribution collapses to a single bucket;
- Hamiltonian-relevant blocks are absent for the target class.

## Stage 1b zero-cost feasibility filter

Stage 1b prevents obviously bad structures from consuming fair-VQE labels. It
is not a ranking stage.

Allowed actions:

```text
pass          candidate remains in the normal pool
soft_flag     candidate remains labelable, but is preferentially boundary
hard_reject   only for extreme/structural failures
```

Default first-version signals:

- `trainability_score`: `metrics/trainability.py::structure_proxy`
- `expressibility_score`: fast structural expressibility proxy for full 4/6/8q
  screening
- `entanglement_score`: two-qubit coverage proxy for entangling Hamiltonians
- `zero_cost_feature_score`: QAS `RewardWeights` weighted score, recorded as a
  feature only; `zero_cost_score_is_ranking_signal=false`

Optional audit signals:

- `expressibility=mmd_relative` or `kl_haar`
- `trainability=gradient_variance` or `gradient_norm`

These optional metrics are too expensive for default full 4/6/8q queue
generation and should be run on small audits or later oracle features.

Rules:

- hard reject only if complexity caps are exceeded or an extreme metric floor
  is crossed;
- soft flag low trainability / low expressibility / low entanglement cases;
- if a metric has no spread in the current pool, record it but do not trigger
  soft flags from it;
- do not sort by `zero_cost_feature_score`;
- soft-flagged candidates should preferentially enter `holdout_boundary`, not
  disappear.

## First v2 label batch

Default label count:

```text
N0 = 96
fallback minimum N0 = 64
holdout_fraction = 0.20
```

Sampling priority:

```text
P0: hamiltonian_class / n_qubits
P1: family
P2: entangler_type
P3: depth_group
P4: topology
P5: n_params bucket
P6: two_q_count bucket
P7: hamiltonian_coverage bucket
```

For `N0=96`, target `P1/P2 >= 3-5` samples per bucket and
`P3/P4 >= 2-3` samples per bucket. Fill the remaining budget with
farthest-first coverage using the fixed Stage-0 distance scale.

For `N0=64`, target `P1/P2 >= 3` and `P3/P4 >= 2`; lower-priority axes are
reported but not hard-gated.

Do not rebuild the 96-row table by reusing the old split. The v2 relabel and
the split repair must happen together:

```text
1. Freeze the revised split sampler.
2. Run the 4q budget-vs-ceiling diagnostic.
3. Generate the new v2 queue with repaired initial_train / holdout_id /
   holdout_boundary / holdout_sparse semantics.
4. Run fair labels with fair_vqe_protocol_v2.
5. Recompute trust-region geometry before oracle calibration.
6. Calibrate the oracle only if split geometry is valid.
```

## Anchor and holdout rules

Anchor selection:

```text
Group by family x entangler_type x topology.
For each non-empty cell, choose one representative candidate.
Prefer the candidate whose n_params and two_q_count are closest to the
cell median. Break ties by canonical_arch_hash.
```

Initial holdout is based on distance to anchors and then rechecked against the
training set geometry:

```text
ID holdout       common/near region
boundary holdout intermediate region
sparse holdout   sparse/far region
```

Default split inside holdout:

```text
ID 40%, boundary 30%, sparse 30%
```

The old v1 split failed this geometry audit:

```text
k_min = 5
d_max = 0.28125

initial_train:
  in-TR rate = 95.65%
  neighbor median = 12

holdout_id:
  in-TR rate = 22.22%
  neighbor median = 0
  diagnosis = not ID; mostly rzz+ring while initial_train had very little rzz

holdout_boundary:
  in-TR rate = 58.33%
  kth_d median near d_max
  diagnosis = closest to intended boundary semantics

holdout_sparse:
  in-TR rate = 50.00%
  diagnosis = not sparse enough; oracle would not abstain often enough
```

Repaired split acceptance criteria:

```text
holdout_id:
  sample from the initial_train neighborhood/distribution
  match initial_train entangler/topology/depth proportions
  require neighbor_count >= k_min, preferably >= 2 * k_min
  target in-TR rate > 0.80

holdout_boundary:
  choose candidates with kth_d close to d_max
  allow mixed in/out trust-region membership
  target in-TR rate = 0.40 .. 0.60

holdout_sparse:
  require neighbor_count < k_min
  prefer neighbor_count <= 2 or kth_d > d_max
  target in-TR rate < 0.20
```

Small-batch smoke exception:

```text
For 4q smoke batches with only 12-16 total labels, use k_min = 3.
Keep d_max fixed at 0.28125.

Reason:
  k_min = 5 was derived for the full 96-row table.
  In a 12-row smoke with about 8-9 train labels, coverage sampling spreads
  train across cx/cz/rzz x linear/ring x L1/L2/L3, so no local region can
  reliably contain five train neighbors.

Full benchmark/oracle calibration:
  restore k_min = 5
  keep d_max = 0.28125 unless a new full-table leave-one-out audit replaces it
```

## Distance and compatibility

All continuous distances use Stage-0 fixed robust scales:

```text
abs(x1 - x2) / IQR_stage0(x)
```

If IQR is zero, use range; if range is zero, use scale `1.0`.

Categorical distance:

```text
same = 0
compatible = 0.5
different = 1
```

Default compatibility is conservative:

- `CX ~ CZ`
- `CZ ~ RZZ`
- `linear ~ ring`
- `HEA ~ RealAmplitudes`
- `QAOA-like ~ problem-inspired`

Compatibility must be frozen with the candidate grammar. First version should
prefer a tight trust region over a falsely broad one.

## Trust-region calibration

Default startup values:

```text
k_min = 5
d_max = 0.28125  # from initial_train leave-one-out geometry audit
A_max = 0.40
sparse_abstain_rate target >= 0.80
TR_coverage target >= 0.20
TR_in_MAE <= 0.5 * TR_out_MAE
```

Interpretation:

- If TR-in and TR-out errors are similar, the trust region is too loose.
- If the trust region is nearly empty, it is too tight.
- If sparse holdout is predicted confidently instead of abstained, coverage
  detection failed.

## Stage 2A and Stage 2B

Stage 2A is local exploitation:

```text
Use trust-region-constrained beam search.
Only rank candidates that remain inside the trust region.
Send out-of-region mutations to Track B instead of extrapolating.
```

Stage 2B is expansion:

```text
Use farthest-first sampling from boundary/OOD candidates.
Maximize distance from the labeled benchmark table.
Do not rank Track-B candidates by oracle score.
```

Batch quotas:

```text
B = 32: local 12, boundary 10, sparse 6, control 4
B = 16: local 6, boundary 5, sparse 3, control 2
```

If the current Track-A mutation abstain rate exceeds `A_max`, reduce local
quota and allocate the freed labels to boundary/sparse expansion.

Abstain-rate denominator:

```text
deduplicated candidates generated by the current Track-A mutation round
```

## Benchmark table requirements

Required provenance columns:

- `protocol_version`
- `batch_id`
- `source`
- `label_status`
- `retry_count`
- `failure_reason`
- `last_error_digest`

Allowed `source` values are defined in `aicir.qas.vqe_qas_protocol.LabelSource`.
Allowed `label_status` values are defined in
`aicir.qas.vqe_qas_protocol.LabelStatus`.

Only rows with:

```text
label_status = completed
protocol_version = current protocol
```

may be used for oracle training or validation.

Retry policy:

```text
max_retry = 2
running_timeout = wall_time_limit * 1.5
```

After the retry limit, set `label_status = failed_nonretryable`.

## Current implementation entry points

Prepare Stage-0/Stage-1.5 files:

```text
python aicir/qas/demos/vqe_qas_prepare_oracle.py --scales 4,6,8 --initial-labels 96 --output-dir outputs/vqe_qas_oracle_prep
```

This writes:

- `stage0_candidates.csv`
- `stage0_anchors.csv`
- `stage0_coverage_report.md`
- `stage1_5_initial_label_queue.csv`
- `benchmark_table_stub.csv`
- `oracle_prep_metadata.json`

Run fair labels from the queue:

```text
python aicir/qas/demos/vqe_qas_run_fair_labels.py --queue outputs/vqe_qas_oracle_prep/stage1_5_initial_label_queue.csv --output outputs/vqe_qas_oracle_prep/benchmark_table_v2.csv
```

For real labels, keep `--protocol aicir/qas/configs/fair_vqe_protocol_v2.json`,
`--backend numpy`, and `--dtype complex128`. `--max-evals` is for smoke tests
only; using it creates non-final labels unless the protocol version is changed.

Calibrate the first retrieval oracle:

```text
python aicir/qas/demos/vqe_qas_calibrate_oracle.py --benchmark-table outputs/vqe_qas_oracle_prep/benchmark_table_v2.csv --output outputs/vqe_qas_oracle_prep/oracle_calibration_v2.json --markdown-output outputs/vqe_qas_oracle_prep/trust_region_calibration_v2.md
```

The markdown report records boundary holdout counts by qubit scale and flags
low-confidence scale-specific boundary calibration, for example `4q: 2`.

Plan the next fair-label batch:

```text
python aicir/qas/demos/vqe_qas_plan_next_batch.py --candidates outputs/vqe_qas_oracle_prep/stage0_candidates.csv --benchmark-table outputs/vqe_qas_oracle_prep/benchmark_table_v2.csv --output outputs/vqe_qas_oracle_prep/next_batch_v2.csv --summary outputs/vqe_qas_oracle_prep/next_batch_v2_summary.json --batch-id batch_v2
```

If completed labels are fewer than `k_min`, the planner disables Track-A oracle
ranking and reallocates local quota to Track-B expansion/control.
