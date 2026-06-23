# VQE-QAS Closed Loop

`aicir.qas.vqe_loop` implements the closed-loop VQE-QAS workflow:

```text
Stage 0/1 preparation
  -> fair VQE labels
  -> Stage-2 oracle + MoG-EA next-batch search
  -> label feedback
  -> oracle calibration
```

The default user entry point is:

```bash
python -m aicir.qas.vqe_loop --hamiltonian h2_terms.json
```

The Hamiltonian JSON can also be passed positionally:

```bash
python -m aicir.qas.vqe_loop h2_terms.json
```

Defaults are chosen for the current NPU fair-label workflow:
`--rounds auto --batch-size auto --backend npu --dtype complex64`.
If `--output-dir` is omitted, the runner derives one from the Hamiltonian file
name, for example `h2_terms.json` -> `outputs/qas_h2_terms_loop`.

`h2_terms.json` should contain literal Pauli terms:

```json
[
  [-0.09706626816762543, "IIII"],
  [0.17141282644776914, "ZIII"],
  [0.12039548242542646, "XXXX"]
]
```

If `--hamiltonian` is omitted, pass `--n-qubits`; the run then uses the TFIM defaults in `fair_label_protocol.json`.

By default, the one-entry runner uses qubit-scaled budgets:

| qubits | initial labels | per-round labels | max rounds |
| --- | ---: | ---: | ---: |
| `<=4` | 12 | 4 | 4 |
| `5-8` | 24 | 6 | 4 |
| `9-12` | 36 | 8 | 2 |
| `>12` | 48 | 12 | 2 |

The per-round labels are split across local / boundary / sparse / control
tracks.  Users can override either the total with `--batch-size` or individual
quotas with `--local`, `--boundary`, `--sparse`, and `--control`.

The loop also stops early when no new best `fair_best_energy` is found for
`--patience` consecutive rounds.  The default is `--patience 2` and
`--min-improvement 1e-8`.

After each Stage-2 round, the next round's quota mix is adapted from oracle
calibration:

- `local`: used only when `overall=true`, enough in-TR holdout exists, the
  in-TR MAE is clearly better than out-of-TR MAE, and sparse abstain is healthy.
- `balanced_explore`: used when there is a weak TR signal (`tr_in_count > 0`,
  `tr_in_mae < tr_out_mae`, and sparse abstain is healthy), but `overall=false`.
- `explore`: used when the oracle has no reliable TR support, TR error is not
  better than out-of-TR error, or sparse abstain is poor.

Higher-qubit runs cap the local fraction more aggressively. For example, a 12q
run with weak TR signal uses `local=1, boundary=2, sparse=3, control=2` for an
8-label round.

## Main Flow

`python -m aicir.qas.vqe_loop` calls `run_vqe_qas_closed_loop()` in `vqe_qas_loop.py`.

The call order is:

```text
vqe_qas_loop.py
  -> preparation.py
  -> stamp_literal_hamiltonian_terms()
  -> labeling.py
  -> stage2.py  # repeated until max rounds or patience stop
       -> next_batch.py
            -> selection_ops.py
       -> labeling.py
       -> calibration.py
```

Module responsibilities:

- `vqe_qas_loop.py`: one-call Python API and multi-round orchestration.
- `__main__.py`: package command-line entry for `python -m aicir.qas.vqe_loop`.
- `preparation.py`: builds Stage-0 candidates, applies Stage-1 filters/soft flags, and writes the initial fair-label queue.
- `labeling.py`: turns queue rows into fair VQE benchmark labels.
- `stage2.py`: runs one Stage-2 loop: plan next batch, label it, append labels, recalibrate oracle.
- `next_batch.py`: reads candidates and benchmark table, builds the oracle view, and writes the next label queue.
- `selection_ops.py`: pure Stage-2 selection operators: MoG-EA proposals, farthest-first expansion, and abstain-rate summaries.
- `calibration.py`: evaluates trust-region oracle quality on holdout rows.
- `fair_vqe.py`: evaluates and optimizes one architecture on one VQE problem using the frozen fair-label policy.
- `sharding.py`: splits a fair-label queue into independent shards for multi-NPU or multi-process runs.
- `sidecars.py`: converts completed fair labels into supernet warm-start/rank sidecars.
- `supernet_screening.py`: optional supernet screening sidecar generation.
- `protocol.py`: benchmark-table schema, label states, append/merge rules, and retry transitions.
- `geometry.py`: candidate distance, trust-region geometry, split sampling, and Hamiltonian feature distances.
- `fair_label_protocol.json`: frozen data protocol for fair VQE labels.

## Protocol Boundaries

`fair_label_protocol.json` answers:

```text
How is a fair VQE label generated?
```

It freezes the label protocol: unmeasured-state energy evaluation, `shots=null`, optimizer defaults, default TFIM parameters, and required output fields. The file name is stable; the internal `protocol_version` remains the data-compatibility key.

`protocol.py` answers:

```text
How are labels represented, merged, retried, and stored?
```

It defines `BENCHMARK_TABLE_FIELDS`, `LabelStatus`, `LabelSource`, append rules, and failure retry transitions.

`geometry.py` answers:

```text
When are candidates or Hamiltonian tasks close enough for the local oracle to trust itself?
```

It defines candidate records, distance scales, trust-region geometry, Hamiltonian feature extraction, and task-aware distances.

`selection_ops.py` answers:

```text
Which candidates are selected for Track-A local search or Track-B expansion?
```

The default local proposal path is MoG-EA/NSGA-II. Farthest-first is used for boundary/sparse expansion, not as a replacement for MoG-EA.

## Useful Commands

Run an automatic multi-round loop:

```bash
python -m aicir.qas.vqe_loop --hamiltonian lih_hamiltonian.json
```

`--hamiltonian` accepts either a legacy Pauli-term list
`[[coeff, pauli], ...]` or a structured specification:

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

Prepare initial candidates and queue:

```bash
python -m aicir.qas.vqe_loop.preparation \
  --scales 4 \
  --initial-labels 24 \
  --output-dir outputs/qas_4q
```

Run fair labels:

```bash
python -m aicir.qas.vqe_loop.labeling \
  --queue outputs/qas_4q/stage1_5_initial_label_queue.csv \
  --output outputs/qas_4q/benchmark_table_4q_v2.csv \
  --protocol aicir/qas/vqe_loop/fair_label_protocol.json \
  --backend numpy \
  --dtype complex128
```

Run fair labels as independent shards:

```bash
python -m aicir.qas.vqe_loop.sharding \
  --queue outputs/qas_stage2/round1/round1_queue.csv \
  --output outputs/qas_stage2/round1/round1_labels.csv \
  --work-dir outputs/qas_stage2/round1/label_shards \
  --protocol aicir/qas/vqe_loop/fair_label_protocol.json \
  --backend npu \
  --dtype complex64 \
  --num-shards 4
```

Run one Stage-2 round:

```bash
python -m aicir.qas.vqe_loop.stage2 \
  --candidates outputs/qas_4q/stage0_candidates.csv \
  --benchmark-table outputs/qas_4q/benchmark_table_4q_v2.csv \
  --output-dir outputs/qas_4q/round1 \
  --batch-id round1
```

## Notes

- `sharding.py` is a task-parallel queue runner; it does not change the fair-label protocol.
- `sidecars.py` and `supernet_screening.py` are optional helpers. They do not produce final labels.
- `fair_vqe.py` is not a general VQE frontend; it is the fair-label execution layer used by VQE-QAS.
- `next_batch.py` is workflow code. `selection_ops.py` is the small operator library it calls.
