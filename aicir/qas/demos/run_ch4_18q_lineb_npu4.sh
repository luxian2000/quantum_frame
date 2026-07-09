#!/usr/bin/env bash
set -euo pipefail

# CH4 18q LineB chemistry-excitation QAS run for a 4-NPU node.
#
# Required:
#   HAM_PATH=/path/to/ch4_18q_terms_or_spec.json bash demos/run_ch4_18q_lineb_npu4.sh
#
# Important defaults target original 18q CH4-style full spin-orbital space:
#   ACTIVE_ELECTRONS=10
#   ACTIVE_SPATIAL_ORBITALS=9
# Override them if the Hamiltonian uses a different active space.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

PYTHON=${PYTHON:-python}
HAM_PATH=${1:-${HAM_PATH:-}}
OUT_DIR=${OUT_DIR:-outputs/ch4_18q_lineb_npu4}

ACTIVE_ELECTRONS=${ACTIVE_ELECTRONS:-10}
ACTIVE_SPATIAL_ORBITALS=${ACTIVE_SPATIAL_ORBITALS:-9}

# P0 seeds: HF/excitation bootstrap. Keep this modest; P1 does the iterative growth.
P0_CANDIDATES=${P0_CANDIDATES:-64}
P0_MAX_EXCITATIONS=${P0_MAX_EXCITATIONS:-6}
P0_SEED=${P0_SEED:-17}

# LineB P1 defaults for 18q: more ADAPT than genetic, wider than smoke defaults.
ROUNDS=${ROUNDS:-6}
PARENT_COUNT=${PARENT_COUNT:-8}
CHILDREN_PER_PARENT=${CHILDREN_PER_PARENT:-16}
FAIR_TOP_K=${FAIR_TOP_K:-8}
CHEMISTRY_GENETIC_WEIGHT=${CHEMISTRY_GENETIC_WEIGHT:-0.3}
CHEMISTRY_ADAPT_GROWTH_WEIGHT=${CHEMISTRY_ADAPT_GROWTH_WEIGHT:-0.7}
CHEMISTRY_GROWTH_MODE=${CHEMISTRY_GROWTH_MODE:-mixed}
CHEMISTRY_ADAPT_APPEND_K=${CHEMISTRY_ADAPT_APPEND_K:-4}
CHEMISTRY_ADAPT_POOL_LIMIT=${CHEMISTRY_ADAPT_POOL_LIMIT:-24}
CHEMISTRY_MAX_EXCITATIONS=${CHEMISTRY_MAX_EXCITATIONS:-32}
MUTATION_TYPES=${MUTATION_TYPES:-chemistry_insert,chemistry_change,chemistry_swap,chemistry_delete,chemistry_adapt_growth}
SELECTOR=${SELECTOR:-e2}
BASELINE_SELECTORS=${BASELINE_SELECTORS:-E2}

EARLY_STOP_EPSILON=${EARLY_STOP_EPSILON:-1e-4}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-3}
MAX_TOTAL_FAIR_CALLS=${MAX_TOTAL_FAIR_CALLS:-100}

NUM_SHARDS=${NUM_SHARDS:-4}
DEVICE_OFFSET=${DEVICE_OFFSET:-0}
BACKEND=${BACKEND:-npu}
DTYPE=${DTYPE:-complex64}
LABEL_SEED=${LABEL_SEED:-5200}
LABEL_N_SEEDS=${LABEL_N_SEEDS:-3}
FAIR_MAX_EVALS=${FAIR_MAX_EVALS:-1000}
SUCCESS_DELTA_REF=${SUCCESS_DELTA_REF:-0.02}
PROTOCOL=${PROTOCOL:-default}
REFERENCE_ENERGY=${REFERENCE_ENERGY:-}

# Cheap selector settings. E2 is a selector only; final comparison uses fair_best_energy.
LIGHT_EVALUATOR=${LIGHT_EVALUATOR:-torch_pauli}
DEVICE=${DEVICE:-cpu}
E2_MAX_EVALS=${E2_MAX_EVALS:-250}
E1_MAX_EVALS=${E1_MAX_EVALS:-20}

if [[ -z "$HAM_PATH" ]]; then
  echo "HAM_PATH or first positional argument is required." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

P0_QUEUE="$OUT_DIR/chemistry_excitation_bootstrap_queue.csv"
P0_LABELS="$OUT_DIR/benchmark_table_p0_chemistry_excitation_npu4.csv"
P0_WORK_DIR="$OUT_DIR/p0_npu4_shards"
P0_SUMMARY="$OUT_DIR/p0_npu4_shard_summary.json"
CURRENT_LABELS="$OUT_DIR/current_labeled_rows.csv"

export OUT_DIR HAM_PATH ACTIVE_ELECTRONS ACTIVE_SPATIAL_ORBITALS P0_CANDIDATES P0_MAX_EXCITATIONS P0_SEED REFERENCE_ENERGY

"$PYTHON" - <<'PY'
import os
from pathlib import Path

from aicir.chemistry.spec import load_hamiltonian_input
from aicir.qas.vqe_loop.benchmark_table import BENCHMARK_TABLE_FIELDS, read_csv_rows, write_csv_rows
from aicir.qas.vqe_loop.p0_bootstrap_fair import ClosedLoopConfig, write_chemistry_excitation_bootstrap_queue

hamiltonian = load_hamiltonian_input(os.environ["HAM_PATH"])
output_dir = Path(os.environ["OUT_DIR"])
config = ClosedLoopConfig(
    output_dir=output_dir,
    n_qubits=int(hamiltonian.n_qubits),
    hamiltonian_terms=hamiltonian.terms,
    hamiltonian_id=str(hamiltonian.hamiltonian_id),
    hamiltonian_class=str(hamiltonian.hamiltonian_class),
    rounds=0,
    initial_labels=0,
    use_chemistry_excitation_pool=True,
    active_electrons=int(os.environ["ACTIVE_ELECTRONS"]),
    active_spatial_orbitals=int(os.environ["ACTIVE_SPATIAL_ORBITALS"]),
    chemistry_excitation_count=int(os.environ["P0_CANDIDATES"]),
    chemistry_excitation_max_excitations=int(os.environ["P0_MAX_EXCITATIONS"]),
    chemistry_excitation_seed=int(os.environ["P0_SEED"]),
)
queue_path, oracle_path, summary_path = write_chemistry_excitation_bootstrap_queue(config, output_dir=output_dir)

reference_energy = os.environ.get("REFERENCE_ENERGY", "").strip()
metadata = dict(getattr(hamiltonian, "metadata", {}) or {})
if not reference_energy:
    for key in (
        "reference_energy",
        "electronic_reference_energy",
        "electronic_reference_energy_old_thread",
        "fci_energy",
        "exact_energy",
    ):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            reference_energy = str(value).strip()
            break
if reference_energy:
    reference_energy = f"{float(reference_energy):.12f}"
    for path in (queue_path, oracle_path):
        rows = read_csv_rows(path)
        for row in rows:
            row["reference_energy"] = reference_energy
        write_csv_rows(path, rows, fieldnames=BENCHMARK_TABLE_FIELDS)
    print(f"reference_energy={reference_energy}")
else:
    print("reference_energy=missing; large literal Hamiltonians will fail before exact diagonalization")

print(f"p0_queue={queue_path}")
print(f"p0_oracle_records={oracle_path}")
print(f"p0_summary={summary_path}")
PY

"$PYTHON" -m aicir.qas.vqe_loop.shard_scheduler \
  --queue "$P0_QUEUE" \
  --output "$P0_LABELS" \
  --work-dir "$P0_WORK_DIR" \
  --summary "$P0_SUMMARY" \
  --protocol "$PROTOCOL" \
  --num-shards "$NUM_SHARDS" \
  --device-offset "$DEVICE_OFFSET" \
  --seed "$LABEL_SEED" \
  --n-seeds "$LABEL_N_SEEDS" \
  --success-delta-ref "$SUCCESS_DELTA_REF" \
  --max-evals "$FAIR_MAX_EVALS" \
  --backend "$BACKEND" \
  --dtype "$DTYPE"

cp "$P0_LABELS" "$CURRENT_LABELS"

best_energy=""
plateau_count=0
fair_calls_so_far=0
stop_reason="max_rounds"

for round in $(seq 1 "$ROUNDS"); do
  if [[ "$fair_calls_so_far" -ge "$MAX_TOTAL_FAIR_CALLS" ]]; then
    stop_reason="max_total_fair_calls"
    break
  fi

  ROUND_DIR="$OUT_DIR/p1_round${round}"
  mkdir -p "$ROUND_DIR"

  "$PYTHON" -m aicir.qas.demos.run_p1_round_demo \
    --preset ch4_18q \
    --hamiltonian-file "$HAM_PATH" \
    --bootstrap-labels-csv "$CURRENT_LABELS" \
    --output-dir "$ROUND_DIR" \
    --rounds 1 \
    --parent-count "$PARENT_COUNT" \
    --children-per-parent "$CHILDREN_PER_PARENT" \
    --fair-top-k "$FAIR_TOP_K" \
    --growth-route line_b_chemistry_excitation \
    --chemistry-growth-mode "$CHEMISTRY_GROWTH_MODE" \
    --chemistry-genetic-weight "$CHEMISTRY_GENETIC_WEIGHT" \
    --chemistry-adapt-growth-weight "$CHEMISTRY_ADAPT_GROWTH_WEIGHT" \
    --chemistry-adapt-append-k "$CHEMISTRY_ADAPT_APPEND_K" \
    --chemistry-adapt-pool-limit "$CHEMISTRY_ADAPT_POOL_LIMIT" \
    --max-layers "$CHEMISTRY_MAX_EXCITATIONS" \
    --mutation-types "$MUTATION_TYPES" \
    --selector "$SELECTOR" \
    --baseline-selectors "$BASELINE_SELECTORS" \
    --selection-policy no_regret \
    --light-evaluator "$LIGHT_EVALUATOR" \
    --device "$DEVICE" \
    --e1-max-evals "$E1_MAX_EVALS" \
    --e2-max-evals "$E2_MAX_EVALS" \
    --seed $((P0_SEED + round)) \
    --batch-id "ch4_18q_lineb_r${round}" \
    --disable-training-free-pruning

  QUEUE="$ROUND_DIR/p1_queue.csv"
  LABELS="$ROUND_DIR/labels_p1.csv"
  WORK_DIR="$ROUND_DIR/p1_npu4_shards"
  SUMMARY="$ROUND_DIR/p1_npu4_shard_summary.json"

  "$PYTHON" -m aicir.qas.vqe_loop.shard_scheduler \
    --queue "$QUEUE" \
    --output "$LABELS" \
    --work-dir "$WORK_DIR" \
    --summary "$SUMMARY" \
    --protocol "$PROTOCOL" \
    --num-shards "$NUM_SHARDS" \
    --device-offset "$DEVICE_OFFSET" \
    --seed "$LABEL_SEED" \
    --n-seeds "$LABEL_N_SEEDS" \
    --success-delta-ref "$SUCCESS_DELTA_REF" \
    --max-evals "$FAIR_MAX_EVALS" \
    --backend "$BACKEND" \
    --dtype "$DTYPE"

  export CURRENT_LABELS LABELS OUT_DIR ROUND_DIR EARLY_STOP_EPSILON best_energy plateau_count fair_calls_so_far round
  eval "$($PYTHON - <<'PY'
import csv
import os
from pathlib import Path

from aicir.qas.vqe_loop.benchmark_table import BENCHMARK_TABLE_FIELDS, append_benchmark_rows, read_csv_rows, write_csv_rows, as_float

current_path = Path(os.environ["CURRENT_LABELS"])
labels_path = Path(os.environ["LABELS"])
current = read_csv_rows(current_path)
labels = read_csv_rows(labels_path)
merged = append_benchmark_rows(current, labels)
write_csv_rows(current_path, merged, fieldnames=BENCHMARK_TABLE_FIELDS)

energies = [float(row["fair_best_energy"]) for row in labels if as_float(row.get("fair_best_energy")) is not None]
round_best = min(energies) if energies else None
old_best_raw = os.environ.get("best_energy", "")
old_best = None if old_best_raw == "" else float(old_best_raw)
epsilon = float(os.environ["EARLY_STOP_EPSILON"])
plateau = int(os.environ.get("plateau_count", "0"))
fair_calls = int(os.environ.get("fair_calls_so_far", "0")) + len(energies)
if round_best is not None:
    if old_best is None:
        new_best = round_best
        plateau = 0
    else:
        new_best = min(old_best, round_best)
        improvement = old_best - new_best
        plateau = plateau + 1 if improvement < epsilon else 0
else:
    new_best = old_best
    plateau += 1
best_text = "" if new_best is None else f"{new_best:.12f}"
print(f"best_energy='{best_text}'")
print(f"plateau_count={plateau}")
print(f"fair_calls_so_far={fair_calls}")
print(f"round_best='{'' if round_best is None else f'{round_best:.12f}'}'")
PY
)"

  echo "round=${round} round_best=${round_best:-} best_energy=${best_energy:-} plateau_count=${plateau_count} fair_calls=${fair_calls_so_far}"

  if [[ "$plateau_count" -ge "$EARLY_STOP_PATIENCE" ]]; then
    stop_reason="early_stop_patience"
    break
  fi
done

cat > "$OUT_DIR/run_summary.json" <<EOF
{
  "hamiltonian": "$HAM_PATH",
  "active_electrons": $ACTIVE_ELECTRONS,
  "active_spatial_orbitals": $ACTIVE_SPATIAL_ORBITALS,
  "route": "line_b_chemistry_excitation",
  "selector": "$SELECTOR",
  "chemistry_genetic_weight": $CHEMISTRY_GENETIC_WEIGHT,
  "chemistry_adapt_growth_weight": $CHEMISTRY_ADAPT_GROWTH_WEIGHT,
  "chemistry_adapt_append_k": $CHEMISTRY_ADAPT_APPEND_K,
  "chemistry_adapt_pool_limit": $CHEMISTRY_ADAPT_POOL_LIMIT,
  "chemistry_max_excitations": $CHEMISTRY_MAX_EXCITATIONS,
  "rounds_requested": $ROUNDS,
  "fair_calls_so_far": $fair_calls_so_far,
  "best_energy": "${best_energy:-}",
  "stop_reason": "$stop_reason",
  "current_labeled_rows": "$CURRENT_LABELS"
}
EOF

echo "final_table=$CURRENT_LABELS"
echo "summary=$OUT_DIR/run_summary.json"

