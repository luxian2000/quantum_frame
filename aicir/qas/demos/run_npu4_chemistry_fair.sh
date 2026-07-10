#!/usr/bin/env bash
set -euo pipefail

# Four-NPU P0 chemistry-excitation fair-label run.
# Usage:
#   HAM_PATH=/path/to/lih.json ACTIVE_ELECTRONS=4 ACTIVE_SPATIAL_ORBITALS=6 bash demos/run_npu4_chemistry_fair.sh
# or:
#   ACTIVE_ELECTRONS=2 ACTIVE_SPATIAL_ORBITALS=2 bash demos/run_npu4_chemistry_fair.sh /path/to/h2.json

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

PYTHON=${PYTHON:-python}
HAM_PATH=${1:-${HAM_PATH:-}}
OUT_DIR=${OUT_DIR:-outputs/npu4_chemistry_excitation_fair}
ACTIVE_ELECTRONS=${ACTIVE_ELECTRONS:-2}
ACTIVE_SPATIAL_ORBITALS=${ACTIVE_SPATIAL_ORBITALS:-2}
CHEMISTRY_EXCITATION_COUNT=${CHEMISTRY_EXCITATION_COUNT:-32}
CHEMISTRY_EXCITATION_MAX_EXCITATIONS=${CHEMISTRY_EXCITATION_MAX_EXCITATIONS:-4}
CHEMISTRY_EXCITATION_SEED=${CHEMISTRY_EXCITATION_SEED:-17}
NUM_SHARDS=${NUM_SHARDS:-4}
DEVICE_OFFSET=${DEVICE_OFFSET:-0}
BACKEND=${BACKEND:-npu}
DTYPE=${DTYPE:-complex64}
LABEL_SEED=${LABEL_SEED:-5200}
LABEL_N_SEEDS=${LABEL_N_SEEDS:-3}
FAIR_MAX_EVALS=${FAIR_MAX_EVALS:-1000}
SUCCESS_DELTA_REF=${SUCCESS_DELTA_REF:-0.02}
PROTOCOL=${PROTOCOL:-default}

if [[ -z "$HAM_PATH" ]]; then
  echo "HAM_PATH or the first positional argument is required." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

export OUT_DIR HAM_PATH ACTIVE_ELECTRONS ACTIVE_SPATIAL_ORBITALS
export CHEMISTRY_EXCITATION_COUNT CHEMISTRY_EXCITATION_MAX_EXCITATIONS CHEMISTRY_EXCITATION_SEED

"$PYTHON" - <<'PY'
import os
from pathlib import Path

from aicir.chemistry.spec import load_hamiltonian_input
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
    chemistry_excitation_count=int(os.environ["CHEMISTRY_EXCITATION_COUNT"]),
    chemistry_excitation_max_excitations=int(os.environ["CHEMISTRY_EXCITATION_MAX_EXCITATIONS"]),
    chemistry_excitation_seed=int(os.environ["CHEMISTRY_EXCITATION_SEED"]),
)
queue_path, oracle_path, summary_path = write_chemistry_excitation_bootstrap_queue(config, output_dir=output_dir)
print(f"queue={queue_path}")
print(f"oracle_records={oracle_path}")
print(f"summary={summary_path}")
PY

QUEUE="$OUT_DIR/chemistry_excitation_bootstrap_queue.csv"
LABELS="$OUT_DIR/benchmark_table_chemistry_excitation_npu4.csv"
WORK_DIR="$OUT_DIR/npu4_shards"
SUMMARY="$OUT_DIR/npu4_shard_summary.json"

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

echo "labels=$LABELS"
echo "shard_summary=$SUMMARY"
