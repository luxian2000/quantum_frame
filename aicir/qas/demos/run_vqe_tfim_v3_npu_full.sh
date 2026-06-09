#!/usr/bin/env bash
set -euo pipefail

CARD_COUNT="${1:-1}"
OUTPUT_DIR="${2:-outputs/vqe_tfim_v3_scaling_npu_full}"

export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

mkdir -p "$OUTPUT_DIR"

mapfile -t DETECTED_DEVICES < <(npu-smi info 2>/dev/null | awk '/910B/{print $2}' | head -n "$CARD_COUNT")
if [ "${#DETECTED_DEVICES[@]}" -lt "$CARD_COUNT" ]; then
  DETECTED_DEVICES=()
  for ((index=0; index<CARD_COUNT; index++)); do
    DETECTED_DEVICES+=("$index")
  done
fi

echo "VQE-QAS v3 NPU full run"
echo "card_count=$CARD_COUNT"
echo "devices=${DETECTED_DEVICES[*]}"
echo "output_dir=$OUTPUT_DIR"

python aicir/qas/demos/vqe_tfim_v3_scaling.py \
  --mode reference \
  --scales 4,6,8 \
  --output-dir "$OUTPUT_DIR"

pids=()
for ((shard=0; shard<CARD_COUNT; shard++)); do
  device="${DETECTED_DEVICES[$shard]}"
  shard_log="$OUTPUT_DIR/shard_${shard}_of_${CARD_COUNT}.log"
  echo "starting shard=$shard/$CARD_COUNT on ASCEND_RT_VISIBLE_DEVICES=$device log=$shard_log"
  (
    export ASCEND_RT_VISIBLE_DEVICES="$device"
    python aicir/qas/demos/vqe_tfim_v3_scaling.py \
      --mode enumerate \
      --scales 4,6,8 \
      --backend npu \
      --no-fallback-to-cpu \
      --dtype complex128 \
      --fair-n-starts 1 \
      --fair-evals-per-param 200 \
      --fair-min-evals 1000 \
      --fair-max-evals 1000000 \
      --init-mode random_uniform_pi \
      --num-shards "$CARD_COUNT" \
      --shard-index "$shard" \
      --verbose \
      --output-dir "$OUTPUT_DIR"
  ) > "$shard_log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

python aicir/qas/demos/vqe_tfim_v3_phase1_analysis.py \
  --input-dir "$OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --scales 4,6,8

echo "done. Results in $OUTPUT_DIR"
