#!/usr/bin/env bash
set -euo pipefail

export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

python aicir/qas/demos/vqe_tfim_v3_scaling.py \
  --mode all \
  --scales 4,6,8 \
  --backend npu \
  --no-fallback-to-cpu \
  --dtype complex128 \
  --fair-n-starts 1 \
  --fair-evals-per-param 200 \
  --fair-min-evals 1000 \
  --fair-max-evals 1000000 \
  --init-mode random_uniform_pi \
  --output-dir outputs/vqe_tfim_v3_scaling_npu_full
