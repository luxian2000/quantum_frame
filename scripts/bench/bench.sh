#!/usr/bin/env bash
# 跨框架基准入口。用法见 scripts/bench/README.md
#
#   scripts/bench/bench.sh --axis parity
#   scripts/bench/bench.sh --axis all --max-qubits 20
#   scripts/bench/bench.sh --archive          # 归档到 docs/evidence/benchmarks/<sha>/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ARCHIVE=0
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --archive) ARCHIVE=1 ;;
    *) ARGS+=("$arg") ;;
  esac
done

# 单线程是跨机器可比的前提；要测多线程扩展性请显式覆盖。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export PYTHONPATH="${PYTHONPATH:-.}"

if [[ "$ARCHIVE" == "1" ]]; then
  SHA="$(git rev-parse HEAD)"
  OUT="docs/evidence/benchmarks/${SHA}/cpu.json"
  mkdir -p "$(dirname "$OUT")"
  ARGS+=(--output-json "$OUT")
  echo "[bench] 归档目标: $OUT"
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "[bench] 警告: 工作区不干净，产出的数字不可复现（清单会记录 worktree_dirty=true）" >&2
  fi
fi

exec python scripts/bench/run_bench.py "${ARGS[@]}"
