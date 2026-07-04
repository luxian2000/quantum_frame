#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${PYTHON:-python}" "$SCRIPT_DIR/run_multi_card_npu.py" "$@"
