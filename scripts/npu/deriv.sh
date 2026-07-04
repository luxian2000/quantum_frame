#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${PYTHON:-python}" "$SCRIPT_DIR/run_npu_tests.py" --suite deriv "$@"
