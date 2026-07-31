#!/usr/bin/env sh
set -eu

NPROC=2
SECTION=all
OUTPUT_JSON=

usage() {
    echo "usage: $0 --nproc-per-node {2|4|8} --section SECTION --output-json PATH" >&2
    exit 2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --nproc-per-node)
            [ "$#" -ge 2 ] || usage
            NPROC=$2
            shift 2
            ;;
        --section)
            [ "$#" -ge 2 ] || usage
            SECTION=$2
            shift 2
            ;;
        --output-json)
            [ "$#" -ge 2 ] || usage
            OUTPUT_JSON=$2
            shift 2
            ;;
        --devices)
            echo "--devices is not supported; torchrun LOCAL_RANK selects npu:LOCAL_RANK" >&2
            exit 2
            ;;
        *)
            usage
            ;;
    esac
done

case "$NPROC" in
    2|4|8) ;;
    *)
        echo "--nproc-per-node must be one of 2, 4, or 8" >&2
        exit 2
        ;;
esac

[ -n "$OUTPUT_JSON" ] || usage

PYTHONPATH=.:${PYTHONPATH:-} torchrun \
    --nproc-per-node="${NPROC}" \
    scripts/npu/distributed_autograd_probe.py \
    --section "${SECTION}" \
    --output-json "${OUTPUT_JSON}"
