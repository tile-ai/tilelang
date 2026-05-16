#!/usr/bin/env bash
# Profile a gqa_fwd_bshd.py variant with Nsight Compute.
#
# Usage:
#   ./profile.sh [-v variant] [-s seq_len] [-o out_name] [-- extra ncu args]
#
# Defaults:
#   variant  = wasp
#   seq_len  = 1024
#   out_name = fa_<variant>_seq<seq_len>
#
# Examples:
#   ./profile.sh                              # wasp, 1024
#   ./profile.sh -v ts2 -s 4096               # ts2, 4096
#   ./profile.sh -v fa4 -s 2048 -o fa4_2k     # named report
#   ./profile.sh -v wasp -- --launch-skip 50  # forward args to ncu
#
# Requires sudo (for PMU access). TMPDIR is overridden because
# /tmp/tvm-debug-mode-tempdirs is locked down on this host.

set -euo pipefail

VARIANT=wasp
SEQ_LEN=1024
OUT_NAME=""

usage() {
  sed -n '2,18p' "$0" >&2
  exit 1
}

while getopts ":v:s:o:h" opt; do
  case "$opt" in
    v) VARIANT="$OPTARG" ;;
    s) SEQ_LEN="$OPTARG" ;;
    o) OUT_NAME="$OPTARG" ;;
    h) usage ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; usage ;;
  esac
done
shift $((OPTIND-1))

case "$VARIANT" in
  ss|ts|ts2|fa4|wasp) ;;
  *) echo "Invalid variant '$VARIANT'. Choose: ss | ts | ts2 | fa4 | wasp" >&2; exit 1 ;;
esac

case "$SEQ_LEN" in
  ''|*[!0-9]*) echo "seq_len must be a positive integer, got '$SEQ_LEN'" >&2; exit 1 ;;
esac

OUT_NAME=${OUT_NAME:-fa_${VARIANT}_seq${SEQ_LEN}}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=/home/yu.cheng/miniconda3/envs/tilelang/bin/python
TMP_DIR=/home/yu.cheng/tmp_tilelang
OUT_PATH="${SCRIPT_DIR}/${OUT_NAME}"

mkdir -p "${TMP_DIR}"

# Re-export the conda env's PATH so ncu can find python's deps when launched via sudo.
export PATH="/home/yu.cheng/miniconda3/envs/tilelang/bin:${PATH}"

echo "Profiling: variant=${VARIANT}, seq_len=${SEQ_LEN}"
echo "Report:    ${OUT_PATH}.ncu-rep"

sudo -E env \
  PATH="${PATH}" \
  PYTHONPATH="${PYTHONPATH:-}" \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  TMPDIR="${TMP_DIR}" \
  ncu --set full \
      -k regex:"main_kernel" \
      --launch-count 1 \
      --launch-skip 10 \
      --target-processes application-only \
      --cache-control none \
      --clock-control none \
      --apply-rules yes \
      --import-source yes \
      --check-exit-code yes \
      "$@" \
      -f -o "${OUT_PATH}" \
      "${PYTHON_BIN}" "${SCRIPT_DIR}/gqa_fwd_bshd.py" \
        --variant "${VARIANT}" --seq_len "${SEQ_LEN}"

echo "Done. Report written to: ${OUT_PATH}.ncu-rep"
