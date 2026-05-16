#!/usr/bin/env bash
# Profile Triton's FlashAttention (tutorial 06) on bench_triton_torch.py.
#
# Usage:
#   ./profile_triton.sh [-m model] [-s seq_len] [-d device] [-o out_name] [-- extra ncu args]
#
# Defaults:
#   model    = glm5         (pick a single model to keep the report small)
#   seq_len  = 4096
#   device   = 0
#   out_name = triton_<model>_seq<seq_len>
#
# Examples:
#   ./profile_triton.sh                            # glm5 + S=4096
#   ./profile_triton.sh -m deepseek_v32 -s 8192
#   ./profile_triton.sh -m llama4_maverick -s 2048 -d 7
#
# Requires sudo (for PMU access). TMPDIR is overridden because
# /tmp/tvm-debug-mode-tempdirs is locked down on this host.

set -euo pipefail

MODEL=glm5
SEQ_LEN=4096
DEVICE=0
OUT_NAME=""

usage() { sed -n '2,18p' "$0" >&2; exit 1; }

while getopts ":m:s:d:o:h" opt; do
  case "$opt" in
    m) MODEL="$OPTARG" ;;
    s) SEQ_LEN="$OPTARG" ;;
    d) DEVICE="$OPTARG" ;;
    o) OUT_NAME="$OPTARG" ;;
    h) usage ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; usage ;;
  esac
done
shift $((OPTIND-1))

case "$MODEL" in
  glm5|llama4_maverick|qwen35|deepseek_v32|minimax_m25) ;;
  *) echo "Invalid model '$MODEL'. Choose: glm5 | llama4_maverick | qwen35 | deepseek_v32 | minimax_m25" >&2; exit 1 ;;
esac
case "$SEQ_LEN" in ''|*[!0-9]*) echo "seq_len must be a positive integer, got '$SEQ_LEN'" >&2; exit 1 ;; esac
case "$DEVICE"  in ''|*[!0-9]*) echo "device must be a non-negative integer, got '$DEVICE'" >&2; exit 1 ;; esac

OUT_NAME=${OUT_NAME:-triton_${MODEL}_seq${SEQ_LEN}}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_SCRIPT=/home/yu.cheng/NUMAKE/script/fa4/bench_triton_torch.py
PYTHON_BIN=/home/yu.cheng/miniconda3/envs/tilelang/bin/python
TMP_DIR=/home/yu.cheng/tmp_tilelang
OUT_PATH="${SCRIPT_DIR}/${OUT_NAME}"

mkdir -p "${TMP_DIR}"

# Re-export the conda env's PATH so ncu can find python's deps when launched via sudo.
export PATH="/home/yu.cheng/miniconda3/envs/tilelang/bin:${PATH}"

echo "Profiling: triton FA  model=${MODEL}  seq_len=${SEQ_LEN}  device=${DEVICE}"
echo "Report:    ${OUT_PATH}.ncu-rep"

# Triton's tutorial-06 forward kernel is named `_attn_fwd*` (with various
# suffixes for tma / warp-specialize / persistent variants). We match any
# of them and skip the torch SDPA path with --skip-torch.
sudo -E env \
  PATH="${PATH}" \
  PYTHONPATH="${PYTHONPATH:-}" \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  TMPDIR="${TMP_DIR}" \
  CUDA_VISIBLE_DEVICES="${DEVICE}" \
  ncu --set full \
      -k regex:"_attn_fwd" \
      --launch-count 1 \
      --launch-skip 500 \
      --target-processes application-only \
      --cache-control none \
      --clock-control none \
      --apply-rules yes \
      --import-source yes \
      --check-exit-code yes \
      "$@" \
      -f -o "${OUT_PATH}" \
      "${PYTHON_BIN}" "${BENCH_SCRIPT}" \
        --device 0 \
        --model "${MODEL}" \
        --seqs "${SEQ_LEN}" \
        --warmup 5 --iters 30 \
        --skip-torch

echo "Done. Report written to: ${OUT_PATH}.ncu-rep"
