#!/bin/bash

# Usage:
#   bash maint/scripts/run_local_ci_test.sh [DEVICE]
#   or
#   bash maint/scripts/run_local_ci_test.sh --device <cuda|hip|cpu|metal>
#
# What it does:
#   - Runs project tests (examples/ and testing/python/) with pytest-xdist.
#   - Loads the CUDA-aware scheduler plugin (-p pytest_cuda_scheduler).
#   - When DEVICE=cuda (default): auto-detects available GPUs (n), selects
#     ceil(n/2) GPUs, and runs W workers per selected GPU (default W=4), i.e.
#     total workers = W * ceil(n/2). Uses the pytest CUDA scheduler plugin.
#   - When DEVICE is hip/cpu/metal: no device auto-detection is performed;
#     runs pytest without the CUDA plugin. Parallelism can be controlled via
#     $PYTEST_XDIST_WORKERS (default 4 if unset).
#
# GPU detection precedence:
#   1) $PYTEST_XDIST_CUDA_DEVICES (comma-separated, e.g. "0,1,2,3")
#   2) $CUDA_VISIBLE_DEVICES
#   3) torch.cuda.device_count() (if torch is available)
#   4) nvidia-smi -L
#
# Environment variables:
#   - PYTEST_XDIST_CUDA_DEVICES: explicit device list to use; if not set and we
#     detect GPUs via torch/nvidia-smi, the script exports this variable.
#   - CUDA_VISIBLE_DEVICES: if set, used as the device list for worker count.
#   - PYTEST_XDIST_CUDA_WORKERS_PER_DEVICE: W (workers per selected GPU), default 4.
#   - PYTEST_XDIST_WORKERS: total workers for non-CUDA runs (hip/cpu/metal), default 4.
#
# Examples:
#   - Use all visible GPUs, auto workers:  bash maint/scripts/run_local_ci_test.sh
#   - Limit to subset of GPUs:             PYTEST_XDIST_CUDA_DEVICES=0,2 bash maint/scripts/run_local_ci_test.sh
#   - Respect an existing visibility:      CUDA_VISIBLE_DEVICES=0,1 bash maint/scripts/run_local_ci_test.sh
#   - Increase workers per GPU:            PYTEST_XDIST_CUDA_WORKERS_PER_DEVICE=8 bash maint/scripts/run_local_ci_test.sh
#
# Requirements:
#   - pytest, pytest-xdist
#   - torch (optional) or nvidia-smi (optional) for auto-detection
#   - NVIDIA drivers and CUDA-capable GPUs for GPU tests

# Parse args
DEVICE="cuda"
if [[ $# -ge 1 ]]; then
  case "$1" in
    --device)
      shift
      DEVICE="${1:-cuda}"
      shift || true
      ;;
    --device=*)
      DEVICE="${1#*=}"
      shift
      ;;
    cuda|hip|cpu|metal)
      DEVICE="$1"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [DEVICE] | --device <cuda|hip|cpu|metal>"; exit 0;;
    *)
      echo "[WARN] Unrecognized arg '$1'; treating as DEVICE." >&2
      DEVICE="$1"; shift;;
  esac
fi

# Normalize DEVICE to lowercase
DEVICE=$(echo "$DEVICE" | tr 'A-Z' 'a-z')
case "$DEVICE" in
  cuda|hip|cpu|metal) ;;
  *) echo "[ERROR] Unsupported DEVICE='$DEVICE'. Choose cuda|hip|cpu|metal." >&2; exit 2;;
esac

# Set ROOT_DIR to the project root (two levels up from this script's directory)
ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)

# Change to the project root directory for local testing of changes
cd "$ROOT_DIR" || exit 1

# Add the project root and plugin directory to PYTHONPATH so Python can find local modules and the pytest plugin
export PYTHONPATH=$ROOT_DIR:$ROOT_DIR/maint/scripts:$PYTHONPATH

# Decide worker count automatically based on selected DEVICE.
# - For cuda: use ceil(num_gpus/2) GPUs, with W workers per GPU (default 4).
# - For hip/cpu/metal: no device detection; use $PYTEST_XDIST_WORKERS or 4.

detect_device_list() {
  # Priority: PYTEST_XDIST_CUDA_DEVICES > CUDA_VISIBLE_DEVICES > torch > nvidia-smi
  if [[ -n "$PYTEST_XDIST_CUDA_DEVICES" ]]; then
    echo "$PYTEST_XDIST_CUDA_DEVICES"
    return
  fi
  if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    echo "$CUDA_VISIBLE_DEVICES"
    return
  fi

  # Try torch
  local torch_cnt
  torch_cnt=$(python - <<'PY' 2>/dev/null
try:
    import torch
    print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
except Exception:
    print(-1)
PY
  )

  if [[ "$torch_cnt" =~ ^[0-9]+$ ]] && [[ $torch_cnt -ge 1 ]]; then
    # Build list 0..torch_cnt-1
    local lst=""
    for ((i=0;i<torch_cnt;i++)); do
      if [[ -z "$lst" ]]; then lst="$i"; else lst="$lst,$i"; fi
    done
    echo "$lst"
    return
  fi

  # Fallback to nvidia-smi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local smi_cnt
    smi_cnt=$(nvidia-smi -L 2>/dev/null | wc -l | awk '{print $1}')
    if [[ "$smi_cnt" =~ ^[0-9]+$ ]] && [[ $smi_cnt -ge 1 ]]; then
      local lst=""
      for ((i=0;i<smi_cnt;i++)); do
        if [[ -z "$lst" ]]; then lst="$i"; else lst="$lst,$i"; fi
      done
      echo "$lst"
      return
    fi
  fi

  echo ""  # no devices detected
}

compute_workers_from_devices() {
  local devlist="$1"
  local workers_per_device="$2"
  # remove whitespace
  devlist=$(echo "$devlist" | tr -d ' ')
  if [[ -z "$devlist" ]]; then
    echo 0
    return
  fi
  local n
  n=$(awk -F, '{print NF}' <<< "$devlist")
  local half
  half=$(( (n + 1) / 2 ))
  echo $(( half * workers_per_device ))
}

# Prepare pytest args and worker counts depending on DEVICE
PYTEST_ARGS_COMMON=(--verbose --color=yes --durations=0 --showlocals --cache-clear)
if [[ "$DEVICE" == "cuda" ]]; then
  DEVLIST=$(detect_device_list)

  # If we had to discover via torch or nvidia-smi (no env set), export PYTEST_XDIST_CUDA_DEVICES
  if [[ -z "$PYTEST_XDIST_CUDA_DEVICES" && -z "$CUDA_VISIBLE_DEVICES" && -n "$DEVLIST" ]]; then
    export PYTEST_XDIST_CUDA_DEVICES="$DEVLIST"
  fi

  # Determine workers per device (sync with plugin via env var)
  WORKERS_PER_DEVICE_DEFAULT=4
  WORKERS_PER_DEVICE=${PYTEST_XDIST_CUDA_WORKERS_PER_DEVICE:-$WORKERS_PER_DEVICE_DEFAULT}
  if ! [[ "$WORKERS_PER_DEVICE" =~ ^[0-9]+$ ]] || [[ "$WORKERS_PER_DEVICE" -le 0 ]]; then
    WORKERS_PER_DEVICE=$WORKERS_PER_DEVICE_DEFAULT
  fi
  export PYTEST_XDIST_CUDA_WORKERS_PER_DEVICE="$WORKERS_PER_DEVICE"

  NWORKERS=$(compute_workers_from_devices "$DEVLIST" "$WORKERS_PER_DEVICE")
  if [[ "$NWORKERS" -le 0 ]]; then
    echo "[ERROR] No CUDA devices detected. Cannot run GPU tests with pytest_cuda_scheduler." >&2
    exit 1
  fi
  echo "[INFO] DEVICE=cuda; devices: ${DEVLIST:-none}. Workers: $NWORKERS ($WORKERS_PER_DEVICE per ceil(n/2))."
  PYTEST_ARGS_DEVICE=(-p pytest_cuda_scheduler -n "$NWORKERS")
else
  # Non-CUDA: do not auto-detect devices; do not load CUDA plugin
  NWORKERS_NONCUDA=${PYTEST_XDIST_WORKERS:-4}
  if ! [[ "$NWORKERS_NONCUDA" =~ ^[0-9]+$ ]] || [[ "$NWORKERS_NONCUDA" -le 0 ]]; then
    NWORKERS_NONCUDA=4
  fi
  echo "[INFO] DEVICE=$DEVICE; running without CUDA plugin. Workers: $NWORKERS_NONCUDA."
  PYTEST_ARGS_DEVICE=(-n "$NWORKERS_NONCUDA")
fi

# Run pytest in parallel for all tests in the examples directory
cd examples || exit 1
python -m pytest "${PYTEST_ARGS_DEVICE[@]}" . "${PYTEST_ARGS_COMMON[@]}"
cd .. || exit 1

# Run pytest in parallel for all tests in the testing/python directory.
# CuTeDSL backend now defaults to GEMM v2 (WGMMA descriptor-based).
# Set TILELANG_USE_GEMM_V1=1 only if you need to debug V1-specific issues.
cd testing/python || exit 1
python -m pytest "${PYTEST_ARGS_DEVICE[@]}" . "${PYTEST_ARGS_COMMON[@]}"
