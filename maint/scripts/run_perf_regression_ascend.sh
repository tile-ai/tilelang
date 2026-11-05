#!/bin/bash
# Ascend performance regression: compare isolated baseline/current installations.
#
# The current examples and driver are frozen once and run with each environment.
# The source checkouts, compiler libraries and object registries are never mixed.
#
# Usage: ./maint/scripts/run_perf_regression_ascend.sh
#
# Environment variables:
#   BASELINE_URL     Baseline remote (default: https://github.com/tile-ai/tilelang.git)
#   BASELINE_BRANCH  Baseline branch (default: main)
#   BASELINE_SHA     Already-fetched commit; avoids fetching a moving branch
#   CURRENT_LABEL    Label for the current checkout
#   WORK_DIR         Dedicated directory outside the checkout for sources/results
#   SKIP_BUILD       Reuse both installed environments when set to 1
#   NINJA_JOBS       Native build parallelism (default: 4)
#   ASCEND_PYTHON    Python with the matching Torch/torch_npu (default: python)
#   ASCEND_SDK_PYTHONPATH  Optional CANN Python paths, excluding compiler checkouts

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
BASELINE_URL="${BASELINE_URL:-https://github.com/tile-ai/tilelang.git}"
BASELINE_BRANCH="${BASELINE_BRANCH:-main}"
NINJA_JOBS="${NINJA_JOBS:-4}"
SKIP_BUILD="${SKIP_BUILD:-0}"
ASCEND_PYTHON="${ASCEND_PYTHON:-python}"

if ! command -v uv >/dev/null 2>&1; then
    echo "The isolated performance environments require uv." >&2
    exit 1
fi
cd "${REPO_ROOT}"
if [[ -z "${WORK_DIR:-}" ]]; then
    WORK_DIR="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/tilelang-perf-regression-ascend.XXXXXX")"
fi
mkdir -p "${WORK_DIR}"
WORK_DIR="$(cd "${WORK_DIR}" && pwd -P)"
case "${WORK_DIR}/" in
    "${REPO_ROOT}/"*)
        echo "WORK_DIR must be outside the repository: ${WORK_DIR}" >&2
        exit 1
        ;;
esac

OLD_JSON="${WORK_DIR}/old.json"
NEW_JSON="${WORK_DIR}/new.json"
RESULT_MD="${WORK_DIR}/regression_result.md"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "result_md=${RESULT_MD}" >> "${GITHUB_OUTPUT}"
fi
MARKER="__TILELANG_PERF_RESULTS_JSON__="
CURRENT_SHA="$(git rev-parse HEAD)"
CURRENT_LABEL="${CURRENT_LABEL:-$(git symbolic-ref --short -q HEAD || git rev-parse --short HEAD)}"

# Freeze the exact current driver and examples without changing either checkout.
# Its relative examples/ascend path is preserved, so all child workers use it too.
HARNESS_ROOT="${WORK_DIR}/harness"
rm -rf "${HARNESS_ROOT}"
mkdir -p "${HARNESS_ROOT}/examples/ascend" "${HARNESS_ROOT}/maint/scripts"
cp -a "${REPO_ROOT}/examples/ascend/." "${HARNESS_ROOT}/examples/ascend/"
cp -a "${REPO_ROOT}/maint/scripts/ascend_perf_regression.py" "${HARNESS_ROOT}/maint/scripts/"

if [[ -n "${BASELINE_SHA:-}" ]]; then
    BASELINE="$(git rev-parse --verify "${BASELINE_SHA}^{commit}")"
else
    git fetch --no-tags "${BASELINE_URL}" "${BASELINE_BRANCH}"
    BASELINE="$(git rev-parse --verify 'FETCH_HEAD^{commit}')"
fi
BASE_WORKTREE="${WORK_DIR}/baseline-${BASELINE}"
if [[ -e "${BASE_WORKTREE}" ]]; then
    [[ "$(git -C "${BASE_WORKTREE}" rev-parse HEAD)" == "${BASELINE}" ]]
    if [[ -n "$(git -C "${BASE_WORKTREE}" status --porcelain)" ]]; then
        echo "Baseline worktree changed; choose a new WORK_DIR: ${BASE_WORKTREE}" >&2
        exit 1
    fi
else
    git worktree add --detach "${BASE_WORKTREE}" "${BASELINE}"
    git -C "${BASE_WORKTREE}" submodule update --init --recursive
fi

# Only reuse the SDK's tensor packages. Each venv gets its own TileLang and
# compiler/FFI wheels; pin Torch versions so installation cannot replace the
# torch_npu combination prepared by the runner. The base environment is read-only.
RUNTIME_CONSTRAINTS="${WORK_DIR}/ascend-torch-constraints.txt"
if [[ "${SKIP_BUILD}" != "1" ]]; then
"${ASCEND_PYTHON}" - "${RUNTIME_CONSTRAINTS}" <<'PYTHON'
from importlib.metadata import version
from pathlib import Path
import sys
Path(sys.argv[1]).write_text("".join(f"{name}=={version(name)}\n" for name in ("torch", "torch_npu")))
PYTHON
fi

install_checkout() {
    local role="$1" repo="$2" venv="${WORK_DIR}/venvs/$1"
    if [[ "${SKIP_BUILD}" == "1" ]]; then
        if [[ ! -x "${venv}/bin/python" ]]; then
            echo "SKIP_BUILD requires the existing ${role} environment: ${venv}" >&2
            exit 1
        fi
        return
    fi
    uv venv --clear --system-site-packages --python "${ASCEND_PYTHON}" "${venv}"
    (
        export USE_CUDA=OFF USE_ASCEND=ON
        export CMAKE_BUILD_PARALLEL_LEVEL="${NINJA_JOBS}"
        export UV_CONSTRAINT="${RUNTIME_CONSTRAINTS}" PIP_CONSTRAINT="${RUNTIME_CONSTRAINTS}"
        unset PYTHONPATH TVM_LIBRARY_PATH TVM_TIRX_LIBRARY_PATH TVM_FFI_LIBRARY_PATH
        if [[ -f "${repo}/tilelang_packaging/compiler_manifest.py" ]]; then
            local staged="${WORK_DIR}/${role}-stage" artifacts="${WORK_DIR}/${role}-artifacts"
            rm -rf "${staged}" "${artifacts}"
            "${venv}/bin/python" "${repo}/maint/scripts/build_native_artifacts.py" stage \
                --source "${repo}" --destination "${staged}"
            "${venv}/bin/python" "${repo}/maint/scripts/build_native_artifacts.py" build \
                --stage "${staged}" --output "${artifacts}" --python "${venv}/bin/python" \
                --frontend uv --wheel-only --install --jobs "${NINJA_JOBS}"
            uv pip install --python "${venv}/bin/python" --find-links "${artifacts}" -r "${repo}/requirements-test.txt"
            (cd "${WORK_DIR}" && "${venv}/bin/python" "${repo}/maint/scripts/verify_native_install.py" \
                --expected-source "${staged}/manifest.json" --output "${role}-install.json")
        else
            uv pip install --python "${venv}/bin/python" -r "${repo}/requirements-test.txt"
            uv pip install --python "${venv}/bin/python" --reinstall-package tilelang "${repo}"
        fi
    )
}

run_driver() {
    local role="$1" out_json="$2"
    (
        cd "${HARNESS_ROOT}"
        env -u TVM_LIBRARY_PATH -u TVM_TIRX_LIBRARY_PATH -u TVM_FFI_LIBRARY_PATH \
            -u TL_TEMPLATE_PATH -u TL_CUTLASS_PATH \
            PYTHONPATH="${HARNESS_ROOT}/examples/ascend${ASCEND_SDK_PYTHONPATH:+:${ASCEND_SDK_PYTHONPATH}}" \
            PYTHONNOUSERSITE=1 ASCEND_NPU_ARCH="${ASCEND_NPU_ARCH:-dav-3510}" \
            TILELANG_DISABLE_CACHE=1 TL_PERF_REGRESSION_FORMAT=json \
            "${WORK_DIR}/venvs/${role}/bin/python" "${HARNESS_ROOT}/maint/scripts/ascend_perf_regression.py"
    ) | tee "${WORK_DIR}/${role}.log" /dev/stderr | grep "^${MARKER}" | tail -1 | sed "s/^${MARKER}//" > "${out_json}"
}

install_checkout old "${BASE_WORKTREE}"
install_checkout new "${REPO_ROOT}"
run_driver old "${OLD_JSON}"
run_driver new "${NEW_JSON}"

# ---- Compare ----
echo ""
echo "===== Results ====="
# Inlined comparison: read both {name: latency} dicts and print a markdown speedup table
# over the UNION of kernels (speedup = old / new, >1.0 means current is faster). A kernel
# that failed to run on one side shows "-" there, so failures stay visible instead of
# silently dropping out of the report.
BASELINE_LABEL="${BASELINE_URL}@${BASELINE_BRANCH}" BASELINE_SHA="${BASELINE}" \
    CURRENT_LABEL="${CURRENT_LABEL}" CURRENT_SHA="${CURRENT_SHA}" \
    OLD_JSON="${OLD_JSON}" NEW_JSON="${NEW_JSON}" RESULT_MD="${RESULT_MD}" python3 - <<'PY'
import json, os

old = json.load(open(os.environ["OLD_JSON"]))
new = json.load(open(os.environ["NEW_JSON"]))


def fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "-"


rows = []
for k in sorted(set(old) | set(new)):
    o = old.get(k)
    n = new.get(k)
    if o is not None and n is not None and n:
        speedup = o / n
        sort_key = speedup
    else:
        speedup = None  # missing on one side -> not comparable
        sort_key = float("inf")  # park incomparable rows at the end
    rows.append((k, o, n, speedup, sort_key))
rows.sort(key=lambda r: r[4])  # worst regressions first, missing-side rows last

lines = [
    f"Baseline: `{os.environ['BASELINE_LABEL']}` (`{os.environ['BASELINE_SHA']}`)",
    "",
    f"Current: `{os.environ['CURRENT_LABEL']}` (`{os.environ['CURRENT_SHA']}`)",
    "",
    "| Kernel | Baseline (ms) | Current (ms) | Speedup (old/new) |",
    "|---|---|---|---|",
]
for k, o, n, s, _ in rows:
    lines.append(f"| {k} | {fmt(o)} | {fmt(n)} | {fmt(s)} |")
table = "\n".join(lines)
print(table)
open(os.environ["RESULT_MD"], "w").write(table + "\n")
print(f"\nSaved markdown to {os.environ['RESULT_MD']}")
PY
