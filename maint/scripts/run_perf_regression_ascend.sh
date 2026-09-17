#!/bin/bash
# Ascend performance regression test: compare current branch vs upstream/asc
#
# Usage:
#   ./maint/scripts/run_perf_regression_ascend.sh
#
# Environment variables:
#   BASELINE_URL    - remote URL to fetch the baseline from
#                     (default: https://github.com/tile-ai/tilelang-deepseek.git)
#   BASELINE_BRANCH - branch on BASELINE_URL to compare against (default: asc)
#   WORK_DIR        - Directory outside the checkout for snapshots and results
#                     (default: unique directory under RUNNER_TEMP, TMPDIR or /tmp)
#   SKIP_BUILD      - Set to 1 to skip the ninja rebuild between checkouts
#   NINJA_JOBS      - Parallelism for ninja (default: 64)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"

BASELINE_URL="${BASELINE_URL:-https://github.com/tile-ai/tilelang-deepseek.git}"
BASELINE_BRANCH="${BASELINE_BRANCH:-asc}"
NINJA_JOBS="${NINJA_JOBS:-64}"

cd "${REPO_ROOT}"
# Keep snapshots outside the checkout so stash and ref switches cannot remove them.
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

# The Ascend examples plus their maintenance driver form the regression harness. Keep
# both constant from the current branch across refs, so only the rebuilt libtvm.so
# varies. The baseline ref may not contain the driver or all examples, so overlay them
# onto each checkout.
HARNESS_DIR="examples/ascend"
DRIVER_PATH="maint/scripts/ascend_perf_regression.py"

echo "============================================"
echo "Ascend Performance Regression Test"
echo "============================================"
echo "Repo root:    ${REPO_ROOT}"
echo "Work dir:     ${WORK_DIR}"
echo "Baseline:     ${BASELINE_URL}@${BASELINE_BRANCH}"
echo ""

# Snapshot the harness dir from the current working tree FIRST (before any stash),
# so the overlay reflects the in-flight edits. We overlay it onto each checkout.
HARNESS_SNAP="${WORK_DIR}/harness_snapshot"
DRIVER_SNAP="${WORK_DIR}/ascend_perf_regression.py"
rm -rf "${HARNESS_SNAP}"
mkdir -p "${HARNESS_SNAP}"
cp -a "${REPO_ROOT}/${HARNESS_DIR}/." "${HARNESS_SNAP}/"
cp -a "${REPO_ROOT}/${DRIVER_PATH}" "${DRIVER_SNAP}"

restore_harness() {
    cp -a "${HARNESS_SNAP}/." "${REPO_ROOT}/${HARNESS_DIR}/"
    cp -a "${DRIVER_SNAP}" "${REPO_ROOT}/${DRIVER_PATH}"
}

# ---- Save current state ----
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Uncommitted changes detected; stashing them:"
    git status --porcelain
    STASHED=1
    git stash push -u -m "perf_regression_ascend_temp_stash"
else
    STASHED=0
fi

CURRENT_REF="$(git rev-parse --abbrev-ref HEAD)"
[[ "${CURRENT_REF}" == "HEAD" ]] && CURRENT_REF="$(git rev-parse HEAD)"
echo "Current ref: ${CURRENT_REF}"

cleanup() {
    echo ""
    echo "Cleaning up: restoring ${CURRENT_REF}..."
    cd "${REPO_ROOT}"
    git checkout -f "${CURRENT_REF}" 2>/dev/null || true
    # An in-flight change may add the maintenance driver on top of a ref that does not
    # track it yet. Remove the overlay before restoring that change from the stash.
    if ! git cat-file -e "${CURRENT_REF}:${DRIVER_PATH}" 2>/dev/null; then
        rm -f "${REPO_ROOT}/${DRIVER_PATH}"
    fi
    git submodule update --init --recursive 2>/dev/null || true
    if [[ "${STASHED}" == "1" ]]; then
        echo "Restoring stashed changes..."
        git stash pop || true
    fi
}
trap cleanup EXIT

# Resolve the baseline into a concrete commit, without depending on any named remote:
# fetch the branch directly from the URL into FETCH_HEAD and pin its SHA.
echo "Fetching baseline ${BASELINE_BRANCH} from ${BASELINE_URL} ..."
git fetch --no-tags "${BASELINE_URL}" "${BASELINE_BRANCH}"
BASELINE="$(git rev-parse --verify FETCH_HEAD)"
echo "Baseline: ${BASELINE_URL}@${BASELINE_BRANCH} -> ${BASELINE}"

build() {
    if [[ "${SKIP_BUILD}" == "1" ]]; then
        echo "Skipping build (SKIP_BUILD=1)"
        return
    fi
    echo "Building libtvm.so (ninja -j${NINJA_JOBS})..."
    USE_CUDA=OFF USE_ASCEND=ON cmake -B build -S . -GNinja >/dev/null
    ninja -j"${NINJA_JOBS}" -C build
}

# Run the Ascend regression suite and capture its {name: latency} JSON marker.
run_driver() {
    local out_json="$1"
    PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/examples/ascend:$PYTHONPATH" \
        ASCEND_NPU_ARCH=dav-3510 TILELANG_DISABLE_CACHE=1 TL_PERF_REGRESSION_FORMAT=json \
        python "${REPO_ROOT}/${DRIVER_PATH}" \
        | tee /dev/stderr | grep "^${MARKER}" | tail -1 | sed "s/^${MARKER}//" > "${out_json}"
}

# ---- Baseline ----
echo ""
echo "===== Baseline (${BASELINE}) ====="
git checkout -f "${BASELINE}"
git submodule update --init --recursive
build
restore_harness   # overlay current-branch harness onto baseline tree
run_driver "${OLD_JSON}"

# ---- Current ----
echo ""
echo "===== Current (${CURRENT_REF}) ====="
git checkout -f "${CURRENT_REF}"
git submodule update --init --recursive
build
restore_harness
run_driver "${NEW_JSON}"

# ---- Compare ----
echo ""
echo "===== Results ====="
# Inlined comparison: read both {name: latency} dicts and print a markdown speedup table
# over the UNION of kernels (speedup = old / new, >1.0 means current is faster). A kernel
# that failed to run on one side shows "-" there, so failures stay visible instead of
# silently dropping out of the report.
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
